from __future__ import annotations

import json
import subprocess
import re
from pathlib import Path

from .config import AppConfig
from .file_tree import build_mac_style_tree, collect_python_files, fill_module_paths
from .function_finder import collect_functions, filter_functions_by_module_paths
from .llm_client import LLMClient
from .models import (
    ExtractedPBTSignal,
    FileInfo,
    FunctionInfo,
    GeneratedPBT,
    RetrievalResult,
    RelatedFileSelection,
)
from .prompts import (
    prompt_build_retrieval_query,
    prompt_extract_pbt_signals_from_files,
    prompt_extract_pbt_signals_from_retrieval,
    prompt_generate_pbt,
    prompt_select_related_files,
)
from .vector_store import VectorStore


class PBTOrchestrator:
    """
    High-level orchestrator implementing steps 1–5 of the workflow.
    This is where you can glue together different LLM/vector backends.
    """

    def __init__(
        self,
        config: AppConfig,
        llm_client: LLMClient,
        vector_store: VectorStore | None = None,
    ) -> None:
        self.config = config
        self.llm = llm_client
        self.vector_store = vector_store

    # ---- Step 1 & 2: scan project, build tree, collect functions ----

    def scan_project(self) -> tuple[str, list[FileInfo], list[FunctionInfo]]:
        project_root = self.config.project.project_dir
        tree_str = build_mac_style_tree(
            project_root,
            max_depth=self.config.pbt.max_file_tree_depth,
            max_entries=self.config.pbt.max_file_tree_entries,
        )
        py_files = collect_python_files(project_root)
        fill_module_paths(project_root, py_files)
        functions = collect_functions(project_root)
        return tree_str, py_files, functions

    # ---- Step 3: select related files & extract signals ----

    async def select_related_files_for_function(
        self,
        project_tree: str,
        all_files: list[FileInfo],
        target_fn: FunctionInfo,
    ) -> RelatedFileSelection:
        prompt = prompt_select_related_files(
            target_function=target_fn,
            project_tree=project_tree,
            python_files=all_files,
            top_n=self.config.pbt.top_n_related_files,
        )
        raw = await self.llm.complete(prompt)
        # Do not crash on JSON issues – keep raw output for debugging.
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {"selected_files": []}

        # Map back to FileInfo
        rel_to_file = {str(f.rel_path): f for f in all_files}
        selected_files: list[FileInfo] = []
        for item in data.get("selected_files", []):
            rel = item.get("rel_path")
            if rel and rel in rel_to_file:
                selected_files.append(rel_to_file[rel])

        return RelatedFileSelection(
            target_function=target_fn,
            related_files=selected_files,
            llm_raw_response=raw,
        )

    async def extract_signals_from_files(
        self,
        selection: RelatedFileSelection,
    ) -> ExtractedPBTSignal:
        project_root = self.config.project.project_dir
        # Read contents of selected files
        contents: dict[str, str] = {}
        for f in selection.related_files:
            text = f.path.read_text(encoding="utf-8")
            # Optionally truncate very long files
            if len(text) > self.config.pbt.max_source_snippet_chars:
                text = text[: self.config.pbt.max_source_snippet_chars] + "\n# ... truncated ..."
            contents[str(f.rel_path)] = text

        prompt = prompt_extract_pbt_signals_from_files(
            target_function=selection.target_function,
            related_files_contents=contents,
        )
        raw = await self.llm.complete(prompt)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {}

        tf = selection.target_function
        return ExtractedPBTSignal(
            target_function=tf,
            description=data.get("description", ""),
            invariants=data.get("invariants", []) or [],
            preconditions=data.get("preconditions", []) or [],
            postconditions=data.get("postconditions", []) or [],
            relationships=[r.get("description", "") for r in data.get("relationships", [])],
            examples=data.get("examples", []) or [],
            metadata={"raw_llm_output": raw},
        )

    # ---- Step 4: build retrieval query, call vector store, extract signals ----

    async def build_retrieval_query(self, target_fn: FunctionInfo) -> RetrievalResult:
        prompt = prompt_build_retrieval_query(target_function=target_fn)
        raw = await self.llm.complete(prompt)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {"retrieval_query": ""}

        query = data.get("retrieval_query", "")
        hits = self.vector_store.search(query, top_k=self.config.pbt.top_n_related_files) if self.vector_store else []
        return RetrievalResult(
            target_function=target_fn,
            query=query,
            hits=hits,
            llm_raw_response=raw,
        )

    async def extract_signals_from_retrieval(
        self,
        retrieval: RetrievalResult,
    ) -> ExtractedPBTSignal:
        prompt = prompt_extract_pbt_signals_from_retrieval(
            target_function=retrieval.target_function,
            query=retrieval.query,
            hits=retrieval.hits,
        )
        raw = await self.llm.complete(prompt)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {}

        tf = retrieval.target_function
        return ExtractedPBTSignal(
            target_function=tf,
            description=data.get("description", ""),
            invariants=data.get("invariants", []) or [],
            preconditions=data.get("preconditions", []) or [],
            postconditions=data.get("postconditions", []) or [],
            relationships=[r.get("description", "") for r in data.get("relationships", [])],
            examples=data.get("examples", []) or [],
            metadata={"raw_llm_output": raw},
        )

    # ---- Step 5: generate tests & run pylint ----

    async def generate_pbt_for_function(
        self,
        target_fn: FunctionInfo,
        signals_files: ExtractedPBTSignal,
        signals_retrieval: ExtractedPBTSignal | None,
        index: int,
    ) -> GeneratedPBT:
        # 使用传入的 index 构造唯一的 test 模块路径与文件名，避免重复
        signals_from_files = {
            "description": signals_files.description,
            "invariants": signals_files.invariants,
            "preconditions": signals_files.preconditions,
            "postconditions": signals_files.postconditions,
            "relationships": signals_files.relationships,
            "examples": signals_files.examples,
        }
        signals_from_retrieval = None
        if signals_retrieval is not None:
            signals_from_retrieval = {
                "description": signals_retrieval.description,
                "invariants": signals_retrieval.invariants,
                "preconditions": signals_retrieval.preconditions,
                "postconditions": signals_retrieval.postconditions,
                "relationships": signals_retrieval.relationships,
                "examples": signals_retrieval.examples,
            }

        prompt = prompt_generate_pbt(
            target_function=target_fn,
            signals_from_files=signals_from_files,
            signals_from_retrieval=signals_from_retrieval,
        )
        raw_response = await self.llm.complete(prompt)
        test_code = extract_python_code_from_response(raw_response)

        output_dir = self.config.project.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        test_file_path = output_dir / f"test_pbt_{index}.py"
        test_file_path.write_text(test_code, encoding="utf-8")

        pylint_output = self._run_pylint_on_file(test_file_path)
        return GeneratedPBT(
            target_function=target_fn,
            test_module_path=test_file_path,
            test_code=test_code,
            pylint_output=pylint_output,
        )

    def _run_pylint_on_file(self, path: Path) -> str:
        try:
            result = subprocess.run(
                [
                    "pylint",
                    # 仅报告错误，关闭所有 warning / convention / refactor 等信息
                    "--disable=all",
                    "--enable=E",
                    str(path),
                ],
                capture_output=True,
                text=True,
                check=False,
                # 确保在被测项目根目录下运行，以避免 import 错误
                cwd=str(self.config.project.project_dir),
            )
            return result.stdout + "\n" + result.stderr
        except FileNotFoundError:
            return "pylint not found in PATH. Please install pylint to enable static checks."


def extract_python_code_from_response(response: str) -> str:
    if not response:
        return ""

    python_block_pattern = re.compile(
        r"```python\s*(?P<code>.+?)```",
        re.DOTALL | re.IGNORECASE,
    )
    match = python_block_pattern.search(response)
    if match:
        return match.group("code").strip()

    generic_block_pattern = re.compile(
        r"```\s*(?P<code>.+?)```",
        re.DOTALL,
    )
    match = generic_block_pattern.search(response)
    if match:
        return match.group("code").strip()

    return response.strip()



