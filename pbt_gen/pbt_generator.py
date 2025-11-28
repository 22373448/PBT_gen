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
    prompt_fix_test,
    prompt_generate_pbt,
    prompt_judge_function_bug,
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

        # 收集所有properties，合并来自files和retrieval的信号
        all_invariants = list(set(signals_files.invariants + (signals_retrieval.invariants if signals_retrieval else [])))
        all_postconditions = list(set(signals_files.postconditions + (signals_retrieval.postconditions if signals_retrieval else [])))
        all_relationships = list(set(signals_files.relationships + (signals_retrieval.relationships if signals_retrieval else [])))
        all_preconditions = list(set(signals_files.preconditions + (signals_retrieval.preconditions if signals_retrieval else [])))

        # 构建property列表，每个property包含类型和描述
        properties = []
        for inv in all_invariants:
            if inv.strip():
                properties.append(("invariant", inv))
        for post in all_postconditions:
            if post.strip():
                properties.append(("postcondition", post))
        for rel in all_relationships:
            if rel.strip():
                properties.append(("relationship", rel))
        for pre in all_preconditions:
            if pre.strip():
                properties.append(("precondition", pre))

        # 如果没有找到任何property，至少生成一个基于description的测试
        if not properties:
            description = signals_files.description or (signals_retrieval.description if signals_retrieval else "")
            if description.strip():
                properties.append(("description", description))

        # 为每个property生成一个测试，运行并验证
        generated_tests = []
        output_dir = self.config.project.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for prop_idx, (prop_type, prop_desc) in enumerate(properties):
            # 生成测试
            prompt = prompt_generate_pbt(
                target_function=target_fn,
                signals_from_files=signals_from_files,
                signals_from_retrieval=signals_from_retrieval,
                property_description=prop_desc,
                property_type=prop_type,
            )
            raw_response = await self.llm.complete(prompt)
            test_code = extract_python_code_from_response(raw_response)
            
            if not test_code.strip():
                continue
            
            # 创建临时测试文件来运行单个测试
            temp_test_file = output_dir / f"_temp_test_{index}_{prop_idx}.py"
            
            # 构建临时测试文件（包含imports和单个测试）
            temp_imports = set()
            test_func_lines = []
            for line in test_code.split('\n'):
                stripped = line.strip()
                if stripped.startswith('import ') or stripped.startswith('from '):
                    temp_imports.add(line)
                else:
                    test_func_lines.append(line)
            
            temp_import_section = '\n'.join(sorted(temp_imports)) if temp_imports else ''
            temp_test_section = '\n'.join(test_func_lines).strip()
            
            if temp_import_section and temp_test_section:
                temp_test_content = f"{temp_import_section}\n\n{temp_test_section}\n"
            elif temp_test_section:
                temp_test_content = f"{temp_test_section}\n"
            else:
                continue
            
            temp_test_file.write_text(temp_test_content, encoding="utf-8")
            
            # 运行测试
            test_passed, error_message = self._run_test_file(temp_test_file)
            
            if test_passed:
                # 测试通过，保留
                generated_tests.append(test_code)
            else:
                # 测试失败，判断是否是function bug
                is_function_bug, judgment = await self._judge_if_function_bug(
                    target_fn=target_fn,
                    test_code=temp_test_content,
                    error_message=error_message,
                    property_description=prop_desc,
                    property_type=prop_type,
                )
                
                if is_function_bug:
                    # 确认是function bug，保留测试
                    print(f"  Property '{prop_desc[:50]}...' failed - confirmed function bug")
                    generated_tests.append(test_code)
                else:
                    # 不是function bug，尝试修正一次
                    print(f"  Property '{prop_desc[:50]}...' failed - attempting to fix test")
                    fixed_test_code = await self._fix_test_code(
                        target_fn=target_fn,
                        original_test_code=temp_test_content,
                        error_message=error_message,
                        property_description=prop_desc,
                        property_type=prop_type,
                    )
                    
                    if fixed_test_code.strip():
                        # 更新临时测试文件
                        temp_imports_fixed = set()
                        test_func_lines_fixed = []
                        for line in fixed_test_code.split('\n'):
                            stripped = line.strip()
                            if stripped.startswith('import ') or stripped.startswith('from '):
                                temp_imports_fixed.add(line)
                            else:
                                test_func_lines_fixed.append(line)
                        
                        temp_import_section_fixed = '\n'.join(sorted(temp_imports_fixed)) if temp_imports_fixed else ''
                        temp_test_section_fixed = '\n'.join(test_func_lines_fixed).strip()
                        
                        if temp_import_section_fixed and temp_test_section_fixed:
                            temp_test_content_fixed = f"{temp_import_section_fixed}\n\n{temp_test_section_fixed}\n"
                        elif temp_test_section_fixed:
                            temp_test_content_fixed = f"{temp_test_section_fixed}\n"
                        else:
                            print(f"  Property '{prop_desc[:50]}...' - fixed test is empty, skipping")
                            temp_test_file.unlink(missing_ok=True)
                            continue
                        
                        temp_test_file.write_text(temp_test_content_fixed, encoding="utf-8")
                        
                        # 再次运行修正后的测试
                        test_passed_fixed, error_message_fixed = self._run_test_file(temp_test_file)
                        
                        if test_passed_fixed:
                            # 修正成功，保留
                            generated_tests.append(fixed_test_code)
                            print(f"  Property '{prop_desc[:50]}...' - test fixed successfully")
                        else:
                            # 修正后仍然失败，放弃
                            print(f"  Property '{prop_desc[:50]}...' - test fix failed, skipping")
                    else:
                        print(f"  Property '{prop_desc[:50]}...' - no fix generated, skipping")
            
            # 清理临时文件
            temp_test_file.unlink(missing_ok=True)

        # 合并所有生成的测试到一个文件中
        # 提取imports并合并，然后合并所有测试代码
        all_imports = set()
        all_test_code_blocks = []
        
        for test_code in generated_tests:
            if not test_code.strip():
                continue
                
            lines = test_code.split('\n')
            non_import_lines = []
            
            for line in lines:
                stripped = line.strip()
                # 收集import语句
                if stripped.startswith('import ') or stripped.startswith('from '):
                    all_imports.add(line)
                else:
                    non_import_lines.append(line)
            
            # 保留非import部分的代码
            non_import_code = '\n'.join(non_import_lines).strip()
            if non_import_code:
                all_test_code_blocks.append(non_import_code)

        # 构建完整的测试文件
        import_section = '\n'.join(sorted(all_imports)) if all_imports else ''
        test_section = '\n\n'.join(all_test_code_blocks) if all_test_code_blocks else ''
        
        if import_section and test_section:
            final_test_code = f"{import_section}\n\n{test_section}\n"
        elif test_section:
            final_test_code = f"{test_section}\n"
        elif import_section:
            final_test_code = f"{import_section}\n\n# No test functions generated\n"
        else:
            final_test_code = "# No tests generated\n"

        init_file_path = output_dir / f"__init__.py"
        init_file_path.write_text("", encoding="utf-8")

        test_file_path = output_dir / f"test_pbt_{index}.py"
        test_file_path.write_text(final_test_code, encoding="utf-8")

        pylint_output = self._run_pylint_on_file(test_file_path)
        return GeneratedPBT(
            target_function=target_fn,
            test_module_path=test_file_path,
            test_code=final_test_code,
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

    def _run_test_file(self, test_file_path: Path) -> tuple[bool, str]:
        """
        Run a test file and return (success, error_message).
        Returns (True, "") if test passes, (False, error_message) if it fails.
        """
        python_exe = self.config.project.python_path or "python"
        
        try:
            result = subprocess.run(
                [
                    python_exe,
                    "-m",
                    "pytest",
                    str(test_file_path),
                    "-v",
                    "--tb=short",
                ],
                capture_output=True,
                text=True,
                check=False,
                cwd=str(self.config.project.project_dir),
            )
            
            if result.returncode == 0:
                return True, ""
            else:
                error_msg = result.stdout + "\n" + result.stderr
                return False, error_msg
        except FileNotFoundError:
            return False, f"Python executable not found: {python_exe}"
        except Exception as e:
            return False, f"Error running test: {str(e)}"

    async def _judge_if_function_bug(
        self,
        target_fn: FunctionInfo,
        test_code: str,
        error_message: str,
        property_description: str,
        property_type: str,
    ) -> tuple[bool, dict]:
        """
        Use LLM to judge if the test failure indicates a real function bug.
        Returns (is_function_bug, judgment_data).
        """
        prompt = prompt_judge_function_bug(
            target_function=target_fn,
            test_code=test_code,
            error_message=error_message,
            property_description=property_description,
            property_type=property_type,
        )
        raw_response = await self.llm.complete(prompt)
        
        try:
            data = json.loads(raw_response)
            is_bug = data.get("is_function_bug", False)
            return is_bug, data
        except json.JSONDecodeError:
            # If JSON parsing fails, be conservative and assume it's not a function bug
            return False, {"error": "Failed to parse LLM response", "raw": raw_response}

    async def _fix_test_code(
        self,
        target_fn: FunctionInfo,
        original_test_code: str,
        error_message: str,
        property_description: str,
        property_type: str,
    ) -> str:
        """
        Use LLM to fix a test that failed due to test generation errors.
        Returns the fixed test code.
        """
        prompt = prompt_fix_test(
            target_function=target_fn,
            original_test_code=original_test_code,
            error_message=error_message,
            property_description=property_description,
            property_type=property_type,
        )
        raw_response = await self.llm.complete(prompt)
        fixed_code = extract_python_code_from_response(raw_response)
        return fixed_code


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



