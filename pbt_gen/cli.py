from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .config import AppConfig, LLMConfig, PBTConfig, ProjectConfig
from .llm_client import OpenAILLMClient
from .models import FunctionInfo, GeneratedPBT
from .pbt_generator import PBTOrchestrator
from .vector_store import ChromaVectorStore
from .function_finder import collect_functions, filter_functions_by_module_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM-based cross-function property-based test generator framework",
    )
    parser.add_argument(
        "--project-dir",
        required=True,
        help="Path to the target Python project root.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write generated PBT test files.",
    )
    parser.add_argument(
        "--functions",
        help="Comma-separated list of target function module paths (e.g. pkg.mod.fn,pkg.mod.Class.method). "
        "If omitted, all discovered functions/methods are used.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Top N related files / retrieval hits to consider.",
    )
    parser.add_argument(
        "--python-path",
        help="Path to Python executable to use for running tests. If not specified, uses system python.",
    )
    return parser.parse_args()


def serialize_pbt_to_dict(pbt: GeneratedPBT) -> dict:
    """
    Convert GeneratedPBT to a serializable dictionary for JSON output.
    """
    return {
        "target_function": {
            "name": pbt.target_function.name,
            "module_path": pbt.target_function.module_path,
            "file": {
                "path": str(pbt.target_function.file.path),
                "rel_path": str(pbt.target_function.file.rel_path),
                "module_path": pbt.target_function.file.module_path,
            },
            "lineno": pbt.target_function.lineno,
            "end_lineno": pbt.target_function.end_lineno,
            "source": pbt.target_function.source,
            "is_method": pbt.target_function.is_method,
            "class_name": pbt.target_function.class_name,
        },
        "test_module_path": str(pbt.test_module_path),
        "test_code": pbt.test_code,
        "pylint_output": pbt.pylint_output,
        "passed": pbt.passed,
    }


async def main_async() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    target_functions = (
        [s.strip() for s in args.functions.split(",") if s.strip()] if args.functions else None
    )

    app_config = AppConfig(
        project=ProjectConfig(
            project_dir=project_dir,
            output_dir=output_dir,
            functions=target_functions,
            python_path=args.python_path,
        ),
        llm=LLMConfig(),
        pbt=PBTConfig(top_n_related_files=args.top_n),
    )

    llm_client = OpenAILLMClient(app_config.llm)

    # 首先构建 orchestrator（暂时不带检索）
    orchestrator = PBTOrchestrator(
        config=app_config,
        llm_client=llm_client,
        vector_store=None,
    )

    project_tree, all_files, all_functions = orchestrator.scan_project()

    # 使用函数源码构建基于 chromadb 的向量检索索引
    vector_store = ChromaVectorStore(collection_name="pbt_gen_functions")
    docs = []
    for fn in all_functions:
        docs.append(
            {
                "id": fn.module_path,
                "content": fn.source,
                "metadata": {
                    "module_path": fn.module_path,
                    "rel_path": str(fn.file.rel_path),
                },
            }
        )
    vector_store.index_documents(docs)
    orchestrator.vector_store = vector_store

    selected_functions = filter_functions_by_module_paths(
        all_functions, target_module_paths=target_functions
    )

    print(
        f"Discovered {len(all_functions)} functions/methods; "
        f"{len(selected_functions)} selected as targets."
    )

    generated_tests = []
    output_jsonl_path = output_dir / "output.jsonl"
    # 创建文件写入锁，确保并行写入安全
    file_write_lock = asyncio.Lock()
    
    async def process_single_function(
        index: int,
        fn: FunctionInfo,
        orchestrator: PBTOrchestrator,
        project_tree: str,
        all_files: list,
        output_path: Path,
        write_lock: asyncio.Lock,
    ) -> list[GeneratedPBT]:
        """
        处理单个函数的异步任务，包括生成测试和写入文件。
        """
        print(f"\n=== Processing target function #{index}: {fn.module_path} ===")

        try:
            # Step 3: file selection & signals
            selection = await orchestrator.select_related_files_for_function(
                project_tree=project_tree,
                all_files=all_files,
                target_fn=fn,
            )
            signals_files = await orchestrator.extract_signals_from_files(selection)

            # Step 4: retrieval-based signals (optional if vector store is empty)
            retrieval = await orchestrator.build_retrieval_query(fn)
            signals_retrieval = None
            if retrieval.query and retrieval.hits:
                signals_retrieval = await orchestrator.extract_signals_from_retrieval(retrieval)

            # Step 5: generate PBT + pylint
            pbt_results = await orchestrator.generate_pbt_for_function(
                target_fn=fn,
                signals_files=signals_files,
                signals_retrieval=signals_retrieval,
                index=index,
            )

            # 使用锁保护文件写入操作，确保线程安全
            async with write_lock:
                with open(output_path, "a", encoding="utf-8") as f:
                    for pbt in pbt_results:
                        pbt_dict = serialize_pbt_to_dict(pbt)
                        f.write(json.dumps(pbt_dict, ensure_ascii=False) + "\n")

            return pbt_results
        except Exception as e:
            print(f"Error processing function {fn.module_path}: {e}")
            return []
    
    # 并行处理所有函数
    tasks = [
        process_single_function(
            index=index,
            fn=fn,
            orchestrator=orchestrator,
            project_tree=project_tree,
            all_files=all_files,
            output_path=output_jsonl_path,
            write_lock=file_write_lock,
        )
        for index, fn in enumerate(selected_functions)
    ]
    
    # 等待所有任务完成并收集结果
    results = await asyncio.gather(*tasks)
    for result_list in results:
        generated_tests.extend(result_list)
    

def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()


