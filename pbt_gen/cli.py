from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from .config import AppConfig, LLMConfig, PBTConfig, ProjectConfig
from .llm_client import OpenAILLMClient
from .pbt_generator import PBTOrchestrator
from .vector_store import DummyVectorStore
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
    return parser.parse_args()


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
        ),
        llm=LLMConfig(),
        pbt=PBTConfig(top_n_related_files=args.top_n),
    )

    llm_client = OpenAILLMClient(app_config.llm)

    # TODO: replace DummyVectorStore with your real implementation and indexing pipeline.
    vector_store = DummyVectorStore(documents=[])

    orchestrator = PBTOrchestrator(
        config=app_config,
        llm_client=llm_client,
        vector_store=vector_store,
    )

    project_tree, all_files, all_functions = orchestrator.scan_project()

    selected_functions = filter_functions_by_module_paths(
        all_functions, target_module_paths=target_functions
    )

    print("Project file tree (macOS style):")
    print(project_tree)
    print()
    print(f"Discovered {len(all_functions)} functions/methods; "
          f"{len(selected_functions)} selected as targets.")

    for fn in selected_functions:
        print(f"\n=== Processing target function: {fn.module_path} ===")

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
        pbt_result = await orchestrator.generate_pbt_for_function(
            target_fn=fn,
            signals_files=signals_files,
            signals_retrieval=signals_retrieval,
        )

        print(f"Generated PBT module for {fn.module_path}: {pbt_result.test_module_path}")
        print("pylint summary:")
        print(pbt_result.pylint_output)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()


