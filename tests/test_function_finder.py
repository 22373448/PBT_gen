from pathlib import Path

from pbt_gen.function_finder import collect_functions, filter_functions_by_module_paths


def get_example_root() -> Path:
    return Path(__file__).parent / "example_project"


def test_collect_functions_discovers_functions_and_methods() -> None:
    root = get_example_root()
    functions = collect_functions(root)
    module_paths = {fn.module_path for fn in functions}

    # 期望发现的函数和方法
    assert "pkg.module_a.add" in module_paths
    assert "pkg.module_a.Encoder.encode" in module_paths
    assert "pkg.subpkg.module_b.mul" in module_paths


def test_filter_functions_by_module_paths() -> None:
    root = get_example_root()
    functions = collect_functions(root)

    target = ["pkg.module_a.add"]
    filtered = filter_functions_by_module_paths(functions, target)
    assert {fn.module_path for fn in filtered} == set(target)


