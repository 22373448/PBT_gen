from pathlib import Path

from pbt_gen.file_tree import (
    build_mac_style_tree,
    collect_python_files,
    fill_module_paths,
    infer_module_path,
)


def get_example_root() -> Path:
    return Path(__file__).parent / "example_project"


def test_build_mac_style_tree_contains_expected_entries() -> None:
    root = get_example_root()
    tree = build_mac_style_tree(root)

    # 根名称应为 example_project
    assert "example_project" in tree
    # 包目录和模块文件应该出现在树中
    assert "pkg" in tree
    assert "module_a.py" in tree
    assert "subpkg" in tree
    assert "module_b.py" in tree


def test_collect_python_files_and_module_paths() -> None:
    root = get_example_root()
    files = collect_python_files(root)
    # example_project 下应该至少有 3 个 .py 文件（__init__ 也会被算上）
    rel_paths = {str(f.rel_path).replace("\\", "/") for f in files}

    assert "pkg/__init__.py" in rel_paths
    assert "pkg/module_a.py" in rel_paths
    assert "pkg/subpkg/module_b.py" in rel_paths

    # 填充 module_path 后检查推断是否正确
    fill_module_paths(root, files)
    path_to_module = {str(f.rel_path).replace("\\", "/"): f.module_path for f in files}

    assert path_to_module["pkg/module_a.py"] == "pkg.module_a"
    assert path_to_module["pkg/subpkg/module_b.py"] == "pkg.subpkg.module_b"


def test_infer_module_path_standalone() -> None:
    root = get_example_root()
    rel = Path("pkg") / "subpkg" / "module_b.py"
    module = infer_module_path(root, rel)
    assert module == "pkg.subpkg.module_b"


