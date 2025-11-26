from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .config import PBTConfig
from .models import FileInfo


def build_mac_style_tree(
    root: Path,
    max_depth: int | None = None,
    max_entries: int | None = None,
    ignore_dirs: Iterable[str] = (".git", "__pycache__", ".venv", "env", ".mypy_cache"),
) -> str:
    """
    Generate a macOS-like tree string (similar to `tree` command).
    """
    root = root.resolve()
    lines: list[str] = [root.name]

    def _walk(dir_path: Path, prefix: str, depth: int, entries_counter: list[int]) -> None:
        if max_depth is not None and depth > max_depth:
            return

        try:
            children = sorted(dir_path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except FileNotFoundError:
            return

        visible_children = [c for c in children if c.name not in ignore_dirs]

        for index, child in enumerate(visible_children):
            if max_entries is not None and entries_counter[0] >= max_entries:
                lines.append(f"{prefix}└── ... (truncated, limit={max_entries})")
                return

            connector = "└── " if index == len(visible_children) - 1 else "├── "
            line = f"{prefix}{connector}{child.name}"
            lines.append(line)
            entries_counter[0] += 1

            if child.is_dir():
                extension = "    " if index == len(visible_children) - 1 else "│   "
                _walk(child, prefix + extension, depth + 1, entries_counter)

    _walk(root, "", 1, [0])
    return "\n".join(lines)


def collect_python_files(project_root: Path) -> list[FileInfo]:
    project_root = project_root.resolve()
    files: list[FileInfo] = []
    for path in project_root.rglob("*.py"):
        if any(part in (".venv", "env", "__pycache__", ".git") for part in path.parts):
            continue
        rel = path.relative_to(project_root)
        files.append(
            FileInfo(
                path=path,
                rel_path=rel,
            )
        )
    return files


def infer_module_path(project_root: Path, rel_path: Path) -> str:
    """
    Convert a relative file path to a Python module path.

    Example:
        src/pkg/module.py -> pkg.module
    """
    parts = list(rel_path.with_suffix("").parts)
    # Drop leading "src" or similar if present – you can adjust this to your layout.
    if parts and parts[0] in {"src"}:
        parts = parts[1:]
    return ".".join(parts)


def fill_module_paths(project_root: Path, files: list[FileInfo]) -> None:
    for f in files:
        f.module_path = infer_module_path(project_root, f.rel_path)


