from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable

from .file_tree import collect_python_files, fill_module_paths
from .models import FileInfo, FunctionInfo


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _iter_functions_in_file(file_info: FileInfo, project_root: Path) -> Iterable[FunctionInfo]:
    source = _read_text(file_info.path)
    try:
        tree = ast.parse(source, filename=str(file_info.path))
    except SyntaxError:
        return []

    lines = source.splitlines()

    def get_source_segment(node: ast.AST) -> str:
        if not hasattr(node, "lineno"):
            return ""
        lineno = node.lineno
        end_lineno = getattr(node, "end_lineno", lineno)
        # ast is 1-based lines
        snippet = "\n".join(lines[lineno - 1 : end_lineno])
        return snippet

    functions: list[FunctionInfo] = []

    module_path = file_info.module_path
    if not module_path:
        from .file_tree import infer_module_path

        module_path = infer_module_path(project_root, file_info.rel_path)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            # Check if method (inside ClassDef)
            class_name: str | None = None
            is_method = False
            parent = getattr(node, "parent", None)
            if isinstance(parent, ast.ClassDef):
                is_method = True
                class_name = parent.name

            fq_module_path = module_path
            if is_method and class_name:
                fq = f"{fq_module_path}.{class_name}.{node.name}"
            else:
                fq = f"{fq_module_path}.{node.name}"

            fn = FunctionInfo(
                name=node.name,
                module_path=fq,
                file=file_info,
                lineno=node.lineno,
                end_lineno=getattr(node, "end_lineno", None),
                source=get_source_segment(node),
                is_method=is_method,
                class_name=class_name,
            )
            functions.append(fn)

    return functions


def _attach_parents(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            setattr(child, "parent", node)


def collect_functions(project_root: Path) -> list[FunctionInfo]:
    project_root = project_root.resolve()
    file_infos = collect_python_files(project_root)
    fill_module_paths(project_root, file_infos)

    all_functions: list[FunctionInfo] = []
    for f in file_infos:
        source = _read_text(f.path)
        try:
            tree = ast.parse(source, filename=str(f.path))
        except SyntaxError:
            continue
        _attach_parents(tree)

        # Reuse parsing result so we don't parse twice
        lines = source.splitlines()

        def get_source_segment(node: ast.AST) -> str:
            if not hasattr(node, "lineno"):
                return ""
            lineno = node.lineno
            end_lineno = getattr(node, "end_lineno", lineno)
            return "\n".join(lines[lineno - 1 : end_lineno])

        module_path = f.module_path or ""
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                parent = getattr(node, "parent", None)
                class_name: str | None = None
                is_method = False
                if isinstance(parent, ast.ClassDef):
                    is_method = True
                    class_name = parent.name

                fq_module_path = module_path
                if is_method and class_name:
                    fq = f"{fq_module_path}.{class_name}.{node.name}"
                else:
                    fq = f"{fq_module_path}.{node.name}"

                all_functions.append(
                    FunctionInfo(
                        name=node.name,
                        module_path=fq,
                        file=f,
                        lineno=node.lineno,
                        end_lineno=getattr(node, "end_lineno", None),
                        source=get_source_segment(node),
                        is_method=is_method,
                        class_name=class_name,
                    )
                )

    return all_functions


def filter_functions_by_module_paths(
    functions: list[FunctionInfo], target_module_paths: list[str] | None
) -> list[FunctionInfo]:
    if not target_module_paths:
        return functions
    targets = set(target_module_paths)
    return [fn for fn in functions if fn.module_path in targets]


