from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class FileInfo:
    path: Path                      # absolute path
    rel_path: Path                  # path relative to project root
    module_path: str | None = None  # e.g. "pkg.sub.module"


@dataclass
class FunctionInfo:
    name: str                       # function or method name
    module_path: str                # "pkg.mod.func" or "pkg.mod.Class.method"
    file: FileInfo
    lineno: int
    end_lineno: int | None
    source: str
    is_method: bool = False
    class_name: str | None = None


@dataclass
class RelatedFileSelection:
    target_function: FunctionInfo
    related_files: list[FileInfo]
    llm_raw_response: str


@dataclass
class RetrievalResult:
    target_function: FunctionInfo
    query: str
    hits: list[dict[str, Any]]  # free-form; must include module_path in metadata
    llm_raw_response: str | None = None


@dataclass
class ExtractedPBTSignal:
    """
    Structured information extracted from related files / retrieval hits
    that is useful for generating property-based tests.
    """

    target_function: FunctionInfo
    description: str
    invariants: list[str] = field(default_factory=list)
    preconditions: list[str] = field(default_factory=list)
    postconditions: list[str] = field(default_factory=list)
    relationships: list[str] = field(default_factory=list)  # cross-function relations
    examples: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneratedPBT:
    target_function: FunctionInfo
    test_module_path: str
    test_code: str
    pylint_output: str | None = None


