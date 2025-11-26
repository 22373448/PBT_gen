from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class LLMConfig:
    model: str = "gpt-4.1"
    temperature: float = 0.1
    max_tokens: int = 3000


@dataclass
class PBTConfig:
    top_n_related_files: int = 10
    max_file_tree_depth: int = 20
    max_file_tree_entries: int = 5000
    max_source_snippet_chars: int = 8000


@dataclass
class ProjectConfig:
    project_dir: Path
    output_dir: Path
    functions: list[str] | None = None  # module paths, if None -> all


@dataclass
class AppConfig:
    project: ProjectConfig
    llm: LLMConfig = LLMConfig()
    pbt: PBTConfig = PBTConfig()


