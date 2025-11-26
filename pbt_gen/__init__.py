"""
LLM-based cross-function property-based test (PBT) generator.

This package provides a framework to:
- scan a Python project
- extract functions/methods and their module paths
- let an LLM analyse cross-file/cross-function relations
- generate property-based tests and run pylint on them
"""

__all__ = [
    "config",
    "file_tree",
    "function_finder",
    "llm_client",
    "models",
    "prompts",
    "pbt_generator",
    "vector_store",
]


