from __future__ import annotations

from textwrap import dedent

from .models import FileInfo, FunctionInfo


def prompt_select_related_files(
    target_function: FunctionInfo,
    project_tree: str,
    python_files: list[FileInfo],
    top_n: int,
) -> str:
    """
    Prompt for step 3: given a target function and the file tree,
    ask LLM to select top-N related files (by property/behavior).
    """
    files_listing = "\n".join(
        f"- {f.rel_path}  (module_path={f.module_path})" for f in python_files
    )

    return dedent(
        f"""
        You are an expert Python engineer and testing specialist.
        Your task is to identify source files that are semantically related to
        the following target function, with a focus on properties, invariants,
        and cross-function relationships that are useful for **property-based testing (PBT)**.

        ## STRICT REQUIREMENT ABOUT METADATA
        - For every file / function / class / method you mention, ALWAYS include its
          **Python module path** metadata (e.g. `package.module`, `package.module.Class.method`).
        - This metadata must be sufficient to import the symbol from test code.

        ## Project file tree (macOS style)
        {project_tree}

        ## All Python files (with module paths)
        {files_listing}

        ## Target function (with module path metadata)
        - module_path: `{target_function.module_path}`
        - defined in file: `{target_function.file.rel_path}`
        - source code:
        ```python
        {target_function.source}
        ```

        ## Your goals
        1. Select **top {top_n} source files** that are most likely to contain:
           - inverse operations (e.g. encode/decode, serialize/deserialize)
           - validation or checking logic
           - invariants / consistency checks
           - higher-level orchestration that relies on the target function
           - alternative implementations with the same behavior
        2. For each selected file, explain **why** it is related from a property/testing perspective.
        3. For each selected file, list notable **functions/classes/methods** (with module paths)
           that might be important when generating PBTs.

        ## Output format (MUST be valid JSON)
        Return a JSON object with the following shape:

        {{
          "target_function_module_path": "<string>",
          "selected_files": [
            {{
              "rel_path": "<relative/path/to/file.py>",
              "module_path": "<python.module.path>",
              "reason": "<why this file is related>",
              "notable_symbols": [
                {{
                  "name": "<symbol name>",
                  "kind": "function|class|method",
                  "module_path": "<python.module.path.for.this.symbol>",
                  "role_for_pbt": "<how this helps define properties/invariants>"
                }}
              ]
            }}
          ]
        }}

        Do not include any comments outside JSON. Do not use trailing commas.
        """
    ).strip()


def prompt_extract_pbt_signals_from_files(
    target_function: FunctionInfo,
    related_files_contents: dict[str, str],  # rel_path -> content
) -> str:
    """
    Second half of step 3: given target function and contents of selected files,
    ask LLM to extract information helpful for PBT.
    """
    files_blob_parts = []
    for rel, content in related_files_contents.items():
        files_blob_parts.append(
            f"# File: {rel}\n"
            f"```python\n"
            f"{content}\n"
            f"```"
        )
    files_blob = "\n\n".join(files_blob_parts)

    return dedent(
        f"""
        You are an expert in property-based testing for Python.
        You are given:
        - A target function (with module path metadata).
        - The full source code of several **related files** picked by another agent.

        Your job is to extract structured information that will help generate
        high-quality **cross-function PBTs**.

        ## STRICT REQUIREMENT ABOUT METADATA
        - Every time you refer to a function/class/method, you MUST include its
          full Python **module path** in the output.
        - This metadata must be sufficient for importing in tests.

        ## Target function
        - module_path: `{target_function.module_path}`
        - file: `{target_function.file.rel_path}`
        - source:
        ```python
        {target_function.source}
        ```

        ## Related files (with source)
        {files_blob}

        ## Your goals
        For the target function, infer:
        - **Invariants**: properties that should always hold about inputs/outputs/state.
        - **Preconditions**: assumptions that must hold for the function to behave correctly.
        - **Postconditions**: guarantees about outputs or state after the function returns.
        - **Cross-function relationships**: how this function interacts with other functions,
          such as encode/decode pairs, round-trips, consistency with validators, etc.
        - **Examples**: a few concrete input/output or usage examples that illustrate the above.

        ## Output format (MUST be valid JSON)
        Return a JSON object:

        {{
          "target_function_module_path": "<string>",
          "description": "<high-level description of the function's behavior and role>",
          "invariants": ["<string>", "..."],
          "preconditions": ["<string>", "..."],
          "postconditions": ["<string>", "..."],
          "relationships": [
            {{
              "description": "<relation, e.g. encode/decode round-trip>",
              "related_symbol_module_path": "<python.module.path>",
              "kind": "function|class|method"
            }}
          ],
          "examples": ["<example 1>", "<example 2>", "..."]
        }}

        Do not include any comments outside JSON. Do not use trailing commas.
        """
    ).strip()


def prompt_build_retrieval_query(target_function: FunctionInfo) -> str:
    """
    Step 4 (first half): ask LLM to generate a dense retrieval query for vector search,
    focusing on cross-function properties.
    """
    return dedent(
        f"""
        You are assisting with semantic vector search for Python source code
        in order to build cross-function property-based tests (PBT).

        ## STRICT REQUIREMENT ABOUT METADATA
        - You must restate the target function's module path exactly, as this will be
          stored as metadata for downstream steps.

        ## Target function
        - module_path: `{target_function.module_path}`
        - file: `{target_function.file.rel_path}`
        - source:
        ```python
        {target_function.source}
        ```

        ## Your goal
        Produce a **single natural-language query** that is well-suited for vector
        similarity search to find:
        - potential inverse/dual functions (e.g. encode/decode, serialize/deserialize)
        - callers and callees that impose stronger invariants
        - validators or checkers enforcing constraints related to this function
        - alternative implementations with same behavior

        The query should:
        - Be descriptive and self-contained (no references like "the function above").
        - Emphasize properties, invariants, and relationships over concrete variable names.

        ## Output format (MUST be valid JSON)
        Return:

        {{
          "target_function_module_path": "<string>",
          "retrieval_query": "<single natural language query>"
        }}

        Do not include comments, only JSON.
        """
    ).strip()


def prompt_extract_pbt_signals_from_retrieval(
    target_function: FunctionInfo,
    query: str,
    hits: list[dict],
) -> str:
    """
    Step 4 (second half): given retrieval hits, ask LLM to extract PBT-relevant info.
    """
    formatted_hits = []
    for i, hit in enumerate(hits):
        content = hit.get("content", "")
        metadata = hit.get("metadata", {})
        module_path = metadata.get("module_path", "<unknown>")
        source_rel_path = metadata.get("rel_path", "<unknown>")
        formatted_hits.append(
            f"### Hit {i}\n"
            f"- module_path: {module_path}\n"
            f"- rel_path: {source_rel_path}\n"
            f"- metadata: {metadata}\n"
            f"```python\n{content}\n```"
        )
    hits_blob = "\n\n".join(formatted_hits)

    return dedent(
        f"""
        You are an expert in property-based testing for Python.
        You are given:
        - A target function.
        - A semantic search query.
        - A set of code/documentation snippets retrieved by vector search.

        ## STRICT REQUIREMENT ABOUT METADATA
        - Each time you reference a symbol from the retrieved snippets, you must include
          its `module_path` metadata exactly as provided (or inferred if clearly implied).

        ## Target function
        - module_path: `{target_function.module_path}`
        - file: `{target_function.file.rel_path}`
        - source:
        ```python
        {target_function.source}
        ```

        ## Retrieval query
        {query}

        ## Retrieval hits
        {hits_blob}

        ## Your goals
        Similar to previous step, but based on retrieval hits:
        - Infer invariants, preconditions, postconditions, cross-function relationships,
          and concrete examples that are useful for generating PBTs.
        - Especially focus on how the target function should behave **in relation to** the
          retrieved symbols (round-trips, consistency, error handling, etc.).

        ## Output format (MUST be valid JSON)

        {{
          "target_function_module_path": "<string>",
          "description": "<summary of new insights from retrieval>",
          "invariants": ["<string>", "..."],
          "preconditions": ["<string>", "..."],
          "postconditions": ["<string>", "..."],
          "relationships": [
            {{
              "description": "<relation>",
              "related_symbol_module_path": "<python.module.path>",
              "kind": "function|class|method"
            }}
          ],
          "examples": ["<example 1>", "<example 2>", "..."]
        }}

        Do not include comments, only JSON.
        """
    ).strip()


def prompt_generate_pbt(
    target_function: FunctionInfo,
    signals_from_files: dict,
    signals_from_retrieval: dict | None,
    property_description: str | None = None,
    property_type: str | None = None,
) -> str:
    """
    Step 5: generate PBT code based on all collected signals.
    If property_description and property_type are provided, generate only ONE test for that specific property.
    Otherwise, generate multiple tests (legacy behavior).
    """
    property_focus = ""
    if property_description and property_type:
        property_focus = f"""
        ## CRITICAL: Generate ONLY ONE test function
        You must generate **exactly ONE property-based test function** for the following specific property:
        - Property type: {property_type}
        - Property description: {property_description}
        
        IMPORTANT:
        - Do NOT generate multiple tests. Generate only one test function (e.g., `def test_...`).
        - Do NOT generate class definitions or setup methods. Only generate the test function itself.
        - The test function will be merged with other tests later, so focus only on the single test function.
        """
    
    return dedent(
        f"""
        You are a senior Python engineer and a property-based testing expert.
        Your task is to generate **property-based tests (PBT)** for the given target
        function using Hypothesis, based on structured signals extracted from
        related files and retrieval hits.

        ## STRICT REQUIREMENTS
        - All imports must use the given **module paths** exactly.
        - Do NOT invent module paths; only use what you are given.
        - The tests must be valid Python code and runnable with pytest + hypothesis.

        ## Target function
        - module_path: `{target_function.module_path}`
        - file: `{target_function.file.rel_path}`
        - source:
        ```python
        {target_function.source}
        ```

        ## Signals from related files 
        ```json
        {signals_from_files}
        ```

        ## Signals from retrieval hits 
        ```json
        {signals_from_retrieval}
        ```
        {property_focus}
        ## Your goals
        - Use Hypothesis (and optionally pytest) to encode the discovered invariants,
          preconditions, postconditions, and cross-function relationships.
        - Prefer **round-trip properties** (e.g. decode(encode(x)) == x) when applicable.
        - Include negative tests where appropriate (violating preconditions).
        {"- Generate exactly ONE test function for the specified property above." if property_description and property_type else "- Keep tests focused but expressive; 2–6 properties per target function is fine."}

        ## Output format
        Return **only** the Python test code (a single test function), no backticks, no explanations.
        {"The output should be a single test function definition, not a complete test module." if property_description and property_type else "Return **only** the Python test module code, no backticks, no explanations."}
        """
    ).strip()


def prompt_judge_function_bug(
    target_function: FunctionInfo,
    test_code: str,
    error_message: str,
    property_description: str,
    property_type: str,
) -> str:
    """
    Prompt for judging whether a test failure indicates a real bug in the function
    or an error in the test generation. Must be very conservative.
    """
    return dedent(
        f"""
        You are an expert Python engineer and testing specialist.
        A property-based test was generated for a function, but it failed when executed.
        Your task is to determine whether the failure indicates a **real bug in the function**
        or an **error in the test generation**.

        ## CRITICAL: Be VERY CONSERVATIVE
        - Only conclude it's a function bug if you have **strong, evidence-based reasons**
        - The property must be **clearly valid** and **well-defined**
        - The test code must be **correctly written** and **properly testing the property**
        - The error must **directly violate** the stated property or function contract
        - If there's any doubt, conclude it's a test generation error

        ## Target function
        - module_path: `{target_function.module_path}`
        - file: `{target_function.file.rel_path}`
        - source:
        ```python
        {target_function.source}
        ```

        ## Property being tested
        - Property type: {property_type}
        - Property description: {property_description}

        ## Generated test code
        ```python
        {test_code}
        ```

        ## Error message from test execution
        ```
        {error_message}
        ```

        ## Your analysis
        Analyze the error carefully:
        1. Is the property clearly valid and well-defined for this function?
        2. Is the test code correctly implementing the property test?
        3. Does the error message indicate the function violated the property?
        4. Could the error be due to:
           - Incorrect test setup (missing imports, wrong module paths)?
           - Incorrect property interpretation?
           - Missing preconditions or assumptions?
           - Test code bugs (wrong assertions, wrong function calls)?

        ## Output format (MUST be valid JSON)
        Return a JSON object:
        {{
          "is_function_bug": <true|false>,
          "confidence": "<high|medium|low>",
          "reasoning": "<detailed explanation of your judgment>",
          "evidence": [
            "<evidence point 1>",
            "<evidence point 2>",
            "..."
          ],
          "property_validity": "<explanation of whether the property is valid>",
          "test_correctness": "<explanation of whether the test is correctly written>"
        }}

        Remember: Only mark is_function_bug as true if you have STRONG evidence that:
        - The property is clearly valid
        - The test correctly implements the property
        - The function clearly violates the property

        Do not include comments, only JSON.
        """
    ).strip()


def prompt_fix_test(
    target_function: FunctionInfo,
    original_test_code: str,
    error_message: str,
    property_description: str,
    property_type: str,
) -> str:
    """
    Prompt for fixing a test that failed due to test generation errors.
    """
    return dedent(
        f"""
        You are an expert Python engineer and testing specialist.
        A property-based test was generated but failed when executed.
        The failure has been determined to be due to a **test generation error**, not a function bug.
        Your task is to fix the test code.

        ## Target function
        - module_path: `{target_function.module_path}`
        - file: `{target_function.file.rel_path}`
        - source:
        ```python
        {target_function.source}
        ```

        ## Property being tested
        - Property type: {property_type}
        - Property description: {property_description}

        ## Original test code (with error)
        ```python
        {original_test_code}
        ```

        ## Error message from test execution
        ```
        {error_message}
        ```

        ## Your task
        Fix the test code to:
        1. Correctly import all necessary modules using the exact module paths
        2. Properly set up the test (e.g., class instantiation if needed)
        3. Correctly implement the property test
        4. Ensure the test is runnable with pytest + hypothesis

        ## Output format
        Return **only** the corrected Python test code (a single test function), no backticks, no explanations.
        The output should be a single test function definition, not a complete test module.
        """
    ).strip()


def prompt_fix_pylint_errors(
    target_function: FunctionInfo,
    original_test_code: str,
    pylint_output: str,
    property_description: str,
    property_type: str,
) -> str:
    """
    Prompt for fixing test code that has pylint syntax errors.
    """
    return dedent(
        f"""
        You are an expert Python engineer and testing specialist.
        A property-based test was generated but has **pylint syntax errors**.
        Your task is to fix the syntax errors in the test code.

        ## Target function
        - module_path: `{target_function.module_path}`
        - file: `{target_function.file.rel_path}`
        - source:
        ```python
        {target_function.source}
        ```

        ## Property being tested
        - Property type: {property_type}
        - Property description: {property_description}

        ## Original test code (with syntax errors)
        ```python
        {original_test_code}
        ```

        ## Pylint error output
        ```
        {pylint_output}
        ```

        ## Your task
        Fix the syntax errors in the test code:
        1. Fix all syntax errors reported by pylint
        2. Ensure all imports are correct and use the exact module paths
        3. Ensure the code is valid Python syntax
        4. Maintain the test logic and property being tested
        5. Ensure the test is runnable with pytest + hypothesis

        ## Output format
        Return **only** the corrected Python test code (a single test function), no backticks, no explanations.
        The output should be a single test function definition, not a complete test module.
        """
    ).strip()


