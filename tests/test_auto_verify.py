"""Tests for the automated verification module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from augmentum.modes.analytical.auto_verify import (
    AutoVerifyResult,
    VerificationCheck,
    check_code_dependencies,
    extract_code_blocks,
    extract_fact_claims,
    extract_math_expressions,
    run_auto_verification,
)

# ==========================================================================
# Math expression extraction
# ==========================================================================


class TestExtractMathExpressions:
    """Tests for extracting math equations from text."""

    def test_simple_addition(self):
        text = "The total is 25 + 17 = 42"
        results = extract_math_expressions(text)
        assert len(results) == 1
        assert results[0] == ("25+17", "42")

    def test_multiplication(self):
        text = "Revenue: 150 * 12 = 1800"
        results = extract_math_expressions(text)
        assert len(results) == 1
        assert results[0] == ("150*12", "1800")

    def test_decimal_numbers(self):
        text = "25.5 * 1.08 = 27.54"
        results = extract_math_expressions(text)
        assert len(results) == 1
        assert results[0] == ("25.5*1.08", "27.54")

    def test_chained_operations(self):
        text = "Total: 100 + 200 + 300 = 600"
        results = extract_math_expressions(text)
        assert len(results) == 1
        assert results[0] == ("100+200+300", "600")

    def test_no_math(self):
        text = "The capital of France is Paris."
        results = extract_math_expressions(text)
        assert len(results) == 0

    def test_multiple_equations(self):
        text = "Step 1: 10 + 5 = 15\nStep 2: 15 * 2 = 30"
        results = extract_math_expressions(text)
        assert len(results) == 2

    def test_numbers_with_commas(self):
        text = "Population: 1,000 + 500 = 1,500"
        results = extract_math_expressions(text)
        assert len(results) == 1
        assert results[0] == ("1000+500", "1500")

    def test_subtraction(self):
        text = "Profit: 500 - 200 = 300"
        results = extract_math_expressions(text)
        assert len(results) == 1

    def test_division(self):
        text = "Average: 100 / 4 = 25"
        results = extract_math_expressions(text)
        assert len(results) == 1


# ==========================================================================
# Code block extraction
# ==========================================================================


class TestExtractCodeBlocks:
    """Tests for extracting Python code blocks from text."""

    def test_python_code_block(self):
        text = "Here is the code:\n```python\ndef hello():\n    print('world')\n```"
        blocks = extract_code_blocks(text)
        assert len(blocks) == 1
        assert "def hello():" in blocks[0]

    def test_py_code_block(self):
        text = "```py\nx = 42\nprint(x)\n```"
        blocks = extract_code_blocks(text)
        assert len(blocks) == 1

    def test_generic_code_block(self):
        text = "```\nimport math\nprint(math.pi)\n```"
        blocks = extract_code_blocks(text)
        assert len(blocks) == 1

    def test_no_code_blocks(self):
        text = "Just regular text without any code."
        blocks = extract_code_blocks(text)
        assert len(blocks) == 0

    def test_short_code_block_skipped(self):
        text = "```python\nx = 1\n```"
        blocks = extract_code_blocks(text)
        assert len(blocks) == 0  # too short (< 10 chars)

    def test_multiple_code_blocks(self):
        text = (
            "First:\n```python\ndef add(a, b):\n    return a + b\n```\n"
            "Second:\n```python\ndef multiply(a, b):\n    return a * b\n```"
        )
        blocks = extract_code_blocks(text)
        assert len(blocks) == 2


# ==========================================================================
# Fact claim extraction
# ==========================================================================


class TestExtractFactClaims:
    """Tests for extracting fact claims that reference search results."""

    def test_according_to_pattern(self):
        text = "The economy is growing. According to the data, the GDP grew by 3.5% in 2025."
        search_context = "GDP growth rate was reported at 3.5% for the year 2025 by the World Bank."
        claims = extract_fact_claims(text, search_context)
        assert len(claims) >= 1

    def test_no_search_context(self):
        text = "According to the data, something happened."
        claims = extract_fact_claims(text, "")
        assert len(claims) == 0

    def test_no_fact_patterns(self):
        text = "The sky is blue and water is wet."
        search_context = "Some search results here."
        claims = extract_fact_claims(text, search_context)
        assert len(claims) == 0


# ==========================================================================
# VerificationCheck and AutoVerifyResult
# ==========================================================================


class TestAutoVerifyResult:
    """Tests for the result data structures."""

    def test_empty_result(self):
        result = AutoVerifyResult()
        assert result.has_checks is False
        assert result.all_passed is True
        assert result.pass_count == 0
        assert result.fail_count == 0

    def test_all_passed(self):
        result = AutoVerifyResult(checks=[
            VerificationCheck("math", "1+1=2", True, "Correct"),
            VerificationCheck("math", "2*3=6", True, "Correct"),
        ])
        result.all_passed = True
        assert result.has_checks is True
        assert result.pass_count == 2
        assert result.fail_count == 0

    def test_mixed_results(self):
        result = AutoVerifyResult(checks=[
            VerificationCheck("math", "1+1=2", True, "Correct"),
            VerificationCheck("math", "2+2=5", False, "Wrong: expected 4"),
        ])
        result.all_passed = False
        assert result.pass_count == 1
        assert result.fail_count == 1


# ==========================================================================
# Code dependency checking
# ==========================================================================


class TestCheckCodeDependencies:
    """Tests for detecting unavailable imports in code blocks."""

    def test_stdlib_only(self):
        code = "import os\nimport json\nprint(os.getcwd())"
        assert check_code_dependencies(code) == []

    def test_available_package_numpy(self):
        code = "import numpy as np\nx = np.array([1, 2, 3])"
        assert check_code_dependencies(code) == []

    def test_available_package_pandas(self):
        code = "import pandas as pd\ndf = pd.DataFrame({'a': [1]})"
        assert check_code_dependencies(code) == []

    def test_available_package_scipy(self):
        code = "from scipy import stats\nresult = stats.norm.pdf(0)"
        assert check_code_dependencies(code) == []

    def test_available_package_sympy(self):
        code = "from sympy import symbols, solve\nx = symbols('x')"
        assert check_code_dependencies(code) == []

    def test_available_package_matplotlib(self):
        code = "import matplotlib.pyplot as plt\nplt.plot([1,2,3])"
        assert check_code_dependencies(code) == []

    def test_unavailable_requests(self):
        code = "import requests\nr = requests.get('https://example.com')"
        unavailable = check_code_dependencies(code)
        assert "requests" in unavailable

    def test_unavailable_sklearn(self):
        code = "from sklearn.linear_model import LinearRegression"
        unavailable = check_code_dependencies(code)
        assert "sklearn" in unavailable

    def test_unavailable_torch(self):
        code = "import torch\nx = torch.tensor([1, 2, 3])"
        unavailable = check_code_dependencies(code)
        assert "torch" in unavailable

    def test_unavailable_transformers(self):
        code = "from transformers import pipeline"
        unavailable = check_code_dependencies(code)
        assert "transformers" in unavailable

    def test_mixed_available_and_unavailable(self):
        code = (
            "import numpy as np\n"
            "import requests\n"
            "import pandas as pd\n"
            "from bs4 import BeautifulSoup\n"
        )
        unavailable = check_code_dependencies(code)
        assert "requests" in unavailable
        assert "bs4" in unavailable
        assert "numpy" not in unavailable
        assert "pandas" not in unavailable

    def test_syntax_error_code_fallback(self):
        code = "import numpy\nimport requests\nthis is not valid python"
        unavailable = check_code_dependencies(code)
        # Should still detect via regex fallback
        assert "requests" in unavailable

    def test_no_imports(self):
        code = "x = 1 + 2\nprint(x)"
        assert check_code_dependencies(code) == []

    def test_from_import(self):
        code = "from PIL import Image"
        unavailable = check_code_dependencies(code)
        assert "PIL" in unavailable

    def test_submodule_import(self):
        code = "import numpy.linalg"
        # Top-level is numpy, which is available
        assert check_code_dependencies(code) == []


# ==========================================================================
# Full auto-verification pipeline
# ==========================================================================


class TestRunAutoVerification:
    """Tests for the full auto-verification pipeline."""

    @pytest.fixture
    def mock_registry(self):
        registry = MagicMock()
        registry.get.return_value = None
        return registry

    @pytest.mark.asyncio
    async def test_no_registry(self):
        result = await run_auto_verification("some output", None)
        assert result.has_checks is False
        assert result.all_passed is True

    @pytest.mark.asyncio
    async def test_no_verifiable_content(self, mock_registry):
        result = await run_auto_verification(
            "The capital of France is Paris.", mock_registry,
        )
        assert result.has_checks is False

    @pytest.mark.asyncio
    async def test_math_verification_pass(self):
        from augmentum.tools.base import ToolResult

        registry = MagicMock()
        math_tool = AsyncMock()
        math_tool.execute = AsyncMock(return_value=ToolResult(
            success=True,
            output="Expression: 10+5\nResult: 15\nExpected: 15\nMatch: YES",
            metadata={"result": 15.0, "expected": 15.0, "match": True},
        ))

        def get_tool(name):
            if name == "math_verify":
                return math_tool
            return None

        registry.get.side_effect = get_tool

        result = await run_auto_verification(
            "The answer is 10 + 5 = 15.", registry,
        )
        assert result.has_checks is True
        assert result.all_passed is True
        assert result.pass_count == 1
        math_tool.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_math_verification_fail(self):
        from augmentum.tools.base import ToolResult

        registry = MagicMock()
        math_tool = AsyncMock()
        math_tool.execute = AsyncMock(return_value=ToolResult(
            success=True,
            output="Expression: 10+5\nResult: 15\nExpected: 20\nMatch: NO",
            metadata={"result": 15.0, "expected": 20.0, "match": False},
        ))

        def get_tool(name):
            if name == "math_verify":
                return math_tool
            return None

        registry.get.side_effect = get_tool

        result = await run_auto_verification(
            "The answer is 10 + 5 = 20.", registry,
        )
        assert result.has_checks is True
        assert result.all_passed is False
        assert result.fail_count == 1

    @pytest.mark.asyncio
    async def test_code_verification_pass(self):
        from augmentum.tools.base import ToolResult

        registry = MagicMock()
        exec_tool = AsyncMock()
        exec_tool.execute = AsyncMock(return_value=ToolResult(
            success=True,
            output="Output:\n42",
            metadata={"stdout": "42"},
        ))

        def get_tool(name):
            if name == "python_exec":
                return exec_tool
            return None

        registry.get.side_effect = get_tool

        apply_output = (
            "Here is the solution:\n"
            "```python\n"
            "result = sum(range(10))\n"
            "print(result)\n"
            "```"
        )
        result = await run_auto_verification(apply_output, registry)
        assert result.has_checks is True
        assert result.all_passed is True
        exec_tool.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_code_verification_fail(self):
        from augmentum.tools.base import ToolResult

        registry = MagicMock()
        exec_tool = AsyncMock()
        exec_tool.execute = AsyncMock(return_value=ToolResult(
            success=False,
            error="NameError: name 'undefined_var' is not defined",
        ))

        def get_tool(name):
            if name == "python_exec":
                return exec_tool
            return None

        registry.get.side_effect = get_tool

        apply_output = (
            "Here is the code:\n"
            "```python\n"
            "print(undefined_var + 1)\n"
            "```"
        )
        result = await run_auto_verification(apply_output, registry)
        assert result.has_checks is True
        assert result.all_passed is False
        assert result.fail_count == 1

    @pytest.mark.asyncio
    async def test_combined_math_and_code(self):
        from augmentum.tools.base import ToolResult

        registry = MagicMock()
        math_tool = AsyncMock()
        math_tool.execute = AsyncMock(return_value=ToolResult(
            success=True,
            output="Match: YES",
            metadata={"match": True},
        ))
        exec_tool = AsyncMock()
        exec_tool.execute = AsyncMock(return_value=ToolResult(
            success=True,
            output="Output: 42",
        ))

        def get_tool(name):
            if name == "math_verify":
                return math_tool
            if name == "python_exec":
                return exec_tool
            return None

        registry.get.side_effect = get_tool

        apply_output = (
            "The calculation: 10 * 4 = 40\n"
            "```python\nprint(10 * 4)\n```"
        )
        result = await run_auto_verification(apply_output, registry)
        assert result.has_checks is True
        assert result.pass_count == 2

    @pytest.mark.asyncio
    async def test_math_tool_exception_handled(self):
        registry = MagicMock()
        math_tool = AsyncMock()
        math_tool.execute = AsyncMock(side_effect=Exception("Tool crashed"))

        def get_tool(name):
            if name == "math_verify":
                return math_tool
            return None

        registry.get.side_effect = get_tool

        result = await run_auto_verification(
            "Result: 10 + 5 = 15", registry,
        )
        assert result.has_checks is True
        assert result.fail_count == 1
        assert "crashed" in result.checks[0].error

    @pytest.mark.asyncio
    async def test_summary_includes_all_checks(self):
        from augmentum.tools.base import ToolResult

        registry = MagicMock()
        math_tool = AsyncMock()
        math_tool.execute = AsyncMock(return_value=ToolResult(
            success=True,
            output="Match: YES",
            metadata={"match": True},
        ))

        def get_tool(name):
            if name == "math_verify":
                return math_tool
            return None

        registry.get.side_effect = get_tool

        result = await run_auto_verification(
            "Answer: 5 + 3 = 8", registry,
        )
        assert "Automated Verification Results" in result.summary
        assert "PASS" in result.summary
        assert "MATH" in result.summary

    @pytest.mark.asyncio
    async def test_caps_math_checks_at_five(self):
        from augmentum.tools.base import ToolResult

        registry = MagicMock()
        math_tool = AsyncMock()
        math_tool.execute = AsyncMock(return_value=ToolResult(
            success=True,
            output="Match: YES",
            metadata={"match": True},
        ))

        def get_tool(name):
            if name == "math_verify":
                return math_tool
            return None

        registry.get.side_effect = get_tool

        # 8 equations
        lines = [f"Step {i}: {i} + {i} = {2*i}" for i in range(1, 9)]
        result = await run_auto_verification("\n".join(lines), registry)
        # Should cap at 5
        assert len(result.checks) <= 5

    @pytest.mark.asyncio
    async def test_code_with_unavailable_deps_skipped(self):
        from augmentum.tools.base import ToolResult

        registry = MagicMock()
        exec_tool = AsyncMock()
        exec_tool.execute = AsyncMock(return_value=ToolResult(
            success=True, output="OK",
        ))

        def get_tool(name):
            if name == "python_exec":
                return exec_tool
            return None

        registry.get.side_effect = get_tool

        apply_output = (
            "```python\n"
            "import requests\n"
            "r = requests.get('https://example.com')\n"
            "print(r.status_code)\n"
            "```"
        )
        result = await run_auto_verification(apply_output, registry)
        assert result.has_checks is True
        assert result.checks[0].skipped is True
        assert result.checks[0].passed is True  # not counted as failure
        assert "requests" in result.checks[0].details
        assert result.all_passed is True
        # Executor should NOT have been called
        exec_tool.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_code_mixed_available_and_unavailable(self):
        from augmentum.tools.base import ToolResult

        registry = MagicMock()
        exec_tool = AsyncMock()
        exec_tool.execute = AsyncMock(return_value=ToolResult(
            success=True, output="42",
        ))

        def get_tool(name):
            if name == "python_exec":
                return exec_tool
            return None

        registry.get.side_effect = get_tool

        apply_output = (
            "Block 1 (available deps):\n"
            "```python\n"
            "import numpy as np\n"
            "print(np.sum([1, 2, 3]))\n"
            "```\n"
            "Block 2 (unavailable deps):\n"
            "```python\n"
            "import torch\n"
            "x = torch.tensor([1])\n"
            "print(x)\n"
            "```"
        )
        result = await run_auto_verification(apply_output, registry)
        assert len(result.checks) == 2
        # First block: executed
        assert result.checks[0].skipped is False
        assert result.checks[0].passed is True
        # Second block: skipped
        assert result.checks[1].skipped is True
        assert "torch" in result.checks[1].details
        # Executor called only once (for the numpy block)
        assert exec_tool.execute.call_count == 1

    @pytest.mark.asyncio
    async def test_skip_count_in_summary(self):
        from augmentum.tools.base import ToolResult

        registry = MagicMock()
        exec_tool = AsyncMock()
        exec_tool.execute = AsyncMock(return_value=ToolResult(
            success=True, output="OK",
        ))

        def get_tool(name):
            if name == "python_exec":
                return exec_tool
            return None

        registry.get.side_effect = get_tool

        apply_output = (
            "```python\n"
            "from sklearn.svm import SVC\n"
            "model = SVC()\nprint(model)\n"
            "```"
        )
        result = await run_auto_verification(apply_output, registry)
        assert result.skip_count == 1
        assert "skipped" in result.summary.lower()
        assert "SKIPPED" in result.summary

    @pytest.mark.asyncio
    async def test_caps_code_blocks_at_three(self):
        from augmentum.tools.base import ToolResult

        registry = MagicMock()
        exec_tool = AsyncMock()
        exec_tool.execute = AsyncMock(return_value=ToolResult(
            success=True,
            output="OK",
        ))

        def get_tool(name):
            if name == "python_exec":
                return exec_tool
            return None

        registry.get.side_effect = get_tool

        # 5 code blocks
        blocks = [
            f"```python\nresult_{i} = {i} * 2\nprint(result_{i})\n```"
            for i in range(5)
        ]
        result = await run_auto_verification("\n".join(blocks), registry)
        assert len(result.checks) <= 3


# ==========================================================================
# Integration with engine state
# ==========================================================================


class TestAutoVerifyIntegration:
    """Tests for auto-verify integration with UARF state model."""

    def test_state_has_auto_verify_field(self):
        from augmentum.modes.analytical.state import AnalyticalState

        state = AnalyticalState()
        assert hasattr(state, "auto_verify_summary")
        assert state.auto_verify_summary == ""

    def test_prompt_accepts_auto_verify_summary(self):
        from augmentum.modes.analytical.prompts import get_phase_prompt

        system, user = get_phase_prompt(
            "verify",
            query="What is 2+2?",
            apply_output="STEP 1: 2+2=4\nPRELIMINARY_ANSWER: 4",
            auto_verify_summary="## Automated Verification Results\n1 check passed.",
        )
        # Summary should appear in user content
        assert "Automated Verification Results" in user
        assert "1 check passed" in user

    def test_prompt_without_auto_verify_summary(self):
        from augmentum.modes.analytical.prompts import get_phase_prompt

        system, user = get_phase_prompt(
            "verify",
            query="What is 2+2?",
            apply_output="STEP 1: 2+2=4\nPRELIMINARY_ANSWER: 4",
        )
        assert "Automated Verification Results" not in user

    def test_verify_prompt_mentions_automated_checks(self):
        from augmentum.modes.analytical.prompts import get_phase_prompt

        system, _ = get_phase_prompt(
            "verify",
            query="test",
            apply_output="test",
        )
        assert "automated verification results" in system.lower()
        assert "ground truth" in system.lower()
