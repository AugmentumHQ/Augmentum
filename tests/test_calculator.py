"""Tests for CalculatorTool — safe mathematical expression evaluation."""

from __future__ import annotations

import math

import pytest

from augmentum.tools.calculator import CalculatorTool, safe_calculate


class TestSafeCalculate:
    """Direct tests on the safe_calculate function."""

    def test_basic_addition(self):
        assert safe_calculate("2 + 3") == 5

    def test_basic_subtraction(self):
        assert safe_calculate("10 - 4") == 6

    def test_basic_multiplication(self):
        assert safe_calculate("6 * 7") == 42

    def test_basic_division(self):
        assert safe_calculate("15 / 3") == 5.0

    def test_floor_division(self):
        assert safe_calculate("7 // 2") == 3

    def test_modulo(self):
        assert safe_calculate("10 % 3") == 1

    def test_exponentiation(self):
        assert safe_calculate("2 ** 10") == 1024

    def test_caret_as_exponent(self):
        assert safe_calculate("2^10") == 1024

    def test_large_numbers(self):
        result = safe_calculate("999999999 * 999999999")
        assert result == 999999999 * 999999999

    def test_floating_point(self):
        result = safe_calculate("0.1 + 0.2")
        assert abs(result - 0.3) < 1e-9

    def test_negative_numbers(self):
        assert safe_calculate("-5 + 3") == -2

    def test_parentheses(self):
        assert safe_calculate("(2 + 3) * 4") == 20

    def test_sqrt_function(self):
        assert safe_calculate("sqrt(144)") == 12.0

    def test_sin_function(self):
        result = safe_calculate("sin(0)")
        assert abs(result) < 1e-9

    def test_pi_constant(self):
        result = safe_calculate("pi")
        assert abs(result - math.pi) < 1e-9

    def test_log_function(self):
        result = safe_calculate("log(100, 10)")
        assert abs(result - 2.0) < 1e-9

    def test_division_by_zero(self):
        with pytest.raises(ZeroDivisionError):
            safe_calculate("1 / 0")

    def test_forbidden_import(self):
        with pytest.raises(ValueError, match="forbidden"):
            safe_calculate("import os")

    def test_forbidden_eval(self):
        with pytest.raises(ValueError, match="forbidden"):
            safe_calculate("eval('1+1')")

    def test_strip_currency_symbol(self):
        assert safe_calculate("$100 + $50") == 150

    def test_strip_trailing_equals(self):
        assert safe_calculate("2 + 3 =") == 5

    def test_strip_thousands_separator(self):
        assert safe_calculate("1,000 + 2,000") == 3000


class TestCalculatorTool:
    """CalculatorTool execute() contract tests."""

    async def test_execute_basic_arithmetic(self):
        tool = CalculatorTool()
        result = await tool.execute(expression="2 + 3 * 4")
        assert result.success is True
        assert result.output == "14"

    async def test_execute_division_by_zero(self):
        tool = CalculatorTool()
        result = await tool.execute(expression="1 / 0")
        assert result.success is False
        assert "error" in result.error.lower()

    async def test_execute_empty_expression(self):
        tool = CalculatorTool()
        result = await tool.execute(expression="")
        assert result.success is False

    async def test_execute_invalid_syntax(self):
        tool = CalculatorTool()
        result = await tool.execute(expression="2 +* 3")
        assert result.success is False

    async def test_execute_metadata_includes_result(self):
        tool = CalculatorTool()
        result = await tool.execute(expression="10 * 5")
        assert result.metadata["result"] == 50

    async def test_tool_properties(self):
        tool = CalculatorTool()
        assert tool.name == "calculator"
        assert tool.timeout == 2.0
        assert tool.cache_ttl == 0.0

    async def test_execute_complex_expression(self):
        tool = CalculatorTool()
        result = await tool.execute(expression="sqrt(16) + pow(2, 3)")
        assert result.success is True
        assert float(result.output) == 12.0

    async def test_execute_factorial(self):
        tool = CalculatorTool()
        result = await tool.execute(expression="factorial(5)")
        assert result.success is True
        assert result.output == "120"
