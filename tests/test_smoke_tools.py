"""Smoke tests — verify every tool module imports and primary classes construct."""

from __future__ import annotations

import importlib

import pytest


# ---------------------------------------------------------------------------
# Module import smoke tests
# ---------------------------------------------------------------------------


class TestToolModuleImports:
    """Every tool module under augmentum/tools/ must import without error."""

    def test_import_registry(self):
        mod = importlib.import_module("augmentum.tools.registry")
        assert hasattr(mod, "ToolRegistry")

    def test_import_base(self):
        mod = importlib.import_module("augmentum.tools.base")
        assert hasattr(mod, "Tool")
        assert hasattr(mod, "ToolResult")
        assert hasattr(mod, "ToolCategory")

    def test_import_web_search(self):
        mod = importlib.import_module("augmentum.tools.web_search")
        assert hasattr(mod, "WebSearchTool")

    def test_import_web_fetch(self):
        mod = importlib.import_module("augmentum.tools.web_fetch")
        assert hasattr(mod, "WebFetchTool")

    def test_import_web(self):
        mod = importlib.import_module("augmentum.tools.web")
        assert hasattr(mod, "WebTool")

    def test_import_wikipedia(self):
        mod = importlib.import_module("augmentum.tools.wikipedia")
        assert hasattr(mod, "WikipediaTool")

    def test_import_youtube(self):
        mod = importlib.import_module("augmentum.tools.youtube")
        assert hasattr(mod, "YouTubeTool")

    def test_import_image_search(self):
        mod = importlib.import_module("augmentum.tools.image_search")
        assert hasattr(mod, "ImageSearchTool")

    def test_import_math_verify(self):
        mod = importlib.import_module("augmentum.tools.math_verify")
        assert hasattr(mod, "MathVerifyTool")

    def test_import_intent(self):
        mod = importlib.import_module("augmentum.tools.intent")
        assert hasattr(mod, "classify_intent")
        assert hasattr(mod, "QueryIntent")

    def test_import_query_formulator(self):
        mod = importlib.import_module("augmentum.tools.query_formulator")
        assert hasattr(mod, "formulate_queries")

    def test_import_python_exec(self):
        mod = importlib.import_module("augmentum.tools.python_exec")
        assert hasattr(mod, "PythonExecTool")

    def test_import_document_parse(self):
        mod = importlib.import_module("augmentum.tools.document_parse")
        assert hasattr(mod, "DocumentParseTool")

    def test_import_parsing(self):
        mod = importlib.import_module("augmentum.tools.parsing")
        assert True  # module imported without error

    def test_import_result_processing(self):
        mod = importlib.import_module("augmentum.tools.result_processing")
        assert hasattr(mod, "truncate_tool_result")

    def test_import_text_analysis(self):
        mod = importlib.import_module("augmentum.tools.text_analysis")
        assert hasattr(mod, "TextAnalysisTool")

    def test_import_hash_tool(self):
        mod = importlib.import_module("augmentum.tools.hash_tool")
        assert hasattr(mod, "HashTool")

    def test_import_unit_converter(self):
        mod = importlib.import_module("augmentum.tools.unit_converter")
        assert hasattr(mod, "UnitConverterTool")

    def test_import_json_tool(self):
        mod = importlib.import_module("augmentum.tools.json_tool")
        assert hasattr(mod, "JsonTool")

    def test_import_datetime_tool(self):
        mod = importlib.import_module("augmentum.tools.datetime_tool")
        assert hasattr(mod, "DateTimeTool")

    def test_import_calculator(self):
        mod = importlib.import_module("augmentum.tools.calculator")
        assert hasattr(mod, "CalculatorTool")

    def test_import_artifact_pipeline(self):
        mod = importlib.import_module("augmentum.tools.artifact_pipeline")
        assert True

    def test_import_artifact_document(self):
        mod = importlib.import_module("augmentum.tools.artifact_document")
        assert hasattr(mod, "DocumentTool")

    def test_import_artifact_presentation(self):
        mod = importlib.import_module("augmentum.tools.artifact_presentation")
        assert hasattr(mod, "PresentationTool")

    def test_import_artifact_spreadsheet(self):
        mod = importlib.import_module("augmentum.tools.artifact_spreadsheet")
        assert hasattr(mod, "SpreadsheetTool")

    def test_import_artifact_chart(self):
        mod = importlib.import_module("augmentum.tools.artifact_chart")
        assert True

    def test_import_artifact_ebook(self):
        mod = importlib.import_module("augmentum.tools.artifact_ebook")
        assert True

    def test_import_artifact_application(self):
        mod = importlib.import_module("augmentum.tools.artifact_application")
        assert True

    def test_import_artifact_normalize(self):
        mod = importlib.import_module("augmentum.tools.artifact_normalize")
        assert hasattr(mod, "normalize_str")
        assert hasattr(mod, "normalize_sections")

    def test_import_artifact_sanitize(self):
        mod = importlib.import_module("augmentum.tools.artifact_sanitize")
        assert hasattr(mod, "sanitize_text")
        assert hasattr(mod, "sanitize_sections")

    def test_import_artifact_storage(self):
        mod = importlib.import_module("augmentum.tools.artifact_storage")
        assert True

    def test_import_artifact_templates(self):
        mod = importlib.import_module("augmentum.tools.artifact_templates")
        assert hasattr(mod, "ArtifactTemplate")

    def test_import_artifact_theme(self):
        mod = importlib.import_module("augmentum.tools.artifact_theme")
        assert True

    def test_import_memory_recall(self):
        mod = importlib.import_module("augmentum.tools.memory_recall")
        assert True

    def test_import_flow_tool(self):
        mod = importlib.import_module("augmentum.tools.flow_tool")
        assert True

    def test_import_custom_flows(self):
        mod = importlib.import_module("augmentum.tools.custom_flows")
        assert True

    def test_import_cache(self):
        mod = importlib.import_module("augmentum.tools.cache")
        assert True

    def test_import_circuit_breaker(self):
        mod = importlib.import_module("augmentum.tools.circuit_breaker")
        assert hasattr(mod, "ToolCircuitBreaker")

    def test_import_filter(self):
        mod = importlib.import_module("augmentum.tools.filter")
        assert True

    def test_import_preferred_sources(self):
        mod = importlib.import_module("augmentum.tools.preferred_sources")
        assert hasattr(mod, "SourceInfo")
        assert hasattr(mod, "domain_quality")

    def test_import_application_references(self):
        mod = importlib.import_module("augmentum.tools.application_references")
        assert True

    def test_import_application_scaffolds(self):
        mod = importlib.import_module("augmentum.tools.application_scaffolds")
        assert True

    def test_import_consistency_check(self):
        mod = importlib.import_module("augmentum.tools.consistency_check")
        assert True

    def test_import_export_tools(self):
        mod = importlib.import_module("augmentum.tools.export_tools")
        assert True

    def test_import_file_ops(self):
        mod = importlib.import_module("augmentum.tools.file_ops")
        assert hasattr(mod, "FileOpsTool")

    def test_import_background_chain(self):
        mod = importlib.import_module("augmentum.tools.background_chain")
        assert True

    def test_import_draft_section(self):
        mod = importlib.import_module("augmentum.tools.draft_section")
        assert True

    def test_import_chain(self):
        mod = importlib.import_module("augmentum.tools.chain")
        assert True


# ---------------------------------------------------------------------------
# Class construction smoke tests
# ---------------------------------------------------------------------------


class TestToolConstruction:
    """Primary tool classes should be constructable with minimal deps."""

    def test_construct_tool_registry(self):
        from augmentum.tools.registry import ToolRegistry
        reg = ToolRegistry()
        assert reg.list_tools() == []

    def test_construct_calculator(self):
        from augmentum.tools.calculator import CalculatorTool
        tool = CalculatorTool()
        assert tool.name == "calculator"
        assert tool.category.value == "verify"

    def test_construct_datetime(self):
        from augmentum.tools.datetime_tool import DateTimeTool
        tool = DateTimeTool()
        assert tool.name == "datetime"

    def test_construct_unit_converter(self):
        from augmentum.tools.unit_converter import UnitConverterTool
        tool = UnitConverterTool()
        assert tool.name == "unit_converter"

    def test_construct_hash_tool(self):
        from augmentum.tools.hash_tool import HashTool
        tool = HashTool()
        assert tool.name == "hash"

    def test_construct_json_tool(self):
        from augmentum.tools.json_tool import JsonTool
        tool = JsonTool()
        assert tool.name == "json_tool"

    def test_construct_text_analysis(self):
        from augmentum.tools.text_analysis import TextAnalysisTool
        tool = TextAnalysisTool()
        assert tool.name == "text_analysis"

    def test_construct_circuit_breaker(self):
        from augmentum.tools.circuit_breaker import ToolCircuitBreaker
        cb = ToolCircuitBreaker(threshold=3, cooldown=60.0)
        assert cb.threshold == 3

    def test_construct_query_intent(self):
        from augmentum.tools.intent import QueryIntent
        qi = QueryIntent()
        assert qi.action == "none"
        assert qi.confidence == 0.0
