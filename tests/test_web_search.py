import copy
from types import SimpleNamespace

from cc_adapter.providers.shared.web_search import (
    WEB_SEARCH_TOOL_DEFINITION,
    inject_web_search_tool,
    is_web_search_enabled,
)


def make_config(provider: str = "", deepseek_key: str = "", brave_key: str = "", tavily_key: str = ""):
    return SimpleNamespace(
        web_search_provider=provider,
        deepseek_api_key=deepseek_key,
        brave_api_key=brave_key,
        tavily_api_key=tavily_key,
    )


class TestWebSearchEnabled:
    def test_disabled_when_no_provider(self):
        assert is_web_search_enabled(make_config()) is False

    def test_disabled_when_empty_provider(self):
        assert is_web_search_enabled(make_config(provider="")) is False

    def test_disabled_when_config_none(self):
        assert is_web_search_enabled(None) is False

    def test_brave_enabled_with_key(self):
        assert is_web_search_enabled(make_config(provider="brave", brave_key="key")) is True

    def test_brave_disabled_without_key(self):
        assert is_web_search_enabled(make_config(provider="brave")) is False

    def test_tavily_enabled_with_key(self):
        assert is_web_search_enabled(make_config(provider="tavily", tavily_key="key")) is True

    def test_tavily_disabled_without_key(self):
        assert is_web_search_enabled(make_config(provider="tavily")) is False

    def test_deepseek_enabled_with_key(self):
        assert is_web_search_enabled(make_config(provider="deepseek", deepseek_key="key")) is True

    def test_deepseek_disabled_without_key(self):
        assert is_web_search_enabled(make_config(provider="deepseek")) is False

    def test_unknown_provider_returns_false(self):
        assert is_web_search_enabled(make_config(provider="unknown_provider")) is False

    def test_case_insensitive_provider_matching(self):
        assert is_web_search_enabled(make_config(provider="BRAVE", brave_key="key")) is True
        assert is_web_search_enabled(make_config(provider="  Brave  ", brave_key="key")) is True
        assert is_web_search_enabled(make_config(provider="DEEPSEEK", deepseek_key="key")) is True
        assert is_web_search_enabled(make_config(provider="TAVILY", tavily_key="key")) is True


class TestInjectWebSearchTool:
    def test_adds_web_search_tool(self):
        tools = [{"name": "other_tool", "input_schema": {}}]
        result = inject_web_search_tool(tools)
        assert len(result) == 2
        assert result[1] == WEB_SEARCH_TOOL_DEFINITION

    def test_does_not_duplicate_when_already_present(self):
        tools = [{"name": "web_search", "input_schema": {}}]
        result = inject_web_search_tool(tools)
        assert len(result) == 1

    def test_does_not_mutate_original_list(self):
        original = [{"name": "other_tool", "input_schema": {}}]
        original_copy = copy.deepcopy(original)
        inject_web_search_tool(original)
        assert original == original_copy

    def test_does_not_mutate_global_tool_definition(self):
        original_def = copy.deepcopy(WEB_SEARCH_TOOL_DEFINITION)
        tools = [{"name": "other_tool", "input_schema": {}}]
        result = inject_web_search_tool(tools)
        result[1]["name"] = "hacked"
        assert WEB_SEARCH_TOOL_DEFINITION == original_def

    def test_returns_new_list_not_reference(self):
        tools = [{"name": "other_tool", "input_schema": {}}]
        result = inject_web_search_tool(tools)
        assert result is not tools


class TestWebSearchToolDefinition:
    def test_has_required_structure(self):
        assert WEB_SEARCH_TOOL_DEFINITION["name"] == "web_search"
        assert "description" in WEB_SEARCH_TOOL_DEFINITION
        assert WEB_SEARCH_TOOL_DEFINITION["input_schema"]["type"] == "object"
        assert "query" in WEB_SEARCH_TOOL_DEFINITION["input_schema"]["properties"]
        assert "query" in WEB_SEARCH_TOOL_DEFINITION["input_schema"]["required"]
