from __future__ import annotations


WEB_SEARCH_TOOL_DEFINITION = {
    "name": "web_search",
    "description": "Search the web for current information",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query"},
        },
        "required": ["query"],
    },
}


def is_web_search_enabled(config) -> bool:
    if not config or not config.web_search_provider:
        return False
    provider = config.web_search_provider
    if provider == "deepseek":
        return bool(config.deepseek_api_key)
    if provider == "brave":
        return bool(config.brave_api_key)
    if provider == "tavily":
        return bool(config.tavily_api_key)
    return False


def inject_web_search_tool(tools: list[dict]) -> list[dict]:
    if any(t.get("name") == "web_search" for t in tools):
        return tools
    return tools + [WEB_SEARCH_TOOL_DEFINITION]
