"""ReAct prompt builder — delegates to centralized prompts module.

v2: Compact tool descriptions, Chinese-first with English fallback.
"""

from agentic_rag.config.prompts import Prompts

# Legacy alias — still used by router.py and react_engine.py
SYSTEM_PROMPT = Prompts.react_system()


def build_react_prompt(tools_description: str) -> str:
    """Build the ReAct system prompt with tool descriptions.

    Uses the Chinese-first prompt by default. Call ``build_react_prompt_en()``
    for the English variant. Conversation history is NOT part of the system
    prompt — the engine appends it as real chat messages after the system
    message, so it arrives with proper roles instead of a flattened string.
    """
    return SYSTEM_PROMPT.format(tools_description=tools_description)


def build_react_prompt_en(tools_description: str) -> str:
    """Build the English variant of the ReAct system prompt."""
    return Prompts.react_system_en().format(tools_description=tools_description)


def build_tools_description(tools) -> str:
    """Build a compact text description of available tools.

    Format (one line per tool + compact param list):
        rag_search: 搜索知识库
          query (required): 搜索关键词
          top_k: 返回数量 [default: 3, max: 3]
    """
    lines = []
    for tool in tools:
        # Handle both ToolDefinition/Pydantic model and plain dict
        if hasattr(tool, 'name'):
            name = tool.name
            desc = tool.description
            params = tool.parameters
        else:
            name = tool.get("name", "unknown")
            desc = tool.get("description", "")
            params = tool.get("parameters", {})

        params_props = params.get("properties", {}) if isinstance(params, dict) else {}
        if isinstance(params_props, dict) and params_props:
            required_params = params.get("required", []) if isinstance(params, dict) else []
            param_strs = []
            for pname, pinfo in params_props.items():
                req_mark = " (必填)" if pname in required_params else ""
                pdesc = pinfo.get("description", "") if isinstance(pinfo, dict) else str(pinfo)
                if isinstance(pinfo, dict) and "default" in pinfo:
                    pdesc += f" [默认: {pinfo['default']}]"
                if isinstance(pinfo, dict) and "enum" in pinfo:
                    pdesc += f" (可选: {', '.join(str(v) for v in pinfo['enum'])})"
                param_strs.append(f"  {pname}: {pdesc}{req_mark}")
            params_block = "\n".join(param_strs)
        else:
            params_block = "  (无参数)"

        lines.append(f"{name}: {desc}\n{params_block}")

    return "\n".join(lines)
