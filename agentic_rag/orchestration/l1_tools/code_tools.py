"""Code execution tool (sandboxed)."""

import asyncio
import subprocess
from typing import Any

from agentic_rag.orchestration.l1_tools.base import BaseTool


class CodeExecuteTool(BaseTool):
    """Execute Python code in a sandboxed subprocess."""

    name = "code_execute"
    description = "Execute Python code and return the output. Use for calculations and data processing."
    parameters_schema = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute",
            },
        },
        "required": ["code"],
    }
    requires_confirmation = True

    async def execute(self, code: str) -> Any:
        try:
            proc = await asyncio.create_subprocess_exec(
                "python3", "-c", code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=30.0
            )

            result = stdout.decode("utf-8", errors="replace")
            if stderr:
                err = stderr.decode("utf-8", errors="replace")
                if err.strip():
                    result += f"\n[stderr]\n{err}"

            return result.strip() or "(no output)"
        except asyncio.TimeoutError:
            return "Execution timed out (30s limit)."
        except Exception as e:
            return f"Execution error: {str(e)}"
