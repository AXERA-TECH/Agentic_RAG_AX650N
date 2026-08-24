"""File operation tools."""

from typing import Any

from agentic_rag.orchestration.l1_tools.base import BaseTool

try:
    import aiofiles
    import aiofiles.os
    HAS_AIOFILES = True
except ImportError:
    HAS_AIOFILES = False


class FileReadTool(BaseTool):
    """Read the contents of a file."""

    name = "file_read"
    description = "Read the contents of a file from the filesystem."
    parameters_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to read",
            },
        },
        "required": ["path"],
    }
    requires_confirmation = True  # Safety: confirm before reading files

    async def execute(self, path: str) -> Any:
        if not HAS_AIOFILES:
            try:
                with open(path, "r") as f:
                    content = f.read()
                return content[:10000] if len(content) > 10000 else content
            except Exception as e:
                return f"Error reading file: {str(e)}"
        try:
            async with aiofiles.open(path, "r") as f:
                content = await f.read()
            return content[:10000] if len(content) > 10000 else content
        except FileNotFoundError:
            return f"File not found: {path}"
        except PermissionError:
            return f"Permission denied: {path}"
        except Exception as e:
            return f"Error reading file: {str(e)}"


class FileWriteTool(BaseTool):
    """Write content to a file."""

    name = "file_write"
    description = "Write text content to a file."
    parameters_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to write the file",
            },
            "content": {
                "type": "string",
                "description": "Content to write",
            },
        },
        "required": ["path", "content"],
    }
    requires_confirmation = True  # Safety: confirm before writing files

    async def execute(self, path: str, content: str) -> Any:
        if not HAS_AIOFILES:
            try:
                with open(path, "w") as f:
                    f.write(content)
                return f"File written successfully: {path} ({len(content)} chars)"
            except Exception as e:
                return f"Error writing file: {str(e)}"
        try:
            async with aiofiles.open(path, "w") as f:
                await f.write(content)
            return f"File written successfully: {path} ({len(content)} chars)"
        except Exception as e:
            return f"Error writing file: {str(e)}"
