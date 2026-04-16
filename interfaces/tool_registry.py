import asyncio
import logging
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from mcp import ClientSession
from mcp.client.sse import sse_client
from pydantic import Field, create_model

logger = logging.getLogger(__name__)


_JSON_TYPE_MAP: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _args_model(tool_name: str, schema: dict[str, Any]) -> type:
    """Build a Pydantic model from an MCP tool's JSON input schema.

    Without this, StructuredTool.from_function(**kwargs) yields a schema with a single
    `kwargs` field — LLM tool calls then land inside `{"kwargs": {...}}` and MCP rejects.
    """
    props: dict[str, Any] = schema.get("properties", {}) or {}
    required = set(schema.get("required", []) or [])
    fields: dict[str, tuple[type, Any]] = {}
    for name, prop in props.items():
        py_type: type = dict  # safe fallback
        if "type" in prop:
            py_type = _JSON_TYPE_MAP.get(prop["type"], dict)
        elif "anyOf" in prop:
            for entry in prop["anyOf"]:
                t = _JSON_TYPE_MAP.get(entry.get("type", ""))
                if t and t is not type(None):
                    py_type = t
                    break
        default = ... if name in required else prop.get("default", None)
        if name not in required:
            py_type = py_type | None  # type: ignore[operator]
        fields[name] = (py_type, Field(default, description=prop.get("description", "")))
    return create_model(f"{tool_name}_Args", **fields)


class MCPToolRegistry:
    """Connects to an MCP server via SSE and exposes tools as LangChain tools."""

    def __init__(self, name: str, url: str) -> None:
        self._name = name
        self._url = url
        self._session: ClientSession | None = None
        self._tools: list[BaseTool] = []
        self._read_stream: Any = None
        self._write_stream: Any = None
        self._task: asyncio.Task | None = None
        self._ready = asyncio.Event()
        self._shutdown = asyncio.Event()

    async def connect(self) -> None:
        """Connect to MCP server and discover tools via a background task."""
        self._task = asyncio.create_task(self._run())
        await self._ready.wait()

    async def _run(self) -> None:
        """Background task that owns the SSE connection lifecycle."""
        try:
            async with sse_client(self._url) as streams:
                read_stream, write_stream = streams
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    self._session = session

                    # Discover tools
                    tools_response = await session.list_tools()
                    for tool_info in tools_response.tools:
                        self._tools.append(self._wrap_tool(tool_info))
                    logger.info(
                        f"MCP {self._name}: connected, {len(self._tools)} tools discovered"
                    )

                    # Signal ready
                    self._ready.set()

                    # Keep alive until shutdown
                    await self._shutdown.wait()

            logger.info(f"MCP {self._name}: disconnected")
        except Exception as e:
            logger.error(f"MCP {self._name}: connection failed: {e}")
            self._ready.set()  # Unblock waiter even on failure

    def _wrap_tool(self, tool_info: Any) -> BaseTool:
        """Wrap an MCP tool as a LangChain StructuredTool with a real args schema."""
        session = self._session
        tool_name = tool_info.name
        input_schema = getattr(tool_info, "inputSchema", None) or {}
        args_schema = _args_model(tool_name, input_schema)

        async def _call(**kwargs: Any) -> str:
            # Strip None values so MCP tools see only explicit args.
            clean = {k: v for k, v in kwargs.items() if v is not None}
            result = await session.call_tool(tool_name, clean)
            if result.content:
                # Concatenate all text content blocks (some tools emit one per row).
                return "\n".join(c.text for c in result.content if hasattr(c, "text"))
            return ""

        return StructuredTool.from_function(
            coroutine=_call,
            name=tool_name,
            description=tool_info.description or f"MCP tool: {tool_name}",
            args_schema=args_schema,
        )

    @property
    def tools(self) -> list[BaseTool]:
        return self._tools

    async def disconnect(self) -> None:
        self._shutdown.set()
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except (asyncio.TimeoutError, Exception):
                self._task.cancel()
        logger.info(f"MCP {self._name}: shutdown complete")
