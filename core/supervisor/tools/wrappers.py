import logging

from langchain_core.tools import BaseTool

from klaudia.interfaces.tool_registry import MCPToolRegistry

logger = logging.getLogger(__name__)


def get_sql_tools(registry: MCPToolRegistry) -> list[BaseTool]:
    """Get MCP-SQLite tools filtered for SQL Agent use."""
    allowed = {
        "tool_get_document",
        "tool_list_documents",
        "tool_list_pages",
        "tool_get_page",
        "tool_get_extraction",
        "tool_get_session_files",
    }
    return [t for t in registry.tools if t.name in allowed]


def get_data_entry_tools(registry: MCPToolRegistry) -> list[BaseTool]:
    """Get all MCP-GSheets tools for Data Entry Team."""
    return registry.tools


def get_read_tools(registry: MCPToolRegistry) -> list[BaseTool]:
    """Get read-only GSheets tools for Read Agent."""
    allowed = {
        "tool_get_sheet_data",
        "tool_get_sheet_formulas",
        "tool_list_sheets",
        "tool_get_spreadsheet_info",
        "tool_get_multiple_sheet_data",
    }
    return [t for t in registry.tools if t.name in allowed]


def get_sheet_tools(registry: MCPToolRegistry) -> list[BaseTool]:
    """Get sheet management tools for Sheet Agent."""
    allowed = {
        "tool_create_sheet",
        "tool_rename_sheet",
        "tool_copy_sheet",
        "tool_delete_sheet",
        "tool_batch_update",
    }
    return [t for t in registry.tools if t.name in allowed]


def get_write_tools(registry: MCPToolRegistry) -> list[BaseTool]:
    """Get write tools for Write Agent."""
    allowed = {
        "tool_update_cells",
        "tool_batch_update_cells",
        "tool_append_rows",
        "tool_add_rows",
        "tool_add_columns",
        "tool_clear_range",
    }
    return [t for t in registry.tools if t.name in allowed]
