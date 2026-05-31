import logging
from typing import Any, Callable, Literal, Optional

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from typing_extensions import TypedDict

from klaudia.core.supervisor._content import coerce_to_text
from klaudia.core.supervisor.agents.data_entry_team.prompts import (
    DATA_ENTRY_SUPERVISOR_PROMPT,
    READ_AGENT_PROMPT,
    SHEET_AGENT_PROMPT,
    WRITE_AGENT_PROMPT,
)
from klaudia.core.supervisor.state import SupervisorState
from klaudia.core.supervisor.tools.wrappers import (
    get_read_tools,
    get_sheet_tools,
    get_write_tools,
)
from klaudia.interfaces.tool_registry import MCPToolRegistry

logger = logging.getLogger(__name__)

MEMBERS = ["read_agent", "sheet_agent", "write_agent"]
VALID_ROUTES = {*MEMBERS, "FINISH"}

TERMINAL_MARKERS = ("[WRITE_DONE]", "[SHEET_DONE]", "[CLARIFY]")
NON_TERMINAL_MARKERS = ("[READ_DONE]",)
COMPLETION_MARKERS = TERMINAL_MARKERS + NON_TERMINAL_MARKERS
WORK_DONE_MARKERS = ("[WRITE_DONE]", "[SHEET_DONE]", "[READ_DONE]")


class TeamRouter(TypedDict):
    next: str


def _normalize_route(raw: str) -> str:
    cleaned = raw.strip()
    for member in MEMBERS:
        if cleaned.startswith(member):
            return member
    if "FINISH" in cleaned.upper():
        return "FINISH"
    logger.warning("Unrecognized route %r, defaulting to FINISH", raw)
    return "FINISH"


def _is_terminal_marker(content: str) -> bool:
    return any(marker in content for marker in TERMINAL_MARKERS)


def _count_tool_messages(messages: list) -> int:
    return sum(1 for m in messages if isinstance(m, ToolMessage))


def _ground_marker_against_tool_calls(
    agent_name: str, messages: list, fallback_question: str
) -> str:
    """Guard against hallucinated *_DONE markers with no actual tool calls."""
    final_text = coerce_to_text(messages[-1].content)
    has_done_marker = any(m in final_text for m in WORK_DONE_MARKERS)
    if not has_done_marker:
        return final_text
    if _count_tool_messages(messages) == 0:
        logger.warning(
            "%s emitted a *_DONE marker without any tool calls — rewriting to [CLARIFY]. "
            "Original: %r",
            agent_name,
            final_text,
        )
        return f"[CLARIFY] {fallback_question}"
    return final_text


def make_data_entry_team(
    routing_llm: BaseChatModel,
    worker_llm: BaseChatModel,
    mcp_gsheets: MCPToolRegistry,
):
    """Build the data entry team subgraph.

    routing_llm: pre-bound with minimal thinking (team_supervisor routing call)
    worker_llm:  pre-bound with low thinking (create_react_agent workers)
    """
    read_agent = create_react_agent(worker_llm, tools=get_read_tools(mcp_gsheets), prompt=READ_AGENT_PROMPT)
    sheet_agent = create_react_agent(worker_llm, tools=get_sheet_tools(mcp_gsheets), prompt=SHEET_AGENT_PROMPT)
    write_agent = create_react_agent(worker_llm, tools=get_write_tools(mcp_gsheets), prompt=WRITE_AGENT_PROMPT)

    async def read_node(state: SupervisorState) -> Command[Literal["supervisor"]]:
        result = await read_agent.ainvoke(state)
        text = _ground_marker_against_tool_calls(
            "read_agent",
            result["messages"],
            "Saya kesulitan membaca sheet yang Anda maksud. Bisakah Anda menjelaskan ulang?",
        )
        return Command(update={"messages": [HumanMessage(content=text, name="read_agent")]}, goto="supervisor")

    async def sheet_node(state: SupervisorState) -> Command[Literal["supervisor"]]:
        result = await sheet_agent.ainvoke(state)
        text = _ground_marker_against_tool_calls(
            "sheet_agent",
            result["messages"],
            "Saya gagal mengeksekusi perubahan struktur sheet. Bisakah Anda menjelaskan ulang?",
        )
        return Command(update={"messages": [HumanMessage(content=text, name="sheet_agent")]}, goto="supervisor")

    async def write_node(state: SupervisorState) -> Command[Literal["supervisor"]]:
        result = await write_agent.ainvoke(state)
        text = _ground_marker_against_tool_calls(
            "write_agent",
            result["messages"],
            "Saya tidak berhasil mengeksekusi operasi tulis. Bisakah Anda menjelaskan ulang?",
        )
        return Command(update={"messages": [HumanMessage(content=text, name="write_agent")]}, goto="supervisor")

    async def team_supervisor(
        state: SupervisorState,
    ) -> Command[Literal["read_agent", "sheet_agent", "write_agent", "__end__"]]:
        msgs = state.get("messages") or []
        if msgs:
            last = msgs[-1]
            last_name = getattr(last, "name", None)
            last_content = coerce_to_text(getattr(last, "content", ""))
            if last_name in MEMBERS and _is_terminal_marker(last_content):
                logger.info(
                    "Team supervisor: deterministic FINISH (terminal marker from %s)", last_name
                )
                return Command(goto=END, update={"next": "FINISH"})

        messages = [{"role": "system", "content": DATA_ENTRY_SUPERVISOR_PROMPT}] + state["messages"]
        # routing_llm is pre-bound with minimal thinking — classification task only.
        response = await routing_llm.with_structured_output(TeamRouter).ainvoke(messages)
        goto = _normalize_route(response["next"])
        if goto == "FINISH":
            goto = END
        return Command(goto=goto, update={"next": goto})

    builder = StateGraph(SupervisorState)
    builder.add_node("supervisor", team_supervisor)
    builder.add_node("read_agent", read_node)
    builder.add_node("sheet_agent", sheet_node)
    builder.add_node("write_agent", write_node)
    builder.add_edge(START, "supervisor")

    return builder.compile()


def make_data_entry_team_node(
    routing_llm: BaseChatModel,
    worker_llm: BaseChatModel,
    mcp_gsheets: MCPToolRegistry,
    on_sheet_mutation: Optional[Callable[[], None]] = None,
):
    """Create the data entry team node for the top-level supervisor.

    on_sheet_mutation: optional zero-arg callback fired when the team reports
    [SHEET_DONE]. Used by SupervisorAgent to invalidate the sheet list cache.
    """
    team_graph = make_data_entry_team(routing_llm, worker_llm, mcp_gsheets)

    async def data_entry_team_node(state: SupervisorState) -> Command[Literal["supervisor"]]:
        response = await team_graph.ainvoke({"messages": state["messages"]})
        last_content = coerce_to_text(response["messages"][-1].content)

        # Cache invalidation: structural sheet change detected.
        if on_sheet_mutation is not None and "[SHEET_DONE]" in last_content:
            logger.info("Sheet mutation detected — invalidating sheet list cache")
            on_sheet_mutation()

        return Command(
            update={"messages": [HumanMessage(content=last_content, name="data_entry_team")]},
            goto="supervisor",
        )

    return data_entry_team_node