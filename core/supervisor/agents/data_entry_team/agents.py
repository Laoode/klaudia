import logging
from typing import Literal

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

# Markers workers append to their final reply so the supervisor can detect
# completion deterministically (no LLM judgement on whether the task is done).
#
# TERMINAL markers definitively end the team's turn — the work is fully done
# (write/structural change committed) or the worker is blocked and needs the
# user (clarify). We short-circuit to END on these.
#
# NON_TERMINAL markers indicate a sub-step finished but the user's overall ask
# may still need another worker. [READ_DONE] is the only one today: a read-only
# request fully terminates on it, but a compound request like "dedup + add
# header" may pass through [READ_DONE] mid-flow if a worker over-uses it. We
# defer those cases to the LLM router instead of force-FINISH.
TERMINAL_MARKERS = ("[WRITE_DONE]", "[SHEET_DONE]", "[CLARIFY]")
NON_TERMINAL_MARKERS = ("[READ_DONE]",)
COMPLETION_MARKERS = TERMINAL_MARKERS + NON_TERMINAL_MARKERS


class TeamRouter(TypedDict):
    next: str


def _normalize_route(raw: str) -> str:
    """Extract just the agent name from an LLM response that may include extra detail.

    e.g. "read_agent.get_sheet_data(...)" -> "read_agent"
    """
    cleaned = raw.strip()
    for member in MEMBERS:
        if cleaned.startswith(member):
            return member
    if "FINISH" in cleaned.upper():
        return "FINISH"
    logger.warning(f"Unrecognized route '{raw}', defaulting to FINISH")
    return "FINISH"


def _has_completion_marker(content: str) -> bool:
    """True if the worker output carries any known marker (terminal or not)."""
    return any(marker in content for marker in COMPLETION_MARKERS)


def _is_terminal_marker(content: str) -> bool:
    """True if the worker's reply carries a marker that should END the team turn.

    [READ_DONE] alone is intentionally NOT terminal here — for compound requests
    (read → write to fulfil 'dedup + add header'), an early [READ_DONE] from a
    misrouted read_agent would otherwise short-circuit the team before the
    write happens. The LLM router decides what to do on [READ_DONE].
    """
    return any(marker in content for marker in TERMINAL_MARKERS)


# Markers that claim "I did some work". Emitting these without actually calling
# any tool is hallucination — Gemini sometimes "promises not actions". The
# worker-node guard rewrites such cases to [CLARIFY] so the user can redirect.
WORK_DONE_MARKERS = ("[WRITE_DONE]", "[SHEET_DONE]", "[READ_DONE]")


def _count_tool_messages(messages: list) -> int:
    return sum(1 for m in messages if isinstance(m, ToolMessage))


def _ground_marker_against_tool_calls(
    agent_name: str, messages: list, fallback_question: str
) -> str:
    """Return the worker's final text, but if it claims success without having
    called any tool, substitute a [CLARIFY] message instead of trusting the
    hallucinated marker. This is the last line of defense before the
    deterministic gate.
    """
    final_text = coerce_to_text(messages[-1].content)
    has_done_marker = any(m in final_text for m in WORK_DONE_MARKERS)
    if not has_done_marker:
        return final_text

    n_tool_calls = _count_tool_messages(messages)
    if n_tool_calls == 0:
        logger.warning(
            f"{agent_name} emitted a *_DONE marker without any tool calls — "
            f"rewriting to [CLARIFY]. Original final text: {final_text!r}"
        )
        return f"[CLARIFY] {fallback_question}"
    return final_text


def make_data_entry_team(llm: BaseChatModel, mcp_gsheets: MCPToolRegistry):
    """Build the data entry team subgraph."""

    # Create worker agents
    read_agent = create_react_agent(llm, tools=get_read_tools(mcp_gsheets), prompt=READ_AGENT_PROMPT)
    sheet_agent = create_react_agent(llm, tools=get_sheet_tools(mcp_gsheets), prompt=SHEET_AGENT_PROMPT)
    write_agent = create_react_agent(llm, tools=get_write_tools(mcp_gsheets), prompt=WRITE_AGENT_PROMPT)

    # Worker nodes — must be async because MCP tools only support async invocation.
    async def read_node(state: SupervisorState) -> Command[Literal["supervisor"]]:
        result = await read_agent.ainvoke(state)
        text = _ground_marker_against_tool_calls(
            "read_agent",
            result["messages"],
            "Saya kesulitan membaca sheet yang Anda maksud. Bisakah Anda menjelaskan ulang sheet/range yang ingin dibaca?",
        )
        return Command(
            update={"messages": [HumanMessage(content=text, name="read_agent")]},
            goto="supervisor",
        )

    async def sheet_node(state: SupervisorState) -> Command[Literal["supervisor"]]:
        result = await sheet_agent.ainvoke(state)
        text = _ground_marker_against_tool_calls(
            "sheet_agent",
            result["messages"],
            "Saya gagal mengeksekusi perubahan struktur sheet. Bisakah Anda menjelaskan ulang langkahnya?",
        )
        return Command(
            update={"messages": [HumanMessage(content=text, name="sheet_agent")]},
            goto="supervisor",
        )

    async def write_node(state: SupervisorState) -> Command[Literal["supervisor"]]:
        result = await write_agent.ainvoke(state)
        text = _ground_marker_against_tool_calls(
            "write_agent",
            result["messages"],
            "Saya tidak berhasil mengeksekusi operasi tulis tersebut. Bisakah Anda menjelaskan ulang langkahnya, atau memecahnya menjadi langkah yang lebih kecil?",
        )
        return Command(
            update={"messages": [HumanMessage(content=text, name="write_agent")]},
            goto="supervisor",
        )

    # Team supervisor node
    def team_supervisor(state: SupervisorState) -> Command[Literal["read_agent", "sheet_agent", "write_agent", "__end__"]]:
        # Deterministic completion gate — if the latest worker reply already carries
        # a TERMINAL marker, finish immediately without an LLM call. Prevents the
        # loop where the LLM re-routes to the same worker and triggers a duplicate
        # write. Non-terminal markers ([READ_DONE]) fall through to the LLM router
        # so compound flows (read→write) aren't cut short.
        msgs = state.get("messages") or []
        if msgs:
            last = msgs[-1]
            last_name = getattr(last, "name", None)
            last_content = coerce_to_text(getattr(last, "content", ""))
            if last_name in MEMBERS and _is_terminal_marker(last_content):
                logger.info(f"Team supervisor: deterministic FINISH (terminal marker from {last_name})")
                return Command(goto=END, update={"next": "FINISH"})

        messages = [
            {"role": "system", "content": DATA_ENTRY_SUPERVISOR_PROMPT},
        ] + state["messages"]
        response = llm.with_structured_output(TeamRouter).invoke(messages)
        goto = _normalize_route(response["next"])
        if goto == "FINISH":
            goto = END
        return Command(goto=goto, update={"next": goto})

    # Build subgraph
    builder = StateGraph(SupervisorState)
    builder.add_node("supervisor", team_supervisor)
    builder.add_node("read_agent", read_node)
    builder.add_node("sheet_agent", sheet_node)
    builder.add_node("write_agent", write_node)
    builder.add_edge(START, "supervisor")

    return builder.compile()


def make_data_entry_team_node(llm: BaseChatModel, mcp_gsheets: MCPToolRegistry):
    """Create the data entry team node for the top-level supervisor."""
    team_graph = make_data_entry_team(llm, mcp_gsheets)

    async def data_entry_team_node(state: SupervisorState) -> Command[Literal["supervisor"]]:
        # Forward full message history so team supervisor + workers can resolve
        # multi-turn references (e.g. sheet name mentioned in earlier turns).
        response = await team_graph.ainvoke({"messages": state["messages"]})
        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content=coerce_to_text(response["messages"][-1].content),
                        name="data_entry_team",
                    )
                ]
            },
            goto="supervisor",
        )

    return data_entry_team_node
