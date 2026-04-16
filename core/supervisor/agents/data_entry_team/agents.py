import logging
from typing import Literal

from langchain_core.messages import HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from typing_extensions import TypedDict

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


def make_data_entry_team(llm: BaseChatModel, mcp_gsheets: MCPToolRegistry):
    """Build the data entry team subgraph."""

    # Create worker agents
    read_agent = create_react_agent(llm, tools=get_read_tools(mcp_gsheets), prompt=READ_AGENT_PROMPT)
    sheet_agent = create_react_agent(llm, tools=get_sheet_tools(mcp_gsheets), prompt=SHEET_AGENT_PROMPT)
    write_agent = create_react_agent(llm, tools=get_write_tools(mcp_gsheets), prompt=WRITE_AGENT_PROMPT)

    # Worker nodes — must be async because MCP tools only support async invocation.
    async def read_node(state: SupervisorState) -> Command[Literal["supervisor"]]:
        result = await read_agent.ainvoke(state)
        return Command(
            update={"messages": [HumanMessage(content=result["messages"][-1].content, name="read_agent")]},
            goto="supervisor",
        )

    async def sheet_node(state: SupervisorState) -> Command[Literal["supervisor"]]:
        result = await sheet_agent.ainvoke(state)
        return Command(
            update={"messages": [HumanMessage(content=result["messages"][-1].content, name="sheet_agent")]},
            goto="supervisor",
        )

    async def write_node(state: SupervisorState) -> Command[Literal["supervisor"]]:
        result = await write_agent.ainvoke(state)
        return Command(
            update={"messages": [HumanMessage(content=result["messages"][-1].content, name="write_agent")]},
            goto="supervisor",
        )

    # Team supervisor node
    def team_supervisor(state: SupervisorState) -> Command[Literal["read_agent", "sheet_agent", "write_agent", "__end__"]]:
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
        response = await team_graph.ainvoke({"messages": state["messages"][-1]})
        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content=response["messages"][-1].content,
                        name="data_entry_team",
                    )
                ]
            },
            goto="supervisor",
        )

    return data_entry_team_node
