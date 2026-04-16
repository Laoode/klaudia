import logging
from typing import Any

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START

from klaudia.core.supervisor.agents.data_entry_team.agents import make_data_entry_team_node
from klaudia.core.supervisor.agents.sql_agent.agent import make_sql_agent_node
from klaudia.core.supervisor.router import make_supervisor_node
from klaudia.core.supervisor.state import SupervisorState
from klaudia.interfaces.tool_registry import MCPToolRegistry
from klaudia.models.message import AgentResponse

logger = logging.getLogger(__name__)


class SupervisorAgent:
    """Klaudia supervisor: LangGraph hierarchical agent teams."""

    def __init__(
        self,
        llm_endpoint: str,
        llm_api_key: str,
        llm_model: str,
        mcp_sqlite: MCPToolRegistry,
        mcp_gsheets: MCPToolRegistry,
    ) -> None:
        self._llm = ChatOpenAI(
            model=llm_model,
            base_url=llm_endpoint,
            api_key=llm_api_key,
            temperature=0.5,
        )
        self._mcp_sqlite = mcp_sqlite
        self._mcp_gsheets = mcp_gsheets
        self._graph = self._build_graph()

    def _build_graph(self):
        """Build the LangGraph supervisor graph."""
        supervisor_node = make_supervisor_node(self._llm)
        sql_agent_node = make_sql_agent_node(self._llm, self._mcp_sqlite)
        data_entry_node = make_data_entry_team_node(self._llm, self._mcp_gsheets)

        builder = StateGraph(SupervisorState)
        builder.add_node("supervisor", supervisor_node)
        builder.add_node("sql_agent", sql_agent_node)
        builder.add_node("data_entry_team", data_entry_node)
        builder.add_edge(START, "supervisor")

        return builder.compile()

    async def process_conversation(
        self,
        messages: list[dict[str, Any]],
        extraction_data: dict[str, Any] | None = None,
    ) -> AgentResponse:
        """Run the supervisor graph on the given messages."""
        state = {
            "messages": messages,
            "extraction_data": extraction_data,
        }

        result = await self._graph.ainvoke(state, {"recursion_limit": 50})

        # Extract final AI message (skip human/system messages)
        final_msg = result["messages"][-1]
        if not isinstance(final_msg, AIMessage):
            logger.warning(f"Last message is {type(final_msg).__name__}, searching for last AIMessage")
            for msg in reversed(result["messages"]):
                if isinstance(msg, AIMessage):
                    final_msg = msg
                    break
        content = final_msg.content if hasattr(final_msg, "content") else str(final_msg)

        # Collect tools used from message names
        tools_used = []
        for msg in result["messages"]:
            name = getattr(msg, "name", None)
            if name and name not in ("user", "system"):
                tools_used.append(name)

        routed_to = result.get("next", "FINISH")

        return AgentResponse(
            content=content,
            tools_called=tools_used,
            metadata={"routed_to": routed_to},
        )
