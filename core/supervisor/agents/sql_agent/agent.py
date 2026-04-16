import logging
from typing import Literal

from langchain_core.messages import HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

from klaudia.core.supervisor.agents.sql_agent.prompts import SQL_AGENT_PROMPT
from klaudia.core.supervisor.state import SupervisorState
from klaudia.interfaces.tool_registry import MCPToolRegistry
from klaudia.core.supervisor.tools.wrappers import get_sql_tools

logger = logging.getLogger(__name__)


def make_sql_agent_node(llm: BaseChatModel, mcp_sqlite: MCPToolRegistry):
    """Create an SQL agent node for the supervisor graph."""
    tools = get_sql_tools(mcp_sqlite)
    agent = create_react_agent(llm, tools=tools, prompt=SQL_AGENT_PROMPT)

    async def sql_agent_node(state: SupervisorState) -> Command[Literal["supervisor"]]:
        result = await agent.ainvoke(state)
        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content=result["messages"][-1].content,
                        name="sql_agent",
                    )
                ]
            },
            goto="supervisor",
        )

    return sql_agent_node
