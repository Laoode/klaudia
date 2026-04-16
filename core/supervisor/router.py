import logging
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END
from langgraph.types import Command
from typing_extensions import TypedDict

from klaudia.core.supervisor.prompts import SUPERVISOR_ROUTING_PROMPT
from klaudia.core.supervisor.state import SupervisorState

logger = logging.getLogger(__name__)

MEMBERS = ["sql_agent", "data_entry_team"]
OPTIONS = ["FINISH"] + MEMBERS


class Router(TypedDict):
    """Worker to route to next. Must be one of: FINISH, sql_agent, data_entry_team."""

    next: Literal["FINISH", "sql_agent", "data_entry_team"]


def make_supervisor_node(llm: BaseChatModel):
    """Create a supervisor node that routes between sub-agents."""

    def supervisor_node(state: SupervisorState) -> Command[Literal["sql_agent", "data_entry_team", "__end__"]]:
        messages = [
            {"role": "system", "content": SUPERVISOR_ROUTING_PROMPT},
        ] + state["messages"]

        response = llm.with_structured_output(Router).invoke(messages)
        goto = response["next"]

        # Treat any value not in MEMBERS as FINISH (LLM sometimes returns conversational text)
        if goto not in MEMBERS:
            logger.info(f"Supervisor routing to FINISH (raw: {goto!r})")
            reply = llm.invoke(state["messages"])
            return Command(goto=END, update={"messages": [reply], "next": "FINISH"})

        logger.info(f"Supervisor routing to: {goto}")
        return Command(goto=goto, update={"next": goto})

    return supervisor_node
