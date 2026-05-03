import logging
from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END
from langgraph.types import Command
from typing_extensions import TypedDict

from klaudia.core.supervisor._content import coerce_to_text, strip_internal_markers
from klaudia.core.supervisor.prompts import SUPERVISOR_ROUTING_PROMPT
from klaudia.core.supervisor.state import SupervisorState

logger = logging.getLogger(__name__)

MEMBERS = ["sql_agent", "data_entry_team"]
OPTIONS = ["FINISH"] + MEMBERS

# Top-level workers whose reply triggers a deterministic FINISH+summarize.
# (Sub-agent worker names like read_agent/write_agent never reach this node;
# they are unwrapped inside the data_entry_team subgraph.)
_WORKER_NAMES = ("data_entry_team", "sql_agent")


_FINAL_REPLY_INSTRUCTION = (
    "You are now writing the user-facing reply as Klaudia. The previous "
    "assistant turn was a worker (data_entry_team / sql_agent) reporting "
    "back. Acknowledge concretely what changed (files written, rows "
    "deleted, query results, …) using the worker's text as ground truth. "
    "Match the user's language. Be concise and friendly. Do NOT repeat "
    "earlier assistant turns verbatim and do NOT echo internal tokens like "
    "[WRITE_DONE]/[READ_DONE]/[SHEET_DONE]/[CLARIFY]."
)


def _prepare_finish_messages(state_messages: list[Any]) -> list[Any]:
    """Build the message list passed to the final-reply LLM.

    - Strips internal completion markers from any worker (`name` in
      _WORKER_NAMES) message so Gemini is not distracted by tokens it has
      no instruction for.
    - Appends a single SystemMessage instructing the model to summarize the
      worker result. Appending (rather than prepending) is intentional:
      Gemini honours the most recent system instruction more reliably, and
      the orchestrator's persona system prompt already sits at index 0.
    """
    cleaned: list[Any] = []
    for msg in state_messages:
        name = getattr(msg, "name", None)
        if name in _WORKER_NAMES:
            text = strip_internal_markers(coerce_to_text(getattr(msg, "content", "")))
            if not text:
                # Nothing left after stripping — drop the message entirely
                # rather than feed an empty turn into Gemini.
                continue
            cleaned.append(HumanMessage(content=text, name=name))
        else:
            cleaned.append(msg)
    cleaned.append(SystemMessage(content=_FINAL_REPLY_INSTRUCTION))
    return cleaned


class Router(TypedDict):
    """Worker to route to next. Must be one of: FINISH, sql_agent, data_entry_team."""

    next: Literal["FINISH", "sql_agent", "data_entry_team"]


def make_supervisor_node(llm: BaseChatModel):
    """Create a supervisor node that routes between sub-agents."""

    def _emit_final_reply(state_messages: list[Any]) -> Command:
        final_llm = llm.with_config({"tags": ["final_answer"]})
        reply = final_llm.invoke(_prepare_finish_messages(state_messages))
        return Command(goto=END, update={"messages": [reply], "next": "FINISH"})

    def supervisor_node(state: SupervisorState) -> Command[Literal["sql_agent", "data_entry_team", "__end__"]]:
        state_messages = state["messages"]

        # Deterministic FINISH gate: if the latest message comes from a top-
        # level worker (data_entry_team / sql_agent), the work for this user
        # turn is done. Skip the router LLM call and go straight to the
        # summarization step. Prevents loops where the router LLM might
        # re-route to the same worker, and removes one LLM hop from the path.
        if state_messages:
            last = state_messages[-1]
            last_name = getattr(last, "name", None)
            if last_name in _WORKER_NAMES:
                logger.info(f"Supervisor: deterministic FINISH (worker reply from {last_name})")
                return _emit_final_reply(state_messages)

        messages = [
            {"role": "system", "content": SUPERVISOR_ROUTING_PROMPT},
        ] + state_messages

        router_llm = llm.with_structured_output(Router).with_config({"tags": ["nostream"]})
        response = router_llm.invoke(messages)
        goto = response["next"]

        # Treat any value not in MEMBERS as FINISH (LLM sometimes returns conversational text)
        if goto not in MEMBERS:
            logger.info(f"Supervisor routing to FINISH (raw: {goto!r})")
            return _emit_final_reply(state_messages)

        logger.info(f"Supervisor routing to: {goto}")
        return Command(goto=goto, update={"next": goto})

    return supervisor_node
