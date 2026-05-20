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

# Minimal thinking config for latency-critical paths (routing + final reply).
# These calls don't need deep reasoning — routing is classification, final reply
# is light summarization. Default thinking level on gemini-3-* is "high", which
# generates ~1000 internal tokens and adds 7-10s for no visible quality gain here.
#
# Applied via .bind() so it only affects these specific calls, not the workers
# (write_agent benefits from reasoning through composition patterns).
_MINIMAL_THINK: dict[str, Any] = {
    "generation_config": {"thinking_config": {"thinking_level": "minimal"}}
}

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


class RouterWithResponse(TypedDict):
    """Combined routing + response for initial FINISH decisions.
 
    When `next` is FINISH and no worker has run yet, `response` contains the
    user-facing reply — skipping the second _emit_final_reply LLM call.
    When routing to a worker, `response` must be "".
    """
    next: Literal["FINISH", "sql_agent", "data_entry_team"]
    response: str


def make_supervisor_node(llm: BaseChatModel):
    """Create a supervisor node that routes between sub-agents."""

    async def _emit_final_reply(state_messages: list[Any]) -> Command:
        # Fix: bind minimal thinking — this call is pure summarization,
        # not reasoning. Reduces 7-10s thinking overhead to ~1-2s.
        final_llm = (
            llm
            .bind(**_MINIMAL_THINK)
            .with_config({"tags": ["final_answer"]})
        )
        reply = await final_llm.ainvoke(_prepare_finish_messages(state_messages))
        return Command(goto=END, update={"messages": [reply], "next": "FINISH"})

    async def supervisor_node(state: SupervisorState) -> Command[...]:
        state_messages = state["messages"]

        # Deterministic FINISH gate — worker just reported, skip router call.
        if state_messages:
            last = state_messages[-1]
            last_name = getattr(last, "name", None)
            if last_name in _WORKER_NAMES:
                logger.info(f"Supervisor: deterministic FINISH (worker reply from {last_name})")
                return await _emit_final_reply(state_messages)

        # Combined routing + response call — one LLM hop instead of two for FINISH path
        # Fix: bind minimal thinking for routing (classification task, ~100 tokens output).
        messages = [
            {"role": "system", "content": SUPERVISOR_ROUTING_PROMPT},
        ] + state_messages

        combined_llm = (
            llm
            .bind(**_MINIMAL_THINK)
            .with_structured_output(RouterWithResponse)
            .with_config({"tags": ["nostream"]})
        )
        result = await combined_llm.ainvoke(messages)
        goto = result.get("next", "FINISH")
        inline_response = (result.get("response") or "").strip()

        if goto not in MEMBERS:
            if inline_response:
                from langchain_core.messages import AIMessage
                logger.info(f"Supervisor: FINISH with inline response ({len(inline_response)} chars)")
                return Command(
                    goto=END,
                    update={"messages": [AIMessage(content=inline_response)], "next": "FINISH"},
                )
            logger.info(f"Supervisor routing to FINISH (raw: {goto!r}), generating reply")
            return await _emit_final_reply(state_messages)

        logger.info(f"Supervisor routing to: {goto}")
        return Command(goto=goto, update={"next": goto})

    return supervisor_node