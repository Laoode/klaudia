import logging
from typing import Any, AsyncIterator, Optional

from langchain_core.messages import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
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
        llm_api_key: str,
        llm_model: str,
        mcp_sqlite: MCPToolRegistry,
        mcp_gsheets: MCPToolRegistry,
        langfuse: Optional[Any] = None,
    ) -> None:
        self._llm = ChatGoogleGenerativeAI(
            model=llm_model,
            google_api_key=llm_api_key,
            temperature=0.5,
        )
        self._mcp_sqlite = mcp_sqlite
        self._mcp_gsheets = mcp_gsheets
        self._langfuse = langfuse
        self._graph = self._build_graph()

    def _graph_config(
        self,
        session_id: Optional[int],
        user_id: Optional[int],
        run_name: str,
    ) -> dict[str, Any]:
        """Base RunnableConfig + optional Langfuse callback + trace metadata."""
        config: dict[str, Any] = {"recursion_limit": 50}
        if self._langfuse is not None:
            lf_cfg = self._langfuse.langchain_config(
                session_id=session_id,
                user_id=user_id,
                run_name=run_name,
                tags=["klaudia", "supervisor"],
            )
            # Merge (callbacks + metadata + run_name) without clobbering recursion_limit
            config.update(lf_cfg)
        return config

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
        session_id: int | None = None,
        user_id: int | None = None,
    ) -> AgentResponse:
        """Run the supervisor graph on the given messages."""
        state = {
            "messages": messages,
            "extraction_data": extraction_data,
        }

        config = self._graph_config(session_id, user_id, run_name="klaudia.supervisor.invoke")
        result = await self._graph.ainvoke(state, config)

        # Extract final AI message (skip human/system messages)
        final_msg = result["messages"][-1]
        if not isinstance(final_msg, AIMessage):
            logger.warning(f"Last message is {type(final_msg).__name__}, searching for last AIMessage")
            for msg in reversed(result["messages"]):
                if isinstance(msg, AIMessage):
                    final_msg = msg
                    break
        raw = final_msg.content if hasattr(final_msg, "content") else str(final_msg)
        content = raw if isinstance(raw, str) else str(raw)

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

    async def stream_conversation(
        self,
        messages: list[dict[str, Any]],
        extraction_data: dict[str, Any] | None = None,
        session_id: int | None = None,
        user_id: int | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream the supervisor graph as structured events.

        Event shape: {"type": <name>, "data": <payload>}
        Types emitted:
          - step:  graph node transition (node, next)
          - token: final-answer token (only tag="final_answer")
          - tool:  tool invocation observed in node output
          - final: end-of-stream aggregate (content, tools_called, metadata)
        """
        state = {
            "messages": messages,
            "extraction_data": extraction_data,
        }

        final_chunks: list[str] = []
        tools_used: list[str] = []
        routed_to = "FINISH"
        final_message_content: str | None = None

        stream_config = self._graph_config(session_id, user_id, run_name="klaudia.supervisor.stream")

        async for mode, payload in self._graph.astream(
            state,
            config=stream_config,
            stream_mode=["messages", "updates"],
        ):
            if mode == "messages":
                msg_chunk, metadata = payload
                tags = metadata.get("tags", []) or []
                if "nostream" in tags or "final_answer" not in tags:
                    continue
                text = getattr(msg_chunk, "content", "")
                if isinstance(text, str) and text:
                    final_chunks.append(text)
                    yield {"type": "token", "data": {"text": text}}
                continue

            if mode == "updates":
                for node, update in payload.items():
                    if not isinstance(update, dict):
                        continue
                    nxt = update.get("next")
                    if nxt is not None:
                        routed_to = nxt
                    yield {"type": "step", "data": {"node": node, "next": nxt}}
                    for msg in update.get("messages", []) or []:
                        name = getattr(msg, "name", None)
                        if name and name not in ("user", "system"):
                            tools_used.append(name)
                            yield {"type": "tool", "data": {"name": name}}
                        if isinstance(msg, AIMessage) and node == "supervisor":
                            raw = msg.content if isinstance(msg.content, str) else str(msg.content)
                            final_message_content = raw

        # Fallback to supervisor's final AIMessage if token-level streaming
        # didn't flow through callbacks (provider-dependent).
        content = "".join(final_chunks) or (final_message_content or "")

        yield {
            "type": "final",
            "data": {
                "content": content,
                "tools_called": tools_used,
                "metadata": {"routed_to": routed_to},
            },
        }
