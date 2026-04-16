from typing import Any

from langgraph.graph import MessagesState


class KlaudiaState(MessagesState):
    """Extended state for the Klaudia supervisor graph."""

    next: str = ""
    session_id: int = 0
    user_id: int = 1
    extraction_data: dict[str, Any] | None = None
    session_files: list[dict[str, Any]] | None = None
