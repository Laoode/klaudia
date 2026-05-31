"""LLM factory — chooses Gemini Developer API vs Vertex AI based on flag.
 
langchain-google-genai>=4.2 unifies both transports under ChatGoogleGenerativeAI:
pass vertexai=True plus project/location to route through Vertex AI; otherwise
the developer API + google_api_key is used.

thinking_level: if provided, binds a generation_config to the model so all
downstream calls (structured output, ainvoke, etc.) use the specified thinking
budget. None = leave model at its default (currently "high" for gemini-3-*).
 
Callers build two variants:
    routing_llm = build_chat_llm(..., thinking_level="minimal")  # router, team_supervisor, _emit_final_reply
    worker_llm  = build_chat_llm(..., thinking_level="low")      # write_agent, read_agent, sql_agent
"""
from __future__ import annotations

from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI

# Gemini 3 models expose thinking_level control via generation_config.
# Models that don't support it (e.g. older 1.x/2.x) will reject this kwarg;
# build_chat_llm skips binding when thinking_level is None so legacy model
# strings are unaffected.
_VALID_THINKING_LEVELS = {"none", "minimal", "low", "medium", "high"}

def build_chat_llm(
    *,
    model: str,
    temperature: float = 0.5,
    use_vertexai: bool = False,
    llm_api_key: Optional[str] = None,
    google_cloud_project: Optional[str] = None,
    google_cloud_location: Optional[str] = "global",
    thinking_level: Optional[str] = None,
) -> BaseChatModel:
    """Build a langchain chat model bound to either Vertex AI or Gemini Dev API.
 
    Args:
        thinking_level: Optional thinking budget level for Gemini 3 models.
            Pass None to use the model's default (high). Pass "none" to disable
            thinking entirely (faster, lower quality on complex tasks).
    """
    if use_vertexai:
        if not google_cloud_project:
            raise ValueError("use_vertexai=True but google_cloud_project is empty")
        llm: BaseChatModel = ChatGoogleGenerativeAI(
            model=model,
            vertexai=True,
            project=google_cloud_project,
            location=google_cloud_location or "global",
            temperature=temperature,
        )
    else:
        if not llm_api_key:
            raise ValueError("llm_api_key is required when use_vertexai=False")
        llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=llm_api_key,
            temperature=temperature,
        )
 
    if thinking_level is not None:
        level = thinking_level.lower()
        if level not in _VALID_THINKING_LEVELS:
            raise ValueError(
                f"Invalid thinking_level={thinking_level!r}. "
                f"Valid values: {sorted(_VALID_THINKING_LEVELS)}"
            )
        llm = llm.bind(
            generation_config={"thinking_config": {"thinking_level": level}}
        )
 
    return llm