"""LLM factory — chooses Gemini Developer API vs Vertex AI based on flag.

Both branches return a langchain ``BaseChatModel`` so downstream graph nodes
(supervisor, sql_agent, data_entry_team) consume them uniformly.

langchain-google-genai>=4.2 unifies both transports under
``ChatGoogleGenerativeAI``: pass ``vertexai=True`` plus ``project``/``location``
to route through Vertex AI; otherwise the developer API + ``google_api_key`` is
used. The legacy ``ChatVertexAI`` (langchain-google-vertexai) is deprecated
since langchain-google-vertexai 3.2.0 in favor of this path.
"""

from __future__ import annotations

from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI


def build_chat_llm(
    *,
    model: str,
    temperature: float = 0.5,
    use_vertexai: bool = False,
    llm_api_key: Optional[str] = None,
    google_cloud_project: Optional[str] = None,
    google_cloud_location: Optional[str] = "global",
) -> BaseChatModel:
    """Build a langchain chat model bound to either Vertex AI or Gemini Dev API.

    Vertex path requires ``google_cloud_project``; ADC is read from the env var
    ``GOOGLE_APPLICATION_CREDENTIALS`` set at process startup. The Dev API path
    requires ``llm_api_key``.
    """
    if use_vertexai:
        if not google_cloud_project:
            raise ValueError(
                "use_vertexai=True but google_cloud_project is empty"
            )
        return ChatGoogleGenerativeAI(
            model=model,
            vertexai=True,
            project=google_cloud_project,
            location=google_cloud_location or "global",
            temperature=temperature,
        )

    if not llm_api_key:
        raise ValueError("llm_api_key is required when use_vertexai=False")

    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=llm_api_key,
        temperature=temperature,
    )
