"""LLM factory — selects the chat backend by provider.

Two providers are supported behind one factory so the rest of the agentic
graph (supervisor, routers, react workers) stays provider-agnostic:

    provider="google"  → ChatGoogleGenerativeAI
        Dual transport: pass use_vertexai=True (+ project/location) to route via
        GCP Vertex AI, otherwise the Gemini Developer API + google_api_key.
        Thinking is controlled with thinking_level (Gemini 3 generation_config).

    provider="openai"  → ChatOpenAI pointed at an OpenAI-compatible server
        (our local vLLM running Qwen). thinking_level is ignored; thinking is
        disabled via extra_body chat_template_kwargs instead (see below).

thinking_level (Gemini only): if provided, binds a generation_config so all
downstream calls (structured output, ainvoke, …) use that thinking budget.
None = leave the model at its default (currently "high" for gemini-3-*).

disable_thinking (OpenAI/vLLM only): when True, every request carries
extra_body={"chat_template_kwargs": {"enable_thinking": False}}. This is the
in-code equivalent of Qwen's `/no_think` soft switch — it forces thinking off
regardless of how the vLLM server was started, without mutating any prompt.

Callers build two variants:
    routing_llm = build_chat_llm(..., thinking_level="minimal")  # router, team_supervisor, _emit_final_reply
    worker_llm  = build_chat_llm(..., thinking_level="low")      # write_agent, read_agent, sql_agent
"""
from __future__ import annotations

from typing import Any, Optional

from langchain_core.language_models.chat_models import BaseChatModel

# Gemini 3 models expose thinking_level control via generation_config.
# Models that don't support it (e.g. older 1.x/2.x) will reject this kwarg;
# build_chat_llm skips binding when thinking_level is None so legacy model
# strings are unaffected.
_VALID_THINKING_LEVELS = {"none", "minimal", "low", "medium", "high"}

# Qwen / vLLM: disable the reasoning trace at the chat-template level. Sent as
# extra_body so it rides on every request the OpenAI client makes. Mirrors the
# `/no_think` inline switch but keeps prompts untouched.
_QWEN_NO_THINK_EXTRA_BODY: dict[str, Any] = {
    "chat_template_kwargs": {"enable_thinking": False}
}

# Placeholder sent when no bearer token is configured. The OpenAI client rejects
# an empty api_key, but a self-hosted vLLM started without --api-key ignores the
# value entirely. Swap in a real token via LLM_OPENAI_API_KEY when vLLM enforces auth.
_VLLM_PLACEHOLDER_KEY = "EMPTY"


def build_chat_llm(
    *,
    model: str,
    provider: str = "google",
    temperature: float = 0.5,
    use_vertexai: bool = False,
    llm_api_key: Optional[str] = None,
    google_cloud_project: Optional[str] = None,
    google_cloud_location: Optional[str] = "global",
    openai_base_url: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    thinking_level: Optional[str] = None,
    disable_thinking: bool = True,
) -> BaseChatModel:
    """Build a chat model for the configured provider.

    Args:
        provider: "google" (Gemini) or "openai" (OpenAI-compatible / vLLM).
        thinking_level: Gemini-only thinking budget. None = model default.
        disable_thinking: OpenAI/vLLM-only. Force thinking off via extra_body.
    """
    normalized = (provider or "google").strip().lower()
    if normalized == "openai":
        return _build_openai_llm(
            model=model,
            temperature=temperature,
            base_url=openai_base_url,
            api_key=openai_api_key,
            disable_thinking=disable_thinking,
        )
    if normalized in ("google", "gemini", "vertexai"):
        return _build_gemini_llm(
            model=model,
            temperature=temperature,
            use_vertexai=use_vertexai,
            llm_api_key=llm_api_key,
            google_cloud_project=google_cloud_project,
            google_cloud_location=google_cloud_location,
            thinking_level=thinking_level,
        )
    raise ValueError(
        f"Unknown MODEL_PROVIDER={provider!r}. Valid values: 'google', 'openai'."
    )


def _build_openai_llm(
    *,
    model: str,
    temperature: float,
    base_url: Optional[str],
    api_key: Optional[str],
    disable_thinking: bool,
) -> BaseChatModel:
    """Build a ChatOpenAI bound to an OpenAI-compatible endpoint (vLLM)."""
    from langchain_openai import ChatOpenAI

    if not base_url:
        raise ValueError(
            "MODEL_PROVIDER=openai requires LLM_ENDPOINT (the vLLM /v1 base URL)."
        )

    extra_body = dict(_QWEN_NO_THINK_EXTRA_BODY) if disable_thinking else None
    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key or _VLLM_PLACEHOLDER_KEY,
        temperature=temperature,
        extra_body=extra_body,
    )


def _build_gemini_llm(
    *,
    model: str,
    temperature: float,
    use_vertexai: bool,
    llm_api_key: Optional[str],
    google_cloud_project: Optional[str],
    google_cloud_location: Optional[str],
    thinking_level: Optional[str],
) -> BaseChatModel:
    """Build a ChatGoogleGenerativeAI bound to Vertex AI or the Gemini Dev API."""
    from langchain_google_genai import ChatGoogleGenerativeAI

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


def _is_openai_llm(llm: Any) -> bool:
    """True if llm is (or wraps) a ChatOpenAI instance.

    Unwraps RunnableBinding layers (.bind()/.with_config()) so the check holds
    even after the model has been decorated.
    """
    base = llm
    while hasattr(base, "bound"):
        base = base.bound
    return base.__class__.__name__ == "ChatOpenAI"


def with_structured(llm: BaseChatModel, schema: Any):
    """Provider-agnostic structured output.

    Gemini uses its native structured-output path. OpenAI-compatible servers
    (vLLM) use method="json_schema" so the constraint rides on response_format /
    guided decoding instead of tool-calling — this keeps the routers working
    even when the vLLM server is started without --enable-auto-tool-choice.
    """
    if _is_openai_llm(llm):
        return llm.with_structured_output(schema, method="json_schema")
    return llm.with_structured_output(schema)
