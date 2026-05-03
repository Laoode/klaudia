"""Content coercion helper shared across supervisor + worker nodes.

Lives in its own module to avoid circular imports between ``agent.py``
(top-level supervisor) and ``agents/*`` (workers). Both depend on it.
"""

import re
from typing import Any


def coerce_to_text(content: Any) -> str:
    """Coerce a LangChain message content payload into a plain string.

    Gemini (and other multimodal LLMs) can return ``content`` as either a
    ``str`` or a list of part-dicts like
    ``[{"type": "text", "text": "...", "extras": {"signature": "..."}}]``.
    Naively ``str()``-ing the list yields a Python ``repr`` (single quotes,
    nested ``extras`` blob) which leaks to users when re-wrapped as a
    ``HumanMessage(content=...)`` for the supervisor.

    Rules:
    - ``str`` → returned trimmed.
    - ``list`` → concatenate ``item["text"]`` for each part dict; raw string
      parts pass through. Anything without a ``text`` key is dropped.
    - ``None`` → empty string.
    - Other types → ``str()`` then trimmed (last-resort).
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                txt = item.get("text") if "text" in item else ""
                if isinstance(txt, str):
                    parts.append(txt)
        return "".join(parts).strip()
    return str(content).strip()


# Strip only the bracketed marker token; preserve any inline summary that the
# worker wrote alongside the marker. Stripping the whole line would drop the
# user-facing summary and break the named-worker fallback in
# `_resolve_final_content` whenever the worker's reply was just one line.
_INTERNAL_MARKER_TOKEN_RE = re.compile(
    r"\[(?:WRITE_DONE|READ_DONE|SHEET_DONE|CLARIFY)\]\s*"
)


def strip_internal_markers(text: str) -> str:
    """Remove [WRITE_DONE]/[READ_DONE]/[SHEET_DONE]/[CLARIFY] tokens while
    keeping the human-readable summary the worker emitted next to them.

    Workers append these markers so the team supervisor's deterministic
    completion gate can fire without an LLM call. Users must never see the
    brackets, but the one-line summary IS valid content for them.
    """
    if not text:
        return ""
    cleaned = _INTERNAL_MARKER_TOKEN_RE.sub("", text)
    return "\n".join(l.rstrip() for l in cleaned.splitlines() if l.strip()).strip()
