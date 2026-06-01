"""Multi-provider LLM client with auto-fallback.

Single entry point: `chat_with_fallback(messages, **kwargs)`. Walks a chain
of providers and returns the first successful response in an OpenAI-shaped
object (`resp.choices[0].message.content`). The chain is:

  1. Groq llama-3.3-70b-versatile  (primary — best quality, free tier)
  2. Groq llama-3.1-8b-instant     (smaller Groq model — separate daily quota)
  3. Groq gemma2-9b-it             (separate daily quota again)
  4. Anthropic Claude Haiku 4.5    (paid safety net when all Groq quotas hit)

Why this order: Groq's free tier is per-model (each model has its own
100K-tokens-per-day budget), so when 70B is exhausted the 8B and Gemma
budgets are usually still untouched. Only when ALL Groq models 429 do we
fall through to Anthropic (which is paid, so we want it last).

Callers do not need to know which provider answered — the response shape
is identical (`resp.choices[0].message.content`).
"""

import os
from openai import OpenAI

_GROQ_KEY  = (os.getenv("GROQ_API_KEY")      or "").strip()
_ANTHRO_KEY = (os.getenv("ANTHROPIC_API_KEY") or "").strip()

GROQ_PRIMARY = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_FALLBACK_MODELS = [GROQ_PRIMARY, "llama-3.1-8b-instant", "gemma2-9b-it"]

CLAUDE_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")

_groq = OpenAI(api_key=_GROQ_KEY or "missing", base_url="https://api.groq.com/openai/v1") if _GROQ_KEY else None

_anthropic = None
if _ANTHRO_KEY:
    try:
        import anthropic
        _anthropic = anthropic.Anthropic(api_key=_ANTHRO_KEY)
    except ImportError:
        _anthropic = None


def _is_rate_limit(err: Exception) -> bool:
    msg = str(err).lower()
    return any(s in msg for s in (
        "rate_limit", "429", "tokens per day", "tpd", "quota", "rate limit",
    ))


class _Msg:
    __slots__ = ("content", "role")
    def __init__(self, content: str, role: str = "assistant"):
        self.content = content
        self.role = role


class _Choice:
    __slots__ = ("message", "index", "finish_reason")
    def __init__(self, content: str):
        self.message = _Msg(content)
        self.index = 0
        self.finish_reason = "stop"


class _ClaudeResponse:
    """OpenAI-shaped wrapper around an Anthropic response."""
    def __init__(self, content: str, model: str):
        self.choices = [_Choice(content)]
        self.model = model


def _call_claude(messages, **kwargs):
    if _anthropic is None:
        raise RuntimeError("Anthropic SDK not configured — set ANTHROPIC_API_KEY and install `anthropic`")

    system_parts = [m["content"] for m in messages if m.get("role") == "system"]
    chat_msgs = []
    for m in messages:
        if m.get("role") == "system":
            continue
        role = m.get("role", "user")
        if role not in ("user", "assistant"):
            role = "user"
        chat_msgs.append({"role": role, "content": m.get("content", "")})

    if not chat_msgs:
        chat_msgs = [{"role": "user", "content": "Hello"}]

    params = {
        "model": CLAUDE_MODEL,
        "max_tokens": kwargs.get("max_tokens") or kwargs.get("max_completion_tokens") or 2048,
        "messages": chat_msgs,
    }
    if system_parts:
        params["system"] = "\n\n".join(system_parts)
    if "temperature" in kwargs:
        params["temperature"] = kwargs["temperature"]

    resp = _anthropic.messages.create(**params)
    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    return _ClaudeResponse(text, CLAUDE_MODEL)


def chat_with_fallback(messages, **kwargs):
    """Run a chat completion, falling back across providers on rate-limit errors.

    Strips any model kwarg the caller passed — this helper chooses the model.
    Non-rate-limit errors propagate immediately so real bugs surface.
    """
    kwargs.pop("model", None)

    last_err = None
    if _groq is not None:
        for model in GROQ_FALLBACK_MODELS:
            try:
                return _groq.chat.completions.create(model=model, messages=messages, **kwargs)
            except Exception as e:
                if not _is_rate_limit(e):
                    raise
                last_err = e
                print(f"[llm] Groq {model} rate-limited, trying next…")

    if _anthropic is not None:
        try:
            print(f"[llm] All Groq models exhausted — falling back to Claude {CLAUDE_MODEL}")
            return _call_claude(messages, **kwargs)
        except Exception as e:
            last_err = e
            print(f"[llm] Claude fallback failed: {e}")

    if last_err is not None:
        raise last_err
    raise RuntimeError("No LLM provider configured — set GROQ_API_KEY or ANTHROPIC_API_KEY")
