"""Shared ADK model helpers for the SEMS agents."""

import asyncio
import logging
from typing import AsyncGenerator

from google.adk.models.lite_llm import LiteLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse

try:
    # RateLimitError is what LiteLLM raises for an HTTP 429; import it so we can
    # distinguish rate limits (retry after waiting) from every other error
    # (re-raise immediately). Guarded so a litellm layout change can't break
    # import — we fall back to message sniffing below.
    from litellm.exceptions import RateLimitError as _LiteLlmRateLimitError
except Exception:  # pragma: no cover - defensive
    _LiteLlmRateLimitError = None

logger = logging.getLogger(__name__)

# Databricks enforces a *per-minute* workspace input-token quota. A retry that
# fires inside the same blown minute just fails again, so a rate-limit retry
# must wait out the ~60s window. We honor the server's Retry-After when present
# and otherwise default to a full window, growing the wait on repeated hits.
_RATE_LIMIT_MAX_RETRIES = 5
_RATE_LIMIT_BASE_WAIT = 60.0  # seconds — one per-minute window
_RATE_LIMIT_MAX_WAIT = 150.0  # cap so a stuck loop can't sleep forever


def _is_rate_limit(exc: BaseException) -> bool:
    """True if exc is a rate-limit / request-limit error worth waiting out."""
    if _LiteLlmRateLimitError is not None and isinstance(exc, _LiteLlmRateLimitError):
        return True
    msg = str(exc).lower()
    return (
        "rate limit" in msg
        or "request limit exceeded" in msg
        or "token per minute" in msg
        or " 429" in msg
    )


def _retry_after_seconds(exc: BaseException) -> float | None:
    """Extract a server-provided Retry-After (seconds) from the error, if any."""
    val = getattr(exc, "retry_after", None)
    if val is None:
        resp = getattr(exc, "response", None)
        headers = getattr(resp, "headers", None) or {}
        try:
            val = headers.get("retry-after") or headers.get("Retry-After")
        except AttributeError:
            val = None
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


class ForceToolLiteLlm(LiteLlm):
    """LiteLlm that forces tool_choice='required' until the named tool has run.

    Maverick (non-reasoning) tends to answer in plain text instead of calling
    its tool, so the first request must carry tool_choice='required'; once the
    request history contains a function response from ``forced_tool``, the
    model relaxes to 'auto' so it can produce the final report.

    tool_choice is computed per request from the request's own contents —
    NOT mutated on shared state from a before_model_callback — so concurrent
    sessions sharing this model instance cannot clobber each other's setting.
    """

    forced_tool: str = ""

    def __init__(self, model: str, *, forced_tool: str, **kwargs):
        # forced_tool is consumed here; it must not reach LiteLlm's kwargs,
        # which are forwarded verbatim to the litellm completion API.
        super().__init__(model="databricks/databricks-claude-sonnet-4-6", **kwargs)
        self.forced_tool = forced_tool

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        called = any(
            getattr(part.function_response, "name", None) == self.forced_tool
            for content in llm_request.contents
            for part in content.parts or []
        )
        attempt = 0
        while True:
            # Give the request its own copy of the model with a private
            # _additional_args dict: the base implementation reads that dict
            # after an await, so mutating it in place would race across sessions.
            per_request = self.model_copy()
            per_request._additional_args = dict(self._additional_args)
            per_request._additional_args["tool_choice"] = (
                "auto" if called else "required"
            )
            yielded = False
            try:
                async for response in LiteLlm.generate_content_async(
                    per_request, llm_request, stream
                ):
                    yielded = True
                    yield response
                return
            except Exception as exc:  # noqa: BLE001 - re-raised unless a rate limit
                # Only swallow rate limits, and only before any token was
                # yielded (a 429 fires at request start; retrying mid-stream
                # would duplicate output). Everything else propagates.
                if (
                    yielded
                    or not _is_rate_limit(exc)
                    or attempt >= _RATE_LIMIT_MAX_RETRIES
                ):
                    raise
                wait = _retry_after_seconds(exc)
                if wait is None:
                    # No Retry-After: wait a full window, growing on repeats.
                    wait = _RATE_LIMIT_BASE_WAIT + attempt * 15.0
                wait = min(wait, _RATE_LIMIT_MAX_WAIT)
                attempt += 1
                logger.warning(
                    "Databricks rate limit hit (attempt %d/%d); sleeping %.0fs "
                    "to clear the per-minute window before retrying",
                    attempt,
                    _RATE_LIMIT_MAX_RETRIES,
                    wait,
                )
                await asyncio.sleep(wait)