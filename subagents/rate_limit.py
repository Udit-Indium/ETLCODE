from __future__ import annotations
import asyncio
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterable, Sequence

logger = logging.getLogger(__name__)

DEFAULT_INPUT_TOKENS_PER_MINUTE = 40_000
DEFAULT_OUTPUT_TOKENS_PER_MINUTE = 8_000
DEFAULT_REQUESTS_PER_MINUTE = 30
DEFAULT_BURST_FRACTION = float(os.environ.get("RATE_LIMIT_BURST_FRACTION", "0.25"))
MAX_WAIT_SECONDS = 300.0
MAX_RETRY_AFTER_SECONDS = 120.0
DEFAULT_MAX_ATTEMPTS = 4
_TOKEN_EPSILON = 1e-6
_MIN_DELAY_SECONDS = 1e-3

MODEL_FALLBACKS: dict[str, list[str]] = {
    "databricks/databricks-claude-opus-4-7": [
        "databricks/databricks-claude-sonnet-4-6",
        "databricks/databricks-claude-sonnet-4-5",
    ],
    "databricks/databricks-claude-sonnet-4-6": [
        "databricks/databricks-claude-sonnet-4-5",
    ],
}
FALLBACKS_ENABLED = os.environ.get(
    "RATE_LIMIT_FALLBACKS", "1"
).strip().lower() not in {"0", "false", "no", "off"}


def estimate_tokens(text: str) -> int:
    """Rough input-token estimate for `text`.

    Deliberately crude (~4 chars per token) and deliberately NOT a tokenizer
    call: this runs before every request, and an exact count would cost a round
    trip to refine a budget that is itself a conservative guess.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def estimate_message_tokens(messages: Iterable[Any]) -> int:
    """Estimate the prompt tokens of a LiteLLM `messages` list.

    Handles both content shapes: a bare string, and the list-of-blocks form
    used for structured content.
    """
    total = 0
    for message in messages or ():
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, Sequence):
            for block in content:
                if isinstance(block, dict):
                    total += estimate_tokens(block.get("text") or "")
        total += 4
    return total


def _env_key(model: str) -> str:
    """Environment-variable suffix for an endpoint name."""
    return re.sub(r"[^0-9A-Za-z]+", "_", model).strip("_").upper()


def _endpoint(model: str) -> str:
    """The bare serving-endpoint name, with any provider prefix stripped.

    "databricks/databricks-claude-opus-4-7" and the bare endpoint name are one
    quota and one cooldown, so the limiter and the router both key on this
    rather than on whichever spelling the caller happened to use.
    """
    return str(model).split("/")[-1]


def _configured(prefix: str, model: str, fallback: int) -> int:
    """Read a per-endpoint quota from the environment."""
    raw = os.environ.get(f"{prefix}_{_env_key(model)}")
    if raw is None:
        return fallback
    try:
        value = int(raw)
    except ValueError:
        return fallback
    return value if value > 0 else fallback


@dataclass
class TokenBucket:
    """A refilling bucket with burst capacity.

    Capacity is what allows a burst: the bucket starts full, so the first
    `capacity` units cost no wait. Once drained, acquisitions are paced by
    `rate_per_minute`.

    The clock and sleep function are injected so behaviour can be exercised
    against a virtual clock — a limiter tested against the wall clock is a test
    that either takes minutes or proves nothing.
    """

    rate_per_minute: float
    capacity: float
    now: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    asleep: Callable[[float], Awaitable[None]] = asyncio.sleep
    _tokens: float = field(default=0.0, init=False)
    _updated: float = field(default=0.0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self) -> None:
        self._tokens = float(self.capacity)
        self._updated = self.now()

    @property
    def _per_second(self) -> float:
        return self.rate_per_minute / 60.0

    def _refill_locked(self) -> None:
        """Add tokens accrued since the last check. Caller holds the lock."""
        current = self.now()
        elapsed = max(0.0, current - self._updated)
        self._updated = current
        self._tokens = min(self.capacity, self._tokens + elapsed * self._per_second)

    def _clamp(self, amount: float) -> float:
        """An amount larger than the bucket can never be satisfied, so clamp it
        rather than deadlock — one oversized request should be slow, not fatal.
        """
        return min(max(0.0, float(amount)), float(self.capacity))

    def _step(self, amount: float, waited: float) -> tuple[bool, float]:
        """One acquisition attempt.

        Returns `(acquired, delay)`. The lock is released before the caller
        waits, which is what lets the sync and async paths share this policy —
        neither ever sleeps while holding it.
        """
        with self._lock:
            self._refill_locked()
            if self._tokens + _TOKEN_EPSILON >= amount:
                self._tokens = max(0.0, self._tokens - amount)
                return True, 0.0
            shortfall = amount - self._tokens
            per_second = self._per_second

        if per_second <= 0:
            raise RuntimeError(f"bucket for rate {self.rate_per_minute} cannot refill")

        delay = min(shortfall / per_second, MAX_WAIT_SECONDS - waited)
        if delay <= 0:
            raise RuntimeError(
                f"rate limiter waited {waited:.0f}s for {amount:.0f} units and gave up; "
                "the configured quota is too small for this request"
            )
        return False, max(delay, _MIN_DELAY_SECONDS)

    def acquire(self, amount: float = 1.0) -> float:
        """Block until `amount` is available; return the seconds spent waiting."""
        amount = self._clamp(amount)
        if amount == 0.0:
            return 0.0
        waited = 0.0
        while True:
            acquired, delay = self._step(amount, waited)
            if acquired:
                return waited
            self.sleep(delay)
            waited += delay

    async def acquire_async(self, amount: float = 1.0) -> float:
        """Await until `amount` is available; return the seconds spent waiting.

        The async twin of `acquire`, and the one ADK must use. The sync version
        blocks the calling thread — inside a coroutine that is the event loop
        itself, so throttling one agent would stall every other concurrent
        request rather than just its own.
        """
        amount = self._clamp(amount)
        if amount == 0.0:
            return 0.0
        waited = 0.0
        while True:
            acquired, delay = self._step(amount, waited)
            if acquired:
                return waited
            await self.asleep(delay)
            waited += delay

    def debit(self, amount: float) -> None:
        """Charge tokens after the fact, even past zero.

        Used when a response spent MORE than was reserved — possible whenever
        the burst capacity is smaller than a single request's real output, since
        `acquire` clamps the reservation to capacity. Going negative is the
        point: the deficit is carried, and the next acquisition waits for it to
        refill, so the long-run rate stays honest instead of silently drifting
        over quota.
        """
        if amount <= 0:
            return
        with self._lock:
            self._refill_locked()
            self._tokens -= float(amount)

    def release(self, amount: float) -> None:
        """Hand unused capacity back, never exceeding the bucket's ceiling."""
        if amount <= 0:
            return
        with self._lock:
            self._refill_locked()
            self._tokens = min(self.capacity, self._tokens + float(amount))


@dataclass
class Reservation:
    """Budget held for one in-flight request.

    Created by `EndpointLimiter.reserve`, closed by `settle`. Until it settles,
    the full `prompt_tokens + max_tokens` is charged against the bucket, so a
    concurrent caller cannot spend the same headroom.
    """

    limiter: "EndpointLimiter"
    prompt_tokens: int
    max_tokens: int
    output_charged: int = 0
    waited_seconds: float = 0.0
    _settled: bool = field(default=False, init=False)

    @property
    def reserved(self) -> int:
        """Total tokens held: the estimated prompt plus the output ceiling."""
        return self.prompt_tokens + self.max_tokens

    def settle(self, usage: Any = None) -> int:
        """Release the reserved budget the response did not use.

        Args:
            usage: the response's usage object or dict, if available. Only the
                OUTPUT budget is reconciled — the prompt estimate stays charged
                even when the reported prompt is smaller, because releasing on
                an under-estimate would let a systematically low estimator
                inflate the effective quota.

        Returns:
            Tokens released. Zero when the response used its full ceiling, or
            when usage is unavailable and the conservative reservation stands.
        """
        if self._settled:
            return 0
        self._settled = True

        completion = _usage_field(usage, "completion_tokens", "output_tokens")
        if completion is None:
            return 0

        charged = self.output_charged or self.max_tokens
        unused = charged - int(completion)
        if unused < 0:
            self.limiter.output_tokens.debit(-unused)
            return 0
        if unused == 0:
            return 0
        self.limiter.output_tokens.release(unused)
        return unused

    def __enter__(self) -> "Reservation":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        if not self._settled:
            self._settled = True
            self.limiter.output_tokens.release(self.output_charged or self.max_tokens)


def _usage_field(usage: Any, *names: str) -> int | None:
    """Read the first present field from a usage object or dict."""
    if usage is None:
        return None
    for name in names:
        value = None
        if isinstance(usage, dict):
            value = usage.get(name)
        else:
            value = getattr(usage, name, None)
        if isinstance(value, (int, float)):
            return int(value)
    return None


class EndpointLimiter:
    """Input, output and request limits for a single serving endpoint.

    Three buckets, because Databricks meters three things independently:

        input_tokens   ITPM — charged the estimated prompt, never refunded
        output_tokens  OTPM — reserved at the `max_tokens` ceiling, refunded
                       down to actual usage once the response lands
        requests       RPM — derived from QPH when RPM is not published

    Only the output bucket uses the reserve/release cycle, because it is the
    only one whose true cost is unknown at request time. The prompt is measured
    before the call, so it is simply charged.
    """

    def __init__(
        self,
        model: str,
        input_tokens_per_minute: int | None = None,
        output_tokens_per_minute: int | None = None,
        requests_per_minute: int | None = None,
        burst_fraction: float = DEFAULT_BURST_FRACTION,
        now: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        asleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.model = model
        itpm = input_tokens_per_minute or _configured(
            "DATABRICKS_ITPM", model, DEFAULT_INPUT_TOKENS_PER_MINUTE
        )
        otpm = output_tokens_per_minute or _configured(
            "DATABRICKS_OTPM", model, DEFAULT_OUTPUT_TOKENS_PER_MINUTE
        )
        rpm = requests_per_minute or _configured(
            "DATABRICKS_RPM", model, DEFAULT_REQUESTS_PER_MINUTE
        )
        self.input_tokens = TokenBucket(
            itpm, itpm * burst_fraction, now=now, sleep=sleep, asleep=asleep
        )
        self.output_tokens = TokenBucket(
            otpm, otpm * burst_fraction, now=now, sleep=sleep, asleep=asleep
        )
        self.requests = TokenBucket(
            rpm, max(1.0, rpm * burst_fraction), now=now, sleep=sleep, asleep=asleep
        )
        self.waited_seconds = 0.0
        self._warned: set[str] = set()

    def _warn_once(self, key: str, level: int, message: str, *args: Any) -> None:
        """Log a sizing complaint the first time it applies, then stay quiet."""
        if key in self._warned:
            return
        self._warned.add(key)
        logger.log(level, message, *args)

    def _output_charge(self, max_tokens: int) -> int:
        """What the output bucket can actually be charged for this ceiling.

        `acquire` clamps anything above capacity, so this mirrors that clamp and
        the reservation remembers it. A ceiling above capacity also means the
        burst allowance is too small for the workload — every such call drains
        the bucket completely and then paces on refill, which is correct but
        worth saying out loud once.
        """
        capacity = self.output_tokens.capacity
        if max_tokens > capacity:
            self._warn_once(
                "output_capacity", logging.WARNING,
                "RateLimiter | %s | max_tokens=%d exceeds the output burst "
                "capacity of %.0f; raise RATE_LIMIT_BURST_FRACTION or lower "
                "max_tokens, otherwise every call waits on refill",
                self.model, max_tokens, capacity,
            )
            return int(capacity)
        return int(max_tokens)

    def _input_charge(self, prompt_tokens: int) -> int:
        """Mirror of `_output_charge` for the input bucket.

        `acquire` clamps silently, so a prompt larger than the burst capacity is
        charged as capacity and the rest is spent unaccounted — the limiter then
        believes it can afford several times the real rate and the 429 arrives
        from the server instead. Output has warned about this since the start;
        input did the same clamp with no warning at all, which is how a prompt
        four times the burst allowance went unnoticed.

        A prompt above the FULL per-minute quota is worse than a tuning problem:
        no pacing can make one request fit in a minute's budget, so that case is
        called out separately.
        """
        capacity = self.input_tokens.capacity
        if prompt_tokens > self.input_tokens.rate_per_minute:
            self._warn_once(
                "input_over_quota", logging.ERROR,
                "RateLimiter | %s | prompt of %d tokens EXCEEDS the whole ITPM "
                "quota of %.0f. No pacing can fit this request into a minute's "
                "budget — shrink the prompt (history/compaction) or raise "
                "DATABRICKS_ITPM_* to the endpoint's real quota. 429s are "
                "expected until then.",
                self.model, prompt_tokens, self.input_tokens.rate_per_minute,
            )
        elif prompt_tokens > capacity:
            self._warn_once(
                "input_capacity", logging.WARNING,
                "RateLimiter | %s | prompt of %d tokens exceeds the input burst "
                "capacity of %.0f; the charge is clamped, so the limiter is "
                "under-counting by %d per call. Raise RATE_LIMIT_BURST_FRACTION "
                "or shrink the prompt.",
                self.model, prompt_tokens, capacity, prompt_tokens - int(capacity),
            )
        return int(prompt_tokens)

    def _debit_input_overflow(self, prompt_tokens: int) -> None:
        """Charge the part of an oversized prompt that `acquire` could not.

        `acquire` clamps to the bucket's capacity, so a prompt larger than the
        burst allowance is only PARTLY charged and the remainder is spent
        unaccounted. Measured against a real run: a 95,835-token prompt on a
        50,000-token capacity left 45,835 tokens invisible on every call, so the
        limiter believed it could afford about twice the real rate, admitted
        twice the requests, and the 429 came back from the server — which is the
        exact failure this class exists to prevent.

        Carrying the difference as a debit is what `Reservation.settle` already
        does for an output overspend. The bucket goes negative, the next
        acquisition waits for it to refill, and the long-run rate stays honest
        at ANY burst fraction — so this is a correctness fix, not a reason to
        retune RATE_LIMIT_BURST_FRACTION.
        """
        overflow = int(prompt_tokens) - int(self.input_tokens.capacity)
        if overflow > 0:
            self.input_tokens.debit(overflow)

    def reserve(self, prompt_tokens: int, max_tokens: int) -> Reservation:
        """Acquire budget for one request, blocking until it is available.

        Ordered cheapest-first — requests, then input, then output — so a call
        that will ultimately wait on the tight output bucket is not also
        sitting on a request slot and input headroom while it waits.
        """
        charged = self._output_charge(max_tokens)
        self._input_charge(prompt_tokens)
        waited = self.requests.acquire(1.0)
        waited += self.input_tokens.acquire(prompt_tokens)
        self._debit_input_overflow(prompt_tokens)
        waited += self.output_tokens.acquire(charged)
        self.waited_seconds += waited
        return Reservation(
            limiter=self,
            prompt_tokens=int(prompt_tokens),
            max_tokens=int(max_tokens),
            output_charged=charged,
            waited_seconds=waited,
        )

    async def reserve_async(self, prompt_tokens: int, max_tokens: int) -> Reservation:
        """Async twin of `reserve`, for callers on an event loop."""
        charged = self._output_charge(max_tokens)
        self._input_charge(prompt_tokens)
        waited = await self.requests.acquire_async(1.0)
        waited += await self.input_tokens.acquire_async(prompt_tokens)
        self._debit_input_overflow(prompt_tokens)
        waited += await self.output_tokens.acquire_async(charged)
        self.waited_seconds += waited
        return Reservation(
            limiter=self,
            prompt_tokens=int(prompt_tokens),
            max_tokens=int(max_tokens),
            output_charged=charged,
            waited_seconds=waited,
        )


class RateLimiter:
    """The pipeline's limiters, one per endpoint.

    Shared deliberately. The quota is per-model across the whole workspace, so
    the converter and the fixers competing for `databricks-claude-sonnet-4-6`
    must draw from the SAME bucket — separate limiters would each stay under the
    limit while their sum went over it, which is the failure this exists to
    prevent.
    """

    def __init__(
        self,
        now: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        asleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._endpoints: dict[str, EndpointLimiter] = {}
        self._lock = threading.Lock()
        self._now = now
        self._sleep = sleep
        self._asleep = asleep

    def for_model(self, model: str) -> EndpointLimiter:
        """Limiter for `model`, created on first use.

        The provider prefix is stripped, so "databricks/databricks-claude-
        sonnet-4-6" and the bare endpoint name share one bucket rather than
        quietly getting one each.
        """
        endpoint = _endpoint(model)
        with self._lock:
            limiter = self._endpoints.get(endpoint)
            if limiter is None:
                limiter = EndpointLimiter(
                    endpoint, now=self._now, sleep=self._sleep, asleep=self._asleep
                )
                self._endpoints[endpoint] = limiter
            return limiter

    def reserve(self, model: str, prompt_tokens: int, max_tokens: int) -> Reservation:
        """Block until `model` has room, then hold the budget."""
        return self.for_model(model).reserve(prompt_tokens, max_tokens)

    async def reserve_async(
        self, model: str, prompt_tokens: int, max_tokens: int
    ) -> Reservation:
        """Await until `model` has room, then hold the budget."""
        return await self.for_model(model).reserve_async(prompt_tokens, max_tokens)

    def report(self) -> dict[str, float]:
        """Seconds spent throttled per endpoint — for logging after a run."""
        with self._lock:
            return {
                name: limiter.waited_seconds
                for name, limiter in self._endpoints.items()
                if limiter.waited_seconds
            }
LIMITER = RateLimiter()


def _fallbacks_for(preferred_model: str) -> list[str]:
    """Fallbacks for `preferred_model`, matched with or without a provider prefix.

    Reads MODEL_FALLBACKS on every call rather than a copy taken at import. An
    import-time snapshot was the obvious optimisation and the wrong one: the
    table is module-level and looks editable, so an edit -- a test's, or a
    deployment pinning a chain at startup -- would be silently ignored while
    appearing to work. The table holds a handful of entries; scanning it costs
    nothing next to the request it is about to make.
    """
    endpoint = _endpoint(preferred_model)
    for key, fallbacks in MODEL_FALLBACKS.items():
        if _endpoint(key) == endpoint:
            return list(fallbacks)
    return []


class ModelRouter:
    """Decides which endpoint a request should actually go to.

    The limiter paces requests against the quota we *believe* an endpoint has.
    The router handles the case where the server disagrees. A 429 says one
    endpoint is out of budget right now -- not that the work has to stop -- so
    the model is parked for as long as the server asked for, and the request is
    re-run immediately against the next model in its chain instead of sleeping
    on the exhausted one.

    Cooldowns are process-wide for the same reason the buckets are: the quota is
    per-model across the whole workspace, so an Opus 429 raised by the converter
    is a fact the parity fixer needs as well. Keeping it private to one caller
    would make every other agent rediscover the same exhausted endpoint, and pay
    a rejected request each time to do it.

    Deliberately knows nothing about tokens or buckets. Routing is "which model",
    pacing is "how fast" -- separating them is what keeps `EndpointLimiter`
    untouched: switching models switches limiters for free, because each
    endpoint already owns its own three buckets.
    """

    def __init__(self, now: Callable[[], float] = time.monotonic) -> None:
        #: endpoint -> the `now()` reading at which it may be used again.
        self._cooldowns: dict[str, float] = {}
        self._lock = threading.Lock()
        self._now = now

    def chain(self, preferred_model: str) -> list[str]:
        """Every model that may serve `preferred_model`, best first.

        Ignores cooldowns -- this is the static preference order, and the
        preferred model always heads it. Duplicates are dropped by endpoint, so
        a chain that names the preferred model again does not get two turns.
        """
        models = [preferred_model]
        if FALLBACKS_ENABLED:
            models.extend(_fallbacks_for(preferred_model))
        seen: set[str] = set()
        ordered: list[str] = []
        for model in models:
            endpoint = _endpoint(model)
            if endpoint not in seen:
                seen.add(endpoint)
                ordered.append(model)
        return ordered

    def models(self, preferred_model: str) -> list[str]:
        """The chain for `preferred_model`, minus whatever is cooling down.

        An empty list means every candidate is parked. That is information, not
        an error: the caller should wait `next_available_in` rather than spend a
        request the router already knows the server will reject.
        """
        now = self._now()
        with self._lock:
            return [
                model
                for model in self.chain(preferred_model)
                if self._cooldowns.get(_endpoint(model), 0.0) <= now
            ]

    def mark_rate_limited(self, model: str, retry_after: float) -> None:
        """Park `model` for `retry_after` seconds.

        Extends an existing cooldown but never shortens one: two callers hitting
        the same 429 should not let the second, smaller Retry-After undo the
        first caller's longer wait.
        """
        if retry_after <= 0:
            return
        until = self._now() + float(retry_after)
        with self._lock:
            endpoint = _endpoint(model)
            self._cooldowns[endpoint] = max(self._cooldowns.get(endpoint, 0.0), until)

    def mark_unavailable(self, model: str) -> None:
        """Park `model` for the rest of the run: the workspace has disowned it.

        Shares the cooldown table with `mark_rate_limited` so `models()` needs
        no second concept -- the only difference is a duration nothing will
        outlast, which is exactly what "gone" means here.
        """
        self.mark_rate_limited(model, MODEL_UNAVAILABLE_COOLDOWN_SECONDS)

    def cooldown_remaining(self, model: str) -> float:
        """Seconds until `model` may be used again; zero if it is available."""
        now = self._now()
        with self._lock:
            return max(0.0, self._cooldowns.get(_endpoint(model), 0.0) - now)

    def next_available_in(self, preferred_model: str) -> float:
        """Seconds until the first model in the chain frees up.

        Zero when one is already free. This is what a caller waits when
        `models` came back empty -- the shortest cooldown, not the preferred
        model's, because any model in the chain will do.
        """
        return min(
            (self.cooldown_remaining(model) for model in self.chain(preferred_model)),
            default=0.0,
        )

    def clear(self) -> None:
        """Forget every cooldown. For tests, and between runs."""
        with self._lock:
            self._cooldowns.clear()

    def report(self) -> dict[str, float]:
        """Endpoints parked right now, and for how much longer -- for logging."""
        now = self._now()
        with self._lock:
            return {
                endpoint: round(until - now, 1)
                for endpoint, until in self._cooldowns.items()
                if until > now
            }
ROUTER = ModelRouter()


def is_rate_limit_error(exc: BaseException) -> bool:
    """True if `exc` looks like a 429 from the serving endpoint."""
    if getattr(exc, "status_code", None) == 429:
        return True
    response = getattr(exc, "response", None)
    if getattr(response, "status_code", None) == 429:
        return True
    text = str(exc).lower()
    return "429" in text or "rate limit" in text or "too many requests" in text

MODEL_UNAVAILABLE_COOLDOWN_SECONDS = 86_400.0
_MODEL_GONE_PATTERN = re.compile(
    r"\bis deprecated\b"
    r"|\bdeprecated\s+endpoint\b"
    r"|\bendpoint\b[^.]{0,60}\b(?:does not exist|not found|no longer)\b"
    r"|\b(?:model|endpoint)\b[^.]{0,40}\bnot (?:supported|available)\b"
    r"|\bunsupported (?:model|endpoint)\b",
    re.I,
)


def is_model_unavailable(exc: BaseException) -> bool:
    """True if the SERVING ENDPOINT is gone, rather than busy or misused.

    A 429 says "come back shortly". This says "do not come back": the endpoint
    was retired or never existed, so every future request to it is wasted, and
    -- because it arrives as a 400 rather than a 429 -- the plain rate-limit
    path would re-raise it and end the run while healthy fallbacks sat unused.

    Matching is on wording because that is all the provider gives: the status
    code is the same 400 a malformed request produces. Narrow by design; when
    in doubt this returns False and the error propagates, which is the right
    failure direction -- a real bug surfaces instead of being routed around.
    """
    if is_rate_limit_error(exc):
        return False
    status = getattr(exc, "status_code", None) or getattr(
        getattr(exc, "response", None), "status_code", None
    )
    if status == 404:
        return True
    return bool(_MODEL_GONE_PATTERN.search(str(exc)))

_INPUT_LIMIT_PATTERN = re.compile(r"input[\s_-]*token|prompt[\s_-]*token|itpm", re.I)


def is_input_token_limit(exc: BaseException) -> bool:
    """True if a 429 names the INPUT token quota specifically.

    ITPM is the limit this pipeline actually runs into -- the prompts carry a
    whole source script plus its conversion history -- and it is the one a
    fallback model fixes outright, because the fallback's input bucket is
    untouched. Used to say so in the logs; the routing itself treats every 429
    as a reason to move on, since any exhausted quota makes that endpoint the
    wrong place to retry.
    """
    if not is_rate_limit_error(exc):
        return False
    return bool(_INPUT_LIMIT_PATTERN.search(str(exc)))


def retry_after_seconds(exc: BaseException, default: float = 20.0) -> float:
    """Seconds to wait before retrying a rate-limited request.

    Providers surface this inconsistently — a response header, a number in the
    message — so this checks the shapes that actually turn up and falls back to
    `default` rather than guessing zero, which would hammer an endpoint that
    just asked to be left alone.
    """
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None and hasattr(headers, "get"):
        for key in ("retry-after", "Retry-After", "x-ratelimit-reset"):
            value = headers.get(key)
            if value:
                try:
                    return min(float(value), MAX_RETRY_AFTER_SECONDS)
                except (TypeError, ValueError):
                    pass

    match = re.search(r"retry[- ]after[:\s]+(\d+(?:\.\d+)?)", str(exc), re.I)
    if match:
        return min(float(match.group(1)), MAX_RETRY_AFTER_SECONDS)
    return default


def _log_reservation(model: str, reservation: "Reservation") -> None:
    """One line per admitted request.

    `waited` is the whole point of the line: zero means the burst buffer
    absorbed the call, anything above zero is the limiter actively pacing the
    pipeline. A run whose waits climb steadily is one whose configured quota is
    below what the workload wants.
    """
    logger.info(
        "RateLimiter | %s | prompt=%d reserved=%d waited=%.2fs",
        model,
        reservation.prompt_tokens,
        reservation.reserved,
        reservation.waited_seconds,
    )


def _log_settlement(model: str, reservation: "Reservation", released: int) -> None:
    """Debug-level record of what the response handed back."""
    if not logger.isEnabledFor(logging.DEBUG):
        return
    bucket = reservation.limiter.output_tokens
    logger.debug(
        "RateLimiter | %s | released=%d remaining=%.0f/%.0f",
        model, released, bucket._tokens, bucket.capacity,
    )


def _release_reservation(reservation: "Reservation") -> None:
    """Hand the whole output ceiling back after a call that failed.

    Nothing was generated that we know of, and holding the budget would only
    throttle whatever comes next -- which, now that routing exists, is usually a
    request to a *different* endpoint that should not be paying for this one.
    """
    reservation.limiter.output_tokens.release(
        reservation.output_charged or reservation.max_tokens
    )
    reservation._settled = True


def _log_route(preferred: str, chosen: str, router: "ModelRouter") -> None:
    """Say which model a request is going to, and why it is not the first choice.

    Called once per round, for its first candidate -- the interesting case being
    a request that skips the preferred model outright because an EARLIER call
    already parked it. Only that case is worth INFO: `_log_reservation` names
    the model on every admitted request anyway, so logging the ordinary path
    here too would double the volume to say the same thing twice.
    """
    if chosen == preferred:
        logger.debug("ModelRouter | Using model: %s", chosen)
        return
    logger.info(
        "ModelRouter | %s is cooling down for another %.0fs. Using model: %s.",
        preferred, router.cooldown_remaining(preferred), chosen,
    )


def _cool_down(
    router: "ModelRouter",
    current: str,
    exc: BaseException,
    attempt: int,
    max_attempts: int,
    following: str | None,
) -> float:
    """Park the endpoint that just 429'd and log where the request goes next."""
    delay = retry_after_seconds(exc)
    router.mark_rate_limited(current, delay)
    reason = "input token limit" if is_input_token_limit(exc) else "rate limit"
    logger.warning(
        "ModelRouter | %s hit the %s (attempt %d/%d). Cooling down for %.0fs.",
        current, reason, attempt, max_attempts, delay,
    )
    if following:
        logger.info("ModelRouter | Switching to %s.", following)
    return delay


def _park_dead(
    router: "ModelRouter", current: str, exc: BaseException, following: str | None
) -> None:
    """Retire an endpoint the workspace has disowned, and say so loudly."""
    router.mark_unavailable(current)
    logger.error(
        "ModelRouter | %s is NOT a usable endpoint (%s). Retiring it for this "
        "run. Remove it from MODEL_FALLBACKS — every process pays one rejected "
        "request to rediscover this.",
        current, str(exc)[:200],
    )
    if following:
        logger.info("ModelRouter | Switching to %s.", following)


def _wait_for_a_model(
    router: "ModelRouter", model: str, attempt: int, max_attempts: int
) -> float | None:
    """How long to wait when every model in the chain is parked.

    None means do not wait at all: the shortest cooldown is longer than any
    rate limit would justify, so the chain is parked because its endpoints are
    GONE rather than busy. Sleeping a day to re-confirm that would turn a clear
    failure into a hung run.
    """
    delay = router.next_available_in(model)
    if delay > MAX_RETRY_AFTER_SECONDS:
        logger.error(
            "ModelRouter | no usable endpoint for %s — every model in its chain "
            "has been retired as unavailable. Fix MODEL_FALLBACKS or the "
            "requested model; retrying cannot help.",
            model,
        )
        return None
    logger.warning(
        "ModelRouter | every model that can serve %s is cooling down; "
        "waiting %.0fs (attempt %d/%d)",
        model, delay, attempt, max_attempts,
    )
    return delay


def call_with_rate_limit(
    complete: Callable[..., Any],
    *,
    model: str,
    messages: Sequence[Any],
    max_tokens: int,
    limiter: RateLimiter | None = None,
    router: ModelRouter | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    sleep: Callable[[float], None] = time.sleep,
    **kwargs: Any,
) -> Any:
    """Run one LiteLLM completion under the local rate limit, routing on 429.

    Two mechanisms, kept apart on purpose:

    * the limiter paces the request -- estimate the prompt, reserve
      prompt + max_tokens against that endpoint's buckets, wait for room;
    * the router picks the endpoint -- if the server rejects the call anyway,
      that model is parked for its Retry-After and the request is re-run
      IMMEDIATELY against the next model in its chain.

    The old shape slept on the exhausted endpoint and tried it again. That wait
    bought nothing: a second endpoint with an untouched quota was sitting idle
    the whole time. Sleeping only happens now when the entire chain is parked.

    Args:
        complete: the completion callable, e.g. `litellm.completion`.
        model: the PREFERRED endpoint, with or without a provider prefix. What
            actually serves the request may be a fallback -- see
            `MODEL_FALLBACKS`.
        messages: the LiteLLM messages list; used for the prompt estimate.
        max_tokens: the request's output ceiling -- reserved in full, refunded
            down to actual usage once the response lands.
        limiter: override the process-wide limiter (tests, isolation).
        router: override the process-wide router (tests, isolation).
        max_attempts: rounds through the chain, including the first. One round
            may issue several requests -- one per available model.
        sleep: injected for testability.
        **kwargs: forwarded to `complete` unchanged.

    Returns:
        Whatever `complete` returns.

    Raises:
        The last rate-limit error if every model was exhausted on every round;
        any non-429 error immediately, unchanged.
    """
    limiter = limiter or LIMITER
    router = router or ROUTER
    prompt_tokens = estimate_message_tokens(messages)
    last_error: BaseException | None = None

    for attempt in range(1, max_attempts + 1):
        candidates = router.models(model)
        if not candidates:
            delay = _wait_for_a_model(router, model, attempt, max_attempts)
            if delay is None:
                break
            sleep(delay)
            candidates = router.models(model) or [model]

        for index, current in enumerate(candidates):
            if index == 0:
                _log_route(model, current, router)
            reservation = limiter.reserve(current, prompt_tokens, max_tokens)
            _log_reservation(current, reservation)
            try:
                response = complete(
                    model=current, messages=messages, max_tokens=max_tokens, **kwargs
                )
            except BaseException as exc:
                _release_reservation(reservation)
                following = candidates[index + 1] if index + 1 < len(candidates) else None
                if is_model_unavailable(exc):
                    last_error = exc
                    _park_dead(router, current, exc, following)
                    continue
                if not is_rate_limit_error(exc):
                    raise
                last_error = exc
                _cool_down(router, current, exc, attempt, max_attempts, following)
                continue

            if current != model:
                logger.info("ModelRouter | %s completed successfully.", current)
            _log_settlement(
                current, reservation, reservation.settle(_response_usage(response))
            )
            return response

    if last_error is not None:
        raise last_error
    raise RuntimeError(
        f"no endpoint available for {model}; cooldowns: {router.report()}"
    )


def _response_usage(response: Any) -> Any:
    """Pull the usage record off a LiteLLM response, whatever shape it is."""
    if response is None:
        return None
    if isinstance(response, dict):
        return response.get("usage")
    return getattr(response, "usage", None)


def _is_stream(response: Any) -> bool:
    """True if the response is a streamed iterator rather than a finished reply.

    A streamed call returns immediately with an iterator; the tokens are only
    produced as it is consumed, so the usage record does not exist yet.
    """
    if response is None or isinstance(response, (dict, str, bytes)):
        return False
    return hasattr(response, "__aiter__") or hasattr(response, "__anext__")



async def acall_with_rate_limit(
    acomplete: Callable[..., Awaitable[Any]],
    *,
    model: str,
    messages: Sequence[Any],
    max_tokens: int,
    limiter: RateLimiter | None = None,
    router: ModelRouter | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    asleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    **kwargs: Any,
) -> Any:
    """Async twin of `call_with_rate_limit`, for `acompletion`.

    Same sequence and the same routing -- reserve, call, park a 429'd endpoint
    and move to the next model in its chain -- but every wait yields to the
    event loop instead of blocking it. This is the one ADK uses, so it is the
    one that matters: a blocking wait here would stall every other concurrent
    agent, not just the throttled one.
    """
    limiter = limiter or LIMITER
    router = router or ROUTER
    prompt_tokens = estimate_message_tokens(messages)
    last_error: BaseException | None = None

    for attempt in range(1, max_attempts + 1):
        candidates = router.models(model)
        if not candidates:
            delay = _wait_for_a_model(router, model, attempt, max_attempts)
            if delay is None:
                break
            await asleep(delay)
            candidates = router.models(model) or [model]

        for index, current in enumerate(candidates):
            if index == 0:
                _log_route(model, current, router)
            reservation = await limiter.reserve_async(current, prompt_tokens, max_tokens)
            _log_reservation(current, reservation)
            try:
                response = await acomplete(
                    model=current, messages=messages, max_tokens=max_tokens, **kwargs
                )
            except BaseException as exc:
                _release_reservation(reservation)
                following = candidates[index + 1] if index + 1 < len(candidates) else None
                if is_model_unavailable(exc):
                    last_error = exc
                    _park_dead(router, current, exc, following)
                    continue
                if not is_rate_limit_error(exc):
                    raise
                last_error = exc
                _cool_down(router, current, exc, attempt, max_attempts, following)
                continue

            if current != model:
                logger.info("ModelRouter | %s completed successfully.", current)

            if _is_stream(response):
                logger.debug(
                    "RateLimiter | %s | streamed response: holding the full %d-token "
                    "output reservation (no usage to reconcile against)",
                    current, reservation.output_charged or reservation.max_tokens,
                )
                return response

            _log_settlement(
                current, reservation, reservation.settle(_response_usage(response))
            )
            return response

    if last_error is not None:
        raise last_error
    raise RuntimeError(
        f"no endpoint available for {model}; cooldowns: {router.report()}"
    )

call_with_rate_limit_async = acall_with_rate_limit
