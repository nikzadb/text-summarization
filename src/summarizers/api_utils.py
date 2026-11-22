import time
from typing import Any, Callable, Tuple, Type


class RateLimiter:
    """Blocks so that successive calls are at least 1 / max_calls_per_second apart."""

    def __init__(self, max_calls_per_second: float = 2.5):
        self.min_interval = 1.0 / max_calls_per_second
        self._last_call = 0.0

    def wait(self):
        delay = self._last_call + self.min_interval - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        self._last_call = time.perf_counter()


def timed_call_with_retry(
    fn: Callable[[], Any],
    rate_limiter: RateLimiter,
    retryable: Tuple[Type[BaseException], ...],
    max_attempts: int = 8,
    initial_backoff: float = 2.0,
    max_backoff: float = 60.0,
) -> Tuple[Any, float]:
    """
    Call fn() and return (result, latency_seconds).

    Latency covers only the successful attempt: from sending the request until the
    response is received. Rate-limit waits, failed attempts and backoff sleeps are
    excluded. Retryable errors (rate limits, overload, timeouts) are retried with
    exponential backoff; any other error, or exhausting all attempts, is raised so
    that no fallback output is ever recorded as a model summary.
    """
    backoff = initial_backoff
    for attempt in range(1, max_attempts + 1):
        rate_limiter.wait()
        start = time.perf_counter()
        try:
            result = fn()
        except retryable as e:
            if attempt == max_attempts:
                raise
            print(f"⚠ API error (attempt {attempt}/{max_attempts}), retrying in {backoff:.0f}s: {e}")
            time.sleep(backoff)
            backoff = min(backoff * 2, max_backoff)
            continue
        return result, time.perf_counter() - start
