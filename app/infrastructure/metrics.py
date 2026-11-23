"""Prometheus metrics for requests, graph nodes, LLM tokens and estimated cost."""

from prometheus_client import Counter, Histogram

from app.config import settings

_LATENCY_BUCKETS = (0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 40, 80, 160)

REQUEST_SECONDS = Histogram(
    "pm_request_seconds", "End-to-end query latency", ["route"], buckets=_LATENCY_BUCKETS
)
NODE_SECONDS = Histogram(
    "pm_node_seconds", "Graph node latency", ["node", "outcome"], buckets=_LATENCY_BUCKETS
)
LLM_SECONDS = Histogram("pm_llm_seconds", "LLM call latency", ["model"], buckets=_LATENCY_BUCKETS)
EMBEDDING_SECONDS = Histogram(
    "pm_embedding_seconds", "Embedding call latency", ["model"], buckets=_LATENCY_BUCKETS
)
FAILURES = Counter("pm_failures_total", "Failed queries", ["error_code"])
RETRIES = Counter("pm_retries_total", "Graph retries")
LLM_TOKENS = Counter("pm_llm_tokens_total", "LLM tokens", ["model", "kind"])
LLM_COST_USD = Counter("pm_llm_cost_usd_total", "Estimated LLM spend", ["model"])


def record_llm_call(
    model: str, seconds: float, prompt_tokens: int | None, completion_tokens: int | None
) -> None:
    """Record one completion; usage can be missing on some OpenAI-compatible servers."""
    LLM_SECONDS.labels(model).observe(seconds)
    prompt = prompt_tokens or 0
    completion = completion_tokens or 0
    LLM_TOKENS.labels(model, "prompt").inc(prompt)
    LLM_TOKENS.labels(model, "completion").inc(completion)
    cost = (
        prompt * settings.llm_cost_per_1k_input + completion * settings.llm_cost_per_1k_output
    ) / 1000
    if cost:
        LLM_COST_USD.labels(model).inc(cost)
