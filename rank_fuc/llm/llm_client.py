"""轻量 LLM 客户端：chat / batch_chat + 线程安全 token 统计。

为 LLM_Rank 各种 ranker（SwissRank、RankGPT、TourRank、Setwise、Pairwise）共用。
参考 ToMEval/src/llm 的设计精简而成，去掉了 StructureClient/Pydantic/reasoning。
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import openai
import yaml
from tqdm import tqdm

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def _load_config() -> dict:
    config_path = Path(os.environ.get("LLM_CONFIG", DEFAULT_CONFIG_PATH)).expanduser()
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    llm_config = config.get("llm", config)
    if not isinstance(llm_config, dict):
        raise ValueError(f"Invalid LLM config in {config_path}: expected a mapping")
    return llm_config


def _get_config_value(config: dict, env_name: str, config_name: str, default):
    env_value = os.environ.get(env_name)
    if env_value not in (None, ""):
        return env_value
    config_value = config.get(config_name)
    if config_value not in (None, ""):
        return config_value
    return default


_CONFIG = _load_config()

DEFAULT_MODEL = _get_config_value(_CONFIG, "LLM_MODEL", "model", "qwen3-8b")
DEFAULT_API_KEY = _get_config_value(_CONFIG, "LLM_API_KEY", "api_key", "")
DEFAULT_API_URL = _get_config_value(
    _CONFIG, "LLM_API_URL", "api_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
DEFAULT_MAX_WORKERS = int(_get_config_value(_CONFIG, "LLM_MAX_WORKERS", "max_workers", 16))
DEFAULT_TEMPERATURE = float(_get_config_value(_CONFIG, "LLM_TEMPERATURE", "temperature", 0.0))
DEFAULT_MAX_TOKENS = int(_get_config_value(_CONFIG, "LLM_MAX_TOKENS", "max_tokens", 2048))
DEFAULT_MAX_RETRY = int(_get_config_value(_CONFIG, "LLM_MAX_RETRY", "max_retry", 5))
DEFAULT_TIMEOUT = float(_get_config_value(_CONFIG, "LLM_TIMEOUT", "timeout", 120.0))
DEFAULT_RATE_LIMIT_BACKOFF_BASE = float(
    _get_config_value(_CONFIG, "LLM_RATE_LIMIT_BACKOFF_BASE", "rate_limit_backoff_base", 5.0)
)
DEFAULT_RATE_LIMIT_BACKOFF_MAX = float(
    _get_config_value(_CONFIG, "LLM_RATE_LIMIT_BACKOFF_MAX", "rate_limit_backoff_max", 60.0)
)
DEFAULT_BACKOFF_JITTER = float(_get_config_value(_CONFIG, "LLM_BACKOFF_JITTER", "backoff_jitter", 0.5))
DEFAULT_REQUESTS_PER_MINUTE = float(
    _get_config_value(_CONFIG, "LLM_REQUESTS_PER_MINUTE", "requests_per_minute", 0)
)
DEFAULT_ENABLE_THINKING = str(
    _get_config_value(_CONFIG, "LLM_ENABLE_THINKING", "enable_thinking", False)
).lower() in {"1", "true", "yes", "on"}
_PRICING = _CONFIG.get("pricing", {})
DEFAULT_PROMPT_PRICE_PER_1K = float(
    _get_config_value(_PRICING, "LLM_PROMPT_PRICE_PER_1K", "prompt_per_1k_tokens", 0.0)
)
DEFAULT_COMPLETION_PRICE_PER_1K = float(
    _get_config_value(_PRICING, "LLM_COMPLETION_PRICE_PER_1K", "completion_per_1k_tokens", 0.0)
)
DEFAULT_PRICE_CURRENCY = _get_config_value(_PRICING, "LLM_PRICE_CURRENCY", "currency", "CNY")


def _classify_error(e: Exception) -> str:
    """把 API 异常归类为 content_filter / rate_limit / timeout / other，用于差异化重试。

    - content_filter：DashScope 内容审核（data_inspection_failed / inappropriate content），
      重试无意义，应立即放弃并整条丢弃该 query。
    - rate_limit：429 / Throttling / 限流，应更长退避后重试。
    - timeout：请求超时，沿用常规指数退避。
    """
    msg = str(e).lower()
    if "data_inspection_failed" in msg or "inappropriate content" in msg:
        return "content_filter"
    if isinstance(e, openai.RateLimitError) or any(
        k in msg for k in ("429", "throttling", "rate limit", "limit_requests", "too many requests")
    ):
        return "rate_limit"
    if isinstance(e, openai.APITimeoutError) or "timed out" in msg or "timeout" in msg:
        return "timeout"
    return "other"


@dataclass
class LLMUsage:
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_calls: int = 0
    failed_calls: int = 0
    content_filter_failed: int = 0
    total_api_requests: int = 0
    failed_api_requests: int = 0
    total_llm_latency_seconds: float = 0.0
    estimated_cost: float = 0.0
    cost_currency: str = DEFAULT_PRICE_CURRENCY


@dataclass
class LLMResponse:
    content: str | None
    reasoning_content: str | None = None
    usage_total_tokens: int = 0


class LLMClient:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        api_key: str = DEFAULT_API_KEY,
        api_url: str = DEFAULT_API_URL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_workers: int = DEFAULT_MAX_WORKERS,
        max_retry: int = DEFAULT_MAX_RETRY,
        timeout: float = DEFAULT_TIMEOUT,
        enable_thinking: bool = DEFAULT_ENABLE_THINKING,
        rate_limit_backoff_base: float = DEFAULT_RATE_LIMIT_BACKOFF_BASE,
        rate_limit_backoff_max: float = DEFAULT_RATE_LIMIT_BACKOFF_MAX,
        backoff_jitter: float = DEFAULT_BACKOFF_JITTER,
        requests_per_minute: float = DEFAULT_REQUESTS_PER_MINUTE,
        prompt_price_per_1k_tokens: float = DEFAULT_PROMPT_PRICE_PER_1K,
        completion_price_per_1k_tokens: float = DEFAULT_COMPLETION_PRICE_PER_1K,
        price_currency: str = DEFAULT_PRICE_CURRENCY,
    ):
        self.model = model_name
        self.api_key = api_key
        self.api_url = api_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_workers = max_workers
        self.max_retry = max_retry
        self.timeout = timeout
        self.enable_thinking = enable_thinking
        self.rate_limit_backoff_base = rate_limit_backoff_base
        self.rate_limit_backoff_max = rate_limit_backoff_max
        self.backoff_jitter = backoff_jitter
        self.requests_per_minute = requests_per_minute
        self.prompt_price_per_1k_tokens = prompt_price_per_1k_tokens
        self.completion_price_per_1k_tokens = completion_price_per_1k_tokens
        self.price_currency = price_currency

        self._client: openai.OpenAI | None = None
        self.usage = LLMUsage(cost_currency=self.price_currency)
        self._lock = threading.Lock()
        # 主动限流（可选）：强制相邻 API 请求最小间隔，避免触发服务端 429。
        self._rate_lock = threading.Lock()
        self._next_request_time = 0.0

    @property
    def client(self) -> openai.OpenAI:
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "LLM API key is not configured. Set LLM_API_KEY or llm.api_key in LLM_Rank/config.yaml."
                )
            self._client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.api_url,
                timeout=self.timeout,
            )
        return self._client

    def _build_extra_body(self) -> dict:
        # enable_thinking / chat_template_kwargs 是 qwen3 混合推理模型专用参数。
        # deepseek-r1/v3、qwen-max、qwen2.5 等不识别，传了会 400，所以只对 qwen3 下发。
        if "qwen3" not in str(self.model).lower():
            return {}
        # 本地 vLLM 只认 chat_template_kwargs（顶层 enable_thinking 会被 400 拒绝）；
        # tokenkey/DashScope 需要顶层 enable_thinking。用 LLM_THINK_STYLE=vllm 切到 vLLM 形式。
        if os.environ.get("LLM_THINK_STYLE", "").strip().lower() == "vllm":
            return {"chat_template_kwargs": {"enable_thinking": self.enable_thinking}}
        return {
            "enable_thinking": self.enable_thinking,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }

    def _track(self, prompt_tokens: int, completion_tokens: int, total_tokens: int, success: bool) -> None:
        estimated_cost = (
            prompt_tokens / 1000 * self.prompt_price_per_1k_tokens
            + completion_tokens / 1000 * self.completion_price_per_1k_tokens
        )
        with self._lock:
            self.usage.total_prompt_tokens += prompt_tokens
            self.usage.total_completion_tokens += completion_tokens
            self.usage.total_tokens += total_tokens
            self.usage.estimated_cost += estimated_cost
            self.usage.total_calls += 1
            if not success:
                self.usage.failed_calls += 1

    def _track_api_request(self, latency_seconds: float, success: bool) -> None:
        with self._lock:
            self.usage.total_api_requests += 1
            self.usage.total_llm_latency_seconds += latency_seconds
            if not success:
                self.usage.failed_api_requests += 1

    def _track_content_filter(self) -> None:
        # 内容审核失败单独计数，不计入 failed_calls/failed_api_requests，
        # 这样断点续跑/重跑判断不会把「数据本身不合规」当成需要重试的瞬时故障。
        with self._lock:
            self.usage.content_filter_failed += 1

    def _throttle(self) -> None:
        """可选主动限流：保证相邻 API 请求间隔 >= 60/requests_per_minute 秒。"""
        if self.requests_per_minute <= 0:
            return
        min_interval = 60.0 / self.requests_per_minute
        with self._rate_lock:
            now = time.monotonic()
            wait = self._next_request_time - now
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            self._next_request_time = now + min_interval

    def reset_usage(self) -> None:
        with self._lock:
            self.usage = LLMUsage(cost_currency=self.price_currency)

    def get_usage(self) -> LLMUsage:
        with self._lock:
            return LLMUsage(
                total_prompt_tokens=self.usage.total_prompt_tokens,
                total_completion_tokens=self.usage.total_completion_tokens,
                total_tokens=self.usage.total_tokens,
                total_calls=self.usage.total_calls,
                failed_calls=self.usage.failed_calls,
                content_filter_failed=self.usage.content_filter_failed,
                total_api_requests=self.usage.total_api_requests,
                failed_api_requests=self.usage.failed_api_requests,
                total_llm_latency_seconds=self.usage.total_llm_latency_seconds,
                estimated_cost=self.usage.estimated_cost,
                cost_currency=self.usage.cost_currency,
            )

    def _chat_once(self, messages: list[dict]) -> LLMResponse:
        self._throttle()
        if self.enable_thinking:
            return self._chat_streaming(messages)
        return self._chat_non_streaming(messages)

    def _chat_non_streaming(self, messages: list[dict]) -> LLMResponse:
        # DashScope qwen3 thinking only works in streaming mode. Non-thinking rankers
        # use the cheaper non-streaming path.
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=messages,
            extra_body=self._build_extra_body(),
        )
        prompt_tokens = completion_tokens = total_tokens = 0
        if getattr(resp, "usage", None):
            prompt_tokens = resp.usage.prompt_tokens or 0
            completion_tokens = resp.usage.completion_tokens or 0
            total_tokens = resp.usage.total_tokens or 0
        content = resp.choices[0].message.content or ""
        self._track(prompt_tokens, completion_tokens, total_tokens, success=True)
        return LLMResponse(content=content, reasoning_content=None, usage_total_tokens=total_tokens)

    def _chat_streaming(self, messages: list[dict]) -> LLMResponse:
        stream = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
            extra_body=self._build_extra_body(),
        )
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        prompt_tokens = completion_tokens = total_tokens = 0
        for chunk in stream:
            if getattr(chunk, "usage", None):
                prompt_tokens = chunk.usage.prompt_tokens or 0
                completion_tokens = chunk.usage.completion_tokens or 0
                total_tokens = chunk.usage.total_tokens or 0
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            delta_data = delta.model_dump()
            content = delta_data.get("content")
            reasoning = delta_data.get("reasoning_content")
            if content:
                content_parts.append(content)
            if reasoning:
                reasoning_parts.append(reasoning)
        self._track(prompt_tokens, completion_tokens, total_tokens, success=True)
        return LLMResponse(
            content="".join(content_parts),
            reasoning_content="".join(reasoning_parts),
            usage_total_tokens=total_tokens,
        )

    def chat(self, messages: list[dict], max_retry: int | None = None) -> LLMResponse:
        retries = self.max_retry if max_retry is None else max_retry
        last_err: Exception | None = None
        for attempt in range(retries):
            start_time = time.perf_counter()
            try:
                resp = self._chat_once(messages)
                self._track_api_request(time.perf_counter() - start_time, success=True)
                return resp
            except Exception as e:
                last_err = e
                kind = _classify_error(e)
                # 内容审核失败：重试无意义，立即放弃（调用方会整条丢弃该 query）。
                if kind == "content_filter":
                    self._track_api_request(time.perf_counter() - start_time, success=False)
                    self._track_content_filter()
                    logging.warning(f"[LLMClient] chat content filtered, giving up (no retry): {e}")
                    return LLMResponse(content=None, reasoning_content=None, usage_total_tokens=0)
                self._track_api_request(time.perf_counter() - start_time, success=False)
                logging.warning(
                    f"[LLMClient] chat attempt {attempt + 1}/{retries} failed ({kind}): {e}"
                )
                if attempt < retries - 1:
                    if kind == "rate_limit":
                        backoff = min(
                            self.rate_limit_backoff_base * (2 ** attempt),
                            self.rate_limit_backoff_max,
                        )
                    else:
                        backoff = min(2 ** attempt, 30)
                    time.sleep(backoff + random.uniform(0, self.backoff_jitter))
        logging.error(f"[LLMClient] chat exhausted {retries} retries; last error: {last_err}")
        self._track(0, 0, 0, success=False)
        return LLMResponse(content=None, reasoning_content=None, usage_total_tokens=0)

    def batch_chat(
        self,
        messages_list: list[list[dict]],
        desc: str | None = None,
    ) -> list[LLMResponse]:
        if not messages_list:
            return []
        results: list[LLMResponse | None] = [None] * len(messages_list)
        with ThreadPoolExecutor(self.max_workers) as ex:
            future_to_idx = {ex.submit(self.chat, m): i for i, m in enumerate(messages_list)}
            completed = as_completed(future_to_idx)
            if desc is not None:
                completed = tqdm(completed, total=len(future_to_idx), desc=desc, miniters=10)
            for fut in completed:
                idx = future_to_idx[fut]
                results[idx] = fut.result()
        return [r if r is not None else LLMResponse(content=None, reasoning_content=None) for r in results]

    def __repr__(self) -> str:
        return f"LLMClient(model={self.model!r}, max_workers={self.max_workers})"
