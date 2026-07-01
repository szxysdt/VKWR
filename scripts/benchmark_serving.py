"""Serving benchmark for VKWR using ShareGPT dataset, modeled after vLLM.

Two modes:
  - Fixed:    --num-prompts 500            (send N requests, stop)
  - Continuous: --duration 300             (loop dataset for 300s, Ctrl+C to stop)

Run:
    PYTHONPATH=. uv run python scripts/benchmark_serving.py \
        --num-prompts 500 --num-workers 32 --warmup-requests 2
    PYTHONPATH=. uv run python scripts/benchmark_serving.py \
        --duration 300 --num-workers 32 --warmup-requests 4
"""

import argparse
import asyncio
import json
import logging
import math
import os
import random
import signal
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field

import httpx
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RequestResult:
    """Per-request metrics collected during benchmark."""

    success: bool = False
    prompt_len: int = 0
    output_tokens: int = 0
    ttft: float = 0.0
    itl: list = field(default_factory=list)
    latency: float = 0.0
    start_time: float = 0.0
    error: str = ""


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_sharegpt_dataset(path: str, seed: int = 0) -> list[dict]:
    """Load ShareGPT dataset and filter entries with >= 2 conversation turns."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    filtered = [entry for entry in data if "conversations" in entry and len(entry["conversations"]) >= 2]
    random.seed(seed)
    random.shuffle(filtered)
    logger.info("Loaded %d entries from ShareGPT (%d with >= 2 turns)", len(data), len(filtered))
    return filtered


def sample_requests(dataset: list[dict], num_requests: int) -> list[dict]:
    """Sample num_requests entries from dataset."""
    return dataset[: min(num_requests, len(dataset))]


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="ShareGPT serving benchmark for VKWR")
    parser.add_argument("-u", "--base-url", default="http://127.0.0.1:8000", help="Server base URL")
    parser.add_argument("--model", default="test", help="Model name")
    parser.add_argument("--system-prompt", default="You are a helpful AI Assistant.", help="System prompt")
    parser.add_argument(
        "--dataset-path",
        default="tmp/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json",
        help="Path to ShareGPT JSON file",
    )
    parser.add_argument("--num-prompts", type=int, default=500, help="Number of benchmark requests")
    parser.add_argument("--num-workers", type=int, default=32, help="Max concurrent in-flight requests")
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Requests per second. 'inf' sends all at once. Otherwise Poisson process.",
    )
    parser.add_argument("--warmup-requests", type=int, default=4, help="Warmup requests per worker (discarded)")
    parser.add_argument("-r", "--report-interval", type=int, default=10, help="Seconds between live reports")
    parser.add_argument(
        "--percentiles",
        type=float,
        nargs="+",
        default=[50, 90, 95, 99],
        help="Percentiles to compute for latency metrics",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--save-results-dir", default="", help="Directory to save results JSON. Auto-generates filename from params + timestamp.")
    parser.add_argument(
        "--duration",
        type=int,
        default=0,
        help="Continuous mode: run for N seconds (Ctrl+C to stop). If 0, sends --num-prompts requests.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-request details for debugging")
    return parser.parse_args()


ARGS = parse_args()

# Module-level state so Ctrl+C handler can still print metrics.
_all_results: list[RequestResult] = []
_benchmark_start: float = 0.0


# ---------------------------------------------------------------------------
# Request generation
# ---------------------------------------------------------------------------


async def generate_requests(total: int, request_rate: float, rng: np.random.Generator, continuous: bool = False, stop_event: asyncio.Event | None = None):
    """Yield request indices at specified rate (Poisson process).

    If continuous is True, loops through indices indefinitely until stop_event is set.
    """
    i = 0
    while True:
        if request_rate != float("inf"):
            interval = rng.exponential(1.0 / request_rate)
            await asyncio.sleep(interval)
        yield i % total
        if stop_event and stop_event.is_set():
            break
        if not continuous:
            i += 1
            if i >= total:
                break
        else:
            i += 1


# ---------------------------------------------------------------------------
# Single request
# ---------------------------------------------------------------------------


async def send_request(client: httpx.AsyncClient, prompt: str, prompt_len: int) -> RequestResult:
    payload = {
        "model": ARGS.model,
        "messages": [],
    }
    if ARGS.system_prompt:
        payload["messages"].append({"role": "system", "content": ARGS.system_prompt})
    payload["messages"].append({"role": "user", "content": prompt})
    payload["max_tokens"] = 1024
    payload["stream"] = True

    result = RequestResult()
    result.prompt_len = prompt_len
    result.start_time = time.perf_counter()
    st = result.start_time
    most_recent_ts = st
    first_token = False
    content_count = 0
    server_prompt_tokens = 0
    server_completion_tokens = 0

    try:
        async with client.stream("POST", f"{ARGS.base_url}/v1/chat/completions", json=payload) as resp:
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}")

            async for line in resp.aiter_lines():
                if not line or line == "data: [DONE]":
                    continue
                if not line.startswith("data: "):
                    continue

                data = json.loads(line[6:])

                # Extract usage from trailing usage chunk (vLLM-style)
                if usage := data.get("usage"):
                    server_prompt_tokens = usage.get("prompt_tokens", 0)
                    server_completion_tokens = usage.get("completion_tokens", 0)
                    continue

                choices = data.get("choices", [])
                if not choices:
                    continue

                if choices[0].get("finish_reason"):
                    continue

                timestamp = time.perf_counter()
                delta = choices[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    content_count += 1
                    if not first_token:
                        first_token = True
                        result.ttft = timestamp - st
                    else:
                        result.itl.append(timestamp - most_recent_ts)
                    most_recent_ts = timestamp

            if first_token:
                result.success = True
                result.latency = time.perf_counter() - st
                # Prefer server-reported token counts from usage chunk
                if server_completion_tokens > 0:
                    result.output_tokens = server_completion_tokens
                else:
                    result.output_tokens = content_count
                if server_prompt_tokens > 0:
                    result.prompt_len = server_prompt_tokens
            else:
                result.success = False
                result.error = "Never received first token"
    except Exception as e:
        result.success = False
        result.error = str(e)

    return result


# ---------------------------------------------------------------------------
# Metrics calculation (vLLM-style per-second bucketing)
# ---------------------------------------------------------------------------


def calculate_metrics(results: list[RequestResult], dur_s: float) -> dict | None:
    ttfts: list[float] = []
    itls: list[float] = []
    e2els: list[float] = []
    total_input = 0
    total_output = 0
    completed = 0
    failed = 0

    for r in results:
        if r.success:
            completed += 1
            total_input += r.prompt_len
            total_output += r.output_tokens
            ttfts.append(r.ttft)
            itls.extend(r.itl)
            e2els.append(r.latency)
        else:
            failed += 1

    if completed == 0:
        logger.error("No successful requests!")
        return None

    request_throughput = completed / dur_s
    output_throughput = total_output / dur_s
    input_throughput = total_input / dur_s

    mean_ttft = float(np.mean(ttfts)) * 1000
    median_ttft = float(np.median(ttfts)) * 1000
    std_ttft = float(np.std(ttfts)) * 1000
    pctl_ttft = [(p, float(np.percentile(ttfts, p) * 1000)) for p in ARGS.percentiles]

    mean_itl = float(np.mean(itls or [0])) * 1000
    median_itl = float(np.median(itls or [0])) * 1000
    std_itl = float(np.std(itls or [0])) * 1000
    pctl_itl = [(p, float(np.percentile(itls or [0], p) * 1000)) for p in ARGS.percentiles]

    mean_e2el = float(np.mean(e2els)) * 1000
    median_e2el = float(np.median(e2els)) * 1000
    std_e2el = float(np.std(e2els)) * 1000
    pctl_e2el = [(p, float(np.percentile(e2els, p) * 1000)) for p in ARGS.percentiles]

    # --- Per-request prefill speed & TPOT (vLLM-style) ---
    # Per-request prefill speed: prompt_len / ttft (tok/s)
    # TPOT: (latency - ttft) / (output_tokens - 1) (s/token), standard vLLM decode metric
    prefill_speeds: list[float] = []
    tpots: list[float] = []  # in seconds per token
    decode_speeds: list[float] = []  # in tok/s

    for i, r in enumerate(results):
        if not r.success:
            continue
        # Prefill speed
        if r.ttft > 0 and r.prompt_len > 0:
            prefill_speeds.append(r.prompt_len / r.ttft)
        # Decode speed & TPOT
        decode_time = r.latency - r.ttft
        if decode_time > 0 and r.output_tokens > 1:
            decode_speed = (r.output_tokens - 1) / decode_time
            decode_speeds.append(decode_speed)
            tpots.append(decode_time / (r.output_tokens - 1))

    mean_prefill_speed = float(np.mean(prefill_speeds)) if prefill_speeds else 0.0
    median_prefill_speed = float(np.median(prefill_speeds)) if prefill_speeds else 0.0
    std_prefill_speed = float(np.std(prefill_speeds)) if prefill_speeds else 0.0
    pctl_prefill_speed = [(p, float(np.percentile(prefill_speeds, p))) for p in ARGS.percentiles] if prefill_speeds else []

    mean_tpot = float(np.mean(tpots)) * 1000 if tpots else 0.0
    median_tpot = float(np.median(tpots)) * 1000 if tpots else 0.0
    std_tpot = float(np.std(tpots)) * 1000 if tpots else 0.0
    pctl_tpot = [(p, float(np.percentile(tpots, p) * 1000)) for p in ARGS.percentiles] if tpots else []

    mean_decode_speed = float(np.mean(decode_speeds)) if decode_speeds else 0.0
    median_decode_speed = float(np.median(decode_speeds)) if decode_speeds else 0.0
    std_decode_speed = float(np.std(decode_speeds)) if decode_speeds else 0.0
    pctl_decode_speed = [(p, float(np.percentile(decode_speeds, p))) for p in ARGS.percentiles] if decode_speeds else []

    # --- Per-second bucketing for peak decode tok/s ---
    peak_decode_tok_s = 0.0
    peak_concurrent = 0

    successful = [r for r in results if r.success]
    if successful:
        min_start = min(r.start_time for r in successful)
        max_end = max(r.start_time + r.latency for r in successful)

        duration_s = int(math.ceil(max_end - min_start)) + 1
        tokens_per_second = np.zeros(duration_s)
        concurrent_per_second = np.zeros(duration_s)

        for r in successful:
            # Decode: each output token mapped by its timestamp
            if r.output_tokens > 0 and r.itl:
                current = r.start_time + r.ttft
                for itl_val in r.itl:
                    bucket = int(current - min_start)
                    if 0 <= bucket < duration_s:
                        tokens_per_second[bucket] += 1
                    current += itl_val
                bucket = int(current - min_start)
                if 0 <= bucket < duration_s:
                    tokens_per_second[bucket] += 1

            # Concurrent requests per second
            s_start = int(r.start_time - min_start)
            s_end = int((r.start_time + r.latency) - min_start)
            for s in range(max(0, s_start), min(duration_s, s_end + 1)):
                concurrent_per_second[s] += 1

        peak_decode_tok_s = float(np.max(tokens_per_second))
        peak_concurrent = int(np.max(concurrent_per_second))

    avg_prefill = total_input / completed if completed > 0 else 0

    return {
        "completed": completed,
        "failed": failed,
        "total_input": total_input,
        "total_output": total_output,
        "dur_s": dur_s,
        "request_throughput": request_throughput,
        "input_throughput": input_throughput,
        "output_throughput": output_throughput,
        "total_token_throughput": (total_input + total_output) / dur_s,
        "avg_prefill": avg_prefill,
        "mean_ttft": mean_ttft,
        "median_ttft": median_ttft,
        "std_ttft": std_ttft,
        "pctl_ttft": pctl_ttft,
        "mean_itl": mean_itl,
        "median_itl": median_itl,
        "std_itl": std_itl,
        "pctl_itl": pctl_itl,
        "mean_e2el": mean_e2el,
        "median_e2el": median_e2el,
        "std_e2el": std_e2el,
        "pctl_e2el": pctl_e2el,
        # Per-request prefill speed (tok/s)
        "mean_prefill_speed": mean_prefill_speed,
        "median_prefill_speed": median_prefill_speed,
        "std_prefill_speed": std_prefill_speed,
        "pctl_prefill_speed": pctl_prefill_speed,
        # Per-request decode speed (tok/s)
        "mean_decode_speed": mean_decode_speed,
        "median_decode_speed": median_decode_speed,
        "std_decode_speed": std_decode_speed,
        "pctl_decode_speed": pctl_decode_speed,
        # TPOT (ms/token) — vLLM standard decode metric
        "mean_tpot": mean_tpot,
        "median_tpot": median_tpot,
        "std_tpot": std_tpot,
        "pctl_tpot": pctl_tpot,
        # Peak decode tok/s from second bucketing
        "peak_decode_tok_s": peak_decode_tok_s,
        "peak_concurrent": peak_concurrent,
    }


def print_metrics(m: dict):
    def p_word(p):
        return str(int(p)) if int(p) == p else str(p)

    print("\n" + "=" * 60)
    print("  SERVING BENCHMARK RESULT")
    print("=" * 60)
    print(f"  Successful requests:           {m['completed']}")
    print(f"  Failed requests:               {m['failed']}")
    print(f"  Total input tokens (prefill):  {m['total_input']}")
    print(f"  Total output tokens (decode):  {m['total_output']}")
    print(f"  Benchmark duration (s):        {m['dur_s']:.2f}")
    print(f"  Request throughput (req/s):    {m['request_throughput']:.2f}")
    print(f"  Input token throughput (tok/s):  {m['input_throughput']:.2f}")
    print(f"  Output token throughput (tok/s): {m['output_throughput']:.2f}")
    print(f"  Total token throughput (tok/s):  {m['total_token_throughput']:.2f}")
    print(f"  Peak decode tok/s:             {m['peak_decode_tok_s']:.1f}")
    print(f"  Peak concurrent requests:      {m['peak_concurrent']}")
    print(f"  Avg prefill tokens / req:      {m['avg_prefill']:.1f}")
    print()
    print("-" * 60)
    print("  PREFILL SPEED (tok/s, per-request)")
    print("-" * 60)
    print(f"  Mean:    {m['mean_prefill_speed']:.2f}")
    print(f"  Median:  {m['median_prefill_speed']:.2f}")
    print(f"  Std:     {m['std_prefill_speed']:.2f}")
    for p, v in m["pctl_prefill_speed"]:
        print(f"  P{p_word(p)}:   {v:.2f}")
    print()
    print("-" * 60)
    print("  DECODE SPEED (tok/s, per-request)")
    print("-" * 60)
    print(f"  Mean:    {m['mean_decode_speed']:.2f}")
    print(f"  Median:  {m['median_decode_speed']:.2f}")
    print(f"  Std:     {m['std_decode_speed']:.2f}")
    for p, v in m["pctl_decode_speed"]:
        print(f"  P{p_word(p)}:   {v:.2f}")
    print()
    print("-" * 60)
    print("  TPOT (ms/token, vLLM-style)")
    print("-" * 60)
    print(f"  Mean:    {m['mean_tpot']:.2f}")
    print(f"  Median:  {m['median_tpot']:.2f}")
    print(f"  Std:     {m['std_tpot']:.2f}")
    for p, v in m["pctl_tpot"]:
        print(f"  P{p_word(p)}:   {v:.2f}")
    print()
    print("-" * 60)
    print("  TTFT (ms)")
    print("-" * 60)
    print(f"  Mean:    {m['mean_ttft']:.2f}")
    print(f"  Median:  {m['median_ttft']:.2f}")
    print(f"  Std:     {m['std_ttft']:.2f}")
    for p, v in m["pctl_ttft"]:
        print(f"  P{p_word(p)}:   {v:.2f}")
    print()
    print("-" * 60)
    print("  ITL (ms)")
    print("-" * 60)
    print(f"  Mean:    {m['mean_itl']:.2f}")
    print(f"  Median:  {m['median_itl']:.2f}")
    print(f"  Std:     {m['std_itl']:.2f}")
    for p, v in m["pctl_itl"]:
        print(f"  P{p_word(p)}:   {v:.2f}")
    print()
    print("-" * 60)
    print("  E2E Latency (ms)")
    print("-" * 60)
    print(f"  Mean:    {m['mean_e2el']:.2f}")
    print(f"  Median:  {m['median_e2el']:.2f}")
    print(f"  Std:     {m['std_e2el']:.2f}")
    for p, v in m["pctl_e2el"]:
        print(f"  P{p_word(p)}:   {v:.2f}")
    print("=" * 60)
    sys.stdout.flush()


def _build_result_path() -> str:
    """Build auto-generated result file path from benchmark params + timestamp."""
    ts = time.strftime("%Y%m%d-%H%M%S")
    dur = f"dur{ARGS.duration}" if ARGS.duration > 0 else f"n{ARGS.num_prompts}"
    rate = str(ARGS.request_rate) if ARGS.request_rate != float("inf") else "inf"
    fname = f"{dur}-w{ARGS.num_workers}-r{rate}-{ARGS.model}-{ts}.json"
    if ARGS.save_results_dir:
        return os.path.join(ARGS.save_results_dir, fname)
    return fname


def print_and_save_metrics():
    """Print metrics from module-level state. Used on normal exit and Ctrl+C."""
    if _benchmark_start <= 0:
        return
    dur_s = time.perf_counter() - _benchmark_start
    metrics = calculate_metrics(_all_results, dur_s)
    if metrics:
        print_metrics(metrics)

        if ARGS.save_results_dir:
            result_path = _build_result_path()
            save_data = {
                "benchmark_params": {
                    "duration": ARGS.duration,
                    "num_prompts": ARGS.num_prompts,
                    "num_workers": ARGS.num_workers,
                    "request_rate": ARGS.request_rate if ARGS.request_rate != float("inf") else "inf",
                    "model": ARGS.model,
                    "base_url": ARGS.base_url,
                    "dataset_path": ARGS.dataset_path,
                    "seed": ARGS.seed,
                    "warmup_requests": ARGS.warmup_requests,
                },
                "duration_s": dur_s,
                "completed": metrics["completed"],
                "failed": metrics["failed"],
                "total_input_tokens": metrics["total_input"],
                "total_output_tokens": metrics["total_output"],
                "request_throughput": metrics["request_throughput"],
                "input_throughput": metrics["input_throughput"],
                "output_throughput": metrics["output_throughput"],
                "total_token_throughput": metrics["total_token_throughput"],
                "avg_prefill_tokens_per_req": metrics["avg_prefill"],
                "peak_decode_tok_s": metrics["peak_decode_tok_s"],
                "peak_concurrent": metrics["peak_concurrent"],
                "mean_prefill_speed_tok_s": metrics["mean_prefill_speed"],
                "median_prefill_speed_tok_s": metrics["median_prefill_speed"],
                "mean_decode_speed_tok_s": metrics["mean_decode_speed"],
                "median_decode_speed_tok_s": metrics["median_decode_speed"],
                "mean_tpot_ms": metrics["mean_tpot"],
                "median_tpot_ms": metrics["median_tpot"],
                "mean_ttft_ms": metrics["mean_ttft"],
                "median_ttft_ms": metrics["median_ttft"],
                "mean_itl_ms": metrics["mean_itl"],
                "median_itl_ms": metrics["median_itl"],
                "mean_e2el_ms": metrics["mean_e2el"],
                "median_e2el_ms": metrics["median_e2el"],
            }
            result_dir = os.path.dirname(result_path) or "."
            os.makedirs(result_dir, exist_ok=True)
            with open(result_path, "w") as f:
                json.dump(save_data, f, indent=2)
            logger.info("Results saved to %s", result_path)

        if metrics["failed"] > 0:
            logger.warning(
                "%d requests failed during benchmark (may be caused by long prompts or server errors)",
                metrics["failed"],
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main():
    continuous = ARGS.duration > 0
    mode_label = f"continuous ({ARGS.duration}s)" if continuous else f"fixed ({ARGS.num_prompts} requests)"
    logger.info("Benchmark mode: %s", mode_label)

    # Load dataset
    dataset = load_sharegpt_dataset(ARGS.dataset_path, seed=ARGS.seed)
    requests_pool = sample_requests(dataset, ARGS.num_prompts)
    pool_size = len(requests_pool)
    logger.info("Dataset pool: %d entries", pool_size)

    rng = np.random.default_rng(ARGS.seed)
    semaphore = asyncio.Semaphore(ARGS.num_workers)
    results_lock = asyncio.Lock()
    live_stats = defaultdict(lambda: 0)
    # Cap results list to prevent memory blowup in continuous mode.
    MAX_RESULTS = 10000

    async def worker(entry: dict):
        prompt = entry["conversations"][0]["value"]
        prompt_len = len(prompt) // 4
        async with semaphore:
            result = await send_request(client, prompt, prompt_len)
        async with results_lock:
            _all_results.append(result)
            if len(_all_results) > MAX_RESULTS:
                _all_results[:] = _all_results[-MAX_RESULTS:]
            if result.success:
                live_stats["completed"] += 1
                live_stats["total_content"] += result.output_tokens
                live_stats["total_input"] += result.prompt_len
            else:
                live_stats["failed"] += 1

    async with httpx.AsyncClient(
        timeout=180.0,
        limits=httpx.Limits(max_connections=10000, max_keepalive_connections=10000),
    ) as client:
        # Wait for server
        logger.info("Waiting for server at %s...", ARGS.base_url)
        for _ in range(30):
            try:
                resp = await client.get(f"{ARGS.base_url}/v1/models")
                if resp.status_code == 200:
                    logger.info("Server ready.")
                    break
            except Exception:
                pass
            await asyncio.sleep(2)
        else:
            logger.error("Server not reachable")
            sys.exit(1)

        # --- Warmup phase ---
        warmup_count = ARGS.warmup_requests * ARGS.num_workers
        if warmup_count > 0:
            logger.info("Running %d warmup requests (discarded)...", warmup_count)
            warmup_entries = [requests_pool[i % pool_size] for i in range(warmup_count)]
            warmup_tasks = [asyncio.create_task(worker(entry)) for entry in warmup_entries]
            await asyncio.gather(*warmup_tasks, return_exceptions=True)
            async with results_lock:
                _all_results.clear()
            live_stats.clear()

        # --- Main benchmark ---
        global _benchmark_start
        _benchmark_start = time.perf_counter()
        rate_str = ARGS.request_rate if ARGS.request_rate != float("inf") else "inf"
        logger.info(
            "Starting benchmark: %s, %d workers, rate=%s req/s",
            mode_label,
            ARGS.num_workers,
            rate_str,
        )

        stop_event = asyncio.Event()
        loop = asyncio.get_event_loop()

        def _handle_signal():
            if not stop_event.is_set():
                stop_event.set()
                logger.info("Signal received, stopping dispatch...")

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _handle_signal)

        async def live_reporter():
            while not stop_event.is_set():
                await asyncio.sleep(ARGS.report_interval)
                if stop_event.is_set():
                    break
                elapsed = time.perf_counter() - _benchmark_start
                async with results_lock:
                    tp_d = live_stats["total_content"] / elapsed if elapsed > 0 else 0
                    tp_p = live_stats["total_input"] / elapsed if elapsed > 0 else 0
                    tp_all = tp_p + tp_d
                    rps = live_stats["completed"] / elapsed if elapsed > 0 else 0
                    logger.info(
                        "=== LIVE | %.1fs | reqs: %d | P: %d | D: %d | P: %.1f | D: %.1f | Total: %.1f tok/s | %.2f RPS | fails: %d ===",
                        elapsed,
                        live_stats["completed"],
                        live_stats["total_input"],
                        live_stats["total_content"],
                        tp_p,
                        tp_d,
                        tp_all,
                        rps,
                        live_stats["failed"],
                    )

        async def duration_timer():
            """Auto-stop after --duration seconds."""
            await asyncio.sleep(ARGS.duration)
            stop_event.set()
            logger.info("Duration limit (%ds) reached.", ARGS.duration)

        reporter_task = asyncio.create_task(live_reporter())
        timer_task = asyncio.create_task(duration_timer()) if continuous else None

        # Dispatch requests
        tasks: list[asyncio.Task] = []
        dispatch_count = 0
        try:
            async for req_idx in generate_requests(pool_size, ARGS.request_rate, rng, continuous=continuous, stop_event=stop_event):
                if stop_event.is_set():
                    break
                tasks.append(asyncio.create_task(worker(requests_pool[req_idx])))
                dispatch_count += 1
                if dispatch_count % 256 == 0:
                    await asyncio.sleep(0)
                    tasks = [t for t in tasks if not t.done()]
        except asyncio.CancelledError:
            pass

        if stop_event.is_set():
            logger.info("Cancelling %d in-flight tasks...", len(tasks))
            for t in tasks:
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

        reporter_task.cancel()
        try:
            await reporter_task
        except asyncio.CancelledError:
            pass
        if timer_task:
            timer_task.cancel()
            try:
                await timer_task
            except asyncio.CancelledError:
                pass

    # Normal exit: print metrics.
    print_and_save_metrics()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted, printing final metrics...")
        print_and_save_metrics()
