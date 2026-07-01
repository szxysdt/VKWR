"""Generate a benchmark comparison markdown report from JSON result files.

Reads all JSON results under a benchmark directory (e.g., docs/benchmark/4060-laptop/),
matches VKWR vs vLLM results by num_workers, computes performance differences,
and writes benchmark.md next to README.md.

Usage:
    uv run python scripts/benchmark_compare.py \
        --benchmark-dir docs/benchmark/4060-laptop \
        --output docs/benchmark/4060-laptop/benchmark.md
"""

import argparse
import json
import os
import sys
from pathlib import Path


def load_json_files(directory: str) -> dict[int, dict]:
    """Load all JSON files from a directory, keyed by num_workers."""
    results = {}
    p = Path(directory)
    if not p.is_dir():
        return results
    for f in sorted(p.glob("*.json")):
        with open(f) as fp:
            data = json.load(fp)
        nw = data.get("benchmark_params", {}).get("num_workers")
        if nw is not None:
            results[nw] = data
    return results


def fmt_pct(v: float) -> str:
    """Format a ratio as a percentage string."""
    if v > 1:
        return f"+{v - 1:.1%}"
    elif v < 1:
        return f"{v - 1:.1%}"
    return "0.0%"


def fmt_pct_lower_better(v: float) -> str:
    """Format percentage for metrics where lower is better."""
    if v < 1:
        return f"{v - 1:.1%} (better)"
    elif v > 1:
        return f"+{v - 1:.1%} (worse)"
    return "0.0%"



def fmt_latency(val: float) -> str:
    """Format a latency value with appropriate unit."""
    if val < 0.1:
        return f"{val * 1000:.1f} us"
    return f"{val:.1f} ms"


def _ratio_str(v_val: float, l_val: float, lower_better: bool = False) -> str:
    """Compute ratio string."""
    if not l_val:
        return "-"
    r = v_val / l_val
    return fmt_pct_lower_better(r) if lower_better else fmt_pct(r)


def generate_report(
    vkwr_dir: str,
    vllm_dir: str,
    title: str,
    gpu: str,
    vkwr_model: str,
    vllm_model: str,
    dataset: str,
    duration: int,
    warmup: int,
    extra_notes: str = "",
) -> str:
    lines = []
    lines.append(f"# {title}\n")
    lines.append("## Environment\n")
    lines.append("| Item | Value |")
    lines.append("|------|-------|")
    lines.append(f"| GPU | {gpu} |")
    lines.append(f"| Dataset | {dataset} |")
    lines.append(f"| Duration | {duration}s per run |")
    lines.append(f"| Warmup requests | {warmup} per worker |")
    lines.append("| Request rate | unlimited (inf) |")
    lines.append("| Max tokens per request | 1024 |")
    lines.append("")
    lines.append("## Models\n")
    lines.append("| Framework | Model | Parameters |")
    lines.append("|-----------|-------|------------|")
    lines.append(f"| VKWR | {vkwr_model} | 1.5B |")
    lines.append(f"| vLLM | {vllm_model} | ~1.8B (LLM portion) |")
    lines.append("")

    vkwr = load_json_files(vkwr_dir)
    vllm = load_json_files(vllm_dir)

    if not vkwr or not vllm:
        lines.append("> **No matching results found.** Ensure both `vkwr/` and `vllm/` subdirectories contain result JSON files.\n")
        return "\n".join(lines)

    all_workers = sorted(set(list(vkwr.keys()) + list(vllm.keys())))

    # --- Throughput summary table (high-level only) ---
    lines.append("## Throughput Summary\n")
    lines.append("| Workers | Output tok/s | Total tok/s | Req/s | Mean TTFT (ms) | Median TTFT (ms) | Mean TPOT (ms) | Median TPOT (ms) |")
    lines.append("|---------|-------------|-------------|-------|--------------|----------------|--------------|----------------|")

    for nw in all_workers:
        has_vkwr = nw in vkwr
        has_vllm = nw in vllm

        if has_vkwr and has_vllm:
            vkwr_d = vkwr[nw]
            vllm_d = vllm[nw]
            lines.append(
                f"| {nw} | "
                f"{vkwr_d['output_throughput']:.0f} / {vllm_d['output_throughput']:.0f} | "
                f"{vkwr_d['total_token_throughput']:.0f} / {vllm_d['total_token_throughput']:.0f} | "
                f"{vkwr_d['request_throughput']:.2f} / {vllm_d['request_throughput']:.2f} | "
                f"{vkwr_d['mean_ttft_ms']:.0f} / {vllm_d['mean_ttft_ms']:.0f} | "
                f"{vkwr_d['median_ttft_ms']:.0f} / {vllm_d['median_ttft_ms']:.0f} | "
                f"{vkwr_d['mean_tpot_ms']:.1f} / {vllm_d['mean_tpot_ms']:.1f} | "
                f"{vkwr_d['median_tpot_ms']:.1f} / {vllm_d['median_tpot_ms']:.1f} |"
            )
        elif has_vkwr:
            d = vkwr[nw]
            lines.append(
                f"| {nw} | "
                f"{d['output_throughput']:.0f} (VKWR) | "
                f"{d['total_token_throughput']:.0f} | "
                f"{d['request_throughput']:.2f} | "
                f"{d['mean_ttft_ms']:.0f} | "
                f"{d['median_ttft_ms']:.0f} | "
                f"{d['mean_tpot_ms']:.1f} | "
                f"{d['median_tpot_ms']:.1f} |"
            )
        else:
            d = vllm[nw]
            lines.append(
                f"| {nw} | "
                f"{d['output_throughput']:.0f} (vLLM) | "
                f"{d['total_token_throughput']:.0f} | "
                f"{d['request_throughput']:.2f} | "
                f"{d['mean_ttft_ms']:.0f} | "
                f"{d['median_ttft_ms']:.0f} | "
                f"{d['mean_tpot_ms']:.1f} | "
                f"{d['median_tpot_ms']:.1f} |"
            )

    lines.append("")
    lines.append("(Format: VKWR / vLLM)\n")

    # --- Per-worker detailed comparison ---
    lines.append("## Detailed Comparison\n")

    for nw in all_workers:
        if nw not in vkwr or nw not in vllm:
            continue

        vkwr_d = vkwr[nw]
        vllm_d = vllm[nw]

        lines.append(f"### Workers = {nw}\n")

        # Request & peak stats
        lines.append("| Metric | VKWR | vLLM |")
        lines.append("|--------|------|------|")
        lines.append(f"| Completed / Failed | {vkwr_d['completed']} / {vkwr_d['failed']} | {vllm_d['completed']} / {vllm_d['failed']} |")
        lines.append(f"| Total input tokens | {vkwr_d['total_input_tokens']:,} | {vllm_d['total_input_tokens']:,} |")
        lines.append(f"| Total output tokens | {vkwr_d['total_output_tokens']:,} | {vllm_d['total_output_tokens']:,} |")
        lines.append(f"| Peak decode tok/s | {vkwr_d['peak_decode_tok_s']:.0f} | {vllm_d['peak_decode_tok_s']:.0f} |")
        lines.append(f"| Peak concurrent | {vkwr_d['peak_concurrent']} | {vllm_d['peak_concurrent']} |")
        lines.append("")

        # Throughput
        lines.append("**Throughput**\n")
        lines.append("| Metric | VKWR | vLLM | Ratio (VKWR/vLLM) |")
        lines.append("|--------|------|------|-------------------|")
        for metric, key in [
            ("Output tok/s", "output_throughput"),
            ("Input tok/s", "input_throughput"),
            ("Total tok/s", "total_token_throughput"),
            ("Req/s", "request_throughput"),
        ]:
            v_val = vkwr_d[key]
            l_val = vllm_d[key]
            lines.append(f"| {metric} | {v_val:.1f} | {l_val:.1f} | {_ratio_str(v_val, l_val)} |")
        lines.append("")

        # Latency
        lines.append("**Latency**\n")
        lines.append("| Metric | VKWR | vLLM | Ratio (lower better) |")
        lines.append("|--------|------|------|----------------------|")
        for metric, key in [
            ("Mean TTFT", "mean_ttft_ms"),
            ("Median TTFT", "median_ttft_ms"),
            ("Mean ITL", "mean_itl_ms"),
            ("Mean E2E latency", "mean_e2el_ms"),
            ("Median E2E latency", "median_e2el_ms"),
        ]:
            v_val = vkwr_d[key]
            l_val = vllm_d[key]
            lines.append(
                f"| {metric} | {fmt_latency(v_val)} | {fmt_latency(l_val)} | {_ratio_str(v_val, l_val, True)} |"
            )
        lines.append("")

        # Per-request speed
        lines.append("**Per-Request Speed**\n")
        lines.append("| Metric | VKWR | vLLM | Ratio (higher better) |")
        lines.append("|--------|------|------|-----------------------|")
        for metric, key in [
            ("Mean prefill (tok/s)", "mean_prefill_speed_tok_s"),
            ("Median prefill (tok/s)", "median_prefill_speed_tok_s"),
            ("Mean decode (tok/s)", "mean_decode_speed_tok_s"),
            ("Median decode (tok/s)", "median_decode_speed_tok_s"),
        ]:
            v_val = vkwr_d[key]
            l_val = vllm_d[key]
            lines.append(f"| {metric} | {v_val:.1f} | {l_val:.1f} | {_ratio_str(v_val, l_val)} |")
        lines.append("")

        # TPOT
        lines.append("**TPOT (ms/token, lower better)**\n")
        lines.append("| Metric | VKWR | vLLM | Ratio (lower better) |")
        lines.append("|--------|------|------|----------------------|")
        for metric, key in [
            ("Mean TPOT", "mean_tpot_ms"),
            ("Median TPOT", "median_tpot_ms"),
        ]:
            v_val = vkwr_d[key]
            l_val = vllm_d[key]
            lines.append(
                f"| {metric} | {v_val:.1f} | {l_val:.1f} | {_ratio_str(v_val, l_val, True)} |"
            )

        # Note TPOT skew if mean >> median
        for label, d in [("VKWR", vkwr_d), ("vLLM", vllm_d)]:
            mean_t = d["mean_tpot_ms"]
            med_t = d["median_tpot_ms"]
            if med_t > 0 and mean_t > med_t * 3:
                lines.append(
                    f"> {label}: Mean TPOT ({mean_t:.0f} ms) >> Median TPOT ({med_t:.0f} ms), "
                    "indicating a long-tail distribution."
                )
        lines.append("")

    # --- Unmatched workers ---
    unmatched_vkwr = [w for w in vkwr if w not in vllm]
    unmatched_vllm = [w for w in vllm if w not in vkwr]
    if unmatched_vkwr or unmatched_vllm:
        lines.append("## Unmatched Configurations\n")
        if unmatched_vkwr:
            lines.append(f"VKWR-only workers: {', '.join(str(w) for w in sorted(unmatched_vkwr))}\n")
            for nw in sorted(unmatched_vkwr):
                vkwr_d = vkwr[nw]
                lines.append(f"### VKWR Workers = {nw}\n")
                lines.append("| Metric | Value |")
                lines.append("|--------|-------|")
                lines.append(f"| Completed / Failed | {vkwr_d['completed']} / {vkwr_d['failed']} |")
                lines.append(f"| Output throughput (tok/s) | {vkwr_d['output_throughput']:.1f} |")
                lines.append(f"| Total throughput (tok/s) | {vkwr_d['total_token_throughput']:.1f} |")
                lines.append(f"| Mean TTFT (ms) | {vkwr_d['mean_ttft_ms']:.1f} |")
                lines.append(f"| Median TTFT (ms) | {vkwr_d['median_ttft_ms']:.1f} |")
                lines.append(f"| Mean TPOT (ms) | {vkwr_d['mean_tpot_ms']:.1f} |")
                lines.append(f"| Median TPOT (ms) | {vkwr_d['median_tpot_ms']:.1f} |")
                lines.append("")
        if unmatched_vllm:
            lines.append(f"vLLM-only workers: {', '.join(str(w) for w in sorted(unmatched_vllm))}\n")
            for nw in sorted(unmatched_vllm):
                vllm_d = vllm[nw]
                lines.append(f"### vLLM Workers = {nw}\n")
                lines.append("| Metric | Value |")
                lines.append("|--------|-------|")
                lines.append(f"| Completed / Failed | {vllm_d['completed']} / {vllm_d['failed']} |")
                lines.append(f"| Output throughput (tok/s) | {vllm_d['output_throughput']:.1f} |")
                lines.append(f"| Total throughput (tok/s) | {vllm_d['total_token_throughput']:.1f} |")
                lines.append(f"| Mean TTFT (ms) | {vllm_d['mean_ttft_ms']:.1f} |")
                lines.append(f"| Median TTFT (ms) | {vllm_d['median_ttft_ms']:.1f} |")
                lines.append(f"| Mean TPOT (ms) | {vllm_d['mean_tpot_ms']:.1f} |")
                lines.append(f"| Median TPOT (ms) | {vllm_d['median_tpot_ms']:.1f} |")
                lines.append("")

    if extra_notes:
        lines.append("## Notes\n")
        lines.append(extra_notes)
        lines.append("")

    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate benchmark comparison report")
    parser.add_argument(
        "--benchmark-dir",
        required=True,
        help="Root benchmark directory containing vkwr/ and vllm/ subdirectories",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for the generated markdown report",
    )
    parser.add_argument("--title", default="Benchmark Comparison Report")
    parser.add_argument("--gpu", default="NVIDIA GeForce RTX 4060 Laptop (8GB)")
    parser.add_argument("--vkwr-model", default="RWKV7-1.5B")
    parser.add_argument("--vllm-model", default="Qwen3.5-2B")
    parser.add_argument("--dataset", default="ShareGPT V3 unfiltered cleaned")
    parser.add_argument("--duration", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--notes", default="", help="Extra notes to append")
    return parser.parse_args()


def main():
    args = parse_args()
    base = args.benchmark_dir

    vkwr_dir = os.path.join(base, "vkwr")
    vllm_dir = os.path.join(base, "vllm")

    if not os.path.isdir(vkwr_dir) and not os.path.isdir(vllm_dir):
        print(f"Error: Neither vkwr/ nor vllm/ found under {base}", file=sys.stderr)
        sys.exit(1)

    report = generate_report(
        vkwr_dir=vkwr_dir,
        vllm_dir=vllm_dir,
        title=args.title,
        gpu=args.gpu,
        vkwr_model=args.vkwr_model,
        vllm_model=args.vllm_model,
        dataset=args.dataset,
        duration=args.duration,
        warmup=args.warmup,
        extra_notes=args.notes,
    )

    out_dir = os.path.dirname(args.output) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
