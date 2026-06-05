"""VKWR CLI entry point.

Usage:
    # Start OpenAI-compatible API server
    vkwr --model /path/to/model.pth
"""

from __future__ import annotations

import argparse
import logging
import sys

from vkwr.config.vkwr import EngineArgs

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vkwr",
        description="VKWR - High-concurrency RWKV inference engine",
    )

    # Model arguments
    parser.add_argument("--model", required=True, help="Model path (.pth file)")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer path (defaults to model directory)")
    parser.add_argument("--tokenizer-mode", default="rwkv", help="Tokenizer mode: rwkv | auto")
    parser.add_argument("--dtype", default="float16", help="Compute dtype (default float16)")
    parser.add_argument("--load-format", default="auto", help="Model load format (default auto)")
    parser.add_argument("--max-model-len", type=int, default=None, help="Maximum sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Scheduler arguments
    parser.add_argument("--max-num-batched-tokens", type=int, default=None, help="Max global tokens per step")
    parser.add_argument("--max-num-seqs", type=int, default=128, help="Maximum concurrent requests")

    # Worker arguments
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92, help="GPU memory utilization cap")
    parser.add_argument("--enforce-eager", action="store_true", help="Force eager mode (debugging)")

    # CUDA Graph arguments
    parser.add_argument("--cudagraph-mode", default="full", choices=["full", "none"], help="CUDA Graph mode")

    # Server arguments
    parser.add_argument("--host", default="0.0.0.0", help="API server host")
    parser.add_argument("--port", type=int, default=8000, help="API server port")

    # Generation defaults (applied when API request omits max_tokens)
    parser.add_argument("--max-tokens", type=int, default=None, help="Maximum tokens to generate (global default, used when API request omits it)")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    engine_args = EngineArgs(
        model=args.model,
        tokenizer=args.tokenizer,
        tokenizer_mode=args.tokenizer_mode,
        dtype=args.dtype,
        load_format=args.load_format,
        max_model_len=args.max_model_len,
        seed=args.seed,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        cudagraph_mode=args.cudagraph_mode,
        default_max_tokens=args.max_tokens,
    )

    _run_server(args, engine_args)


def _run_server(args: argparse.Namespace, engine_args: EngineArgs) -> None:
    """Start the OpenAI-compatible API server."""
    try:
        import uvicorn
    except ImportError:
        logger.error("uvicorn is not installed. Install with: uv pip install fastapi uvicorn")
        sys.exit(1)

    from vkwr.entrypoints.api import create_app

    logger.info("Starting VKWR API server...")
    logger.info("Model: %s", engine_args.model)
    logger.info("Listening on %s:%d", args.host, args.port)

    app = create_app(engine_args)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
