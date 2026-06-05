"""VKWR CLI entry point.

Usage:
    # Start OpenAI-compatible API server
    vkwr --model /path/to/model.pth

    # CLI generation (direct mode)
    vkwr --model /path/to/model.pth --generate "Hello, world!"
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

    # Direct generation mode
    parser.add_argument("--generate", type=str, default=None, help="Direct generation mode: pass a prompt and print output")
    parser.add_argument("--max-tokens", type=int, default=None, help="Maximum tokens to generate (global default, used when API request omits it)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (--generate mode)")
    parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling parameter (--generate mode)")
    parser.add_argument("--top-k", type=int, default=-1, help="Top-k sampling parameter (--generate mode)")

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

    if args.generate is not None:
        _run_generate(args, engine_args)
    else:
        _run_server(args, engine_args)


def _run_generate(args: argparse.Namespace, engine_args: EngineArgs) -> None:
    """Direct generation mode: load model, run one generation, print results, and exit."""
    from vkwr.engine.request import SamplingParams
    from vkwr.entrypoints.llm import LLM

    logger.info("Starting VKWR in generate mode...")
    logger.info("Model: %s", engine_args.model)

    llm = LLM(**vars(engine_args))

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )

    try:
        outputs = llm.generate(args.generate, sampling_params)
    except Exception as e:
        logger.error("Generation failed: %s", e, exc_info=True)
        sys.exit(1)

    if not outputs:
        logger.warning("No output generated")
        return

    for out in outputs:
        if out.outputs:
            print(out.outputs[0].text, end="")
            print()
        else:
            print("(empty output)")


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
