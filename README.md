# VKWR

A high-performance inference engine for the RWKV linear language model, with optimized CUDA kernels and continuous batching support.

> **Note:** This project is under active development. Dependency versions may not have been exhaustively tested. A comprehensive test suite is provided — if you relax any dependency constraints, please run the tests first to verify correctness.

---

## Usage

### Installation

```bash
uv sync --no-build-isolation
```

After modifying native code, rebuild:

```bash
uv pip install -e . --no-build-isolation
```

### Start the API Server

After installation, use the `vkwr` CLI to launch an OpenAI-compatible API server:

```bash
# Minimal — starts server on 0.0.0.0:8000
vkwr --model /path/to/model.pth

# With common options
vkwr --model /path/to/model.pth \
  --max-model-len 8192 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 2048 \
  --port 8000

```

### CLI Options

| Option            | Default                  | Description                                                           |
| ----------------- | ------------------------ | --------------------------------------------------------------------- |
| `--model`         | *(required)*             | Path to `.pth` model file                                             |
| `--tokenizer`     | model's parent directory | Path to `rwkv_vocab_v20230424.txt` or its parent directory (optional) |
| `--host`          | `0.0.0.0`                | API server bind address                                               |
| `--port`          | `8000`                   | API server port                                                       |
| `--max-model-len` | None (no limit)          | Maximum sequence length                                               |
| `--max-tokens`    | 2048 (final fallback)    | Default max generation tokens (applied when API request omits it)     |

For the full list of CLI options, see [`docs/cli-reference.md`](docs/cli-reference.md).

### API Endpoints

The server provides an OpenAI-compatible API with endpoints for text completion (`/v1/completions`) and chat (`/v1/chat/completions`).
See [`docs/api.md`](docs/api.md) for details and examples.

## Supported Models

| Model | Parameters | Layers | Hidden Dim | Download |
| ----- | ---------- | ------ | ---------- | -------- |
| **RWKV7-G1** | 0.1B | 12 | 768 | [HuggingFace](https://huggingface.co/BlinkDL/rwkv7-g1) · [ModelScope](https://modelscope.cn/models/Blink_DL/rwkv7-g1/) |
| **RWKV7-G1** | 0.4B | 24 | 1024 | Same as above |
| **RWKV7-G1** | 1.5B | 24 | 2048 | Same as above |
| **RWKV7-G1** | 2.9B | 32 | 2560 | Same as above |
| **RWKV7-G1** | 7.2B | 32 | 4096 | Same as above |
| **RWKV7-G1** | 13.3B | 61 | 4096 | Same as above |

All models share a vocabulary size of 65536 and a head size of 64.

## Requirements

- Python 3.10+
- CUDA 12.x (CUDA arch auto-detected from PyTorch; override with `VKWR_CUDA_ARCH`)
- PyTorch 2.x
- `uv` for environment management and installation


## Acknowledgments

VKWR draws inspiration from and builds upon the following outstanding open-source projects:

- [RWKV-LM](https://github.com/BlinkDL/RWKV-LM/) — The original RWKV language model architecture by BlinkDL.
- [Albatross](https://github.com/BlinkDL/Albatross/) — A high-performance RWKV inference engine with optimized CUDA kernels, by BlinkDL.
- [rwkv_lightning](https://github.com/RWKV-Vibe/rwkv_lightning/) — A fast RWKV batch inference framework by RWKV-Vibe.
- [vLLM](https://github.com/vllm-project/vllm) — A high-throughput, memory-efficient LLM inference and serving engine.
