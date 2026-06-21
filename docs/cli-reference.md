# CLI Reference

## Common Options

| Option            | Default                  | Description                                                           |
| ----------------- | ------------------------ | --------------------------------------------------------------------- |
| `--model`         | *(required)*             | Path to `.pth` model file                                             |
| `--tokenizer`     | model's parent directory | Path to `rwkv_vocab_v20230424.txt` or its parent directory (optional) |
| `--host`          | `0.0.0.0`                | API server bind address                                               |
| `--port`          | `8000`                   | API server port                                                       |
| `--max-model-len` | None (no limit)          | Maximum sequence length                                               |
| `--max-tokens`    | 2048 (final fallback)    | Default max generation tokens (applied when API request omits it)      |

## Advanced Options

> **Status: Under active development.** Some parameters below are currently locked to fixed values or reserved for future use.

| Option                     | Default   | Description                                                       | Status |
| -------------------------- | --------- | ----------------------------------------------------------------- | ------ |
| `--dtype`                  | `float16` | Compute dtype                                                     | **Locked** — only `float16` is supported. Any other value will raise `ValueError` at startup (`vkwr/config/engine.py:27`). |
| `--max-num-batched-tokens` | `2048`    | Max tokens per scheduling step                                    | **Constrained** — must be `>= --max-num-seqs`, otherwise `ValueError` at startup (`vkwr/config/engine.py:30`). |
| `--cudagraph-mode`         | `full`    | CUDA Graph mode: `full` \| `none`                                 | Enforced by argparse `choices` |
| `--max-num-seqs`           | `64`      | Maximum concurrent requests                                       | Must be `<= --max-num-batched-tokens` |
| `--load-format`            | `auto`    | Model load format                                                 | **Locked** — no downstream logic branches on this value; always behaves as `"auto"` |
| `--seed`                   | `42`      | Random seed                                                       | — |
| `--enforce-eager`          | off       | Force eager mode (disables CUDA graphs, for debugging)            | — |
| `--gpu-memory-utilization` | `0.92`    | GPU memory utilization cap                                        | **Reserved** — parameter chain exists but `determine_available_memory()` is never called by any executor; no effect currently |
