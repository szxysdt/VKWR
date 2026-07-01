# VKWR vs vLLM — RTX 4060 Laptop Benchmark

## Environment

| Item | Value |
|------|-------|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU (8GB) |
| Dataset | ShareGPT V3 unfiltered cleaned |
| Duration | 300s per run |
| Warmup requests | 3 per worker |
| Request rate | unlimited (inf) |
| Max tokens per request | 1024 |

## Models

| Framework | Model | Parameters |
|-----------|-------|------------|
| VKWR | RWKV7-1.5B | 1.5B |
| vLLM | Qwen3.5-2B | ~1.8B (LLM portion) |

## Throughput Summary

| Workers | Output tok/s | Total tok/s | Req/s | Mean TTFT (ms) | Median TTFT (ms) | Mean TPOT (ms) | Median TPOT (ms) |
|---------|-------------|-------------|-------|--------------|----------------|--------------|----------------|
| 32 | 1287 / 1139 | 1930 / 1655 | 1.91 / 1.60 | 597 / 615 | 521 / 539 | 20.4 / 26.9 | 18.7 / 25.2 |
| 64 | 1953 / 1489 | 2864 / 2164 | 2.80 / 2.12 | 733 / 833 | 591 / 616 | 39.9 / 39.7 | 24.5 / 38.0 |
| 128 | 2677 / 1319 | 3867 / 1945 | 3.57 / 1.93 | 1258 / 28783 | 717 / 31856 | 222.8 / 42.3 | 35.6 / 40.8 |
| 192 | 2541 / 1350 | 3839 / 1968 | 3.91 / 1.92 | 2302 / 53379 | 1150 / 61424 | 684.1 / 42.8 | 51.9 / 40.6 |

(Format: VKWR / vLLM)

## Detailed Comparison

### Workers = 32

| Metric | VKWR | vLLM |
|--------|------|------|
| Completed / Failed | 595 / 0 | 499 / 0 |
| Total input tokens | 200,310 | 160,785 |
| Total output tokens | 400,643 | 355,061 |
| Peak decode tok/s | 3632 | 4052 |
| Peak concurrent | 49 | 53 |

**Throughput**

| Metric | VKWR | vLLM | Ratio (VKWR/vLLM) |
|--------|------|------|-------------------|
| Output tok/s | 1286.9 | 1139.2 | +13.0% |
| Input tok/s | 643.4 | 515.9 | +24.7% |
| Total tok/s | 1930.3 | 1655.1 | +16.6% |
| Req/s | 1.9 | 1.6 | +19.4% |

**Latency**

| Metric | VKWR | vLLM | Ratio (lower better) |
|--------|------|------|----------------------|
| Mean TTFT | 596.9 ms | 615.3 ms | -3.0% (better) |
| Median TTFT | 521.1 ms | 539.4 ms | -3.4% (better) |
| Mean ITL | 18.7 ms | 25.0 ms | -25.2% (better) |
| Mean E2E latency | 15690.6 ms | 18481.1 ms | -15.1% (better) |
| Median E2E latency | 19400.2 ms | 22106.3 ms | -12.2% (better) |

**Per-Request Speed**

| Metric | VKWR | vLLM | Ratio (higher better) |
|--------|------|------|-----------------------|
| Mean prefill (tok/s) | 588.7 | 530.3 | +11.0% |
| Median prefill (tok/s) | 268.1 | 207.6 | +29.1% |
| Mean decode (tok/s) | 52.8 | 39.1 | +35.1% |
| Median decode (tok/s) | 53.4 | 39.8 | +34.2% |

**TPOT (ms/token, lower better)**

| Metric | VKWR | vLLM | Ratio (lower better) |
|--------|------|------|----------------------|
| Mean TPOT | 20.4 | 26.9 | -24.4% (better) |
| Median TPOT | 18.7 | 25.2 | -25.5% (better) |

### Workers = 64

| Metric | VKWR | vLLM |
|--------|------|------|
| Completed / Failed | 874 / 1 | 660 / 1 |
| Total input tokens | 284,658 | 210,211 |
| Total output tokens | 609,971 | 463,505 |
| Peak decode tok/s | 6865 | 4818 |
| Peak concurrent | 89 | 92 |

**Throughput**

| Metric | VKWR | vLLM | Ratio (VKWR/vLLM) |
|--------|------|------|-------------------|
| Output tok/s | 1952.7 | 1488.6 | +31.2% |
| Input tok/s | 911.3 | 675.1 | +35.0% |
| Total tok/s | 2864.0 | 2163.7 | +32.4% |
| Req/s | 2.8 | 2.1 | +32.0% |

**Latency**

| Metric | VKWR | vLLM | Ratio (lower better) |
|--------|------|------|----------------------|
| Mean TTFT | 733.0 ms | 832.8 ms | -12.0% (better) |
| Median TTFT | 590.8 ms | 616.3 ms | -4.1% (better) |
| Mean ITL | 24.7 ms | 37.9 ms | -34.9% (better) |
| Mean E2E latency | 21100.2 ms | 27486.9 ms | -23.2% (better) |
| Median E2E latency | 25293.6 ms | 32157.5 ms | -21.3% (better) |

**Per-Request Speed**

| Metric | VKWR | vLLM | Ratio (higher better) |
|--------|------|------|-----------------------|
| Mean prefill (tok/s) | 492.8 | 446.8 | +10.3% |
| Median prefill (tok/s) | 219.5 | 164.7 | +33.3% |
| Mean decode (tok/s) | 40.2 | 26.0 | +54.4% |
| Median decode (tok/s) | 40.9 | 26.3 | +55.2% |

**TPOT (ms/token, lower better)**

| Metric | VKWR | vLLM | Ratio (lower better) |
|--------|------|------|----------------------|
| Mean TPOT | 39.9 | 39.7 | +0.5% (worse) |
| Median TPOT | 24.5 | 38.0 | -35.6% (better) |

### Workers = 128

| Metric | VKWR | vLLM |
|--------|------|------|
| Completed / Failed | 1111 / 3 | 601 / 1 |
| Total input tokens | 370,075 | 195,355 |
| Total output tokens | 832,857 | 411,611 |
| Peak decode tok/s | 11001 | 3990 |
| Peak concurrent | 180 | 145 |

**Throughput**

| Metric | VKWR | vLLM | Ratio (VKWR/vLLM) |
|--------|------|------|-------------------|
| Output tok/s | 2677.3 | 1319.3 | +102.9% |
| Input tok/s | 1189.6 | 626.2 | +90.0% |
| Total tok/s | 3866.9 | 1945.5 | +98.8% |
| Req/s | 3.6 | 1.9 | +85.4% |

**Latency**

| Metric | VKWR | vLLM | Ratio (lower better) |
|--------|------|------|----------------------|
| Mean TTFT | 1258.2 ms | 28783.5 ms | -95.6% (better) |
| Median TTFT | 717.3 ms | 31856.4 ms | -97.7% (better) |
| Mean ITL | 35.8 ms | 40.7 ms | -12.1% (better) |
| Mean E2E latency | 32293.7 ms | 56750.6 ms | -43.1% (better) |
| Median E2E latency | 36568.3 ms | 59676.2 ms | -38.7% (better) |

**Per-Request Speed**

| Metric | VKWR | vLLM | Ratio (higher better) |
|--------|------|------|-----------------------|
| Mean prefill (tok/s) | 390.3 | 41.4 | +843.3% |
| Median prefill (tok/s) | 156.4 | 5.7 | +2645.6% |
| Mean decode (tok/s) | 27.5 | 24.2 | +13.8% |
| Median decode (tok/s) | 28.1 | 24.5 | +14.6% |

**TPOT (ms/token, lower better)**

| Metric | VKWR | vLLM | Ratio (lower better) |
|--------|------|------|----------------------|
| Mean TPOT | 222.8 | 42.3 | +427.2% (worse) |
| Median TPOT | 35.6 | 40.8 | -12.7% (better) |
> VKWR: Mean TPOT (223 ms) >> Median TPOT (36 ms), indicating a long-tail distribution.

### Workers = 192

| Metric | VKWR | vLLM |
|--------|------|------|
| Completed / Failed | 1218 / 11 | 599 / 0 |
| Total input tokens | 404,352 | 192,957 |
| Total output tokens | 791,179 | 421,301 |
| Peak decode tok/s | 12525 | 4556 |
| Peak concurrent | 241 | 214 |

**Throughput**

| Metric | VKWR | vLLM | Ratio (VKWR/vLLM) |
|--------|------|------|-------------------|
| Output tok/s | 2540.6 | 1350.0 | +88.2% |
| Input tok/s | 1298.5 | 618.3 | +110.0% |
| Total tok/s | 3839.1 | 1968.3 | +95.1% |
| Req/s | 3.9 | 1.9 | +103.8% |

**Latency**

| Metric | VKWR | vLLM | Ratio (lower better) |
|--------|------|------|----------------------|
| Mean TTFT | 2301.8 ms | 53379.3 ms | -95.7% (better) |
| Median TTFT | 1150.0 ms | 61423.5 ms | -98.1% (better) |
| Mean ITL | 52.4 ms | 40.5 ms | +29.6% (worse) |
| Mean E2E latency | 43451.3 ms | 81916.1 ms | -47.0% (better) |
| Median E2E latency | 52398.6 ms | 85469.6 ms | -38.7% (better) |

**Per-Request Speed**

| Metric | VKWR | vLLM | Ratio (higher better) |
|--------|------|------|-----------------------|
| Mean prefill (tok/s) | 273.2 | 30.4 | +799.9% |
| Median prefill (tok/s) | 95.4 | 3.6 | +2576.0% |
| Mean decode (tok/s) | 18.6 | 24.2 | -23.4% |
| Median decode (tok/s) | 19.3 | 24.7 | -21.8% |

**TPOT (ms/token, lower better)**

| Metric | VKWR | vLLM | Ratio (lower better) |
|--------|------|------|----------------------|
| Mean TPOT | 684.1 | 42.8 | +1499.7% (worse) |
| Median TPOT | 51.9 | 40.6 | +27.9% (worse) |
> VKWR: Mean TPOT (684 ms) >> Median TPOT (52 ms), indicating a long-tail distribution.

## Notes

Models are not directly comparable (RWKV7-1.5B vs Qwen3.5-2B ~1.8B LLM params), but both are in the 1.5-2B parameter range, making this a reasonable comparison for serving throughput on consumer hardware.
