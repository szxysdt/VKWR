import torch


def log(message: str) -> None:
    print(f"[rwkv7_fast_v3a] {message}", flush=True)


def cuda_mem() -> str:
    if not torch.cuda.is_available():
        return "cuda=unavailable"
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    used = total - free
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    return f"gpu_mem used={used / 2**30:.2f}GiB allocated={allocated / 2**30:.2f}GiB reserved={reserved / 2**30:.2f}GiB total={total / 2**30:.2f}GiB"
