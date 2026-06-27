import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

logger = logging.getLogger(__name__)
ROOT_DIR = Path(__file__).parent.resolve()


def _find_ninja() -> Path | None:
    ninja = shutil.which("ninja")
    if ninja:
        return Path(ninja)

    scripts_dir = Path(sys.executable).parent
    for name in ("ninja", "ninja.exe"):
        candidate = scripts_dir / name
        if candidate.is_file():
            return candidate

    return None


def _is_ninja_available() -> bool:
    return _find_ninja() is not None


def _get_mem_available_kb() -> int | None:
    """Read MemAvailable from /proc/meminfo, return value in kB or None."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    return int(parts[1])
    except Exception as e:
        logger.warning("Failed to read /proc/meminfo: %s", e)
    return None


def _compute_num_jobs():
    """Determine number of compilation jobs.

    Limits applied in order:
    1. CPU count (os.sched_getaffinity or os.cpu_count)
    2. Memory limit from /proc/meminfo MemAvailable
    3. MAX_JOBS env var (user override, acts as upper cap)
    4. NVCC_THREADS adjustment
    """
    # Step 1: CPU limit
    try:
        cpu_count = len(os.sched_getaffinity(0))
    except AttributeError:
        cpu_count = os.cpu_count() or 1
    num_jobs = cpu_count

    # Step 2: Memory limit from /proc/meminfo
    job_mem_mb = int(os.environ.get("VKWR_JOB_MEMORY_MB", "4096"))
    mem_fraction = float(os.environ.get("VKWR_MEMORY_FRACTION", "0.7"))

    if os.environ.get("VKWR_MEMORY_LIMIT", "1") != "0":
        avail_mem_kb = _get_mem_available_kb()
        if avail_mem_kb is not None:
            avail_mem_mb = avail_mem_kb // 1024
            safe_mem_mb = int(avail_mem_mb * mem_fraction)
            mem_limited_jobs = max(1, safe_mem_mb // job_mem_mb)
            num_jobs = min(num_jobs, mem_limited_jobs)
            logger.info(
                "Memory-limited jobs: %d (MemAvailable=%d MB, safe=%.0f MB, per-job=%d MB)",
                mem_limited_jobs,
                avail_mem_mb,
                safe_mem_mb,
                job_mem_mb,
            )
        else:
            logger.info("Could not read MemAvailable, skipping memory-based job limit.")

    # Step 3: MAX_JOBS user override (upper cap)
    max_jobs = os.environ.get("MAX_JOBS")
    if max_jobs is not None:
        max_jobs = int(max_jobs)
        num_jobs = min(num_jobs, max_jobs)
        logger.info("Capped by MAX_JOBS=%d, final=%d.", max_jobs, num_jobs)

    # Step 4: NVCC_THREADS adjustment
    nvcc_threads = os.environ.get("NVCC_THREADS")
    if nvcc_threads is not None:
        nvcc_threads = int(nvcc_threads)
        logger.info("Using NVCC_THREADS=%d as the number of nvcc threads.", nvcc_threads)
        num_jobs = max(1, num_jobs // nvcc_threads)
    else:
        nvcc_threads = None

    logger.info("[vkwr] Using %d parallel compilation jobs.", num_jobs)
    return num_jobs, nvcc_threads


class CMakeBuild(build_ext):
    def run(self):
        self.build_cmake()

    def build_cmake(self):
        import torch

        build_temp = ROOT_DIR / "build"
        build_temp.mkdir(exist_ok=True)

        device = os.environ.get("VKWR_TARGET_DEVICE", "cuda")

        # Auto-detect CUDA architecture from the local GPU, similar to vLLM's approach.
        # If VKWR_CUDA_ARCH is set, use it explicitly; otherwise detect from GPU.
        cuda_arch = os.environ.get("VKWR_CUDA_ARCH")
        if cuda_arch is None and device == "cuda":
            try:
                major, minor = torch.cuda.get_device_capability()
                cuda_arch = f"{major}{minor}"
                print(f"[vkwr] Auto-detected CUDA architecture: {cuda_arch}")
            except Exception as e:
                raise RuntimeError(
                    f"Cannot auto-detect CUDA architecture. "
                    f"Set VKWR_CUDA_ARCH environment variable to override. "
                    f"Error: {e}"
                ) from e

        cfg = os.environ.get("CMAKE_BUILD_TYPE") or "Release"
        torch_dir = Path(torch.utils.cmake_prefix_path)

        cmake_args = [
            f"-DVKWR_TARGET_DEVICE={device}",
            f"-DVKWR_CUDA_ARCH={cuda_arch}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={ROOT_DIR / 'vkwr'}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DCMAKE_PYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_PREFIX_PATH={torch_dir}",
        ]

        num_jobs, _ = _compute_num_jobs()

        if sys.platform == "win32":
            build_tool = ["-G", "Visual Studio 17 2022", "-A", "x64"]
        elif _is_ninja_available():
            build_tool = ["-G", "Ninja"]
            cmake_args += [
                "-DCMAKE_JOB_POOL_COMPILE:STRING=compile",
                f"-DCMAKE_JOB_POOLS:STRING=compile={num_jobs}",
            ]
        else:
            build_tool = []

        other_cmake_args = os.environ.get("CMAKE_ARGS")
        if other_cmake_args:
            cmake_args += other_cmake_args.split()

        print("[vkwr] Running cmake configure...")
        final_args = [str(ROOT_DIR), *build_tool, *cmake_args]
        print(f"[vkwr] CMake args: {final_args}")
        subprocess.check_call(["cmake", *final_args], cwd=str(build_temp))

        print("[vkwr] Running cmake build...")
        build_cmd = ["cmake", "--build", str(build_temp), f"-j={num_jobs}"]
        if not _is_ninja_available():
            build_cmd += ["--config", cfg]
            if sys.platform == "win32":
                build_cmd += ["--", "/maxcpucount"]
        subprocess.check_call(build_cmd, cwd=str(build_temp))


ext_modules = []
for name in [
    "_sampling_C",
    "_v1_wkv_fp16_C",
    "_v1_wkv_fp32_C",
    "_v1_linear_C",
    "_v1_norm_C",
    "_v1_mix_C",
    "_v1_rank_C",
    "_v1_5_wkv_fp16_C",
    "_v1_5_wkv_fp32_C",
    "_v1_5_mix_C",
    "_v1_5_norm_C",
]:
    ext_modules.append(Extension(f"vkwr.{name}", sources=[]))


setup(
    name="vkwr",
    version="0.1.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": CMakeBuild},
)
