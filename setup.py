import os
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

ROOT_DIR = Path(__file__).parent.resolve()


def _find_ninja() -> Path | None:
    """Find the ninja executable.

    Checks in order:
    1. PATH via shutil.which
    2. Same directory as sys.executable (venv Scripts/ or bin/)
    """
    ninja = shutil.which("ninja")
    if ninja:
        return Path(ninja)

    # Fallback: check the Scripts/ (Windows) or bin/ (Unix) directory
    # where pip-installed executables live.
    scripts_dir = Path(sys.executable).parent
    for name in ("ninja", "ninja.exe"):
        candidate = scripts_dir / name
        if candidate.is_file():
            return candidate

    return None


def _is_ninja_available() -> bool:
    return _find_ninja() is not None


class CMakeBuild(build_ext):
    def run(self):
        self.build_cmake()

    def build_cmake(self):
        import torch

        build_temp = ROOT_DIR / "build"
        build_temp.mkdir(exist_ok=True)

        device = os.environ.get("VKWR_TARGET_DEVICE", "cuda")
        cuda_arch = os.environ.get("VKWR_CUDA_ARCH", "89")

        cfg = "Release"
        torch_dir = Path(torch.utils.cmake_prefix_path)

        cmake_args = [
            str(ROOT_DIR),
            f"-DVKWR_TARGET_DEVICE={device}",
            f"-DVKWR_CUDA_ARCH={cuda_arch}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={ROOT_DIR / 'vkwr'}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DCMAKE_PYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_PREFIX_PATH={torch_dir}",
        ]

        # setup.py 修改第 71-77 行
        if sys.platform == "win32":
            build_tool = ["-G", "Visual Studio 17 2022", "-A", "x64"]
        elif _is_ninja_available():
            build_tool = ["-G", "Ninja"]
        else:
            build_tool = []

        print("[vkwr] Running cmake configure...")
        final_args = [str(ROOT_DIR), *build_tool, *cmake_args]
        print(f"[vkwr] CMake args: {final_args}")
        subprocess.check_call(["cmake", *final_args], cwd=str(build_temp))

        print("[vkwr] Running cmake build...")
        build_cmd = ["cmake", "--build", str(build_temp)]
        if not _is_ninja_available():
            build_cmd += ["--config", cfg]
            if sys.platform == "win32":
                build_cmd += ["--", "/maxcpucount"]
        subprocess.check_call(build_cmd, cwd=str(build_temp))


ext_modules = []
for name in ["_rwkv_C", "_state_C", "_sampling_C"]:
    ext_modules.append(Extension(f"vkwr.{name}", sources=[]))


setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": CMakeBuild},
)
