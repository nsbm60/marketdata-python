"""C-extension build for the vendored "Let's Be Rational" solver.

Project metadata lives in pyproject.toml; this file exists only to declare the native extension, which
setuptools cannot yet take from pyproject. The editable reinstall in deploy/deploy-python-remote.sh
(`pip install -e`) runs build_ext, so the extension is compiled on drogon at deploy time. It needs a
C++ compiler there (build-essential / g++).

The extension is a plain shared object over Jäckel's reference C++ (greeks/native/lbr/, his licence
preserved). It has no PyInit — greeks.solver._native loads it by file path with ctypes and calls the
one C entry point, lbr_implied_vol_batch. Compiled with -O2 as specified.
"""

from setuptools import Extension, setup

_LBR = "greeks/native/lbr"

lbr_ext = Extension(
    name="greeks.solver._lbr",
    sources=[
        f"{_LBR}/erf_cody.cpp",
        f"{_LBR}/lets_be_rational.cpp",
        f"{_LBR}/normaldistribution.cpp",
        f"{_LBR}/rationalcubic.cpp",
        f"{_LBR}/lbr_batch.cpp",
    ],
    include_dirs=[_LBR],
    language="c++",
    # -DNO_XL_API drops Jäckel's Excel-XLL integration (XLFunctions.h etc.), leaving the pure math.
    extra_compile_args=["-O2", "-std=c++17", "-DNDEBUG", "-DNO_XL_API"],
)

setup(ext_modules=[lbr_ext])
