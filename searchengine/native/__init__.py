"""ctypes loader for the native query kernels. Auto-compiles the dylib with
clang -O2 if missing or stale (zero build-system dependencies)."""
import ctypes
import os
import subprocess

_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_DIR, "maxscore.c")
_LIB = os.path.join(_DIR, "libmaxscore.dylib")


def _ensure_built() -> None:
    if (not os.path.exists(_LIB)
            or os.path.getmtime(_LIB) < os.path.getmtime(_SRC)):
        subprocess.run(
            ["clang", "-O2", "-shared", "-o", _LIB, _SRC], check=True)


_ensure_built()
lib = ctypes.CDLL(_LIB)
lib.maxscore_query.restype = ctypes.c_int64
lib.maxscore_query.argtypes = [
    ctypes.POINTER(ctypes.c_void_p),   # docids ptrs
    ctypes.POINTER(ctypes.c_void_p),   # impacts ptrs
    ctypes.POINTER(ctypes.c_int64),    # lens
    ctypes.POINTER(ctypes.c_float),    # max impacts
    ctypes.c_int32, ctypes.c_int32,    # nterms, k
    ctypes.POINTER(ctypes.c_uint32),   # out ids
    ctypes.POINTER(ctypes.c_float),    # out scores
    ctypes.POINTER(ctypes.c_int64),    # stats[2]
]
