"""torchcodec's CUDA build dlopens NVIDIA NPP (libnppicc.so.12 ...), which isn't on the loader
path; preload it from the nvidia-npp-cu12 wheel. Import before anything that loads torchcodec."""

import ctypes
import glob
import os

try:
    import nvidia.npp

    for lib in sorted(glob.glob(os.path.join(list(nvidia.npp.__path__)[0], "lib", "libnpp*.so*"))):
        ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
except ImportError:
    pass
