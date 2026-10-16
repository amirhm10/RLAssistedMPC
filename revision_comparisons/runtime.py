"""Worker-local numerical settings and provenance; never change the parent environment."""
from __future__ import annotations

import os
import random
import sys


def numerical_environment(threads):
    env = dict(os.environ)
    env.update(OPENBLAS_NUM_THREADS=str(threads), OMP_NUM_THREADS=str(threads),
               MKL_NUM_THREADS=str(threads), NUMEXPR_NUM_THREADS=str(threads),
               OPENBLAS_CORETYPE="NEHALEM", NPY_DISABLE_CPU_FEATURES="X86_V3,AVX2,FMA3",
               PYTHONFAULTHANDLER="1", PYTHONUNBUFFERED="1")
    return env


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metadata():
    import numpy as np
    import scipy
    import torch
    from threadpoolctl import threadpool_info
    return {"python": sys.version, "executable": sys.executable, "numpy": np.__version__,
            "scipy": scipy.__version__, "torch": torch.__version__,
            "torch_cpu_threads": torch.get_num_threads(), "cuda_available": torch.cuda.is_available(),
            "threadpools": threadpool_info(),
            "worker_environment": {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                "OPENBLAS_CORETYPE", "NPY_DISABLE_CPU_FEATURES", "PYTHONFAULTHANDLER")}}
