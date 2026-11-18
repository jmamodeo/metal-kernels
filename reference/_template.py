# SPDX-License-Identifier: Apache-2.0
"""Template for a NumPy reference.

Copy this file to ``reference/<name>.py``. Keep the implementation in NumPy.
Do not import MLX or PyTorch. The pytest for the kernel compares Metal
outputs against the function you write here.
"""

from __future__ import annotations

import numpy as np


def kernel(x: np.ndarray) -> np.ndarray:
    """Reference for one kernel.

    Replace ``kernel`` with the real function name when copying this file.
    Check shapes and dtypes here so a bad input fails in the oracle.
    """
    raise NotImplementedError(
        "This is the reference template, not a kernel implementation. "
        f"Received shape {tuple(x.shape)} and dtype {x.dtype}."
    )
