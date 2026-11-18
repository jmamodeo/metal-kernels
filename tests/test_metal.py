# SPDX-License-Identifier: Apache-2.0
"""Metal execution tests.

The body below is the hook the first real kernel replaces. Off macOS the
``metal`` marker skips it during collection. On macOS it stays skipped until
a kernel exists.
"""

from __future__ import annotations

import pytest


@pytest.mark.metal
def test_kernel_matches_numpy_reference():
    pytest.skip("No Metal kernels are implemented yet")
