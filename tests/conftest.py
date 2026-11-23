# SPDX-License-Identifier: Apache-2.0
"""Skip Metal execution tests when no Metal device is available."""

from __future__ import annotations

import pytest
from metal_kernels.platform import metal_device_available


def pytest_collection_modifyitems(config, items):
    if metal_device_available():
        return
    skip = pytest.mark.skip(reason="no Metal device is available")
    for item in items:
        if "metal" in item.keywords:
            item.add_marker(skip)
