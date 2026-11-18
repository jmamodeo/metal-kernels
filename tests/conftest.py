# SPDX-License-Identifier: Apache-2.0
"""Skip Metal execution tests on hosts that are not macOS."""

from __future__ import annotations

import pytest

from metal_kernels.platform import host_is_macos


def pytest_collection_modifyitems(config, items):
    if host_is_macos():
        return
    skip = pytest.mark.skip(reason="Metal kernels run only on macOS")
    for item in items:
        if "metal" in item.keywords:
            item.add_marker(skip)
