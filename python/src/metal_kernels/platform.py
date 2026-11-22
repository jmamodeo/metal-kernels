# SPDX-License-Identifier: Apache-2.0
"""Host checks used by the test suite.

A usable GPU is left for the first kernel that actually launches. The only
signal today is the operating system.
"""

from __future__ import annotations

import sys


def host_is_macos() -> bool:
    """Return True when the host is macOS.

    Metal kernels are exercised only there. This does not probe a GPU.
    """
    return sys.platform == "darwin"
