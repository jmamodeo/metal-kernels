# SPDX-License-Identifier: Apache-2.0
"""Host checks used by the test suite."""

from __future__ import annotations

import ctypes
import sys

_METAL_FRAMEWORK = "/System/Library/Frameworks/Metal.framework/Metal"


def host_is_macos() -> bool:
    """Return True when the host is macOS.

    Metal kernels are exercised only there. This does not probe a GPU.
    """
    return sys.platform == "darwin"


def metal_device_available() -> bool:
    """Return True when a Metal default device can be created.

    False on non-macOS hosts and on macOS without a Metal GPU, such as
    some hosted CI VMs. This is a device probe, not a performance check.
    """
    if not host_is_macos():
        return False
    try:
        metal = ctypes.cdll.LoadLibrary(_METAL_FRAMEWORK)
        metal.MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
        metal.MTLCreateSystemDefaultDevice.argtypes = []
        return bool(metal.MTLCreateSystemDefaultDevice())
    except OSError:
        return False
