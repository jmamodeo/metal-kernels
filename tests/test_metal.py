# SPDX-License-Identifier: Apache-2.0
"""Metal execution tests live next to the kernel they launch.

``test_conv3x3_metal.py`` compares conv3x3 with the NumPy reference.
The ``metal`` marker on those tests skips them when the host is not macOS.
"""
