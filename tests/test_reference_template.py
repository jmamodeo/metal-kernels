# SPDX-License-Identifier: Apache-2.0
"""The reference template is present and is not an implementation."""

from __future__ import annotations

import pytest


def test_reference_template_is_unfinished():
    numpy = pytest.importorskip("numpy")
    from reference._template import kernel

    with pytest.raises(NotImplementedError, match="not a kernel implementation"):
        kernel(numpy.zeros((2, 3), dtype=numpy.float32))
