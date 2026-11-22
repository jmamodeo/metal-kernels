# SPDX-License-Identifier: Apache-2.0
"""FlashAttention hd32 checks.

The launch-geometry and source checks run everywhere. The ``metal`` tests
launch the kernel through MLX and PyTorch MPS and compare it to
``reference.sdpa.sdpa_reference`` in fp32, on inputs already rounded to bf16.
"""

from __future__ import annotations

import math
import subprocess
import sys

import numpy as np
import pytest
from metal_kernels import list_kernels, metal_source
from metal_kernels.flash_attention_hd32 import (
    KERNEL_NAME,
    THREADGROUP_SIZE,
    launch_geometry,
    resolve_scale,
)

from reference.sdpa import sdpa_reference

# bf16 output tolerance against the fp32 reference (MTL-11).
RTOL = 2e-2
ATOL = 1e-2

SAM3_ENCODER = (1, 8, 5184, 5184)
SAM3_CROSS = (1, 8, 201, 5184)

# (batch, heads, query_length, key_length)
SMALL_SHAPES = [
    (1, 1, 1, 1),
    (1, 2, 5, 7),
    (2, 3, 17, 33),
    (1, 8, 64, 64),
    (1, 2, 9, 300),
]


def test_kernel_is_registered(monkeypatch):
    monkeypatch.delenv("METAL_KERNELS_ROOT", raising=False)
    assert KERNEL_NAME in list_kernels()
    source = metal_source(KERNEL_NAME)
    assert "kernel void flash_attention_hd32(" in source
    assert "inline void flash_attention_hd32_row(" in source


def test_import_does_not_load_gpu_frameworks():
    script = (
        "import metal_kernels.flash_attention_hd32, sys; "
        "assert 'torch' not in sys.modules; "
        "assert 'mlx' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        ((1, 1, 1, 1), ((1, 1, 1), 32, 32)),
        ((1, 2, 5, 7), ((10, 5, 7), 320, 256)),
        (SAM3_ENCODER, ((41472, 5184, 5184), 41472 * 32, THREADGROUP_SIZE)),
        (SAM3_CROSS, ((1608, 201, 5184), 1608 * 32, THREADGROUP_SIZE)),
    ],
)
def test_launch_geometry(shape, expected):
    batch, heads, query_length, key_length = shape
    q_shape = (batch, heads, query_length, 32)
    kv_shape = (batch, heads, key_length, 32)
    dims, threads, threadgroup = launch_geometry(q_shape, kv_shape, kv_shape)
    assert (dims, threads, threadgroup) == expected
    assert threads % 32 == 0 and threadgroup % 32 == 0


def test_launch_geometry_rejects_bad_shapes():
    q = (1, 2, 4, 32)
    kv = (1, 2, 6, 32)
    with pytest.raises(ValueError, match="shape"):
        launch_geometry(q[:3], kv, kv)
    with pytest.raises(ValueError, match="head dimension 32"):
        launch_geometry((1, 2, 4, 64), (1, 2, 6, 64), (1, 2, 6, 64))
    with pytest.raises(ValueError, match="head dimension 32"):
        launch_geometry(q, kv, (1, 2, 6, 16))
    with pytest.raises(ValueError, match="batch and head"):
        launch_geometry(q, (1, 3, 6, 32), (1, 3, 6, 32))
    with pytest.raises(ValueError, match="k and v sequence lengths"):
        launch_geometry(q, kv, (1, 2, 5, 32))
    with pytest.raises(ValueError, match="positive"):
        launch_geometry((1, 2, 0, 32), kv, kv)
    with pytest.raises(ValueError, match="positive"):
        launch_geometry(q, (1, 2, 0, 32), (1, 2, 0, 32))
    with pytest.raises(ValueError, match="at most"):
        launch_geometry((1, 1, 2**26, 32), (1, 1, 1, 32), (1, 1, 1, 32))


def test_resolve_scale():
    assert resolve_scale(None) == pytest.approx(1 / math.sqrt(32))
    assert resolve_scale(0.25) == 0.25
    with pytest.raises(ValueError, match="finite"):
        resolve_scale(float("nan"))


# Metal launches -----------------------------------------------------------


def _random_inputs(shape, seed):
    batch, heads, query_length, key_length = shape
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((batch, heads, query_length, 32), dtype=np.float32)
    k = rng.standard_normal((batch, heads, key_length, 32), dtype=np.float32)
    v = rng.standard_normal((batch, heads, key_length, 32), dtype=np.float32)
    return q, k, v


def _reference(q, k, v, scale=None):
    # One head at a time keeps the score matrix of the SAM 3 shape small.
    out = np.empty_like(q)
    for b in range(q.shape[0]):
        for h in range(q.shape[1]):
            sl = (slice(b, b + 1), slice(h, h + 1))
            out[sl] = sdpa_reference(q[sl], k[sl], v[sl], scale=scale)
    return out


def _mlx():
    mx = pytest.importorskip("mlx.core")
    from metal_kernels.flash_attention_hd32 import flash_attention_hd32_mlx

    return mx, flash_attention_hd32_mlx


def _torch_mps():
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("PyTorch MPS is not available")
    from metal_kernels.flash_attention_hd32 import flash_attention_hd32_mps

    return torch, flash_attention_hd32_mps


def _run_mlx(q, k, v, scale=None):
    mx, launch = _mlx()
    q_mx, k_mx, v_mx = (mx.array(x).astype(mx.bfloat16) for x in (q, k, v))
    out = launch(q_mx, k_mx, v_mx, scale=scale)
    mx.eval(out)
    assert out.dtype == mx.bfloat16
    assert tuple(out.shape) == q.shape
    rounded = [np.array(x.astype(mx.float32)) for x in (q_mx, k_mx, v_mx)]
    return np.array(out.astype(mx.float32)), rounded


def _run_mps(q, k, v, scale=None):
    torch, launch = _torch_mps()
    tensors = [torch.from_numpy(x).to("mps", torch.bfloat16) for x in (q, k, v)]
    out = launch(*tensors, scale=scale)
    torch.mps.synchronize()
    assert out.dtype == torch.bfloat16
    assert tuple(out.shape) == q.shape
    rounded = [x.float().cpu().numpy() for x in tensors]
    return out.float().cpu().numpy(), rounded


def _mps_sdpa_error(rounded, expected):
    """Max abs error of PyTorch MPS bf16 SDPA against the fp32 reference.

    ``rounded`` holds q, k, v already rounded to bf16, so the cast is exact.
    """
    torch, _ = _torch_mps()
    import torch.nn.functional as functional

    tensors = [torch.from_numpy(x).to("mps", torch.bfloat16) for x in rounded]
    out = functional.scaled_dot_product_attention(*tensors)
    return float(np.max(np.abs(out.float().cpu().numpy() - expected)))


_HOSTS = [pytest.param(_run_mlx, id="mlx"), pytest.param(_run_mps, id="mps")]


@pytest.mark.metal
@pytest.mark.parametrize("run", _HOSTS)
@pytest.mark.parametrize("shape", SMALL_SHAPES)
def test_matches_reference_small(run, shape):
    q, k, v = _random_inputs(shape, seed=sum(shape))
    actual, rounded = run(q, k, v)
    expected = _reference(*rounded)
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.metal
@pytest.mark.parametrize("run", _HOSTS)
def test_matches_reference_with_explicit_scale(run):
    q, k, v = _random_inputs((1, 2, 16, 40), seed=3)
    actual, rounded = run(q, k, v, scale=0.3)
    expected = _reference(*rounded, scale=0.3)
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.metal
@pytest.mark.parametrize("run", _HOSTS)
def test_large_scores_stay_finite(run):
    # Scores far outside exp range: the online softmax must not overflow.
    q, k, v = _random_inputs((1, 2, 8, 50), seed=4)
    q *= 16
    k *= 16
    actual, rounded = run(q, k, v)
    expected = _reference(*rounded)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.metal
@pytest.mark.parametrize("run", _HOSTS)
def test_single_key_returns_the_value(run):
    q, k, v = _random_inputs((1, 2, 3, 1), seed=5)
    actual, rounded = run(q, k, v)
    np.testing.assert_array_equal(actual, np.broadcast_to(rounded[2], q.shape))


@pytest.mark.metal
@pytest.mark.parametrize("run", _HOSTS)
@pytest.mark.parametrize("shape", [SAM3_ENCODER, SAM3_CROSS], ids=["encoder", "cross"])
def test_sam3_shapes_match_reference_and_mps_sdpa(run, shape):
    q, k, v = _random_inputs(shape, seed=11)
    actual, rounded = run(q, k, v)
    expected = _reference(*rounded)
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)
    ours = float(np.max(np.abs(actual - expected)))
    assert ours <= _mps_sdpa_error(rounded, expected)
