# SPDX-License-Identifier: Apache-2.0
"""FlashAttention hd32 checks.

The shape, bias-layout and backend checks run everywhere. The ``metal`` tests
launch the kernel through MLX and PyTorch MPS and compare it to
``reference.sdpa.sdpa_reference`` in fp32. bf16 cases use inputs already
rounded to bf16.
"""

from __future__ import annotations

import math
import subprocess
import sys

import numpy as np
import pytest
from metal_kernels import list_kernels, metal_source
from metal_kernels.flash_attention import (
    KERNEL_NAME,
    QUERY_BLOCK,
    THREADGROUP_MEMORY_BYTES,
    THREADGROUP_SIZE,
    bias_strides,
    flash_attention_hd32,
    launch_geometry,
    resolve_scale,
)

from reference.sdpa import sdpa_reference

# Output tolerance against the fp32 reference, per input dtype (MTL-15).
TOLERANCE = {"bfloat16": (2e-2, 1e-2), "float32": (1e-3, 1e-4)}

SAM3_ENCODER = (1, 8, 5184, 5184)
SAM3_CROSS = (1, 8, 201, 5184)

# (batch, heads, query_length, key_length). 100 and 1000 leave a key tail
# that is not a multiple of 64.
SMALL_SHAPES = [
    (1, 1, 1, 1),
    (1, 2, 5, 7),
    (2, 3, 17, 33),
    (1, 8, 64, 64),
    (1, 2, 9, 100),
    (1, 2, 9, 1000),
]


def test_kernel_is_registered(monkeypatch):
    monkeypatch.delenv("METAL_KERNELS_ROOT", raising=False)
    assert KERNEL_NAME in list_kernels()
    source = metal_source(KERNEL_NAME)
    assert '[[host_name("flash_attention_hd32_" #NAME)]]' in source
    assert "FA_HD32_INSTANTIATE(bfloat, ushort)" in source
    assert "FA_HD32_INSTANTIATE(float, float)" in source
    assert "inline void flash_attention_hd32_block(" in source
    assert "constant constexpr uint FA_HD32_BLOCK_Q = 16;" in source
    assert "constant constexpr uint FA_HD32_BLOCK_K = 64;" in source


def test_launch_constants_match_the_shader():
    # 16 query rows per threadgroup, 8 per simdgroup: two simdgroups.
    assert (QUERY_BLOCK, THREADGROUP_SIZE) == (16, 64)
    assert THREADGROUP_MEMORY_BYTES == 2 * 64 * 32 * 4


def test_import_does_not_load_gpu_frameworks():
    script = (
        "import metal_kernels.flash_attention, sys; "
        "assert 'torch' not in sys.modules; "
        "assert 'mlx' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


@pytest.mark.parametrize("backend", ["cuda", "MLX", "", None])
def test_rejects_invalid_backend(backend):
    with pytest.raises(ValueError, match="backend must be one of"):
        flash_attention_hd32(None, None, None, backend=backend)


def test_backend_is_keyword_only():
    with pytest.raises(TypeError):
        flash_attention_hd32(None, None, None, None, None, "mlx")


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        ((1, 1, 1, 1), ((1, 1, 1, 1), 64, 64)),
        ((1, 2, 5, 7), ((10, 2, 5, 7), 128, 64)),
        ((2, 3, 17, 33), ((102, 3, 17, 33), 6 * 2 * 64, 64)),
        (SAM3_ENCODER, ((41472, 8, 5184, 5184), 8 * 324 * 64, THREADGROUP_SIZE)),
        (SAM3_CROSS, ((1608, 8, 201, 5184), 8 * 13 * 64, THREADGROUP_SIZE)),
    ],
)
def test_launch_geometry(shape, expected):
    batch, heads, query_length, key_length = shape
    q_shape = (batch, heads, query_length, 32)
    kv_shape = (batch, heads, key_length, 32)
    dims, threads, threadgroup = launch_geometry(q_shape, kv_shape, kv_shape)
    assert (dims, threads, threadgroup) == expected
    assert threads % threadgroup == 0 and threadgroup % 32 == 0


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
    # One query per head still launches a full 64-thread group.
    with pytest.raises(ValueError, match="at most"):
        launch_geometry((2**25, 1, 1, 32), (2**25, 1, 1, 32), (2**25, 1, 1, 32))


@pytest.mark.parametrize(
    ("bias_shape", "expected"),
    [
        ((2, 3, 4, 5), ((2, 3, 4, 5), (60, 20, 5, 1))),
        ((4, 5), ((1, 1, 4, 5), (0, 0, 5, 1))),
        ((5,), ((1, 1, 1, 5), (0, 0, 0, 1))),
        ((), ((1, 1, 1, 1), (0, 0, 0, 0))),
        ((2, 1, 4, 1), ((2, 1, 4, 1), (4, 0, 1, 0))),
        ((1, 3, 1, 5), ((1, 3, 1, 5), (0, 5, 0, 1))),
    ],
)
def test_bias_strides(bias_shape, expected):
    assert bias_strides(bias_shape, (2, 3, 4, 5)) == expected


def test_bias_strides_with_unit_target_dimensions():
    assert bias_strides((1, 1, 1, 1), (1, 1, 1, 1)) == ((1, 1, 1, 1), (0, 0, 0, 0))


@pytest.mark.parametrize(
    "bias_shape",
    [(2, 3, 4, 6), (3, 4, 4), (2, 3, 1, 5, 1), (1, 2, 3, 4, 5), (7,), (2, 2, 4, 5)],
)
def test_bias_strides_rejects_non_broadcastable(bias_shape):
    with pytest.raises(ValueError, match="broadcastable"):
        bias_strides(bias_shape, (2, 3, 4, 5))


def test_resolve_scale():
    assert resolve_scale(None) == pytest.approx(1 / math.sqrt(32))
    assert resolve_scale(0.25) == 0.25
    with pytest.raises(ValueError, match="finite"):
        resolve_scale(float("nan"))


# Metal launches -----------------------------------------------------------


def _random_inputs(shape, seed, bias_shape=None):
    batch, heads, query_length, key_length = shape
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((batch, heads, query_length, 32), dtype=np.float32)
    k = rng.standard_normal((batch, heads, key_length, 32), dtype=np.float32)
    v = rng.standard_normal((batch, heads, key_length, 32), dtype=np.float32)
    bias = None
    if bias_shape is not None:
        bias = np.asarray(3 * rng.standard_normal(bias_shape, dtype=np.float32))
    return q, k, v, bias


def _reference(q, k, v, bias=None, scale=None):
    # One head at a time keeps the score matrix of the SAM 3 shape small.
    out = np.empty_like(q)
    full_bias = None
    if bias is not None:
        target = (*q.shape[:3], k.shape[2])
        full_bias = np.broadcast_to(bias.reshape((1,) * (4 - bias.ndim) + bias.shape), target)
    for b in range(q.shape[0]):
        for h in range(q.shape[1]):
            sl = (slice(b, b + 1), slice(h, h + 1))
            head_bias = None if full_bias is None else np.ascontiguousarray(full_bias[sl])
            out[sl] = sdpa_reference(q[sl], k[sl], v[sl], head_bias, scale=scale)
    return out


def _run_mlx(arrays, dtype, scale=None):
    mx = pytest.importorskip("mlx.core")
    mx_dtype = getattr(mx, dtype)
    inputs = [None if x is None else mx.array(x).astype(mx_dtype) for x in arrays]
    out = flash_attention_hd32(*inputs, scale=scale, backend="mlx")
    mx.eval(out)
    assert out.dtype == mx_dtype
    assert tuple(out.shape) == arrays[0].shape
    rounded = [None if x is None else np.array(x.astype(mx.float32)) for x in inputs]
    return np.array(out.astype(mx.float32)), rounded


def _run_mps(arrays, dtype, scale=None):
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("PyTorch MPS is not available")
    torch_dtype = getattr(torch, dtype)
    inputs = [None if x is None else torch.from_numpy(x).to("mps", torch_dtype) for x in arrays]
    out = flash_attention_hd32(*inputs, scale=scale, backend="mps")
    torch.mps.synchronize()
    assert out.dtype == torch_dtype
    assert tuple(out.shape) == arrays[0].shape
    rounded = [None if x is None else x.float().cpu().numpy() for x in inputs]
    return out.float().cpu().numpy(), rounded


def _mps_sdpa_error(rounded, expected):
    """Max abs error of PyTorch MPS bf16 SDPA against the fp32 reference.

    ``rounded`` holds q, k, v already rounded to bf16, so the cast is exact.
    """
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("PyTorch MPS is not available")
    import torch.nn.functional as functional

    tensors = [torch.from_numpy(x).to("mps", torch.bfloat16) for x in rounded]
    out = functional.scaled_dot_product_attention(*tensors)
    return float(np.max(np.abs(out.float().cpu().numpy() - expected)))


def _check(run, arrays, dtype, scale=None):
    actual, rounded = run(arrays, dtype, scale=scale)
    expected = _reference(*rounded, scale=scale)
    rtol, atol = TOLERANCE[dtype]
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
    return actual, rounded, expected


_HOSTS = [pytest.param(_run_mlx, id="mlx"), pytest.param(_run_mps, id="mps")]
_DTYPES = ["bfloat16", "float32"]


@pytest.mark.metal
@pytest.mark.parametrize("run", _HOSTS)
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("shape", SMALL_SHAPES)
def test_matches_reference_small(run, dtype, shape):
    q, k, v, _ = _random_inputs(shape, seed=sum(shape))
    _check(run, [q, k, v, None], dtype)


@pytest.mark.metal
@pytest.mark.parametrize("run", _HOSTS)
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("key_length", [7, 100, 1000])
@pytest.mark.parametrize(
    "bias_shape",
    ["full", (1, 1, 1, -1), (-2, -1), (2, 1, -2, 1), ()],
    ids=["full", "keys", "query-key", "batch-query", "scalar"],
)
def test_matches_reference_with_bias(run, dtype, key_length, bias_shape):
    shape = (2, 3, 9, key_length)
    if bias_shape == "full":
        bias_shape = shape
    bias_shape = tuple({-1: key_length, -2: 9}.get(size, size) for size in bias_shape)
    q, k, v, bias = _random_inputs(shape, seed=key_length, bias_shape=bias_shape)
    if bias.size > 2:
        bias.flat[::3] = -np.inf
    _check(run, [q, k, v, bias], dtype)


@pytest.mark.metal
@pytest.mark.parametrize("run", _HOSTS)
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("key_length", [3, 100, 1000])
def test_fully_masked_rows_are_zero(run, dtype, key_length):
    q, k, v, bias = _random_inputs((1, 2, 6, key_length), seed=7, bias_shape=(1, 2, 6, key_length))
    bias[0, 0, 1, :] = -np.inf
    bias[0, 1, 5, :] = -np.inf
    bias[0, 1, 2, : key_length // 2] = -np.inf
    actual, _, _ = _check(run, [q, k, v, bias], dtype)
    np.testing.assert_array_equal(actual[0, 0, 1], 0)
    np.testing.assert_array_equal(actual[0, 1, 5], 0)
    assert np.any(actual[0, 1, 2] != 0)


@pytest.mark.metal
@pytest.mark.parametrize("run", _HOSTS)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_finite_minimum_bias_is_not_a_mask(run, dtype):
    # finfo.min is a common mask value. It is finite, so a row masked with it
    # everywhere averages v, as the reference and the built-ins do.
    q, k, v, bias = _random_inputs((1, 2, 6, 100), seed=8, bias_shape=(1, 2, 6, 100))
    minimum = float(np.finfo(np.float32).min) if dtype == "float32" else -3.3895313892515355e38
    bias[0, 0, 1, :] = minimum
    bias[0, 1, 2, :50] = minimum
    actual, rounded, _ = _check(run, [q, k, v, bias], dtype)
    rtol, atol = TOLERANCE[dtype]
    np.testing.assert_allclose(actual[0, 0, 1], rounded[2][0, 0].mean(axis=0), rtol=rtol, atol=atol)


@pytest.mark.metal
@pytest.mark.parametrize("run", _HOSTS)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_matches_reference_with_explicit_scale(run, dtype):
    q, k, v, bias = _random_inputs((1, 2, 16, 40), seed=3, bias_shape=(16, 40))
    _check(run, [q, k, v, bias], dtype, scale=0.3)


@pytest.mark.metal
@pytest.mark.parametrize("run", _HOSTS)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_large_scores_stay_finite(run, dtype):
    # Scores far outside exp range: the online softmax must not overflow.
    q, k, v, _ = _random_inputs((1, 2, 8, 50), seed=4)
    _check(run, [16 * q, 16 * k, v, None], dtype)


@pytest.mark.metal
@pytest.mark.parametrize("run", _HOSTS)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_single_key_returns_the_value(run, dtype):
    q, k, v, _ = _random_inputs((1, 2, 3, 1), seed=5)
    actual, rounded = run([q, k, v, None], dtype)
    np.testing.assert_array_equal(actual, np.broadcast_to(rounded[2], q.shape))


@pytest.mark.metal
@pytest.mark.parametrize("run", _HOSTS)
@pytest.mark.parametrize("shape", [SAM3_ENCODER, SAM3_CROSS], ids=["encoder", "cross"])
def test_sam3_shapes_match_reference_and_mps_sdpa(run, shape):
    q, k, v, _ = _random_inputs(shape, seed=11)
    actual, rounded, expected = _check(run, [q, k, v, None], "bfloat16")
    ours = float(np.max(np.abs(actual - expected)))
    assert ours <= _mps_sdpa_error(rounded[:3], expected)


@pytest.mark.metal
@pytest.mark.parametrize("run", _HOSTS)
def test_sam3_cross_attention_with_bias(run):
    q, k, v, bias = _random_inputs(SAM3_CROSS, seed=12, bias_shape=SAM3_CROSS)
    _check(run, [q, k, v, bias], "bfloat16")
