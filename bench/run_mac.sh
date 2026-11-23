#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Test, benchmark and record one FlashAttention hd32 version on Apple silicon.
#
#   bench/run_mac.sh <version> [design]
#   bench/run_mac.sh swift
#
# `swift` checks the Swift host instead: swift build, swift test (the GPU
# correctness run), then the release-built timing mode, FlashAttentionHD32Bench,
# at [1, 8, 5184, 32] in bf16 and fp32. It prints and records nothing else.
#
# Installs the Mac extra, runs the metal tests, then runs
# bench/flash_attention_hd32.py with --record. The record goes to
# bench/results/flash_attention_hd32.jsonl through bench/record.py. The
# benchmark interleaves the frozen v6 baseline, the candidate and the built-in
# on every iteration, in a balanced rotating order (102 timed plus 12 warmup by
# default), and prints the max and mean abs error against sdpa_reference,
# median/min/p10/p90 ms per call for all six variants, the speed gate (paired
# bootstrap 95% CI of the median candidate/v6 ratio entirely below 1), the
# reported candidate/MPS SDPA ratio CI, and the git SHA. Nothing is committed.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 || -z "$1" ]]; then
    echo "usage: bench/run_mac.sh <version> [design] | bench/run_mac.sh swift" >&2
    exit 2
fi
version="$1"

if [[ "$version" == "swift" ]]; then
    if [[ $# -ne 1 ]]; then
        echo "usage: bench/run_mac.sh swift" >&2
        exit 2
    fi
    cd "$(dirname "${BASH_SOURCE[0]}")/.."
    echo "git SHA: $(git rev-parse HEAD)"
    swift build
    swift test
    swift run -c release FlashAttentionHD32Bench
    echo "git SHA: $(git rev-parse HEAD)"
    exit 0
fi

case "$version" in
    v0)
        default_design="per-row online softmax: one simdgroup per query row, one lane per head-dim column, simd_sum dot products, fp32 math, no threadgroup memory"
        ;;
    v1)
        default_design="tiled simdgroup_matrix: one threadgroup of 4 simdgroups per 32 query rows, 8 rows per simdgroup, 64-key K/V blocks loaded per lane from device into registers as vectors (no threadgroup memory, no barriers), bf16 simdgroup_matrix inputs with fp32 S and O accumulators (P stays fp32), exp2 online softmax with shuffle-xor row reductions"
        ;;
    v6)
        default_design="v1 tiling (32 query rows, 4 simdgroups x 8 rows, 64-key blocks, bf16 simdgroup_matrix inputs, fp32 P and accumulators, always-applied exp2 correction) with each K/V block loaded once by all 128 threads into threadgroup memory (8 KB for bf16, 16 KB for fp32) shared by the 4 simdgroups, zero-filled past key_length, barrier after the load and before the next block"
        ;;
    v7)
        default_design="v6 with the K/V stage double buffered in threadgroup memory (16 KB for bf16, 32 KB for fp32): the next 64-key block's device loads are issued into registers before the current block's math and stored into the other buffer after it, one barrier per block instead of two"
        ;;
    *)
        default_design=""
        ;;
esac
design="${2:-$default_design}"
if [[ -z "$design" ]]; then
    echo "bench/run_mac.sh: no default design for $version; pass it as the second argument" >&2
    exit 2
fi

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ -x .venv/bin/python ]]; then
    python=.venv/bin/python
else
    python=python3
fi

echo "git SHA: $(git rev-parse HEAD)"
"$python" -m pip install -e "./python[mac]"
"$python" -m pytest -m metal
"$python" bench/flash_attention_hd32.py --record --version "$version" --design "$design"
echo "git SHA: $(git rev-parse HEAD)"
