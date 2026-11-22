#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Test, benchmark and record one FlashAttention hd32 version on Apple silicon.
#
#   bench/run_mac.sh <version> [design]
#
# Installs the Mac extra, runs the metal tests, then runs
# bench/flash_attention_hd32.py with --record. The record goes to
# bench/results/flash_attention_hd32.jsonl through bench/record.py. Prints the
# max and mean abs error against sdpa_reference and ms/call for all four
# variants, and the git SHA. Nothing is committed.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 || -z "$1" ]]; then
    echo "usage: bench/run_mac.sh <version> [design]" >&2
    exit 2
fi
version="$1"

case "$version" in
    v0)
        default_design="per-row online softmax: one simdgroup per query row, one lane per head-dim column, simd_sum dot products, fp32 math, no threadgroup memory"
        ;;
    v1)
        default_design="tiled simdgroup_matrix: one threadgroup of 2 simdgroups per 16 query rows, 8 rows per simdgroup, 64-key K/V blocks staged as fp32 in threadgroup memory (16 KB), fp32 simdgroup_float8x8 S=QK^T and O+=PV, exp2 online softmax with shuffle-xor row reductions"
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
