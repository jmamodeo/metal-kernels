#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Compile every .metal file in the repository with the macOS Metal compiler,
# then link each .air into a .metallib. Compile only: nothing is executed.
# Outputs go to $RUNNER_TEMP (CI) or a fresh temporary directory, never into
# the repository. Exits non-zero if any file fails to compile or link.
#
# Usage, from anywhere inside a checkout on a Mac:
#   tools/compile_metal.sh
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

out_dir="${RUNNER_TEMP:-}"
if [[ -z "$out_dir" ]]; then
  out_dir="$(mktemp -d)"
fi
out_dir="$out_dir/metal-compile"
mkdir -p "$out_dir"

# Tracked and untracked-but-not-ignored files, so build directories are skipped.
files=()
while IFS= read -r -d '' f; do
  files+=("$f")
done < <(git ls-files -z --cached --others --exclude-standard -- '*.metal')

if [[ ${#files[@]} -eq 0 ]]; then
  echo "No .metal files found; nothing to compile."
  exit 0
fi

xcrun --sdk macosx metal --version

failed=()
for f in "${files[@]}"; do
  # Name outputs after the path so two files with the same basename don't collide.
  name="${f%.metal}"
  name="${name//\//__}"
  air="$out_dir/$name.air"
  lib="$out_dir/$name.metallib"
  echo "==> $f"
  if ! xcrun --sdk macosx metal -std=metal3.1 -c "$f" -o "$air"; then
    echo "::error file=$f::Metal compile failed: $f"
    failed+=("$f")
    continue
  fi
  if ! xcrun --sdk macosx metallib "$air" -o "$lib"; then
    echo "::error file=$f::metallib link failed: $f"
    failed+=("$f")
    continue
  fi
done

if [[ ${#failed[@]} -gt 0 ]]; then
  echo "FAILED (${#failed[@]} of ${#files[@]}):"
  printf '  %s\n' "${failed[@]}"
  exit 1
fi

echo "OK: compiled and linked ${#files[@]} .metal file(s) into $out_dir"
