#!/usr/bin/env bash
# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Check the closed-form fractal_sync (aggr, id) encoding against the reference
# generator, for every tree size we instantiate. See fsync_reqs_oracle.c.
#
# The reference header hardcodes a 4x4 grid in three #defines with no include
# guards, so we compile against a copy that wraps them in #ifndef and pass the
# geometry with -D. The reference .c is copied BYTE-FOR-BYTE beside that header
# rather than compiled in place: a quoted #include resolves relative to the
# including file's own directory first, so building it from the dep would find
# the dep's unguarded header no matter what -I/-D we pass, silently pinning
# every run to 4x4. (That is exactly how this script failed first time -- only
# the 4x4 case passed, because only 4x4 was ever actually built.)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEP="$ROOT/hardware/deps/fractal_sync/sw"
OUT="${TMPDIR:-/tmp}/fsync_oracle.$$"

[ -f "$DEP/fractal_sync_req_gen.c" ] || {
  echo "fractal_sync dep not checked out: run 'bender checkout'" >&2; exit 2; }

mkdir -p "$OUT"
trap 'rm -rf "$OUT"' EXIT
sed -E 's/^#define (__FSYNC_N_CU_X__|__FSYNC_N_CU_Y__|__FSYNC_N_LVL__)( *)\(([0-9]+)\)/#ifndef \1\n#define \1 (\3)\n#endif/' \
  "$DEP/fractal_sync_req_gen.h" > "$OUT/fractal_sync_req_gen.h"
cp "$DEP/fractal_sync_req_gen.c" "$OUT/fractal_sync_req_gen.c"

rc=0
for n in 4 8 16 32; do
  case $n in 4) lvl=4;; 8) lvl=6;; 16) lvl=8;; 32) lvl=10;; esac   # 2*log2(n)
  cc -O1 -Wall -Wextra -I"$OUT" -I"$(dirname "$0")" \
     -I"$ROOT/software/runtime" \
     -DNUM_TILES_PER_GROUP=4 -DNUM_X=4 \
     -DNUM_BANKS_PER_TILE=1 -DNUM_CORES_PER_TILE=1 \
     -D__FSYNC_N_CU_X__="$n" -D__FSYNC_N_CU_Y__="$n" -D__FSYNC_N_LVL__="$lvl" \
     -o "$OUT/oracle_$n" "$(dirname "$0")/fsync_reqs_oracle.c" \
     "$OUT/fractal_sync_req_gen.c"
  "$OUT/oracle_$n" || rc=1
done
[ $rc -eq 0 ] && echo "fsync oracle: PASS" || echo "fsync oracle: FAIL"
exit $rc
