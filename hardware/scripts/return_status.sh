#!/usr/bin/env bash

# Copyright 2021 ETH Zurich and University of Bologna.
# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51

# Author: Samuel Riedel, ETH Zurich

# Snitch banks wakes arriving while a core is awake in a 3-bit counter that
# WRAPS (snitch.sv:130,138,447); the eighth unconsumed wake silently discards
# the other seven, after which a wfi can sleep on an already-true predicate.
# snitch.sv:2987 prints on exactly that event but only as a $display, so the
# run would otherwise pass or hang with no explanation. Fail loudly instead:
# any hit means some core received more wakes than it spent, and the sleeping
# barriers' credit-neutrality assumption is broken somewhere.
MISSED=$(grep -c 'Missed wake-up' $1)
if [[ "${MISSED}" -ne 0 ]]; then
  echo "FAIL: ${MISSED} missed wake-up(s) -- a core's 3-bit wake counter wrapped."
  echo "      Some core banked 8 unconsumed wakes. Find the wake source that is"
  echo "      not matched by a wfi (RedMulE EOC, fsync, or an out-of-scope"
  echo "      wake_up_all), or move that wait to the polling barrier."
  grep -m5 'Missed wake-up' $1
  exit 1
fi

RET=$(grep '\[EOC\]' $1 | grep -Poh -- '(?<=\(retval = )-?[0-9]+(?=\))')
[[ -z "${RET}" ]] && echo "Simulation did not finish" && exit 1
echo "Simulation returned ${RET}"
[[ "${RET}" -eq 0 ]] && exit 0 || exit 1
