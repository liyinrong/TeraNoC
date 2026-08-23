# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Stage-1 S2/M4 with a 1024-bit Spatz vector register. Keep a distinct build
# path so the guest cannot be confused with the legacy 512-bit S2 binary.
config_build_path := dse_s2_m4_spatz1024
noc_req_wide_ro_channel_num  := 4
noc_req_wide_rw_channel_num  := 1
noc_resp_wide_rw_channel_num := 4
noc_resp_wide_wo_channel_num := 1
vlen := 1024
include $(MEMPOOL_DIR)/config/dse_s2_common.mk
