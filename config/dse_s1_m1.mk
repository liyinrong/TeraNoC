# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Stage-1 DSE S1 / M1 baseline: (RO, RWreq, RWresp, WO) = (0, 1, 1, 0).
config_build_path := dse_s1_m1
noc_req_wide_ro_channel_num  := 0
noc_req_wide_rw_channel_num  := 1
noc_resp_wide_rw_channel_num := 1
noc_resp_wide_wo_channel_num := 0
include $(MEMPOOL_DIR)/config/dse_s1_common.mk
