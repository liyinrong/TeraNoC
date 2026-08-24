# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Stage-1 DSE S8 / M6: (RO, RWreq, RWresp, WO) = (8, 2, 4, 2).
config_build_path := dse_s8_m6
noc_req_wide_ro_channel_num  := 8
noc_req_wide_rw_channel_num  := 2
noc_resp_wide_rw_channel_num := 4
noc_resp_wide_wo_channel_num := 2
include $(MEMPOOL_DIR)/config/dse_s8_common.mk
