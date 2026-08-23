# Stage-1 DSE S1 / M3: (RO, RWreq, RWresp, WO) = (2, 1, 2, 1).
config_build_path := dse_s1_m3
noc_req_wide_ro_channel_num  := 2
noc_req_wide_rw_channel_num  := 1
noc_resp_wide_rw_channel_num := 2
noc_resp_wide_wo_channel_num := 1
include $(MEMPOOL_DIR)/config/dse_s1_common.mk
