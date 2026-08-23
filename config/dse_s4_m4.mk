# Stage-1 DSE S4 / M4: (RO, RWreq, RWresp, WO) = (4, 1, 4, 1).
config_build_path := dse_s4_m4
noc_req_wide_ro_channel_num  := 4
noc_req_wide_rw_channel_num  := 1
noc_resp_wide_rw_channel_num := 4
noc_resp_wide_wo_channel_num := 1
include $(MEMPOOL_DIR)/config/dse_s4_common.mk
