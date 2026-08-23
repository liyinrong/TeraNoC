# Stage-1 DSE S5 / M6: (RO, RWreq, RWresp, WO) = (8, 2, 4, 2).
config_build_path := dse_s5_m6
noc_req_wide_ro_channel_num  := 8
noc_req_wide_rw_channel_num  := 2
noc_resp_wide_rw_channel_num := 4
noc_resp_wide_wo_channel_num := 2
include $(MEMPOOL_DIR)/config/dse_s5_common.mk
