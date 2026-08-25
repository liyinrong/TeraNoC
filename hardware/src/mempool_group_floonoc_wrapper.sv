// Copyright 2024 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

`include "mempool/mempool.svh"
`include "reqrsp_interface/typedef.svh"
`include "common_cells/registers.svh"

module mempool_group_floonoc_wrapper
  import mempool_pkg::*;
  import cf_math_pkg::idx_width;
  import floo_pkg::*;
  import floo_terapool_noc_pkg::*;
#(
  // Parameters for mempool_group
  parameter addr_t       TCDMBaseAddr = 32'b0,
  parameter logic [31:0] BootAddr     = 32'h0000_1000
) (
  // Clock and reset
  input  logic                                          clk_i,
  input  logic                                          rst_ni,
  input  logic                                          testmode_i,

  // Scan chain
  input  logic                                          scan_enable_i,
  input  logic                                          scan_data_i,
  output logic                                          scan_data_o,

  // Group ID
  input  logic              [idx_width(NumGroups)-1:0]  group_id_i,
  input  id_t                                           floo_id_i,

  // TCDM Router interface
  output floo_tcdm_req_if_t [West:North]                floo_tcdm_req_o,
  output floo_tcdm_rsp_if_t [West:North]                floo_tcdm_rsp_o,
  input  floo_tcdm_req_if_t [West:North]                floo_tcdm_req_i,
  input  floo_tcdm_rsp_if_t [West:North]                floo_tcdm_rsp_i,

  // AXI Router interface
  output floo_req_t         [West:North]                floo_axi_req_o,
  output floo_rsp_t         [West:North]                floo_axi_rsp_o,
  output floo_wide_t        [West:North]                floo_axi_wide_o,
  input  floo_req_t         [West:North]                floo_axi_req_i,
  input  floo_rsp_t         [West:North]                floo_axi_rsp_i,
  input  floo_wide_t        [West:North]                floo_axi_wide_i,

  // Wake up interface
  input  logic              [NumCoresPerGroup-1:0]      wake_up_i,

  // RO-Cache configuration
  input  `STRUCT_PORT(ro_cache_ctrl_t)                  ro_cache_ctrl_i,

  // DMA request
  input  `STRUCT_PORT(dma_req_t)                        dma_req_i,
  input  logic                                          dma_req_valid_i,
  output logic                                          dma_req_ready_o,

  // DMA status
  output `STRUCT_PORT(dma_meta_t)                       dma_meta_o
);

// narrow RO request NoC
`ifdef USE_RO_NARROW_REQ_CHANNEL
  floo_tcdm_narrow_ro_req_t [West:North][NumTilesPerGroup-1:0][NumNarrowRoReqPortsPerTile-1:0] floo_tcdm_narrow_ro_req_out;
  logic              [West:North][NumTilesPerGroup-1:0][NumNarrowRoReqPortsPerTile-1:0] floo_tcdm_narrow_ro_req_valid_out;
  logic              [West:North][NumTilesPerGroup-1:0][NumNarrowRoReqPortsPerTile-1:0] floo_tcdm_narrow_ro_req_ready_in;
  floo_tcdm_narrow_ro_req_t [West:North][NumTilesPerGroup-1:0][NumNarrowRoReqPortsPerTile-1:0] floo_tcdm_narrow_ro_req_in;
  logic              [West:North][NumTilesPerGroup-1:0][NumNarrowRoReqPortsPerTile-1:0] floo_tcdm_narrow_ro_req_valid_in;
  logic              [West:North][NumTilesPerGroup-1:0][NumNarrowRoReqPortsPerTile-1:0] floo_tcdm_narrow_ro_req_ready_out;

  for (genvar i = North; i <= West; i++) begin : gen_tcdm_narrow_ro_req_if_i
    for (genvar j = 0; j < NumTilesPerGroup; j++) begin : gen_tcdm_narrow_ro_req_if_j
      for (genvar k = 0; k < NumNarrowRoReqPortsPerTile; k++) begin : gen_tcdm_narrow_ro_req_if_k
        assign floo_tcdm_req_o[i].floo_tcdm_req[j].narrow_ro_req[k].req   = floo_tcdm_narrow_ro_req_out     [i][j][k];
        assign floo_tcdm_req_o[i].floo_tcdm_req[j].narrow_ro_req[k].valid = floo_tcdm_narrow_ro_req_valid_out[i][j][k];
        assign floo_tcdm_req_o[i].floo_tcdm_req[j].narrow_ro_req[k].ready = floo_tcdm_narrow_ro_req_ready_out[i][j][k];
        assign floo_tcdm_narrow_ro_req_in       [i][j][k] = floo_tcdm_req_i[i].floo_tcdm_req[j].narrow_ro_req[k].req;
        assign floo_tcdm_narrow_ro_req_valid_in [i][j][k] = floo_tcdm_req_i[i].floo_tcdm_req[j].narrow_ro_req[k].valid;
        assign floo_tcdm_narrow_ro_req_ready_in [i][j][k] = floo_tcdm_req_i[i].floo_tcdm_req[j].narrow_ro_req[k].ready;
      end
    end
  end
`endif

// narrow RW request NoC
floo_tcdm_narrow_rw_req_t [West:North][NumTilesPerGroup-1:0][NumNarrowPayloadReqPortsPerTile-1:0] floo_tcdm_narrow_rw_req_out;
logic                [West:North][NumTilesPerGroup-1:0][NumNarrowPayloadReqPortsPerTile-1:0] floo_tcdm_narrow_rw_req_valid_out;
logic                [West:North][NumTilesPerGroup-1:0][NumNarrowPayloadReqPortsPerTile-1:0] floo_tcdm_narrow_rw_req_ready_in;
floo_tcdm_narrow_rw_req_t [West:North][NumTilesPerGroup-1:0][NumNarrowPayloadReqPortsPerTile-1:0] floo_tcdm_narrow_rw_req_in;
logic                [West:North][NumTilesPerGroup-1:0][NumNarrowPayloadReqPortsPerTile-1:0] floo_tcdm_narrow_rw_req_valid_in;
logic                [West:North][NumTilesPerGroup-1:0][NumNarrowPayloadReqPortsPerTile-1:0] floo_tcdm_narrow_rw_req_ready_out;

for (genvar i = North; i <= West; i++) begin : gen_tcdm_narrow_rw_req_if_i
  for(genvar j = 0; j < NumTilesPerGroup; j++) begin : gen_tcdm_narrow_rw_req_if_j
    for(genvar k = 0; k < NumNarrowPayloadReqPortsPerTile; k++) begin : gen_tcdm_narrow_rw_req_if_k
      assign floo_tcdm_req_o[i].floo_tcdm_req[j].narrow_rw_req[k].req    = floo_tcdm_narrow_rw_req_out      [i][j][k];
      assign floo_tcdm_req_o[i].floo_tcdm_req[j].narrow_rw_req[k].valid  = floo_tcdm_narrow_rw_req_valid_out[i][j][k];
      assign floo_tcdm_req_o[i].floo_tcdm_req[j].narrow_rw_req[k].ready  = floo_tcdm_narrow_rw_req_ready_out[i][j][k];
      assign floo_tcdm_narrow_rw_req_in        [i][j][k] = floo_tcdm_req_i[i].floo_tcdm_req[j].narrow_rw_req[k].req;
      assign floo_tcdm_narrow_rw_req_valid_in  [i][j][k] = floo_tcdm_req_i[i].floo_tcdm_req[j].narrow_rw_req[k].valid;
      assign floo_tcdm_narrow_rw_req_ready_in  [i][j][k] = floo_tcdm_req_i[i].floo_tcdm_req[j].narrow_rw_req[k].ready;
    end
  end
end

// narrow RD response NoC
floo_tcdm_narrow_rw_resp_t     [West:North][NumTilesPerGroup-1:0][NumNarrowRemoteRespPortsPerTile-1:1] floo_tcdm_narrow_rw_resp_out;
logic                [West:North][NumTilesPerGroup-1:0][NumNarrowRemoteRespPortsPerTile-1:1] floo_tcdm_narrow_rw_resp_valid_out;
logic                [West:North][NumTilesPerGroup-1:0][NumNarrowRemoteRespPortsPerTile-1:1] floo_tcdm_narrow_rw_resp_ready_in;
floo_tcdm_narrow_rw_resp_t     [West:North][NumTilesPerGroup-1:0][NumNarrowRemoteRespPortsPerTile-1:1] floo_tcdm_narrow_rw_resp_in;
logic                [West:North][NumTilesPerGroup-1:0][NumNarrowRemoteRespPortsPerTile-1:1] floo_tcdm_narrow_rw_resp_valid_in;
logic                [West:North][NumTilesPerGroup-1:0][NumNarrowRemoteRespPortsPerTile-1:1] floo_tcdm_narrow_rw_resp_ready_out;

for (genvar i = North; i <= West; i++) begin : gen_tcdm_narrow_rw_resp_if_i
  for(genvar j = 0; j < NumTilesPerGroup; j++) begin : gen_tcdm_narrow_rw_resp_if_j
    for(genvar k = 1; k < NumNarrowRemoteRespPortsPerTile; k++) begin : gen_tcdm_narrow_rw_resp_if_k
      assign floo_tcdm_rsp_o[i].floo_tcdm_narrow_rw_resp[j][k].resp   = floo_tcdm_narrow_rw_resp_out       [i][j][k];
      assign floo_tcdm_rsp_o[i].floo_tcdm_narrow_rw_resp[j][k].valid  = floo_tcdm_narrow_rw_resp_valid_out [i][j][k];
      assign floo_tcdm_rsp_o[i].floo_tcdm_narrow_rw_resp[j][k].ready  = floo_tcdm_narrow_rw_resp_ready_out [i][j][k];
      assign floo_tcdm_narrow_rw_resp_in        [i][j][k] = floo_tcdm_rsp_i[i].floo_tcdm_narrow_rw_resp[j][k].resp;
      assign floo_tcdm_narrow_rw_resp_valid_in  [i][j][k] = floo_tcdm_rsp_i[i].floo_tcdm_narrow_rw_resp[j][k].valid;
      assign floo_tcdm_narrow_rw_resp_ready_in  [i][j][k] = floo_tcdm_rsp_i[i].floo_tcdm_narrow_rw_resp[j][k].ready;
    end
  end
end

// Parse the address width to calculate the offset
localparam integer unsigned NumTilesPerGroupWidth = idx_width(NumTilesPerGroup);
localparam integer unsigned NumBanksPerTileWidth  = idx_width(NumBanksPerTile);
localparam integer unsigned TileBankRowOffset     = TCDMAddrMemWidth + NumBanksPerTileWidth;
localparam integer unsigned TileOffset            = NumBanksPerTileWidth + ByteOffset;

// TCDM Master interfaces
tcdm_master_req_t  [NumTilesPerGroup-1:0][NumNarrowRemoteReqPortsPerTile-1:1] tcdm_master_req;
logic              [NumTilesPerGroup-1:0][NumNarrowRemoteReqPortsPerTile-1:1] tcdm_master_req_valid;
logic              [NumTilesPerGroup-1:0][NumNarrowRemoteReqPortsPerTile-1:1] tcdm_master_req_ready;
tcdm_master_resp_t [NumTilesPerGroup-1:0][NumNarrowRemoteRespPortsPerTile-1:1] tcdm_master_resp;
logic              [NumTilesPerGroup-1:0][NumNarrowRemoteRespPortsPerTile-1:1] tcdm_master_resp_valid;
logic              [NumTilesPerGroup-1:0][NumNarrowRemoteRespPortsPerTile-1:1] tcdm_master_resp_ready;

// TCDM Slave interfaces
tcdm_slave_req_t   [NumTilesPerGroup-1:0][NumNarrowRemoteReqPortsPerTile-1:1] tcdm_slave_req;
logic              [NumTilesPerGroup-1:0][NumNarrowRemoteReqPortsPerTile-1:1] tcdm_slave_req_valid;
logic              [NumTilesPerGroup-1:0][NumNarrowRemoteReqPortsPerTile-1:1] tcdm_slave_req_ready;
tcdm_slave_resp_t  [NumTilesPerGroup-1:0][NumNarrowRemoteRespPortsPerTile-1:1] tcdm_slave_resp;
logic              [NumTilesPerGroup-1:0][NumNarrowRemoteRespPortsPerTile-1:1] tcdm_slave_resp_valid;
logic              [NumTilesPerGroup-1:0][NumNarrowRemoteRespPortsPerTile-1:1] tcdm_slave_resp_ready;

axi_tile_req_t  axi_mst_req;
axi_tile_resp_t axi_mst_resp;
axi_tile_req_t  axi_mst_req_from_group;
axi_tile_resp_t axi_mst_resp_to_group;
axi_tile_req_t  axi_mst_req_splitter;
axi_tile_resp_t axi_mst_resp_splitter;

// Wide TCDM plane (WP2): the group's inter-group wide ports, numbered as on the
// tile: requests {RO, WO}, responses {RW, WO}. FlooNoC packets are constructed
// and decoded below, next to the wide routers.
`ifdef USE_WIDE_TCDM_PLANE
tcdm_wide_master_req_t  [NumTilesPerGroup-1:0][NumWideRemoteReqPortsPerTile-1:NumWideIntraGroupPortsPerTile]  tcdm_wide_master_req;
logic                   [NumTilesPerGroup-1:0][NumWideRemoteReqPortsPerTile-1:NumWideIntraGroupPortsPerTile]  tcdm_wide_master_req_valid;
logic                   [NumTilesPerGroup-1:0][NumWideRemoteReqPortsPerTile-1:NumWideIntraGroupPortsPerTile]  tcdm_wide_master_req_ready;
tcdm_wide_slave_req_t   [NumTilesPerGroup-1:0][NumWideRemoteReqPortsPerTile-1:NumWideIntraGroupPortsPerTile]  tcdm_wide_slave_req;
logic                   [NumTilesPerGroup-1:0][NumWideRemoteReqPortsPerTile-1:NumWideIntraGroupPortsPerTile]  tcdm_wide_slave_req_valid;
logic                   [NumTilesPerGroup-1:0][NumWideRemoteReqPortsPerTile-1:NumWideIntraGroupPortsPerTile]  tcdm_wide_slave_req_ready;
tcdm_wide_master_resp_t [NumTilesPerGroup-1:0][NumWideRemoteRespPortsPerTile-1:NumWideIntraGroupPortsPerTile] tcdm_wide_master_resp;
logic                   [NumTilesPerGroup-1:0][NumWideRemoteRespPortsPerTile-1:NumWideIntraGroupPortsPerTile] tcdm_wide_master_resp_valid;
logic                   [NumTilesPerGroup-1:0][NumWideRemoteRespPortsPerTile-1:NumWideIntraGroupPortsPerTile] tcdm_wide_master_resp_ready;
tcdm_wide_slave_resp_t  [NumTilesPerGroup-1:0][NumWideRemoteRespPortsPerTile-1:NumWideIntraGroupPortsPerTile] tcdm_wide_slave_resp;
logic                   [NumTilesPerGroup-1:0][NumWideRemoteRespPortsPerTile-1:NumWideIntraGroupPortsPerTile] tcdm_wide_slave_resp_valid;
logic                   [NumTilesPerGroup-1:0][NumWideRemoteRespPortsPerTile-1:NumWideIntraGroupPortsPerTile] tcdm_wide_slave_resp_ready;
`endif

// Instantiate the mempool_group
mempool_group #(
  .TCDMBaseAddr             (TCDMBaseAddr                   ),
  .BootAddr                 (BootAddr                       )
) i_mempool_group (
  .clk_i                    (clk_i                          ),
  .rst_ni                   (rst_ni                         ),
  .testmode_i               (testmode_i                     ),
  .scan_enable_i            (scan_enable_i                  ),
  .scan_data_i              (scan_data_i                    ),
  .scan_data_o              (scan_data_o                    ),
  .group_id_i               (group_id_i                     ),
  .tcdm_master_req_o        (tcdm_master_req                ),
  .tcdm_master_req_valid_o  (tcdm_master_req_valid          ),
  .tcdm_master_req_ready_i  (tcdm_master_req_ready          ),
  .tcdm_master_resp_i       (tcdm_master_resp               ),
  .tcdm_master_resp_valid_i (tcdm_master_resp_valid         ),
  .tcdm_master_resp_ready_o (tcdm_master_resp_ready         ),
  .tcdm_slave_req_i         (tcdm_slave_req                 ),
  .tcdm_slave_req_valid_i   (tcdm_slave_req_valid           ),
  .tcdm_slave_req_ready_o   (tcdm_slave_req_ready           ),
  .tcdm_slave_resp_o        (tcdm_slave_resp                ),
  .tcdm_slave_resp_valid_o  (tcdm_slave_resp_valid          ),
  .tcdm_slave_resp_ready_i  (tcdm_slave_resp_ready          ),
  .wake_up_i                (wake_up_i                      ),
  .ro_cache_ctrl_i          (ro_cache_ctrl_i                ),
  .dma_req_i                (dma_req_i                      ),
  .dma_req_valid_i          (dma_req_valid_i                ),
  .dma_req_ready_o          (dma_req_ready_o                ),
  .dma_meta_o               (dma_meta_o                     ),
  `ifdef USE_WIDE_TCDM_PLANE
  .tcdm_wide_master_req_o        (tcdm_wide_master_req        ),
  .tcdm_wide_master_req_valid_o  (tcdm_wide_master_req_valid  ),
  .tcdm_wide_master_req_ready_i  (tcdm_wide_master_req_ready  ),
  .tcdm_wide_slave_req_i         (tcdm_wide_slave_req         ),
  .tcdm_wide_slave_req_valid_i   (tcdm_wide_slave_req_valid   ),
  .tcdm_wide_slave_req_ready_o   (tcdm_wide_slave_req_ready   ),
  .tcdm_wide_master_resp_i       (tcdm_wide_master_resp       ),
  .tcdm_wide_master_resp_valid_i (tcdm_wide_master_resp_valid ),
  .tcdm_wide_master_resp_ready_o (tcdm_wide_master_resp_ready ),
  .tcdm_wide_slave_resp_o        (tcdm_wide_slave_resp        ),
  .tcdm_wide_slave_resp_valid_o  (tcdm_wide_slave_resp_valid  ),
  .tcdm_wide_slave_resp_ready_i  (tcdm_wide_slave_resp_ready  ),
  `endif
  .axi_mst_req_o            (axi_mst_req_from_group         ),
  .axi_mst_resp_i           (axi_mst_resp_to_group          )
);

// Splitting logic based on DmaBurstLen and Interleave
generate
  if (DmaBurstLen > Interleave) begin : gen_axi_splitter
    axi_wide_splitter #(
      .MaxReadTxns  (16                             ),
      .MaxWriteTxns (16                             ),
      .AddrWidth    (AddrWidth                      ),
      .DataWidth    (AxiDataWidth                   ),
      .IdWidth      (AxiTileIdWidth                 ),
      .UserWidth    (1                              ),
      .axi_req_t    (axi_tile_req_t                 ),
      .axi_resp_t   (axi_tile_resp_t                )
    ) i_axi_wide_splitter (
      .clk_i        (clk_i                          ),
      .rst_ni       (rst_ni                         ),
      .slv_req_i    (axi_mst_req_from_group         ),
      .slv_resp_o   (axi_mst_resp_to_group          ),
      .mst_req_o    (axi_mst_req_splitter           ),
      .mst_resp_i   (axi_mst_resp_splitter          )
    );
  end else begin : gen_axi_splitter_bypass
    assign axi_mst_req_splitter = axi_mst_req_from_group;
    assign axi_mst_resp_to_group = axi_mst_resp_splitter;
  end
endgenerate

axi_L2_interleaver #(
  .NumAXIMasters            (1                              ),
  .NumL2                    (NumL2Banks                     ),
  .L2Size                   (L2Size                         )
) i_axi_L2_interleaver (
  .clk_i                    (clk_i                          ),
  .rst_ni                   (rst_ni                         ),
  .axi_l2_req_i             (axi_mst_req_splitter           ),
  .axi_l2_resp_o            (axi_mst_resp_splitter          ),
  .axi_l2_req_interleaved_o (axi_mst_req                    ),
  .axi_l2_resp_interleaved_i(axi_mst_resp                   )
);

// Instantiate the floo_tcdm_router for each tile
`ifdef USE_RO_NARROW_REQ_CHANNEL
  floo_tcdm_narrow_ro_req_t  [NumTilesPerGroup-1:0][NumNarrowRoReqPortsPerTile-1:0] floo_tcdm_narrow_ro_req_to_router;
  floo_tcdm_narrow_ro_req_t  [NumTilesPerGroup-1:0][NumNarrowRoReqPortsPerTile-1:0] floo_tcdm_narrow_ro_req_from_router;
  floo_tcdm_narrow_ro_req_t  [NumTilesPerGroup-1:0][NumNarrowRoReqPortsPerTile-1:0] floo_tcdm_narrow_ro_req_from_router_vc;
  logic               [NumTilesPerGroup-1:0][NumNarrowRoReqPortsPerTile-1:0] floo_tcdm_narrow_ro_req_to_router_valid;
  logic               [NumTilesPerGroup-1:0][NumNarrowRoReqPortsPerTile-1:0] floo_tcdm_narrow_ro_req_to_router_ready;
  logic               [NumTilesPerGroup-1:0][NumNarrowRoReqPortsPerTile-1:0] floo_tcdm_narrow_ro_req_to_router_vc_valid;
  logic               [NumTilesPerGroup-1:0][NumNarrowRoReqPortsPerTile-1:0] floo_tcdm_narrow_ro_req_to_router_vc_ready;
  logic               [NumTilesPerGroup-1:0][NumNarrowRoReqPortsPerTile-1:0] floo_tcdm_narrow_ro_req_from_router_vc_valid;
  logic               [NumTilesPerGroup-1:0][NumNarrowRoReqPortsPerTile-1:0] floo_tcdm_narrow_ro_req_from_router_vc_ready;
`endif

floo_tcdm_narrow_rw_req_t  [NumTilesPerGroup-1:0][NumNarrowPayloadReqPortsPerTile-1:0] floo_tcdm_narrow_rw_req_to_router;
floo_tcdm_narrow_rw_req_t  [NumTilesPerGroup-1:0][NumNarrowPayloadReqPortsPerTile-1:0] floo_tcdm_narrow_rw_req_from_router;
floo_tcdm_narrow_rw_req_t  [NumTilesPerGroup-1:0][NumNarrowPayloadReqPortsPerTile-1:0] floo_tcdm_narrow_rw_req_from_router_vc;
floo_tcdm_narrow_rw_req_t  [NumTilesPerGroup-1:0][NumNarrowRemoteReqPortsPerTile-1:1    ] floo_tcdm_narrow_req_from_router;
floo_tcdm_narrow_rw_req_t  [NumTilesPerGroup-1:0][NumNarrowRemoteReqPortsPerTile-1:1    ] floo_tcdm_narrow_req_from_router_interleaved;
floo_tcdm_narrow_rw_resp_t      [NumTilesPerGroup-1:0][NumNarrowRemoteRespPortsPerTile-1:1   ] floo_tcdm_narrow_rw_resp_to_router;
floo_tcdm_narrow_rw_resp_t      [NumTilesPerGroup-1:0][NumNarrowRemoteRespPortsPerTile-1:1   ] floo_tcdm_narrow_rw_resp_from_router;
logic                 [NumTilesPerGroup-1:0][NumNarrowPayloadReqPortsPerTile-1:0] floo_tcdm_narrow_rw_req_to_router_valid;
logic                 [NumTilesPerGroup-1:0][NumNarrowPayloadReqPortsPerTile-1:0] floo_tcdm_narrow_rw_req_to_router_ready;
logic                 [NumTilesPerGroup-1:0][NumNarrowPayloadReqPortsPerTile-1:0] floo_tcdm_narrow_rw_req_to_router_vc_valid;
logic                 [NumTilesPerGroup-1:0][NumNarrowPayloadReqPortsPerTile-1:0] floo_tcdm_narrow_rw_req_to_router_vc_ready;
logic                 [NumTilesPerGroup-1:0][NumNarrowPayloadReqPortsPerTile-1:0] floo_tcdm_narrow_rw_req_from_router_vc_valid;
logic                 [NumTilesPerGroup-1:0][NumNarrowPayloadReqPortsPerTile-1:0] floo_tcdm_narrow_rw_req_from_router_vc_ready;
logic                 [NumTilesPerGroup-1:0][NumNarrowRemoteRespPortsPerTile-1:1   ] floo_tcdm_narrow_rw_resp_to_router_valid;
logic                 [NumTilesPerGroup-1:0][NumNarrowRemoteRespPortsPerTile-1:1   ] floo_tcdm_narrow_rw_resp_to_router_ready;
logic                 [NumTilesPerGroup-1:0][NumNarrowRemoteRespPortsPerTile-1:1   ] floo_tcdm_narrow_rw_resp_to_router_vc_valid;
logic                 [NumTilesPerGroup-1:0][NumNarrowRemoteRespPortsPerTile-1:1   ] floo_tcdm_narrow_rw_resp_to_router_vc_ready;

// ------------------------------------------------------------------ //
// Remapping: From MemPool "Master Request" to FlooNoC "TCDM request" //
// ------------------------------------------------------------------ //
if (NocRouterRemapping == 1 || NocRouterRemapping == 3) begin: gen_req_remapping

  `ifdef USE_RO_NARROW_REQ_CHANNEL
    floo_tcdm_narrow_ro_req_t   [NumTilesPerGroup-1:0][NumNarrowRoReqPortsPerTile-1:0] floo_tcdm_narrow_ro_req_to_remapper;
    logic                [NumTilesPerGroup-1:0][NumNarrowRoReqPortsPerTile-1:0] floo_tcdm_narrow_ro_req_to_remapper_valid;
    logic                [NumTilesPerGroup-1:0][NumNarrowRoReqPortsPerTile-1:0] floo_tcdm_narrow_ro_req_to_remapper_ready;
  `endif

  floo_tcdm_narrow_rw_req_t [NumTilesPerGroup-1:0][NumNarrowPayloadReqPortsPerTile-1:0]   floo_tcdm_narrow_rw_req_to_remapper;
  logic                [NumTilesPerGroup-1:0][NumNarrowPayloadReqPortsPerTile-1:0]   floo_tcdm_narrow_rw_req_to_remapper_valid;
  logic                [NumTilesPerGroup-1:0][NumNarrowPayloadReqPortsPerTile-1:0]   floo_tcdm_narrow_rw_req_to_remapper_ready;

  for (genvar i = 0; i < NumTilesPerGroup; i++) begin : gen_master_req_to_remapper_req_i
    `ifdef USE_RO_NARROW_REQ_CHANNEL
      for (genvar j = 0; j < NumNarrowRoReqPortsPerTile; j++) begin : gen_master_narrow_ro_req_to_remapper_req_j
        assign floo_tcdm_narrow_ro_req_to_remapper[i][j] = floo_tcdm_narrow_ro_req_t'{
          hdr: floo_tcdm_req_meta_t'{
            meta_id : tcdm_master_req[i][j+(1)].wdata.meta_id,                      // For Register File
            core_id : tcdm_master_req[i][j+(1)].wdata.core_id,                      // For Core
            src_tile_id : i,                                                        // For Crossbar when response back
            src_id: group_xy_of(group_id_i),                             // For NoC Router when response back
            dst_id: group_xy_of(tcdm_master_req[i][j+(1)].tgt_group_id), // For NoC Router when request send
            tgt_addr: tcdm_master_req[i][j+(1)].tgt_addr,                           // For Crossbar when request send (bank rows per Group)
            last : 1'b1,                                                            // Non Burst Request
            collective_op : 1'b0                                                    // Not used
          }
        };
        assign floo_tcdm_narrow_ro_req_to_remapper_valid[i][j] = tcdm_master_req_valid[i][j+(1)];
        assign tcdm_master_req_ready[i][j+(1)] = floo_tcdm_narrow_ro_req_to_remapper_ready[i][j];
      end : gen_master_narrow_ro_req_to_remapper_req_j
    `endif

    for (genvar j = 1 + NumNarrowRoReqPortsPerTile; j < NumNarrowRemoteReqPortsPerTile; j++) begin : gen_master_narrow_payload_req_to_remapper_req_j
      assign floo_tcdm_narrow_rw_req_to_remapper[i][j-(1 + NumNarrowRoReqPortsPerTile)] = floo_tcdm_narrow_rw_req_t'{
        payload: floo_tcdm_narrow_req_payload_t'{
          amo : tcdm_master_req[i][j].wdata.amo,
          wen : tcdm_master_req[i][j].wen,
          be  : tcdm_master_req[i][j].be,
          data: tcdm_master_req[i][j].wdata.data
        },
        hdr: floo_tcdm_req_meta_t'{
          meta_id : tcdm_master_req[i][j].wdata.meta_id,                      // For Register File
          core_id : tcdm_master_req[i][j].wdata.core_id,                      // For Core
          src_tile_id : i,                                                    // For Crossbar when response back
          src_id: group_xy_of(group_id_i),                         // For NoC Router when response back
          dst_id: group_xy_of(tcdm_master_req[i][j].tgt_group_id), // For NoC Router when request send
          tgt_addr: tcdm_master_req[i][j].tgt_addr,                           // For Crossbar when request send (bank rows per Group)
          last : 1'b1,                                                        // Non Burst Request
          collective_op : 1'b0                                                // Not used
        }
      };
      assign floo_tcdm_narrow_rw_req_to_remapper_valid[i][j-(1 + NumNarrowRoReqPortsPerTile)] = tcdm_master_req_valid[i][j];
      assign tcdm_master_req_ready[i][j] = floo_tcdm_narrow_rw_req_to_remapper_ready[i][j-(1 + NumNarrowRoReqPortsPerTile)];
    end : gen_master_narrow_payload_req_to_remapper_req_j
  end : gen_master_req_to_remapper_req_i

  `ifdef USE_RO_NARROW_REQ_CHANNEL
    floo_remapper #(
      .NumInp     (NumTilesPerGroup * NumNarrowRoReqPortsPerTile),
      .NumOut     (NumTilesPerGroup * NumNarrowRoReqPortsPerTile),
      .payload_t  (floo_tcdm_narrow_ro_req_t                               ),
      .GroupSize  (RouterRemapGroupSize                             ),
      .Interleaved(1'b1                                             )
    ) i_floo_tcdm_narrow_ro_req_remapper (
      .clk_i      (clk_i                                            ),
      .rst_ni     (rst_ni                                           ),
      .inp_data_i (floo_tcdm_narrow_ro_req_to_remapper                     ),
      .inp_valid_i(floo_tcdm_narrow_ro_req_to_remapper_valid               ),
      .inp_ready_o(floo_tcdm_narrow_ro_req_to_remapper_ready               ),
      .oup_data_o (floo_tcdm_narrow_ro_req_to_router                       ),
      .oup_valid_o(floo_tcdm_narrow_ro_req_to_router_valid                 ),
      .oup_ready_i(floo_tcdm_narrow_ro_req_to_router_ready                 )
    );
  `endif

  floo_remapper #(
    .NumInp     (NumTilesPerGroup * NumNarrowPayloadReqPortsPerTile),
    .NumOut     (NumTilesPerGroup * NumNarrowPayloadReqPortsPerTile),
    .payload_t  (floo_tcdm_narrow_rw_req_t                           ),
    .GroupSize  (RouterRemapGroupSize                           ),
    .Interleaved(1'b1                                           )
  ) i_floo_tcdm_narrow_rw_req_remapper (
    .clk_i      (clk_i                                          ),
    .rst_ni     (rst_ni                                         ),
    .inp_data_i (floo_tcdm_narrow_rw_req_to_remapper                 ),
    .inp_valid_i(floo_tcdm_narrow_rw_req_to_remapper_valid           ),
    .inp_ready_o(floo_tcdm_narrow_rw_req_to_remapper_ready           ),
    .oup_data_o (floo_tcdm_narrow_rw_req_to_router                   ),
    .oup_valid_o(floo_tcdm_narrow_rw_req_to_router_valid             ),
    .oup_ready_i(floo_tcdm_narrow_rw_req_to_router_ready             )
  );

end else begin: gen_req_remapping_bypass
  for (genvar i = 0; i < NumTilesPerGroup; i++) begin : gen_master_req_to_router_req_i
    `ifdef USE_RO_NARROW_REQ_CHANNEL
      for (genvar j = 0; j < NumNarrowRoReqPortsPerTile; j++) begin : gen_master_narrow_ro_req_to_router_req_j
        assign floo_tcdm_narrow_ro_req_to_router[i][j] = floo_tcdm_narrow_ro_req_t'{
          hdr: floo_tcdm_req_meta_t'{
            meta_id : tcdm_master_req[i][j+(1)].wdata.meta_id,                      // For Register File
            core_id : tcdm_master_req[i][j+(1)].wdata.core_id,                      // For Core
            src_tile_id : i,                                                        // For Crossbar when response back
            src_id: group_xy_of(group_id_i),                             // For NoC Router when response back
            dst_id: group_xy_of(tcdm_master_req[i][j+(1)].tgt_group_id), // For NoC Router when request send
            tgt_addr: tcdm_master_req[i][j+(1)].tgt_addr,                           // For Crossbar when request send (bank rows per Group)
            last : 1'b1,                                                            // Non Burst Request
            collective_op : 1'b0                                                    // Not used
          }
        };
        assign floo_tcdm_narrow_ro_req_to_router_valid[i][j] = tcdm_master_req_valid[i][j+(1)];
        assign tcdm_master_req_ready[i][j+(1)] = floo_tcdm_narrow_ro_req_to_router_ready[i][j];
      end : gen_master_narrow_ro_req_to_router_req_j
    `endif

    for (genvar j = 1 + NumNarrowRoReqPortsPerTile; j < NumNarrowRemoteReqPortsPerTile; j++) begin : gen_master_narrow_payload_req_to_router_req_j
      assign floo_tcdm_narrow_rw_req_to_router[i][j-(1 + NumNarrowRoReqPortsPerTile)] = floo_tcdm_narrow_rw_req_t'{
        payload: floo_tcdm_narrow_req_payload_t'{
          amo : tcdm_master_req[i][j].wdata.amo,
          wen : tcdm_master_req[i][j].wen,
          be  : tcdm_master_req[i][j].be,
          data: tcdm_master_req[i][j].wdata.data
        },
        hdr: floo_tcdm_req_meta_t'{
          meta_id : tcdm_master_req[i][j].wdata.meta_id,                      // For Register File
          core_id : tcdm_master_req[i][j].wdata.core_id,                      // For Core
          src_tile_id : i,                                                    // For Crossbar when response back
          src_id: group_xy_of(group_id_i),                         // For NoC Router when response back
          dst_id: group_xy_of(tcdm_master_req[i][j].tgt_group_id), // For NoC Router when request send
          tgt_addr: tcdm_master_req[i][j].tgt_addr,                           // For Crossbar when request send (bank rows per Group)
          last : 1'b1,                                                        // Non Burst Request
          collective_op : 1'b0                                                // Not used
        }
      };
      assign floo_tcdm_narrow_rw_req_to_router_valid[i][j-(1 + NumNarrowRoReqPortsPerTile)] = tcdm_master_req_valid[i][j];
      assign tcdm_master_req_ready[i][j] = floo_tcdm_narrow_rw_req_to_router_ready[i][j-(1 + NumNarrowRoReqPortsPerTile)];
    end : gen_master_narrow_payload_req_to_router_req_j
  end : gen_master_req_to_router_req_i
end

`ifdef USE_RO_NARROW_REQ_CHANNEL
  for (genvar i = 0; i < NumTilesPerGroup; i++) begin : gen_master_narrow_ro_req_to_router_vc_req_i
    for (genvar j = 0; j < NumNarrowRoReqPortsPerTile; j++) begin : gen_master_narrow_ro_req_to_router_vc_req_j
      assign floo_tcdm_narrow_ro_req_to_router_vc_valid[i][j] = floo_tcdm_narrow_ro_req_to_router_valid[i][j];
      assign floo_tcdm_narrow_ro_req_to_router_ready[i][j] = floo_tcdm_narrow_ro_req_to_router_vc_ready[i][j];
    end : gen_master_narrow_ro_req_to_router_vc_req_j
  end : gen_master_narrow_ro_req_to_router_vc_req_i
`endif

for (genvar i = 0; i < NumTilesPerGroup; i++) begin : gen_master_narrow_payload_req_to_router_vc_req_i
  for (genvar j = 0; j < NumNarrowPayloadReqPortsPerTile; j++) begin : gen_master_narrow_payload_req_to_router_vc_req_j
    assign floo_tcdm_narrow_rw_req_to_router_vc_valid[i][j] = floo_tcdm_narrow_rw_req_to_router_valid[i][j];
    assign floo_tcdm_narrow_rw_req_to_router_ready[i][j] = floo_tcdm_narrow_rw_req_to_router_vc_ready[i][j];
  end : gen_master_narrow_payload_req_to_router_vc_req_j
end : gen_master_narrow_payload_req_to_router_vc_req_i

// ------------------------------------------------------------------ //
// Crossbar: FlooNoC "TCDM request" input select target tile          //
// TODO: This is a consitent assignment to Tile's port,               //
//       Should be improved for confict resolution.                   //
// ------------------------------------------------------------------ //
tile_group_id_t       [NumTilesPerGroup-1:0][NumNarrowRemoteReqPortsPerTile-1:1] req_tile_sel;
floo_tcdm_narrow_rw_req_t  [NumTilesPerGroup-1:0][NumNarrowRemoteReqPortsPerTile-1:1] floo_tcdm_narrow_req_from_router_after_xbar;
logic                 [NumTilesPerGroup-1:0][NumNarrowRemoteReqPortsPerTile-1:1] floo_tcdm_narrow_req_from_router_after_xbar_valid;
logic                 [NumTilesPerGroup-1:0][NumNarrowRemoteReqPortsPerTile-1:1] floo_tcdm_narrow_req_from_router_after_xbar_ready;
logic                 [NumTilesPerGroup-1:0][NumNarrowRemoteReqPortsPerTile-1:1] floo_tcdm_narrow_req_from_router_before_xbar_valid;
logic                 [NumTilesPerGroup-1:0][NumNarrowRemoteReqPortsPerTile-1:1] floo_tcdm_narrow_req_from_router_before_xbar_ready;
logic                 [NumTilesPerGroup-1:0][NumNarrowRemoteReqPortsPerTile-1:1] floo_tcdm_narrow_req_from_router_before_xbar_valid_interleaved;
logic                 [NumTilesPerGroup-1:0][NumNarrowRemoteReqPortsPerTile-1:1] floo_tcdm_narrow_req_from_router_before_xbar_ready_interleaved;

generate
  for(genvar i = 0; i < NumTilesPerGroup; i++) begin: gen_floo_tcdm_narrow_req_from_router_interleaved_i
    for (genvar j = 1; j < NumNarrowRemoteReqPortsPerTile; j++) begin : gen_floo_tcdm_narrow_req_from_router_interleaved_j
      if(j < 1 + NumNarrowRoReqPortsPerTile) begin
        `ifdef USE_RO_NARROW_REQ_CHANNEL
          assign floo_tcdm_narrow_req_from_router_interleaved[i][j].hdr      = floo_tcdm_narrow_ro_req_from_router[i][j-1].hdr;
          assign floo_tcdm_narrow_req_from_router_interleaved[i][j].payload  = '0;
        `endif
      end else begin
        assign floo_tcdm_narrow_req_from_router_interleaved[i][j] = floo_tcdm_narrow_rw_req_from_router[i][j-(1+NumNarrowRoReqPortsPerTile)];
      end
    end : gen_floo_tcdm_narrow_req_from_router_interleaved_j
  end : gen_floo_tcdm_narrow_req_from_router_interleaved_i
endgenerate

// For now, only support 1 RO + 1 payload-bearing RW lane, or all RW
`ifdef USE_RO_NARROW_REQ_CHANNEL
  generate
    for(genvar i = 0; i < NumTilesPerGroup/RouterRemapGroupSize; i++) begin
      assign floo_tcdm_narrow_req_from_router[0+i*RouterRemapGroupSize][0+1] = floo_tcdm_narrow_req_from_router_interleaved[0+i*RouterRemapGroupSize][0+1];
      assign floo_tcdm_narrow_req_from_router[0+i*RouterRemapGroupSize][1+1] = floo_tcdm_narrow_req_from_router_interleaved[1+i*RouterRemapGroupSize][0+1];
      assign floo_tcdm_narrow_req_from_router[1+i*RouterRemapGroupSize][0+1] = floo_tcdm_narrow_req_from_router_interleaved[2+i*RouterRemapGroupSize][0+1];
      assign floo_tcdm_narrow_req_from_router[1+i*RouterRemapGroupSize][1+1] = floo_tcdm_narrow_req_from_router_interleaved[3+i*RouterRemapGroupSize][0+1];
      assign floo_tcdm_narrow_req_from_router[2+i*RouterRemapGroupSize][0+1] = floo_tcdm_narrow_req_from_router_interleaved[0+i*RouterRemapGroupSize][1+1];
      assign floo_tcdm_narrow_req_from_router[2+i*RouterRemapGroupSize][1+1] = floo_tcdm_narrow_req_from_router_interleaved[1+i*RouterRemapGroupSize][1+1];
      assign floo_tcdm_narrow_req_from_router[3+i*RouterRemapGroupSize][0+1] = floo_tcdm_narrow_req_from_router_interleaved[2+i*RouterRemapGroupSize][1+1];
      assign floo_tcdm_narrow_req_from_router[3+i*RouterRemapGroupSize][1+1] = floo_tcdm_narrow_req_from_router_interleaved[3+i*RouterRemapGroupSize][1+1];

      assign floo_tcdm_narrow_req_from_router_before_xbar_valid[0+i*RouterRemapGroupSize][0+1] = floo_tcdm_narrow_req_from_router_before_xbar_valid_interleaved[0+i*RouterRemapGroupSize][0+1];
      assign floo_tcdm_narrow_req_from_router_before_xbar_valid[0+i*RouterRemapGroupSize][1+1] = floo_tcdm_narrow_req_from_router_before_xbar_valid_interleaved[1+i*RouterRemapGroupSize][0+1];
      assign floo_tcdm_narrow_req_from_router_before_xbar_valid[1+i*RouterRemapGroupSize][0+1] = floo_tcdm_narrow_req_from_router_before_xbar_valid_interleaved[2+i*RouterRemapGroupSize][0+1];
      assign floo_tcdm_narrow_req_from_router_before_xbar_valid[1+i*RouterRemapGroupSize][1+1] = floo_tcdm_narrow_req_from_router_before_xbar_valid_interleaved[3+i*RouterRemapGroupSize][0+1];
      assign floo_tcdm_narrow_req_from_router_before_xbar_valid[2+i*RouterRemapGroupSize][0+1] = floo_tcdm_narrow_req_from_router_before_xbar_valid_interleaved[0+i*RouterRemapGroupSize][1+1];
      assign floo_tcdm_narrow_req_from_router_before_xbar_valid[2+i*RouterRemapGroupSize][1+1] = floo_tcdm_narrow_req_from_router_before_xbar_valid_interleaved[1+i*RouterRemapGroupSize][1+1];
      assign floo_tcdm_narrow_req_from_router_before_xbar_valid[3+i*RouterRemapGroupSize][0+1] = floo_tcdm_narrow_req_from_router_before_xbar_valid_interleaved[2+i*RouterRemapGroupSize][1+1];
      assign floo_tcdm_narrow_req_from_router_before_xbar_valid[3+i*RouterRemapGroupSize][1+1] = floo_tcdm_narrow_req_from_router_before_xbar_valid_interleaved[3+i*RouterRemapGroupSize][1+1];

      assign floo_tcdm_narrow_req_from_router_before_xbar_ready_interleaved[0+i*RouterRemapGroupSize][0+1] = floo_tcdm_narrow_req_from_router_before_xbar_ready[0+i*RouterRemapGroupSize][0+1];
      assign floo_tcdm_narrow_req_from_router_before_xbar_ready_interleaved[1+i*RouterRemapGroupSize][0+1] = floo_tcdm_narrow_req_from_router_before_xbar_ready[0+i*RouterRemapGroupSize][1+1];
      assign floo_tcdm_narrow_req_from_router_before_xbar_ready_interleaved[2+i*RouterRemapGroupSize][0+1] = floo_tcdm_narrow_req_from_router_before_xbar_ready[1+i*RouterRemapGroupSize][0+1];
      assign floo_tcdm_narrow_req_from_router_before_xbar_ready_interleaved[3+i*RouterRemapGroupSize][0+1] = floo_tcdm_narrow_req_from_router_before_xbar_ready[1+i*RouterRemapGroupSize][1+1];
      assign floo_tcdm_narrow_req_from_router_before_xbar_ready_interleaved[0+i*RouterRemapGroupSize][1+1] = floo_tcdm_narrow_req_from_router_before_xbar_ready[2+i*RouterRemapGroupSize][0+1];
      assign floo_tcdm_narrow_req_from_router_before_xbar_ready_interleaved[1+i*RouterRemapGroupSize][1+1] = floo_tcdm_narrow_req_from_router_before_xbar_ready[2+i*RouterRemapGroupSize][1+1];
      assign floo_tcdm_narrow_req_from_router_before_xbar_ready_interleaved[2+i*RouterRemapGroupSize][1+1] = floo_tcdm_narrow_req_from_router_before_xbar_ready[3+i*RouterRemapGroupSize][0+1];
      assign floo_tcdm_narrow_req_from_router_before_xbar_ready_interleaved[3+i*RouterRemapGroupSize][1+1] = floo_tcdm_narrow_req_from_router_before_xbar_ready[3+i*RouterRemapGroupSize][1+1];
    end
  endgenerate
`else
  assign floo_tcdm_narrow_req_from_router = floo_tcdm_narrow_req_from_router_interleaved;
  assign floo_tcdm_narrow_req_from_router_before_xbar_valid = floo_tcdm_narrow_req_from_router_before_xbar_valid_interleaved;
  assign floo_tcdm_narrow_req_from_router_before_xbar_ready_interleaved = floo_tcdm_narrow_req_from_router_before_xbar_ready;
`endif

if (NumTilesPerGroup == 1) begin
  assign req_tile_sel = '0;
end else begin
  for (genvar i = 0; i < NumTilesPerGroup; i++) begin : gen_req_sel_tgt_tile_i
    for (genvar j = 1; j < NumNarrowRemoteReqPortsPerTile; j++) begin : gen_req_sel_tgt_tile_j
      assign req_tile_sel[i][j] = floo_tcdm_narrow_req_from_router[i][j].hdr.tgt_addr[NumTilesPerGroupWidth-1:0];
    end : gen_req_sel_tgt_tile_j
  end : gen_req_sel_tgt_tile_i
end

tile_group_id_t     [NumNarrowRemoteReqPortsPerTile-1:1][NumTilesPerGroup-1:0] req_tile_sel_per_port;
floo_tcdm_narrow_rw_req_t[NumNarrowRemoteReqPortsPerTile-1:1][NumTilesPerGroup-1:0] floo_tcdm_narrow_req_from_router_per_port;
logic               [NumNarrowRemoteReqPortsPerTile-1:1][NumTilesPerGroup-1:0] floo_tcdm_narrow_req_from_router_before_xbar_valid_per_port;
logic               [NumNarrowRemoteReqPortsPerTile-1:1][NumTilesPerGroup-1:0] floo_tcdm_narrow_req_from_router_before_xbar_ready_per_port;
floo_tcdm_narrow_rw_req_t[NumNarrowRemoteReqPortsPerTile-1:1][NumTilesPerGroup-1:0] floo_tcdm_narrow_req_from_router_after_xbar_per_port;
logic               [NumNarrowRemoteReqPortsPerTile-1:1][NumTilesPerGroup-1:0] floo_tcdm_narrow_req_from_router_after_xbar_valid_per_port;
logic               [NumNarrowRemoteReqPortsPerTile-1:1][NumTilesPerGroup-1:0] floo_tcdm_narrow_req_from_router_after_xbar_ready_per_port;

for (genvar i = 0; i < NumTilesPerGroup; i++) begin : gen_floo_tcdm_narrow_req_from_router_per_port_i
  for (genvar j = 1; j < NumNarrowRemoteReqPortsPerTile; j++) begin : gen_floo_tcdm_narrow_req_from_router_per_port_j
    assign req_tile_sel_per_port                                 [j][i] = req_tile_sel                                          [i][j];
    assign floo_tcdm_narrow_req_from_router_per_port                    [j][i] = floo_tcdm_narrow_req_from_router                             [i][j];
    assign floo_tcdm_narrow_req_from_router_before_xbar_valid_per_port  [j][i] = floo_tcdm_narrow_req_from_router_before_xbar_valid           [i][j];
    assign floo_tcdm_narrow_req_from_router_before_xbar_ready           [i][j] = floo_tcdm_narrow_req_from_router_before_xbar_ready_per_port  [j][i];
    assign floo_tcdm_narrow_req_from_router_after_xbar                  [i][j] = floo_tcdm_narrow_req_from_router_after_xbar_per_port         [j][i];
    assign floo_tcdm_narrow_req_from_router_after_xbar_valid            [i][j] = floo_tcdm_narrow_req_from_router_after_xbar_valid_per_port   [j][i];
    assign floo_tcdm_narrow_req_from_router_after_xbar_ready_per_port   [j][i] = floo_tcdm_narrow_req_from_router_after_xbar_ready            [i][j];
  end : gen_floo_tcdm_narrow_req_from_router_per_port_j
end : gen_floo_tcdm_narrow_req_from_router_per_port_i

for (genvar i = 1; i < NumNarrowRemoteReqPortsPerTile; i++) begin : floo_tcdm_req_xbar
  stream_xbar #(
    .NumInp   (NumTilesPerGroup                                     ),
    .NumOut   (NumTilesPerGroup                                     ),
    .payload_t(floo_tcdm_narrow_rw_req_t                                 )
  ) i_local_req_interco (
    .clk_i  (clk_i                                                  ),
    .rst_ni (rst_ni                                                 ),
    .flush_i(1'b0                                                   ),
    // External priority flag
    .rr_i   ('0                                                     ),
    // Master
    .data_i (floo_tcdm_narrow_req_from_router_per_port                  [i]),
    .valid_i(floo_tcdm_narrow_req_from_router_before_xbar_valid_per_port[i]),
    .ready_o(floo_tcdm_narrow_req_from_router_before_xbar_ready_per_port[i]),
    .sel_i  (req_tile_sel_per_port                               [i]),
    // Slave
    .data_o (floo_tcdm_narrow_req_from_router_after_xbar_per_port      [i] ),
    .valid_o(floo_tcdm_narrow_req_from_router_after_xbar_valid_per_port[i] ),
    .ready_i(floo_tcdm_narrow_req_from_router_after_xbar_ready_per_port[i] ),
    .idx_o  (/* Unused, TODO?: this is the data comes from index */ )
  );
end : floo_tcdm_req_xbar

// ------------------------------------------------------------------ //
// Remapping: From FlooNoC "TCDM request" to MemPool "Slave Request"  //
// ------------------------------------------------------------------ //
for (genvar i = 0; i < NumTilesPerGroup; i++) begin : gen_router_req_to_slave_req_i
  for (genvar j = 1; j < NumNarrowRemoteReqPortsPerTile; j++) begin : gen_router_req_to_slave_req_j
    assign tcdm_slave_req[i][j] = tcdm_slave_req_t'{
      wdata: tcdm_payload_t'{
        meta_id : floo_tcdm_narrow_req_from_router_after_xbar[i][j].hdr.meta_id,  // For Register File
        core_id : floo_tcdm_narrow_req_from_router_after_xbar[i][j].hdr.core_id,  // For Core
        amo     : floo_tcdm_narrow_req_from_router_after_xbar[i][j].payload.amo,
        data    : floo_tcdm_narrow_req_from_router_after_xbar[i][j].payload.data
      },
      wen          : floo_tcdm_narrow_req_from_router_after_xbar[i][j].payload.wen,
      be           : floo_tcdm_narrow_req_from_router_after_xbar[i][j].payload.be,
      tgt_addr     : floo_tcdm_narrow_req_from_router_after_xbar[i][j].hdr.tgt_addr[
                       (NumTilesPerGroup == 1 ? 0 : NumTilesPerGroupWidth) +:
                       (idx_width(NumBanksPerTile) + TCDMAddrMemWidth)
                     ], // For TCDM Bank, remove tile offset
      ini_addr     : floo_tcdm_narrow_req_from_router_after_xbar[i][j].hdr.src_tile_id, // For Crossbar response
      src_group_id : group_id_t'({
                        floo_tcdm_narrow_req_from_router_after_xbar[i][j].hdr.src_id.x,
                        floo_tcdm_narrow_req_from_router_after_xbar[i][j].hdr.src_id.y
                      }) // For NoC Router when response back
    };
    assign tcdm_slave_req_valid[i][j]                    = floo_tcdm_narrow_req_from_router_after_xbar_valid[i][j];
    assign floo_tcdm_narrow_req_from_router_after_xbar_ready[i][j] = tcdm_slave_req_ready[i][j];
  end : gen_router_req_to_slave_req_j
end : gen_router_req_to_slave_req_i

// -------------------------------------------------------------------- //
// Remapping: From MemPool "Slave Response" to FlooNoC "TCDM Response" //
// -------------------------------------------------------------------- //
if (NocRouterRemapping == 2 || NocRouterRemapping == 3) begin: gen_resp_remapping

  floo_tcdm_narrow_rw_resp_t [NumTilesPerGroup-1:0][NumNarrowRemoteRespPortsPerTile-1:1] floo_tcdm_narrow_rw_resp_to_remapper;
  for (genvar i = 0; i < NumTilesPerGroup; i++) begin : gen_slave_resp_to_remapper_resp_i
    for (genvar j = 1; j < NumNarrowRemoteRespPortsPerTile; j++) begin : gen_slave_resp_to_remapper_resp_j
      assign floo_tcdm_narrow_rw_resp_to_remapper[i][j] = floo_tcdm_narrow_rw_resp_t'{
        payload: floo_tcdm_narrow_resp_payload_t'{
          amo  : tcdm_slave_resp[i][j].rdata.amo,
          data : tcdm_slave_resp[i][j].rdata.data
        },
        hdr: floo_tcdm_resp_meta_t'{
          meta_id : tcdm_slave_resp[i][j].rdata.meta_id,                          // For Register File
          core_id : tcdm_slave_resp[i][j].rdata.core_id,                          // For Core
          tile_id : tcdm_slave_resp[i][j].ini_addr,                               // Sender's Tile ID
          src_id  : group_xy_of(group_id_i),                         // For NoC Router
          dst_id  : group_xy_of(tcdm_slave_resp[i][j].src_group_id), // Sender's Group ID
          last    : 1'b1,                                                         // Non Burst Request
          collective_op : 1'b0                                                    // Not used
        }
      };
    end : gen_slave_resp_to_remapper_resp_j
  end : gen_slave_resp_to_remapper_resp_i

  floo_remapper #(
    .NumInp      (NumTilesPerGroup * (NumNarrowRemoteRespPortsPerTile - 1)),
    .NumOut      (NumTilesPerGroup * (NumNarrowRemoteRespPortsPerTile - 1)),
    .payload_t   (floo_tcdm_narrow_rw_resp_t                                  ),
    .GroupSize   (RouterRemapGroupSize                              ),
    .Interleaved (1'b1                                              )
  ) i_floo_tcdm_narrow_rw_resp_remapper (
    .clk_i        (clk_i                                            ),
    .rst_ni       (rst_ni                                           ),
    .inp_data_i   (floo_tcdm_narrow_rw_resp_to_remapper                       ),
    .inp_valid_i  (tcdm_slave_resp_valid                            ),
    .inp_ready_o  (tcdm_slave_resp_ready                            ),
    .oup_data_o   (floo_tcdm_narrow_rw_resp_to_router                         ),
    .oup_valid_o  (floo_tcdm_narrow_rw_resp_to_router_valid                   ),
    .oup_ready_i  (floo_tcdm_narrow_rw_resp_to_router_ready                   )
  );

end else begin: gen_resp_remapping_bypass
  for (genvar i = 0; i < NumTilesPerGroup; i++) begin : gen_slave_resp_to_router_resp_i
    for (genvar j = 1; j < NumNarrowRemoteRespPortsPerTile; j++) begin : gen_slave_resp_to_router_resp_j
      assign floo_tcdm_narrow_rw_resp_to_router[i][j] = floo_tcdm_narrow_rw_resp_t'{
        payload: floo_tcdm_narrow_resp_payload_t'{
          amo : tcdm_slave_resp[i][j].rdata.amo,
          data: tcdm_slave_resp[i][j].rdata.data
        },
        hdr: floo_tcdm_resp_meta_t'{
          meta_id : tcdm_slave_resp[i][j].rdata.meta_id,                      // For Register File
          core_id : tcdm_slave_resp[i][j].rdata.core_id,                      // For Core
          tile_id : tcdm_slave_resp[i][j].ini_addr,                           // For Crossbar when response back (Sender's Tile ID, propagated from request)
          src_id: group_xy_of(group_id_i),                         // For NoC Router when response back
          dst_id: group_xy_of(tcdm_slave_resp[i][j].src_group_id), // For NoC Router when response back (Sender's Group ID, propagated from request)
          last : 1'b1,                                                        // Non Burst Request
          collective_op : 1'b0                                                // Not used
        }
      };
    end : gen_slave_resp_to_router_resp_j
  end : gen_slave_resp_to_router_resp_i
  assign floo_tcdm_narrow_rw_resp_to_router_valid = tcdm_slave_resp_valid;
  assign tcdm_slave_resp_ready = floo_tcdm_narrow_rw_resp_to_router_ready;
end

for (genvar i = 0; i < NumTilesPerGroup; i++) begin : gen_slave_resp_to_router_vc_resp_i
  for (genvar j = 1; j < NumNarrowRemoteRespPortsPerTile; j++) begin : gen_slave_resp_to_router_vc_resp_j
    assign floo_tcdm_narrow_rw_resp_to_router_vc_valid[i][j] = floo_tcdm_narrow_rw_resp_to_router_valid[i][j];
    assign floo_tcdm_narrow_rw_resp_to_router_ready[i][j] = floo_tcdm_narrow_rw_resp_to_router_vc_ready[i][j];
  end : gen_slave_resp_to_router_vc_resp_j
end : gen_slave_resp_to_router_vc_resp_i

// ------------------------------------------------------------------ //
// Crossbar: FlooNoC "TCDM reponse" input select target tile          //
// TODO: This is a consitent assignment to Tile's port,               //
//       Should be improved for confict resolution.                   //
// ------------------------------------------------------------------ //
tile_group_id_t [NumTilesPerGroup-1:0][NumNarrowRemoteRespPortsPerTile-1:1] resp_tile_sel;
floo_tcdm_narrow_rw_resp_t[NumTilesPerGroup-1:0][NumNarrowRemoteRespPortsPerTile-1:1] floo_tcdm_narrow_rw_resp_from_router_after_xbar;
logic           [NumTilesPerGroup-1:0][NumNarrowRemoteRespPortsPerTile-1:1] floo_tcdm_narrow_rw_resp_from_router_after_xbar_valid;
logic           [NumTilesPerGroup-1:0][NumNarrowRemoteRespPortsPerTile-1:1] floo_tcdm_narrow_rw_resp_from_router_after_xbar_ready;
logic           [NumTilesPerGroup-1:0][NumNarrowRemoteRespPortsPerTile-1:1] floo_tcdm_narrow_rw_resp_from_router_before_xbar_valid;
logic           [NumTilesPerGroup-1:0][NumNarrowRemoteRespPortsPerTile-1:1] floo_tcdm_narrow_rw_resp_from_router_before_xbar_ready;
floo_tcdm_narrow_rw_resp_t[NumTilesPerGroup-1:0][NumNarrowRemoteRespPortsPerTile-1:1] floo_tcdm_narrow_rw_resp_from_router_vc;
logic           [NumTilesPerGroup-1:0][NumNarrowRemoteRespPortsPerTile-1:1] floo_tcdm_narrow_rw_resp_from_router_vc_valid;
logic           [NumTilesPerGroup-1:0][NumNarrowRemoteRespPortsPerTile-1:1] floo_tcdm_narrow_rw_resp_from_router_vc_ready;

if (NumTilesPerGroup == 1) begin
  assign resp_tile_sel = '0;
end else begin
  for (genvar i = 0; i < NumTilesPerGroup; i++) begin : gen_resp_sel_tgt_tile_i
    for (genvar j = 1; j < NumNarrowRemoteRespPortsPerTile; j++) begin : gen_resp_sel_tgt_tile_j
      assign resp_tile_sel[i][j] = floo_tcdm_narrow_rw_resp_from_router[i][j].hdr.tile_id;
    end : gen_resp_sel_tgt_tile_j
  end : gen_resp_sel_tgt_tile_i
end

tile_group_id_t [NumNarrowRemoteRespPortsPerTile-1:1][NumTilesPerGroup-1:0] resp_tile_sel_per_port;
floo_tcdm_narrow_rw_resp_t[NumNarrowRemoteRespPortsPerTile-1:1][NumTilesPerGroup-1:0] floo_tcdm_narrow_rw_resp_from_router_per_port;
logic           [NumNarrowRemoteRespPortsPerTile-1:1][NumTilesPerGroup-1:0] floo_tcdm_narrow_rw_resp_from_router_before_xbar_valid_per_port;
logic           [NumNarrowRemoteRespPortsPerTile-1:1][NumTilesPerGroup-1:0] floo_tcdm_narrow_rw_resp_from_router_before_xbar_ready_per_port;

floo_tcdm_narrow_rw_resp_t[NumNarrowRemoteRespPortsPerTile-1:1][NumTilesPerGroup-1:0] floo_tcdm_narrow_rw_resp_from_router_after_xbar_per_port;
logic           [NumNarrowRemoteRespPortsPerTile-1:1][NumTilesPerGroup-1:0] floo_tcdm_narrow_rw_resp_from_router_after_xbar_valid_per_port;
logic           [NumNarrowRemoteRespPortsPerTile-1:1][NumTilesPerGroup-1:0] floo_tcdm_narrow_rw_resp_from_router_after_xbar_ready_per_port;

for (genvar i = 0; i < NumTilesPerGroup; i++) begin : gen_floo_tcdm_narrow_rw_resp_from_router_per_port_i
  for (genvar j = 1; j < NumNarrowRemoteRespPortsPerTile; j++) begin : gen_floo_tcdm_narrow_rw_resp_from_router_per_port_j
    assign resp_tile_sel_per_port                                 [j][i] = resp_tile_sel                                          [i][j];
    assign floo_tcdm_narrow_rw_resp_from_router_per_port                    [j][i] = floo_tcdm_narrow_rw_resp_from_router                             [i][j];
    assign floo_tcdm_narrow_rw_resp_from_router_before_xbar_valid_per_port  [j][i] = floo_tcdm_narrow_rw_resp_from_router_before_xbar_valid           [i][j];
    assign floo_tcdm_narrow_rw_resp_from_router_before_xbar_ready           [i][j] = floo_tcdm_narrow_rw_resp_from_router_before_xbar_ready_per_port  [j][i];

    assign floo_tcdm_narrow_rw_resp_from_router_after_xbar                  [i][j] = floo_tcdm_narrow_rw_resp_from_router_after_xbar_per_port         [j][i];
    assign floo_tcdm_narrow_rw_resp_from_router_after_xbar_valid            [i][j] = floo_tcdm_narrow_rw_resp_from_router_after_xbar_valid_per_port   [j][i];
    assign floo_tcdm_narrow_rw_resp_from_router_after_xbar_ready_per_port   [j][i] = floo_tcdm_narrow_rw_resp_from_router_after_xbar_ready            [i][j];
  end : gen_floo_tcdm_narrow_rw_resp_from_router_per_port_j
end : gen_floo_tcdm_narrow_rw_resp_from_router_per_port_i

for (genvar i = 1; i < NumNarrowRemoteRespPortsPerTile; i++) begin : floo_tcdm_narrow_rw_resp_xbar
  stream_xbar #(
    .NumInp   (NumTilesPerGroup                                    ),
    .NumOut   (NumTilesPerGroup                                    ),
    .payload_t(floo_tcdm_narrow_rw_resp_t                                    )
  ) i_local_resp_interco (
    .clk_i  (clk_i                                                 ),
    .rst_ni (rst_ni                                                ),
    .flush_i(1'b0                                                  ),
    // External priority flag
    .rr_i   ('0                                                    ),
    // Master
    .data_i (floo_tcdm_narrow_rw_resp_from_router_per_port                    [i] ),
    .valid_i(floo_tcdm_narrow_rw_resp_from_router_before_xbar_valid_per_port  [i] ),
    .ready_o(floo_tcdm_narrow_rw_resp_from_router_before_xbar_ready_per_port  [i] ),
    .sel_i  (resp_tile_sel_per_port                                 [i] ),
    // Slave
    .data_o (floo_tcdm_narrow_rw_resp_from_router_after_xbar_per_port         [i] ),
    .valid_o(floo_tcdm_narrow_rw_resp_from_router_after_xbar_valid_per_port   [i] ),
    .ready_i(floo_tcdm_narrow_rw_resp_from_router_after_xbar_ready_per_port   [i] ),
    .idx_o  (/* Unused, TODO?: this is the data comes from index */     )
  );
end : floo_tcdm_narrow_rw_resp_xbar

// --------------------------------------------------------------------- //
// Remapping: From FlooNoC "TCDM response" to MemPool "Master Response"  //
// --------------------------------------------------------------------- //
for (genvar i = 0; i < NumTilesPerGroup; i++) begin : gen_router_resp_to_master_resp_i
  for (genvar j = 1; j < NumNarrowRemoteRespPortsPerTile; j++) begin : gen_router_resp_to_master_resp_j
    assign tcdm_master_resp[i][j] = tcdm_master_resp_t'{
      rdata: tcdm_payload_t'{
        meta_id : floo_tcdm_narrow_rw_resp_from_router_after_xbar[i][j].hdr.meta_id, // For Register File
        core_id : floo_tcdm_narrow_rw_resp_from_router_after_xbar[i][j].hdr.core_id, // For Core
        amo     : floo_tcdm_narrow_rw_resp_from_router_after_xbar[i][j].payload.amo,
        data    : floo_tcdm_narrow_rw_resp_from_router_after_xbar[i][j].payload.data
      }
    };
    assign tcdm_master_resp_valid[i][j] = floo_tcdm_narrow_rw_resp_from_router_after_xbar_valid[i][j];
    assign floo_tcdm_narrow_rw_resp_from_router_after_xbar_ready[i][j] = tcdm_master_resp_ready[i][j];
  end : gen_router_resp_to_master_resp_j
end : gen_router_resp_to_master_resp_i

// ------------------------------------------------------------------ //
// ----------------------       Router      --------------------------//
// ------------------------------------------------------------------ //
// Narrow RO request NoC
`ifdef USE_RO_NARROW_REQ_CHANNEL
floo_tcdm_narrow_ro_req_t    [NumTilesPerGroup-1:0][NumNarrowRoReqPortsPerTile-1:0][West:North] floo_tcdm_narrow_ro_req_out_trans;
logic                 [NumTilesPerGroup-1:0][NumNarrowRoReqPortsPerTile-1:0][West:North] floo_tcdm_narrow_ro_req_valid_out_trans;
logic                 [NumTilesPerGroup-1:0][NumNarrowRoReqPortsPerTile-1:0][West:North] floo_tcdm_narrow_ro_req_ready_in_trans;
floo_tcdm_narrow_ro_req_t    [NumTilesPerGroup-1:0][NumNarrowRoReqPortsPerTile-1:0][West:North] floo_tcdm_narrow_ro_req_in_trans;
logic                 [NumTilesPerGroup-1:0][NumNarrowRoReqPortsPerTile-1:0][West:North] floo_tcdm_narrow_ro_req_valid_in_trans;
logic                 [NumTilesPerGroup-1:0][NumNarrowRoReqPortsPerTile-1:0][West:North] floo_tcdm_narrow_ro_req_ready_out_trans;
`endif

// Narrow payload-bearing RW/WO request NoC
floo_tcdm_narrow_rw_req_t  [NumTilesPerGroup-1:0][NumNarrowPayloadReqPortsPerTile-1:0][West:North] floo_tcdm_narrow_rw_req_out_trans;
logic                 [NumTilesPerGroup-1:0][NumNarrowPayloadReqPortsPerTile-1:0][West:North] floo_tcdm_narrow_rw_req_valid_out_trans;
logic                 [NumTilesPerGroup-1:0][NumNarrowPayloadReqPortsPerTile-1:0][West:North] floo_tcdm_narrow_rw_req_ready_in_trans;
floo_tcdm_narrow_rw_req_t  [NumTilesPerGroup-1:0][NumNarrowPayloadReqPortsPerTile-1:0][West:North] floo_tcdm_narrow_rw_req_in_trans;
logic                 [NumTilesPerGroup-1:0][NumNarrowPayloadReqPortsPerTile-1:0][West:North] floo_tcdm_narrow_rw_req_valid_in_trans;
logic                 [NumTilesPerGroup-1:0][NumNarrowPayloadReqPortsPerTile-1:0][West:North] floo_tcdm_narrow_rw_req_ready_out_trans;


floo_tcdm_narrow_rw_resp_t      [NumTilesPerGroup-1:0][NumNarrowRemoteRespPortsPerTile-1:1][West:North] floo_tcdm_narrow_rw_resp_out_trans;
logic                 [NumTilesPerGroup-1:0][NumNarrowRemoteRespPortsPerTile-1:1][West:North] floo_tcdm_narrow_rw_resp_valid_out_trans;
logic                 [NumTilesPerGroup-1:0][NumNarrowRemoteRespPortsPerTile-1:1][West:North] floo_tcdm_narrow_rw_resp_ready_in_trans;
floo_tcdm_narrow_rw_resp_t      [NumTilesPerGroup-1:0][NumNarrowRemoteRespPortsPerTile-1:1][West:North] floo_tcdm_narrow_rw_resp_in_trans;
logic                 [NumTilesPerGroup-1:0][NumNarrowRemoteRespPortsPerTile-1:1][West:North] floo_tcdm_narrow_rw_resp_valid_in_trans;
logic                 [NumTilesPerGroup-1:0][NumNarrowRemoteRespPortsPerTile-1:1][West:North] floo_tcdm_narrow_rw_resp_ready_out_trans;

for (genvar i = 0; i < NumTilesPerGroup; i++) begin : gen_router_router_connection_i
  `ifdef USE_RO_NARROW_REQ_CHANNEL
    for (genvar j = 0; j < NumNarrowRoReqPortsPerTile; j++) begin : gen_router_router_narrow_ro_req_connection_j
      for (genvar k = North; k <= West; k++) begin: gen_router_router_narrow_ro_req_connection_k
        assign floo_tcdm_narrow_ro_req_out              [k][i][j] = floo_tcdm_narrow_ro_req_out_trans       [i][j][k];
        assign floo_tcdm_narrow_ro_req_valid_out        [k][i][j] = floo_tcdm_narrow_ro_req_valid_out_trans [i][j][k];
        assign floo_tcdm_narrow_ro_req_ready_out        [k][i][j] = floo_tcdm_narrow_ro_req_ready_out_trans [i][j][k];
        assign floo_tcdm_narrow_ro_req_in_trans         [i][j][k] = floo_tcdm_narrow_ro_req_in              [k][i][j];
        assign floo_tcdm_narrow_ro_req_ready_in_trans   [i][j][k] = floo_tcdm_narrow_ro_req_ready_in        [k][i][j];
        assign floo_tcdm_narrow_ro_req_valid_in_trans   [i][j][k] = floo_tcdm_narrow_ro_req_valid_in        [k][i][j];
      end : gen_router_router_narrow_ro_req_connection_k
    end : gen_router_router_narrow_ro_req_connection_j
  `endif

  for (genvar j = 0; j < NumNarrowPayloadReqPortsPerTile; j++) begin : gen_router_router_narrow_rw_req_connection_j
    for (genvar k = North; k <= West; k++) begin: gen_router_router_narrow_rw_req_connection_k
      assign floo_tcdm_narrow_rw_req_out                [k][i][j] = floo_tcdm_narrow_rw_req_out_trans         [i][j][k];
      assign floo_tcdm_narrow_rw_req_valid_out          [k][i][j] = floo_tcdm_narrow_rw_req_valid_out_trans   [i][j][k];
      assign floo_tcdm_narrow_rw_req_ready_out          [k][i][j] = floo_tcdm_narrow_rw_req_ready_out_trans   [i][j][k];
      assign floo_tcdm_narrow_rw_req_in_trans           [i][j][k] = floo_tcdm_narrow_rw_req_in                [k][i][j];
      assign floo_tcdm_narrow_rw_req_ready_in_trans     [i][j][k] = floo_tcdm_narrow_rw_req_ready_in          [k][i][j];
      assign floo_tcdm_narrow_rw_req_valid_in_trans     [i][j][k] = floo_tcdm_narrow_rw_req_valid_in          [k][i][j];
    end : gen_router_router_narrow_rw_req_connection_k
  end : gen_router_router_narrow_rw_req_connection_j

  for (genvar j = 1; j < NumNarrowRemoteRespPortsPerTile; j++) begin : gen_router_router_narrow_rw_resp_connection_j
    for (genvar k = North; k <= West; k++) begin: gen_router_router_narrow_rw_resp_connection_k
      assign floo_tcdm_narrow_rw_resp_out              [k][i][j] = floo_tcdm_narrow_rw_resp_out_trans       [i][j][k];
      assign floo_tcdm_narrow_rw_resp_valid_out        [k][i][j] = floo_tcdm_narrow_rw_resp_valid_out_trans [i][j][k];
      assign floo_tcdm_narrow_rw_resp_ready_out        [k][i][j] = floo_tcdm_narrow_rw_resp_ready_out_trans [i][j][k];

      assign floo_tcdm_narrow_rw_resp_in_trans         [i][j][k] = floo_tcdm_narrow_rw_resp_in              [k][i][j];
      assign floo_tcdm_narrow_rw_resp_ready_in_trans   [i][j][k] = floo_tcdm_narrow_rw_resp_ready_in        [k][i][j];
      assign floo_tcdm_narrow_rw_resp_valid_in_trans   [i][j][k] = floo_tcdm_narrow_rw_resp_valid_in        [k][i][j];
    end : gen_router_router_narrow_rw_resp_connection_k
  end : gen_router_router_narrow_rw_resp_connection_j
end : gen_router_router_connection_i

group_xy_id_t group_xy_id;
assign group_xy_id = group_xy_of(group_id_i);

for (genvar i = 0; i < NumTilesPerGroup; i++) begin : gen_router_router_i
`ifdef USE_RO_NARROW_REQ_CHANNEL
  for (genvar j = 0; j < NumNarrowRoReqPortsPerTile; j++) begin : gen_router_narrow_ro_req_router_j
    if (NocRoutingAlgorithm == 1) begin: gen_id_table_routing
      floo_router #(
        .NumRoutes        (mempool_pkg::NumDirections                         ),
        .NumVirtChannels  (1                                                  ),
        .flit_t           (floo_tcdm_narrow_ro_req_t                                 ),
        .InFifoDepth      (mempool_pkg::NumNarrowReqRouterInFifoDepth                  ), // Input buffer depth
        .OutFifoDepth     (mempool_pkg::NumNarrowReqRouterOutFifoDepth                 ), // Output buffer depth, can try to set it to 0 for -1 cycle latency
        .RouteAlgo        (IdTable                                            ),
        .VcImpl           (VcPreemptValid                                     ), // suppress StableValid warning
        .id_t             (routing_table_pkg::routing_rule_addr_t             ),
        .NumAddrRules     (NumGroups                                          ),
        .addr_rule_t      (routing_table_pkg::routing_rule_t                  )
      ) i_floo_tcdm_narrow_ro_req_router (
        .clk_i,
        .rst_ni,
        .test_enable_i  (1'b0                                                                                     ),
        .xy_id_i        ('0                                                                                       ),
        .id_route_map_i (routing_table_pkg::RoutingTables[group_xy_id.x][group_xy_id.y]                           ),
        .valid_i        ({floo_tcdm_narrow_ro_req_to_router_vc_valid[i][j],   floo_tcdm_narrow_ro_req_valid_in_trans[i][j]} ),
        .ready_o        ({floo_tcdm_narrow_ro_req_to_router_vc_ready[i][j],   floo_tcdm_narrow_ro_req_ready_out_trans[i][j]}),
        .data_i         ({floo_tcdm_narrow_ro_req_to_router[i][j],            floo_tcdm_narrow_ro_req_in_trans      [i][j]} ),
        .valid_o        ({floo_tcdm_narrow_ro_req_from_router_vc_valid[i][j], floo_tcdm_narrow_ro_req_valid_out_trans[i][j]}),
        .ready_i        ({floo_tcdm_narrow_ro_req_from_router_vc_ready[i][j], floo_tcdm_narrow_ro_req_ready_in_trans[i][j]} ),
        .data_o         ({floo_tcdm_narrow_ro_req_from_router_vc[i][j],       floo_tcdm_narrow_ro_req_out_trans      [i][j]}),
        .credit_o       (                                                                                          ),
        .credit_i       ('0                                                                                        ),
        .offload_req_o  (                                                                                          ),
        .offload_rsp_i  ('0                                                                                        )
      );
    end else begin: gen_xy_routing
      floo_router #(
        .NumRoutes        (mempool_pkg::NumDirections                         ),
        .NumVirtChannels  (1                                                  ),
        .flit_t           (floo_tcdm_narrow_ro_req_t                                 ),
        .InFifoDepth      (mempool_pkg::NumNarrowReqRouterInFifoDepth                  ), // Input buffer depth
        .OutFifoDepth     (mempool_pkg::NumNarrowReqRouterOutFifoDepth                 ), // Output buffer depth, can try to set it to 0 for -1 cycle latency
        .RouteAlgo        (XYRouting                                          ),
        .VcImpl           (VcPreemptValid                                     ), // suppress StableValid warning
        .id_t             (group_xy_id_t                                      ),
        .NumAddrRules     (1                                                  ),
        .addr_rule_t      (logic                                              )
      ) i_floo_tcdm_narrow_ro_req_router (
        .clk_i,
        .rst_ni,
        .test_enable_i  (1'b0                                                                                     ),
        .xy_id_i        (group_xy_id                                                                              ),
        .id_route_map_i ('0                                                                                       ),
        .valid_i        ({floo_tcdm_narrow_ro_req_to_router_vc_valid[i][j],   floo_tcdm_narrow_ro_req_valid_in_trans[i][j]} ),
        .ready_o        ({floo_tcdm_narrow_ro_req_to_router_vc_ready[i][j],   floo_tcdm_narrow_ro_req_ready_out_trans[i][j]}),
        .data_i         ({floo_tcdm_narrow_ro_req_to_router[i][j],            floo_tcdm_narrow_ro_req_in_trans      [i][j]} ),
        .valid_o        ({floo_tcdm_narrow_ro_req_from_router_vc_valid[i][j], floo_tcdm_narrow_ro_req_valid_out_trans[i][j]}),
        .ready_i        ({floo_tcdm_narrow_ro_req_from_router_vc_ready[i][j], floo_tcdm_narrow_ro_req_ready_in_trans[i][j]} ),
        .data_o         ({floo_tcdm_narrow_ro_req_from_router_vc[i][j],       floo_tcdm_narrow_ro_req_out_trans      [i][j]}),
        .credit_o       (                                                                                          ),
        .credit_i       ('0                                                                                        ),
        .offload_req_o  (                                                                                          ),
        .offload_rsp_i  ('0                                                                                        )
      );
    end
    assign floo_tcdm_narrow_ro_req_from_router[i][j] = floo_tcdm_narrow_ro_req_from_router_vc[i][j];
    assign floo_tcdm_narrow_req_from_router_before_xbar_valid_interleaved[i][j+(1)] = floo_tcdm_narrow_ro_req_from_router_vc_valid[i][j];
    assign floo_tcdm_narrow_ro_req_from_router_vc_ready[i][j] = floo_tcdm_narrow_req_from_router_before_xbar_ready_interleaved[i][j+(1)];
  end : gen_router_narrow_ro_req_router_j
`endif

  for (genvar j = 0; j < NumNarrowPayloadReqPortsPerTile; j++) begin : gen_router_narrow_rw_req_router_j
    if (NocRoutingAlgorithm == 1) begin: gen_id_table_routing
      floo_router #(
        .NumRoutes        (mempool_pkg::NumDirections                         ),
        .NumVirtChannels  (1                                                  ),
        .flit_t           (floo_tcdm_narrow_rw_req_t                               ),
        .InFifoDepth      (mempool_pkg::NumNarrowReqRouterInFifoDepth                  ), // Input buffer depth
        .OutFifoDepth     (mempool_pkg::NumNarrowReqRouterOutFifoDepth                 ), // Output buffer depth, can try to set it to 0 for -1 cycle latency
        .RouteAlgo        (IdTable                                            ),
        .VcImpl           (VcPreemptValid                                     ), // suppress StableValid warning
        .id_t             (routing_table_pkg::routing_rule_addr_t             ),
        .NumAddrRules     (NumGroups                                          ),
        .addr_rule_t      (routing_table_pkg::routing_rule_t                  )
      ) i_floo_tcdm_narrow_rw_req_router (
        .clk_i,
        .rst_ni,
        .test_enable_i  (1'b0                                                                                     ),
        .xy_id_i        ('0                                                                                       ),
        .id_route_map_i (routing_table_pkg::RoutingTables[group_xy_id.x][group_xy_id.y]                           ),
        .valid_i        ({floo_tcdm_narrow_rw_req_to_router_vc_valid[i][j],   floo_tcdm_narrow_rw_req_valid_in_trans[i][j]} ),
        .ready_o        ({floo_tcdm_narrow_rw_req_to_router_vc_ready[i][j],   floo_tcdm_narrow_rw_req_ready_out_trans[i][j]}),
        .data_i         ({floo_tcdm_narrow_rw_req_to_router[i][j],            floo_tcdm_narrow_rw_req_in_trans      [i][j]} ),
        .valid_o        ({floo_tcdm_narrow_rw_req_from_router_vc_valid[i][j], floo_tcdm_narrow_rw_req_valid_out_trans[i][j]}),
        .ready_i        ({floo_tcdm_narrow_rw_req_from_router_vc_ready[i][j], floo_tcdm_narrow_rw_req_ready_in_trans[i][j]} ),
        .data_o         ({floo_tcdm_narrow_rw_req_from_router_vc[i][j],       floo_tcdm_narrow_rw_req_out_trans      [i][j]}),
        .credit_o       (                                                                                          ),
        .credit_i       ('0                                                                                        ),
        .offload_req_o  (                                                                                          ),
        .offload_rsp_i  ('0                                                                                        )
      );
    end else begin: gen_xy_routing
      floo_router #(
        .NumRoutes        (mempool_pkg::NumDirections                         ),
        .NumVirtChannels  (1                                                  ),
        .flit_t           (floo_tcdm_narrow_rw_req_t                               ),
        .InFifoDepth      (mempool_pkg::NumNarrowReqRouterInFifoDepth                  ), // Input buffer depth
        .OutFifoDepth     (mempool_pkg::NumNarrowReqRouterOutFifoDepth                 ), // Output buffer depth, can try to set it to 0 for -1 cycle latency
        .RouteAlgo        (XYRouting                                          ),
        .VcImpl           (VcPreemptValid                                     ), // suppress StableValid warning
        .id_t             (group_xy_id_t                                      ),
        .NumAddrRules     (1                                                  ),
        .addr_rule_t      (logic                                              )
      ) i_floo_tcdm_narrow_rw_req_router (
        .clk_i,
        .rst_ni,
        .test_enable_i  (1'b0                                                                                     ),
        .xy_id_i        (group_xy_id                                                                              ),
        .id_route_map_i ('0                                                                                       ),
        .valid_i        ({floo_tcdm_narrow_rw_req_to_router_vc_valid[i][j],   floo_tcdm_narrow_rw_req_valid_in_trans[i][j]} ),
        .ready_o        ({floo_tcdm_narrow_rw_req_to_router_vc_ready[i][j],   floo_tcdm_narrow_rw_req_ready_out_trans[i][j]}),
        .data_i         ({floo_tcdm_narrow_rw_req_to_router[i][j],            floo_tcdm_narrow_rw_req_in_trans      [i][j]} ),
        .valid_o        ({floo_tcdm_narrow_rw_req_from_router_vc_valid[i][j], floo_tcdm_narrow_rw_req_valid_out_trans[i][j]}),
        .ready_i        ({floo_tcdm_narrow_rw_req_from_router_vc_ready[i][j], floo_tcdm_narrow_rw_req_ready_in_trans[i][j]} ),
        .data_o         ({floo_tcdm_narrow_rw_req_from_router_vc[i][j],       floo_tcdm_narrow_rw_req_out_trans      [i][j]}),
        .credit_o       (                                                                                          ),
        .credit_i       ('0                                                                                        ),
        .offload_req_o  (                                                                                          ),
        .offload_rsp_i  ('0                                                                                        )
      );
    end
    assign floo_tcdm_narrow_rw_req_from_router[i][j] = floo_tcdm_narrow_rw_req_from_router_vc[i][j];
    assign floo_tcdm_narrow_req_from_router_before_xbar_valid_interleaved[i][j+(1+NumNarrowRoReqPortsPerTile)] = floo_tcdm_narrow_rw_req_from_router_vc_valid[i][j];
    assign floo_tcdm_narrow_rw_req_from_router_vc_ready[i][j] = floo_tcdm_narrow_req_from_router_before_xbar_ready_interleaved[i][j+(1+NumNarrowRoReqPortsPerTile)];
  end : gen_router_narrow_rw_req_router_j

  for (genvar j = 1; j < NumNarrowRemoteRespPortsPerTile; j++) begin : gen_router_narrow_rw_resp_router_j
    if (NocRoutingAlgorithm == 1) begin: gen_id_table_routing
      floo_router #(
        .NumRoutes        (mempool_pkg::NumDirections                         ),
        .NumVirtChannels  (1                                                  ),
        .flit_t           (floo_tcdm_narrow_rw_resp_t                                   ),
        .InFifoDepth      (mempool_pkg::NumNarrowRespRouterInFifoDepth                  ), // Input buffer depth
        .OutFifoDepth     (mempool_pkg::NumNarrowRespRouterOutFifoDepth                 ), // Output buffer depth, can try to set it to 0 for -1 cycle latency
        .RouteAlgo        (IdTable                                            ),
        .VcImpl           (VcPreemptValid                                     ), // suppress StableValid warning
        .id_t             (routing_table_pkg::routing_rule_addr_t             ),
        .NumAddrRules     (NumGroups                                          ),
        .addr_rule_t      (routing_table_pkg::routing_rule_t                  )
      ) i_floo_tcdm_narrow_rw_resp_router (
        .clk_i,
        .rst_ni,
        .test_enable_i  (1'b0                                                                             ),
        .xy_id_i        ('0                                                                               ),
        .id_route_map_i (routing_table_pkg::RoutingTables[group_xy_id.x][group_xy_id.y]                   ),
        .valid_i        ({floo_tcdm_narrow_rw_resp_to_router_vc_valid[i][j],   floo_tcdm_narrow_rw_resp_valid_in_trans[i][j]} ),
        .ready_o        ({floo_tcdm_narrow_rw_resp_to_router_vc_ready[i][j],   floo_tcdm_narrow_rw_resp_ready_out_trans[i][j]}),
        .data_i         ({floo_tcdm_narrow_rw_resp_to_router[i][j],            floo_tcdm_narrow_rw_resp_in_trans[i][j]}       ),
        .valid_o        ({floo_tcdm_narrow_rw_resp_from_router_vc_valid[i][j], floo_tcdm_narrow_rw_resp_valid_out_trans[i][j]}),
        .ready_i        ({floo_tcdm_narrow_rw_resp_from_router_vc_ready[i][j], floo_tcdm_narrow_rw_resp_ready_in_trans[i][j]} ),
        .data_o         ({floo_tcdm_narrow_rw_resp_from_router_vc[i][j],       floo_tcdm_narrow_rw_resp_out_trans[i][j]}      ),
        .credit_o       (                                                                                  ),
        .credit_i       ('0                                                                                ),
        .offload_req_o  (                                                                                  ),
        .offload_rsp_i  ('0                                                                                )
      );
    end else begin: gen_xy_routing
      floo_router #(
        .NumRoutes        (mempool_pkg::NumDirections                         ),
        .NumVirtChannels  (1                                                  ),
        .flit_t           (floo_tcdm_narrow_rw_resp_t                                   ),
        .InFifoDepth      (mempool_pkg::NumNarrowRespRouterInFifoDepth                  ), // Input buffer depth
        .OutFifoDepth     (mempool_pkg::NumNarrowRespRouterOutFifoDepth                 ), // Output buffer depth, can try to set it to 0 for -1 cycle latency
        .RouteAlgo        (XYRouting                                          ),
        .VcImpl           (VcPreemptValid                                     ), // suppress StableValid warning
        .id_t             (group_xy_id_t                                      ),
        .NumAddrRules     (1                                                  ),
        .addr_rule_t      (logic                                              )
      ) i_floo_tcdm_narrow_rw_resp_router (
        .clk_i,
        .rst_ni,
        .test_enable_i  (1'b0                                                                             ),
        .xy_id_i        (group_xy_id                                                                      ),
        .id_route_map_i ('0                                                                               ),
        .valid_i        ({floo_tcdm_narrow_rw_resp_to_router_vc_valid[i][j],   floo_tcdm_narrow_rw_resp_valid_in_trans[i][j]} ),
        .ready_o        ({floo_tcdm_narrow_rw_resp_to_router_vc_ready[i][j],   floo_tcdm_narrow_rw_resp_ready_out_trans[i][j]}),
        .data_i         ({floo_tcdm_narrow_rw_resp_to_router[i][j],            floo_tcdm_narrow_rw_resp_in_trans[i][j]}       ),
        .valid_o        ({floo_tcdm_narrow_rw_resp_from_router_vc_valid[i][j], floo_tcdm_narrow_rw_resp_valid_out_trans[i][j]}),
        .ready_i        ({floo_tcdm_narrow_rw_resp_from_router_vc_ready[i][j], floo_tcdm_narrow_rw_resp_ready_in_trans[i][j]} ),
        .data_o         ({floo_tcdm_narrow_rw_resp_from_router_vc[i][j],       floo_tcdm_narrow_rw_resp_out_trans[i][j]}      ),
        .credit_o       (                                                                                  ),
        .credit_i       ('0                                                                                ),
        .offload_req_o  (                                                                                  ),
        .offload_rsp_i  ('0                                                                                )
      );
    end
    assign floo_tcdm_narrow_rw_resp_from_router[i][j] = floo_tcdm_narrow_rw_resp_from_router_vc[i][j];
    assign floo_tcdm_narrow_rw_resp_from_router_before_xbar_valid[i][j] = floo_tcdm_narrow_rw_resp_from_router_vc_valid[i][j];
    assign floo_tcdm_narrow_rw_resp_from_router_vc_ready[i][j] = floo_tcdm_narrow_rw_resp_from_router_before_xbar_ready[i][j];
  end : gen_router_narrow_rw_resp_router_j
end : gen_router_router_i

// ====================================================================== //
// ----   Wide TCDM plane (WP2): separate single-beat inter-group net  --- //
// ====================================================================== //
// A physically-wide (WIDE_W) FlooNoC net that carries RedMulE lines and Spatz
// bursts as one flit, built like the narrow plane above: every RO request and
// RW response port of every tile owns a router, an optional floo_remapper
// rotates port p of a group of tiles, and a crossbar per port delivers what
// arrives to the addressed tile. Writes are rare, so the WO request and WO
// response ports of NumWideWoMerge tiles share one router, through an arbiter
// on the way out and a crossbar to the addressed tile on the way in. It rides
// the same boundary bundle as the 32-bit net, whole-struct wired group-to-group
// by the cluster wrapper, but keeps its own routers, valid/ready and FIFOs.
`ifdef USE_WIDE_TCDM_PLANE
  localparam int unsigned WideRoBase     = NumWideIntraGroupPortsPerTile;
  localparam int unsigned WideWoBase     = WideRoBase + NumWideRoReqPortsPerTile;
  localparam int unsigned WideRwBase     = NumWideIntraGroupPortsPerTile;
  localparam int unsigned WideWoRespBase = WideRwBase + NumWideRwRespPortsPerTile;
  localparam int unsigned WideTileSelW   = (NumTilesPerGroup == 1) ? 1 : NumTilesPerGroupWidth;

  // ======================= RO request ports =======================
  floo_tcdm_wide_ro_req_t [NumTilesPerGroup-1:0][NumWideRoReqPortsPerTile-1:0]
      floo_tcdm_wide_ro_req_to_router, floo_tcdm_wide_ro_req_from_router;
  logic [NumTilesPerGroup-1:0][NumWideRoReqPortsPerTile-1:0]
      floo_tcdm_wide_ro_req_to_router_valid, floo_tcdm_wide_ro_req_to_router_ready;
  logic [NumTilesPerGroup-1:0][NumWideRoReqPortsPerTile-1:0]
      floo_tcdm_wide_ro_req_from_router_valid, floo_tcdm_wide_ro_req_from_router_ready;
  floo_tcdm_wide_ro_req_t [NumTilesPerGroup-1:0][NumWideRoReqPortsPerTile-1:0][West:North]
      floo_tcdm_wide_ro_req_out_trans, floo_tcdm_wide_ro_req_in_trans;
  logic [NumTilesPerGroup-1:0][NumWideRoReqPortsPerTile-1:0][West:North]
      floo_tcdm_wide_ro_req_valid_out_trans, floo_tcdm_wide_ro_req_ready_in_trans;
  logic [NumTilesPerGroup-1:0][NumWideRoReqPortsPerTile-1:0][West:North]
      floo_tcdm_wide_ro_req_valid_in_trans, floo_tcdm_wide_ro_req_ready_out_trans;
  floo_tcdm_wide_ro_req_t [NumTilesPerGroup-1:0][NumWideRoReqPortsPerTile-1:0]
      floo_tcdm_wide_ro_req_to_remapper;
  logic [NumTilesPerGroup-1:0][NumWideRoReqPortsPerTile-1:0]
      floo_tcdm_wide_ro_req_to_remapper_valid, floo_tcdm_wide_ro_req_to_remapper_ready;

  for (genvar t = 0; t < NumTilesPerGroup; t++) begin : gen_wide_ro_req_packet_t
    for (genvar k = 0; k < NumWideRoReqPortsPerTile; k++) begin : gen_wide_ro_req_packet_k
      assign floo_tcdm_wide_ro_req_to_remapper[t][k] = '{
        hdr: '{
          meta_id:       tcdm_wide_master_req[t][WideRoBase+k].wdata.meta_id,
          core_id:       tcdm_wide_master_req[t][WideRoBase+k].wdata.core_id,
          src_tile_id:   tcdm_wide_master_req[t][WideRoBase+k].src_tile_id,
          src_id:        group_xy_of(group_id_i),
          dst_id:        group_xy_of(tcdm_wide_master_req[t][WideRoBase+k].tgt_group_id),
          tgt_addr:      tcdm_wide_master_req[t][WideRoBase+k].tgt_addr,
          last:          1'b1,
          collective_op: 1'b0
        }
      };
      assign floo_tcdm_wide_ro_req_to_remapper_valid[t][k] =
          tcdm_wide_master_req_valid[t][WideRoBase+k];
      assign tcdm_wide_master_req_ready[t][WideRoBase+k] =
          floo_tcdm_wide_ro_req_to_remapper_ready[t][k];
    end
  end

  // Rotation over port k of WideRoReqRemapGroupSize tiles, Interleaved over the
  // flattened {tile, port} index so a rotation never changes the port.
  if (WideRouterRemapping[0]) begin : gen_wide_ro_req_remapping
    floo_remapper #(
      .NumInp     (NumTilesPerGroup * NumWideRoReqPortsPerTile),
      .NumOut     (NumTilesPerGroup * NumWideRoReqPortsPerTile),
      .payload_t  (floo_tcdm_wide_ro_req_t                    ),
      .GroupSize  (WideRoReqRemapGroupSize                    ),
      .Interleaved(1'b1                                       )
    ) i_floo_tcdm_wide_ro_req_remapper (
      .clk_i      (clk_i                                  ),
      .rst_ni     (rst_ni                                 ),
      .inp_data_i (floo_tcdm_wide_ro_req_to_remapper      ),
      .inp_valid_i(floo_tcdm_wide_ro_req_to_remapper_valid),
      .inp_ready_o(floo_tcdm_wide_ro_req_to_remapper_ready),
      .oup_data_o (floo_tcdm_wide_ro_req_to_router        ),
      .oup_valid_o(floo_tcdm_wide_ro_req_to_router_valid  ),
      .oup_ready_i(floo_tcdm_wide_ro_req_to_router_ready  )
    );
  end else begin : gen_wide_ro_req_remapping_bypass
    assign floo_tcdm_wide_ro_req_to_router         = floo_tcdm_wide_ro_req_to_remapper;
    assign floo_tcdm_wide_ro_req_to_router_valid   = floo_tcdm_wide_ro_req_to_remapper_valid;
    assign floo_tcdm_wide_ro_req_to_remapper_ready = floo_tcdm_wide_ro_req_to_router_ready;
  end

  for (genvar i = North; i <= West; i++) begin : gen_wide_ro_req_bundle_i
    for (genvar t = 0; t < NumTilesPerGroup; t++) begin : gen_wide_ro_req_bundle_t
      for (genvar k = 0; k < NumWideRoReqPortsPerTile; k++) begin : gen_wide_ro_req_bundle_k
      assign floo_tcdm_req_o[i].floo_tcdm_req[t].wide_ro_req[k].req =
          floo_tcdm_wide_ro_req_out_trans[t][k][i];
      assign floo_tcdm_req_o[i].floo_tcdm_req[t].wide_ro_req[k].valid =
          floo_tcdm_wide_ro_req_valid_out_trans[t][k][i];
      assign floo_tcdm_req_o[i].floo_tcdm_req[t].wide_ro_req[k].ready =
          floo_tcdm_wide_ro_req_ready_out_trans[t][k][i];
      assign floo_tcdm_wide_ro_req_in_trans[t][k][i] =
          floo_tcdm_req_i[i].floo_tcdm_req[t].wide_ro_req[k].req;
      assign floo_tcdm_wide_ro_req_valid_in_trans[t][k][i] =
          floo_tcdm_req_i[i].floo_tcdm_req[t].wide_ro_req[k].valid;
      assign floo_tcdm_wide_ro_req_ready_in_trans[t][k][i] =
          floo_tcdm_req_i[i].floo_tcdm_req[t].wide_ro_req[k].ready;
      end
    end
  end

  for (genvar t = 0; t < NumTilesPerGroup; t++) begin : gen_wide_ro_req_router_t
    for (genvar k = 0; k < NumWideRoReqPortsPerTile; k++) begin : gen_wide_ro_req_router_k
      if (NocRoutingAlgorithm == 1) begin: gen_id_table_routing
        floo_router #(
          .NumRoutes       (mempool_pkg::NumDirections                  ),
          .NumVirtChannels (1                                           ),
          .flit_t          (floo_tcdm_wide_ro_req_t                     ),
          .InFifoDepth     (mempool_pkg::NumWideReqRouterInFifoDepth    ),
          .OutFifoDepth    (mempool_pkg::NumWideReqRouterOutFifoDepth   ),
          .RouteAlgo       (IdTable                                     ),
          .VcImpl          (VcPreemptValid                              ),
          .id_t            (routing_table_pkg::routing_rule_addr_t      ),
          .NumAddrRules    (NumGroups                                   ),
          .addr_rule_t     (routing_table_pkg::routing_rule_t           )
        ) i_floo_tcdm_wide_ro_req_router (
          .clk_i,
          .rst_ni,
          .test_enable_i  (1'b0),
          .xy_id_i        ('0),
          .id_route_map_i (routing_table_pkg::RoutingTables[group_xy_id.x][group_xy_id.y]),
          .valid_i        ({floo_tcdm_wide_ro_req_to_router_valid[t][k],   floo_tcdm_wide_ro_req_valid_in_trans[t][k]}),
          .ready_o        ({floo_tcdm_wide_ro_req_to_router_ready[t][k],   floo_tcdm_wide_ro_req_ready_out_trans[t][k]}),
          .data_i         ({floo_tcdm_wide_ro_req_to_router[t][k],         floo_tcdm_wide_ro_req_in_trans[t][k]}),
          .valid_o        ({floo_tcdm_wide_ro_req_from_router_valid[t][k], floo_tcdm_wide_ro_req_valid_out_trans[t][k]}),
          .ready_i        ({floo_tcdm_wide_ro_req_from_router_ready[t][k], floo_tcdm_wide_ro_req_ready_in_trans[t][k]}),
          .data_o         ({floo_tcdm_wide_ro_req_from_router[t][k],       floo_tcdm_wide_ro_req_out_trans[t][k]}),
          .credit_o       (),
          .credit_i       ('0),
          .offload_req_o  (),
          .offload_rsp_i  ('0)
        );
      end else begin: gen_xy_routing
        floo_router #(
          .NumRoutes       (mempool_pkg::NumDirections                  ),
          .NumVirtChannels (1                                           ),
          .flit_t          (floo_tcdm_wide_ro_req_t                     ),
          .InFifoDepth     (mempool_pkg::NumWideReqRouterInFifoDepth    ),
          .OutFifoDepth    (mempool_pkg::NumWideReqRouterOutFifoDepth   ),
          .RouteAlgo       (XYRouting                                   ),
          .VcImpl          (VcPreemptValid                              ),
          .id_t            (group_xy_id_t                               ),
          .NumAddrRules    (1                                           ),
          .addr_rule_t     (logic                                       )
        ) i_floo_tcdm_wide_ro_req_router (
          .clk_i,
          .rst_ni,
          .test_enable_i  (1'b0),
          .xy_id_i        (group_xy_id),
          .id_route_map_i ('0),
          .valid_i        ({floo_tcdm_wide_ro_req_to_router_valid[t][k],   floo_tcdm_wide_ro_req_valid_in_trans[t][k]}),
          .ready_o        ({floo_tcdm_wide_ro_req_to_router_ready[t][k],   floo_tcdm_wide_ro_req_ready_out_trans[t][k]}),
          .data_i         ({floo_tcdm_wide_ro_req_to_router[t][k],         floo_tcdm_wide_ro_req_in_trans[t][k]}),
          .valid_o        ({floo_tcdm_wide_ro_req_from_router_valid[t][k], floo_tcdm_wide_ro_req_valid_out_trans[t][k]}),
          .ready_i        ({floo_tcdm_wide_ro_req_from_router_ready[t][k], floo_tcdm_wide_ro_req_ready_in_trans[t][k]}),
          .data_o         ({floo_tcdm_wide_ro_req_from_router[t][k],       floo_tcdm_wide_ro_req_out_trans[t][k]}),
          .credit_o       (),
          .credit_i       ('0),
          .offload_req_o  (),
          .offload_rsp_i  ('0)
        );
      end
    end
  end

  // Crossbar per port: an RO request arriving on port k of any tile's router
  // goes to port k of the addressed tile.
  for (genvar k = 0; k < NumWideRoReqPortsPerTile; k++) begin : gen_wide_ro_req_xbar_k
    floo_tcdm_wide_ro_req_t [NumTilesPerGroup-1:0] xbar_in, xbar_out;
    logic [NumTilesPerGroup-1:0] xbar_in_valid, xbar_in_ready, xbar_out_valid, xbar_out_ready;
    logic [NumTilesPerGroup-1:0][WideTileSelW-1:0] xbar_sel;
    for (genvar t = 0; t < NumTilesPerGroup; t++) begin : gen_wide_ro_req_xbar_t
      assign xbar_in[t]       = floo_tcdm_wide_ro_req_from_router[t][k];
      assign xbar_in_valid[t] = floo_tcdm_wide_ro_req_from_router_valid[t][k];
      assign floo_tcdm_wide_ro_req_from_router_ready[t][k] = xbar_in_ready[t];
      assign xbar_sel[t] = (NumTilesPerGroup == 1) ? '0 :
          WideTileSelW'(floo_tcdm_wide_ro_req_from_router[t][k].hdr.tgt_addr);

      // A read: no payload.
      assign tcdm_wide_slave_req[t][WideRoBase+k] = '{
        wdata: '{
          meta_id: xbar_out[t].hdr.meta_id,
          core_id: xbar_out[t].hdr.core_id,
          amo:     '0,
          data:    '0
        },
        wen:          1'b0,
        be:           '0,
        tgt_addr:     xbar_out[t].hdr.tgt_addr,
        ini_addr:     xbar_out[t].hdr.src_tile_id,
        src_group_id: group_id_t'({xbar_out[t].hdr.src_id.x, xbar_out[t].hdr.src_id.y})
      };
      assign tcdm_wide_slave_req_valid[t][WideRoBase+k] = xbar_out_valid[t];
      assign xbar_out_ready[t] = tcdm_wide_slave_req_ready[t][WideRoBase+k];
    end

    stream_xbar #(
      .NumInp   (NumTilesPerGroup       ),
      .NumOut   (NumTilesPerGroup       ),
      .payload_t(floo_tcdm_wide_ro_req_t)
    ) i_wide_ro_req_xbar (
      .clk_i  (clk_i         ),
      .rst_ni (rst_ni        ),
      .flush_i(1'b0          ),
      .rr_i   ('0            ),
      .data_i (xbar_in       ),
      .valid_i(xbar_in_valid ),
      .ready_o(xbar_in_ready ),
      .sel_i  (xbar_sel      ),
      .data_o (xbar_out      ),
      .valid_o(xbar_out_valid),
      .ready_i(xbar_out_ready),
      .idx_o  (/* unused */  )
    );
  end

  // ======================= WO request channels =======================
  // Port w of tile t is input t * NumWideWoReqPortsPerTile + w; channel c merges
  // the inputs congruent to c, i.e. tiles t and t + NumTilesPerGroup / merge.
  floo_tcdm_wide_rw_req_t [NumWideWoReqChannelsPerGroup-1:0]
      floo_tcdm_wide_wo_req_to_router, floo_tcdm_wide_wo_req_from_router;
  logic [NumWideWoReqChannelsPerGroup-1:0]
      floo_tcdm_wide_wo_req_to_router_valid, floo_tcdm_wide_wo_req_to_router_ready;
  logic [NumWideWoReqChannelsPerGroup-1:0]
      floo_tcdm_wide_wo_req_from_router_valid, floo_tcdm_wide_wo_req_from_router_ready;
  floo_tcdm_wide_rw_req_t [NumWideWoReqChannelsPerGroup-1:0][West:North]
      floo_tcdm_wide_wo_req_out_trans, floo_tcdm_wide_wo_req_in_trans;
  logic [NumWideWoReqChannelsPerGroup-1:0][West:North]
      floo_tcdm_wide_wo_req_valid_out_trans, floo_tcdm_wide_wo_req_ready_in_trans;
  logic [NumWideWoReqChannelsPerGroup-1:0][West:North]
      floo_tcdm_wide_wo_req_valid_in_trans, floo_tcdm_wide_wo_req_ready_out_trans;
  floo_tcdm_wide_rw_req_t [NumTilesPerGroup*NumWideWoReqPortsPerTile-1:0]
      floo_tcdm_wide_wo_req_to_merge;
  logic [NumTilesPerGroup*NumWideWoReqPortsPerTile-1:0]
      floo_tcdm_wide_wo_req_to_merge_valid, floo_tcdm_wide_wo_req_to_merge_ready;

  for (genvar t = 0; t < NumTilesPerGroup; t++) begin : gen_wide_wo_req_packet_t
    for (genvar w = 0; w < NumWideWoReqPortsPerTile; w++) begin : gen_wide_wo_req_packet_w
      localparam int unsigned Inp = t * NumWideWoReqPortsPerTile + w;
      assign floo_tcdm_wide_wo_req_to_merge[Inp] = '{
        payload: '{
          amo:  tcdm_wide_master_req[t][WideWoBase+w].wdata.amo,
          wen:  tcdm_wide_master_req[t][WideWoBase+w].wen,
          be:   tcdm_wide_master_req[t][WideWoBase+w].be,
          data: tcdm_wide_master_req[t][WideWoBase+w].wdata.data
        },
        hdr: '{
          meta_id:       tcdm_wide_master_req[t][WideWoBase+w].wdata.meta_id,
          core_id:       tcdm_wide_master_req[t][WideWoBase+w].wdata.core_id,
          src_tile_id:   tcdm_wide_master_req[t][WideWoBase+w].src_tile_id,
          src_id:        group_xy_of(group_id_i),
          dst_id:        group_xy_of(tcdm_wide_master_req[t][WideWoBase+w].tgt_group_id),
          tgt_addr:      tcdm_wide_master_req[t][WideWoBase+w].tgt_addr,
          last:          1'b1,
          collective_op: 1'b0
        }
      };
      assign floo_tcdm_wide_wo_req_to_merge_valid[Inp] = tcdm_wide_master_req_valid[t][WideWoBase+w];
      assign tcdm_wide_master_req_ready[t][WideWoBase+w] = floo_tcdm_wide_wo_req_to_merge_ready[Inp];
    end
  end

  for (genvar c = 0; c < NumWideWoReqChannelsPerGroup; c++) begin : gen_wide_wo_req_merge_c
    floo_tcdm_wide_rw_req_t [NumWideWoMerge-1:0] merge_in;
    logic [NumWideWoMerge-1:0] merge_in_valid, merge_in_ready;
    for (genvar m = 0; m < NumWideWoMerge; m++) begin : gen_wide_wo_req_merge_m
      localparam int unsigned Inp = c + m * NumWideWoReqChannelsPerGroup;
      assign merge_in[m]       = floo_tcdm_wide_wo_req_to_merge[Inp];
      assign merge_in_valid[m] = floo_tcdm_wide_wo_req_to_merge_valid[Inp];
      assign floo_tcdm_wide_wo_req_to_merge_ready[Inp] = merge_in_ready[m];
    end
    stream_arbiter #(
      .DATA_T (floo_tcdm_wide_rw_req_t),
      .N_INP  (NumWideWoMerge         ),
      .ARBITER("rr"                   )
    ) i_wide_wo_req_merge (
      .clk_i      (clk_i                                    ),
      .rst_ni     (rst_ni                                   ),
      .inp_data_i (merge_in                                 ),
      .inp_valid_i(merge_in_valid                           ),
      .inp_ready_o(merge_in_ready                           ),
      .oup_data_o (floo_tcdm_wide_wo_req_to_router[c]       ),
      .oup_valid_o(floo_tcdm_wide_wo_req_to_router_valid[c] ),
      .oup_ready_i(floo_tcdm_wide_wo_req_to_router_ready[c] )
    );
  end

  for (genvar i = North; i <= West; i++) begin : gen_wide_wo_req_bundle_i
    for (genvar c = 0; c < NumWideWoReqChannelsPerGroup; c++) begin : gen_wide_wo_req_bundle_c
      assign floo_tcdm_req_o[i].floo_tcdm_wide_req.wide_wo_req[c].req =
          floo_tcdm_wide_wo_req_out_trans[c][i];
      assign floo_tcdm_req_o[i].floo_tcdm_wide_req.wide_wo_req[c].valid =
          floo_tcdm_wide_wo_req_valid_out_trans[c][i];
      assign floo_tcdm_req_o[i].floo_tcdm_wide_req.wide_wo_req[c].ready =
          floo_tcdm_wide_wo_req_ready_out_trans[c][i];
      assign floo_tcdm_wide_wo_req_in_trans[c][i] =
          floo_tcdm_req_i[i].floo_tcdm_wide_req.wide_wo_req[c].req;
      assign floo_tcdm_wide_wo_req_valid_in_trans[c][i] =
          floo_tcdm_req_i[i].floo_tcdm_wide_req.wide_wo_req[c].valid;
      assign floo_tcdm_wide_wo_req_ready_in_trans[c][i] =
          floo_tcdm_req_i[i].floo_tcdm_wide_req.wide_wo_req[c].ready;
    end
  end

  for (genvar c = 0; c < NumWideWoReqChannelsPerGroup; c++) begin : gen_wide_wo_req_router_c
    if (NocRoutingAlgorithm == 1) begin: gen_id_table_routing
      floo_router #(
        .NumRoutes       (mempool_pkg::NumDirections                  ),
        .NumVirtChannels (1                                           ),
        .flit_t          (floo_tcdm_wide_rw_req_t                     ),
        .InFifoDepth     (mempool_pkg::NumWideReqRouterInFifoDepth    ),
        .OutFifoDepth    (mempool_pkg::NumWideReqRouterOutFifoDepth   ),
        .RouteAlgo       (IdTable                                     ),
        .VcImpl          (VcPreemptValid                              ),
        .id_t            (routing_table_pkg::routing_rule_addr_t      ),
        .NumAddrRules    (NumGroups                                   ),
        .addr_rule_t     (routing_table_pkg::routing_rule_t           )
      ) i_floo_tcdm_wide_wo_req_router (
        .clk_i,
        .rst_ni,
        .test_enable_i  (1'b0),
        .xy_id_i        ('0),
        .id_route_map_i (routing_table_pkg::RoutingTables[group_xy_id.x][group_xy_id.y]),
        .valid_i        ({floo_tcdm_wide_wo_req_to_router_valid[c],   floo_tcdm_wide_wo_req_valid_in_trans[c]}),
        .ready_o        ({floo_tcdm_wide_wo_req_to_router_ready[c],   floo_tcdm_wide_wo_req_ready_out_trans[c]}),
        .data_i         ({floo_tcdm_wide_wo_req_to_router[c],         floo_tcdm_wide_wo_req_in_trans[c]}),
        .valid_o        ({floo_tcdm_wide_wo_req_from_router_valid[c], floo_tcdm_wide_wo_req_valid_out_trans[c]}),
        .ready_i        ({floo_tcdm_wide_wo_req_from_router_ready[c], floo_tcdm_wide_wo_req_ready_in_trans[c]}),
        .data_o         ({floo_tcdm_wide_wo_req_from_router[c],       floo_tcdm_wide_wo_req_out_trans[c]}),
        .credit_o       (),
        .credit_i       ('0),
        .offload_req_o  (),
        .offload_rsp_i  ('0)
      );
    end else begin: gen_xy_routing
      floo_router #(
        .NumRoutes       (mempool_pkg::NumDirections                  ),
        .NumVirtChannels (1                                           ),
        .flit_t          (floo_tcdm_wide_rw_req_t                     ),
        .InFifoDepth     (mempool_pkg::NumWideReqRouterInFifoDepth    ),
        .OutFifoDepth    (mempool_pkg::NumWideReqRouterOutFifoDepth   ),
        .RouteAlgo       (XYRouting                                   ),
        .VcImpl          (VcPreemptValid                              ),
        .id_t            (group_xy_id_t                               ),
        .NumAddrRules    (1                                           ),
        .addr_rule_t     (logic                                       )
      ) i_floo_tcdm_wide_wo_req_router (
        .clk_i,
        .rst_ni,
        .test_enable_i  (1'b0),
        .xy_id_i        (group_xy_id),
        .id_route_map_i ('0),
        .valid_i        ({floo_tcdm_wide_wo_req_to_router_valid[c],   floo_tcdm_wide_wo_req_valid_in_trans[c]}),
        .ready_o        ({floo_tcdm_wide_wo_req_to_router_ready[c],   floo_tcdm_wide_wo_req_ready_out_trans[c]}),
        .data_i         ({floo_tcdm_wide_wo_req_to_router[c],         floo_tcdm_wide_wo_req_in_trans[c]}),
        .valid_o        ({floo_tcdm_wide_wo_req_from_router_valid[c], floo_tcdm_wide_wo_req_valid_out_trans[c]}),
        .ready_i        ({floo_tcdm_wide_wo_req_from_router_ready[c], floo_tcdm_wide_wo_req_ready_in_trans[c]}),
        .data_o         ({floo_tcdm_wide_wo_req_from_router[c],       floo_tcdm_wide_wo_req_out_trans[c]}),
        .credit_o       (),
        .credit_i       ('0),
        .offload_req_o  (),
        .offload_rsp_i  ('0)
      );
    end
  end

  // A write arriving on channel c goes to the addressed tile's port c % ports,
  // the port it was merged from.
  floo_tcdm_wide_rw_req_t [NumTilesPerGroup*NumWideWoReqPortsPerTile-1:0] floo_tcdm_wide_wo_req_after_xbar;
  logic [NumTilesPerGroup*NumWideWoReqPortsPerTile-1:0] floo_tcdm_wide_wo_req_after_xbar_valid, floo_tcdm_wide_wo_req_after_xbar_ready;
  logic [NumWideWoReqChannelsPerGroup-1:0]
      [idx_width(NumTilesPerGroup*NumWideWoReqPortsPerTile)-1:0] floo_tcdm_wide_wo_req_xbar_sel;
  for (genvar c = 0; c < NumWideWoReqChannelsPerGroup; c++) begin : gen_wide_wo_req_xbar_sel
    assign floo_tcdm_wide_wo_req_xbar_sel[c] = idx_width(NumTilesPerGroup*NumWideWoReqPortsPerTile)'(
        ((NumTilesPerGroup == 1) ? 0 :
         WideTileSelW'(floo_tcdm_wide_wo_req_from_router[c].hdr.tgt_addr)) *
        NumWideWoReqPortsPerTile + c % NumWideWoReqPortsPerTile);
  end

  stream_xbar #(
    .NumInp   (NumWideWoReqChannelsPerGroup                 ),
    .NumOut   (NumTilesPerGroup * NumWideWoReqPortsPerTile  ),
    .payload_t(floo_tcdm_wide_rw_req_t                      )
  ) i_wide_wo_req_xbar (
    .clk_i  (clk_i                                   ),
    .rst_ni (rst_ni                                  ),
    .flush_i(1'b0                                    ),
    .rr_i   ('0                                      ),
    .data_i (floo_tcdm_wide_wo_req_from_router       ),
    .valid_i(floo_tcdm_wide_wo_req_from_router_valid ),
    .ready_o(floo_tcdm_wide_wo_req_from_router_ready ),
    .sel_i  (floo_tcdm_wide_wo_req_xbar_sel                                ),
    .data_o (floo_tcdm_wide_wo_req_after_xbar                                ),
    .valid_o(floo_tcdm_wide_wo_req_after_xbar_valid                          ),
    .ready_i(floo_tcdm_wide_wo_req_after_xbar_ready                          ),
    .idx_o  (/* unused */                            )
  );

  for (genvar t = 0; t < NumTilesPerGroup; t++) begin : gen_wide_wo_req_slave_t
    for (genvar w = 0; w < NumWideWoReqPortsPerTile; w++) begin : gen_wide_wo_req_slave_w
      localparam int unsigned Oup = t * NumWideWoReqPortsPerTile + w;
      assign tcdm_wide_slave_req[t][WideWoBase+w] = '{
        wdata: '{
          meta_id: floo_tcdm_wide_wo_req_after_xbar[Oup].hdr.meta_id,
          core_id: floo_tcdm_wide_wo_req_after_xbar[Oup].hdr.core_id,
          amo:     floo_tcdm_wide_wo_req_after_xbar[Oup].payload.amo,
          data:    floo_tcdm_wide_wo_req_after_xbar[Oup].payload.data
        },
        wen:          floo_tcdm_wide_wo_req_after_xbar[Oup].payload.wen,
        be:           floo_tcdm_wide_wo_req_after_xbar[Oup].payload.be,
        tgt_addr:     floo_tcdm_wide_wo_req_after_xbar[Oup].hdr.tgt_addr,
        ini_addr:     floo_tcdm_wide_wo_req_after_xbar[Oup].hdr.src_tile_id,
        src_group_id: group_id_t'({floo_tcdm_wide_wo_req_after_xbar[Oup].hdr.src_id.x, floo_tcdm_wide_wo_req_after_xbar[Oup].hdr.src_id.y})
      };
      assign tcdm_wide_slave_req_valid[t][WideWoBase+w] = floo_tcdm_wide_wo_req_after_xbar_valid[Oup];
      assign floo_tcdm_wide_wo_req_after_xbar_ready[Oup] = tcdm_wide_slave_req_ready[t][WideWoBase+w];
    end
  end

  // ======================= RW response ports =======================
  floo_tcdm_wide_rd_resp_t [NumTilesPerGroup-1:0][NumWideRwRespPortsPerTile-1:0]
      floo_tcdm_wide_rw_resp_to_router, floo_tcdm_wide_rw_resp_from_router;
  logic [NumTilesPerGroup-1:0][NumWideRwRespPortsPerTile-1:0]
      floo_tcdm_wide_rw_resp_to_router_valid, floo_tcdm_wide_rw_resp_to_router_ready;
  logic [NumTilesPerGroup-1:0][NumWideRwRespPortsPerTile-1:0]
      floo_tcdm_wide_rw_resp_from_router_valid, floo_tcdm_wide_rw_resp_from_router_ready;
  floo_tcdm_wide_rd_resp_t [NumTilesPerGroup-1:0][NumWideRwRespPortsPerTile-1:0][West:North]
      floo_tcdm_wide_rw_resp_out_trans, floo_tcdm_wide_rw_resp_in_trans;
  logic [NumTilesPerGroup-1:0][NumWideRwRespPortsPerTile-1:0][West:North]
      floo_tcdm_wide_rw_resp_valid_out_trans, floo_tcdm_wide_rw_resp_ready_in_trans;
  logic [NumTilesPerGroup-1:0][NumWideRwRespPortsPerTile-1:0][West:North]
      floo_tcdm_wide_rw_resp_valid_in_trans, floo_tcdm_wide_rw_resp_ready_out_trans;
  floo_tcdm_wide_rd_resp_t [NumTilesPerGroup-1:0][NumWideRwRespPortsPerTile-1:0]
      floo_tcdm_wide_rw_resp_to_remapper;
  logic [NumTilesPerGroup-1:0][NumWideRwRespPortsPerTile-1:0]
      floo_tcdm_wide_rw_resp_to_remapper_valid, floo_tcdm_wide_rw_resp_to_remapper_ready;

  for (genvar t = 0; t < NumTilesPerGroup; t++) begin : gen_wide_rw_resp_packet_t
    for (genvar k = 0; k < NumWideRwRespPortsPerTile; k++) begin : gen_wide_rw_resp_packet_k
      assign floo_tcdm_wide_rw_resp_to_remapper[t][k] = '{
        payload: '{data: tcdm_wide_slave_resp[t][WideRwBase+k].rdata.data},
        hdr: '{
          meta_id:       tcdm_wide_slave_resp[t][WideRwBase+k].rdata.meta_id,
          core_id:       tcdm_wide_slave_resp[t][WideRwBase+k].rdata.core_id,
          tile_id:       tcdm_wide_slave_resp[t][WideRwBase+k].ini_addr,
          src_id:        group_xy_of(group_id_i),
          dst_id:        group_xy_of(tcdm_wide_slave_resp[t][WideRwBase+k].src_group_id),
          last:          1'b1,
          collective_op: 1'b0
        }
      };
      assign floo_tcdm_wide_rw_resp_to_remapper_valid[t][k] =
          tcdm_wide_slave_resp_valid[t][WideRwBase+k];
      assign tcdm_wide_slave_resp_ready[t][WideRwBase+k] =
          floo_tcdm_wide_rw_resp_to_remapper_ready[t][k];
    end
  end

  // Same rotation as the RO request ports, on the response's outgoing edge: the
  // read payload comes back here, so remapping only the request would move the
  // limit instead of removing it.
  if (WideRouterRemapping[1]) begin : gen_wide_rw_resp_remapping
    floo_remapper #(
      .NumInp     (NumTilesPerGroup * NumWideRwRespPortsPerTile),
      .NumOut     (NumTilesPerGroup * NumWideRwRespPortsPerTile),
      .payload_t  (floo_tcdm_wide_rd_resp_t                    ),
      .GroupSize  (WideRwRespRemapGroupSize                    ),
      .Interleaved(1'b1                                        )
    ) i_floo_tcdm_wide_rw_resp_remapper (
      .clk_i      (clk_i                                   ),
      .rst_ni     (rst_ni                                  ),
      .inp_data_i (floo_tcdm_wide_rw_resp_to_remapper      ),
      .inp_valid_i(floo_tcdm_wide_rw_resp_to_remapper_valid),
      .inp_ready_o(floo_tcdm_wide_rw_resp_to_remapper_ready),
      .oup_data_o (floo_tcdm_wide_rw_resp_to_router        ),
      .oup_valid_o(floo_tcdm_wide_rw_resp_to_router_valid  ),
      .oup_ready_i(floo_tcdm_wide_rw_resp_to_router_ready  )
    );
  end else begin : gen_wide_rw_resp_remapping_bypass
    assign floo_tcdm_wide_rw_resp_to_router         = floo_tcdm_wide_rw_resp_to_remapper;
    assign floo_tcdm_wide_rw_resp_to_router_valid   = floo_tcdm_wide_rw_resp_to_remapper_valid;
    assign floo_tcdm_wide_rw_resp_to_remapper_ready = floo_tcdm_wide_rw_resp_to_router_ready;
  end

  for (genvar i = North; i <= West; i++) begin : gen_wide_rw_resp_bundle_i
    for (genvar t = 0; t < NumTilesPerGroup; t++) begin : gen_wide_rw_resp_bundle_t
      for (genvar k = 0; k < NumWideRwRespPortsPerTile; k++) begin : gen_wide_rw_resp_bundle_k
      assign floo_tcdm_rsp_o[i].floo_tcdm_wide_rw_resp[t][k].resp =
          floo_tcdm_wide_rw_resp_out_trans[t][k][i];
      assign floo_tcdm_rsp_o[i].floo_tcdm_wide_rw_resp[t][k].valid =
          floo_tcdm_wide_rw_resp_valid_out_trans[t][k][i];
      assign floo_tcdm_rsp_o[i].floo_tcdm_wide_rw_resp[t][k].ready =
          floo_tcdm_wide_rw_resp_ready_out_trans[t][k][i];
      assign floo_tcdm_wide_rw_resp_in_trans[t][k][i] =
          floo_tcdm_rsp_i[i].floo_tcdm_wide_rw_resp[t][k].resp;
      assign floo_tcdm_wide_rw_resp_valid_in_trans[t][k][i] =
          floo_tcdm_rsp_i[i].floo_tcdm_wide_rw_resp[t][k].valid;
      assign floo_tcdm_wide_rw_resp_ready_in_trans[t][k][i] =
          floo_tcdm_rsp_i[i].floo_tcdm_wide_rw_resp[t][k].ready;
      end
    end
  end

  for (genvar t = 0; t < NumTilesPerGroup; t++) begin : gen_wide_rw_resp_router_t
    for (genvar k = 0; k < NumWideRwRespPortsPerTile; k++) begin : gen_wide_rw_resp_router_k
      if (NocRoutingAlgorithm == 1) begin: gen_id_table_routing
        floo_router #(
          .NumRoutes       (mempool_pkg::NumDirections                  ),
          .NumVirtChannels (1                                           ),
          .flit_t          (floo_tcdm_wide_rd_resp_t                    ),
          .InFifoDepth     (mempool_pkg::NumWideRespRouterInFifoDepth   ),
          .OutFifoDepth    (mempool_pkg::NumWideRespRouterOutFifoDepth  ),
          .RouteAlgo       (IdTable                                     ),
          .VcImpl          (VcPreemptValid                              ),
          .id_t            (routing_table_pkg::routing_rule_addr_t      ),
          .NumAddrRules    (NumGroups                                   ),
          .addr_rule_t     (routing_table_pkg::routing_rule_t           )
        ) i_floo_tcdm_wide_rw_resp_router (
          .clk_i,
          .rst_ni,
          .test_enable_i  (1'b0),
          .xy_id_i        ('0),
          .id_route_map_i (routing_table_pkg::RoutingTables[group_xy_id.x][group_xy_id.y]),
          .valid_i        ({floo_tcdm_wide_rw_resp_to_router_valid[t][k],   floo_tcdm_wide_rw_resp_valid_in_trans[t][k]}),
          .ready_o        ({floo_tcdm_wide_rw_resp_to_router_ready[t][k],   floo_tcdm_wide_rw_resp_ready_out_trans[t][k]}),
          .data_i         ({floo_tcdm_wide_rw_resp_to_router[t][k],         floo_tcdm_wide_rw_resp_in_trans[t][k]}),
          .valid_o        ({floo_tcdm_wide_rw_resp_from_router_valid[t][k], floo_tcdm_wide_rw_resp_valid_out_trans[t][k]}),
          .ready_i        ({floo_tcdm_wide_rw_resp_from_router_ready[t][k], floo_tcdm_wide_rw_resp_ready_in_trans[t][k]}),
          .data_o         ({floo_tcdm_wide_rw_resp_from_router[t][k],       floo_tcdm_wide_rw_resp_out_trans[t][k]}),
          .credit_o       (),
          .credit_i       ('0),
          .offload_req_o  (),
          .offload_rsp_i  ('0)
        );
      end else begin: gen_xy_routing
        floo_router #(
          .NumRoutes       (mempool_pkg::NumDirections                  ),
          .NumVirtChannels (1                                           ),
          .flit_t          (floo_tcdm_wide_rd_resp_t                    ),
          .InFifoDepth     (mempool_pkg::NumWideRespRouterInFifoDepth   ),
          .OutFifoDepth    (mempool_pkg::NumWideRespRouterOutFifoDepth  ),
          .RouteAlgo       (XYRouting                                   ),
          .VcImpl          (VcPreemptValid                              ),
          .id_t            (group_xy_id_t                               ),
          .NumAddrRules    (1                                           ),
          .addr_rule_t     (logic                                       )
        ) i_floo_tcdm_wide_rw_resp_router (
          .clk_i,
          .rst_ni,
          .test_enable_i  (1'b0),
          .xy_id_i        (group_xy_id),
          .id_route_map_i ('0),
          .valid_i        ({floo_tcdm_wide_rw_resp_to_router_valid[t][k],   floo_tcdm_wide_rw_resp_valid_in_trans[t][k]}),
          .ready_o        ({floo_tcdm_wide_rw_resp_to_router_ready[t][k],   floo_tcdm_wide_rw_resp_ready_out_trans[t][k]}),
          .data_i         ({floo_tcdm_wide_rw_resp_to_router[t][k],         floo_tcdm_wide_rw_resp_in_trans[t][k]}),
          .valid_o        ({floo_tcdm_wide_rw_resp_from_router_valid[t][k], floo_tcdm_wide_rw_resp_valid_out_trans[t][k]}),
          .ready_i        ({floo_tcdm_wide_rw_resp_from_router_ready[t][k], floo_tcdm_wide_rw_resp_ready_in_trans[t][k]}),
          .data_o         ({floo_tcdm_wide_rw_resp_from_router[t][k],       floo_tcdm_wide_rw_resp_out_trans[t][k]}),
          .credit_o       (),
          .credit_i       ('0),
          .offload_req_o  (),
          .offload_rsp_i  ('0)
        );
      end
    end
  end

  // Crossbar per port: read data arriving on port k goes to port k of the
  // requesting tile.
  for (genvar k = 0; k < NumWideRwRespPortsPerTile; k++) begin : gen_wide_rw_resp_xbar_k
    floo_tcdm_wide_rd_resp_t [NumTilesPerGroup-1:0] xbar_in, xbar_out;
    logic [NumTilesPerGroup-1:0] xbar_in_valid, xbar_in_ready, xbar_out_valid, xbar_out_ready;
    logic [NumTilesPerGroup-1:0][WideTileSelW-1:0] xbar_sel;
    for (genvar t = 0; t < NumTilesPerGroup; t++) begin : gen_wide_rw_resp_xbar_t
      assign xbar_in[t]       = floo_tcdm_wide_rw_resp_from_router[t][k];
      assign xbar_in_valid[t] = floo_tcdm_wide_rw_resp_from_router_valid[t][k];
      assign floo_tcdm_wide_rw_resp_from_router_ready[t][k] = xbar_in_ready[t];
      assign xbar_sel[t] = (NumTilesPerGroup == 1) ? '0 :
          WideTileSelW'(floo_tcdm_wide_rw_resp_from_router[t][k].hdr.tile_id);

      assign tcdm_wide_master_resp[t][WideRwBase+k] = '{
        rdata: '{
          meta_id: xbar_out[t].hdr.meta_id,
          core_id: xbar_out[t].hdr.core_id,
          amo:     '0,
          data:    xbar_out[t].payload.data
        },
        tile_id: xbar_out[t].hdr.tile_id
      };
      assign tcdm_wide_master_resp_valid[t][WideRwBase+k] = xbar_out_valid[t];
      assign xbar_out_ready[t] = tcdm_wide_master_resp_ready[t][WideRwBase+k];
    end

    stream_xbar #(
      .NumInp   (NumTilesPerGroup        ),
      .NumOut   (NumTilesPerGroup        ),
      .payload_t(floo_tcdm_wide_rd_resp_t)
    ) i_wide_rw_resp_xbar (
      .clk_i  (clk_i         ),
      .rst_ni (rst_ni        ),
      .flush_i(1'b0          ),
      .rr_i   ('0            ),
      .data_i (xbar_in       ),
      .valid_i(xbar_in_valid ),
      .ready_o(xbar_in_ready ),
      .sel_i  (xbar_sel      ),
      .data_o (xbar_out      ),
      .valid_o(xbar_out_valid),
      .ready_i(xbar_out_ready),
      .idx_o  (/* unused */  )
    );
  end

  // ======================= WO response channels =======================
  // Merged as the WO request channels: write acks of tiles t and
  // t + NumTilesPerGroup / merge share channel t % channels.
  floo_tcdm_wide_wr_resp_t [NumWideWoRespChannelsPerGroup-1:0]
      floo_tcdm_wide_wo_resp_to_router, floo_tcdm_wide_wo_resp_from_router;
  logic [NumWideWoRespChannelsPerGroup-1:0]
      floo_tcdm_wide_wo_resp_to_router_valid, floo_tcdm_wide_wo_resp_to_router_ready;
  logic [NumWideWoRespChannelsPerGroup-1:0]
      floo_tcdm_wide_wo_resp_from_router_valid, floo_tcdm_wide_wo_resp_from_router_ready;
  floo_tcdm_wide_wr_resp_t [NumWideWoRespChannelsPerGroup-1:0][West:North]
      floo_tcdm_wide_wo_resp_out_trans, floo_tcdm_wide_wo_resp_in_trans;
  logic [NumWideWoRespChannelsPerGroup-1:0][West:North]
      floo_tcdm_wide_wo_resp_valid_out_trans, floo_tcdm_wide_wo_resp_ready_in_trans;
  logic [NumWideWoRespChannelsPerGroup-1:0][West:North]
      floo_tcdm_wide_wo_resp_valid_in_trans, floo_tcdm_wide_wo_resp_ready_out_trans;
  floo_tcdm_wide_wr_resp_t [NumTilesPerGroup*NumWideWoRespPortsPerTile-1:0]
      floo_tcdm_wide_wo_resp_to_merge;
  logic [NumTilesPerGroup*NumWideWoRespPortsPerTile-1:0]
      floo_tcdm_wide_wo_resp_to_merge_valid, floo_tcdm_wide_wo_resp_to_merge_ready;

  for (genvar t = 0; t < NumTilesPerGroup; t++) begin : gen_wide_wo_resp_packet_t
    for (genvar w = 0; w < NumWideWoRespPortsPerTile; w++) begin : gen_wide_wo_resp_packet_w
      localparam int unsigned Inp = t * NumWideWoRespPortsPerTile + w;
      assign floo_tcdm_wide_wo_resp_to_merge[Inp] = '{
        hdr: '{
          meta_id:       tcdm_wide_slave_resp[t][WideWoRespBase+w].rdata.meta_id,
          core_id:       tcdm_wide_slave_resp[t][WideWoRespBase+w].rdata.core_id,
          tile_id:       tcdm_wide_slave_resp[t][WideWoRespBase+w].ini_addr,
          src_id:        group_xy_of(group_id_i),
          dst_id:        group_xy_of(tcdm_wide_slave_resp[t][WideWoRespBase+w].src_group_id),
          last:          1'b1,
          collective_op: 1'b0
        }
      };
      assign floo_tcdm_wide_wo_resp_to_merge_valid[Inp] =
          tcdm_wide_slave_resp_valid[t][WideWoRespBase+w];
      assign tcdm_wide_slave_resp_ready[t][WideWoRespBase+w] =
          floo_tcdm_wide_wo_resp_to_merge_ready[Inp];
    end
  end

  for (genvar c = 0; c < NumWideWoRespChannelsPerGroup; c++) begin : gen_wide_wo_resp_merge_c
    floo_tcdm_wide_wr_resp_t [NumWideWoMerge-1:0] merge_in;
    logic [NumWideWoMerge-1:0] merge_in_valid, merge_in_ready;
    for (genvar m = 0; m < NumWideWoMerge; m++) begin : gen_wide_wo_resp_merge_m
      localparam int unsigned Inp = c + m * NumWideWoRespChannelsPerGroup;
      assign merge_in[m]       = floo_tcdm_wide_wo_resp_to_merge[Inp];
      assign merge_in_valid[m] = floo_tcdm_wide_wo_resp_to_merge_valid[Inp];
      assign floo_tcdm_wide_wo_resp_to_merge_ready[Inp] = merge_in_ready[m];
    end
    stream_arbiter #(
      .DATA_T (floo_tcdm_wide_wr_resp_t),
      .N_INP  (NumWideWoMerge          ),
      .ARBITER("rr"                    )
    ) i_wide_wo_resp_merge (
      .clk_i      (clk_i                                     ),
      .rst_ni     (rst_ni                                    ),
      .inp_data_i (merge_in                                  ),
      .inp_valid_i(merge_in_valid                            ),
      .inp_ready_o(merge_in_ready                            ),
      .oup_data_o (floo_tcdm_wide_wo_resp_to_router[c]       ),
      .oup_valid_o(floo_tcdm_wide_wo_resp_to_router_valid[c] ),
      .oup_ready_i(floo_tcdm_wide_wo_resp_to_router_ready[c] )
    );
  end

  for (genvar i = North; i <= West; i++) begin : gen_wide_wo_resp_bundle_i
    for (genvar c = 0; c < NumWideWoRespChannelsPerGroup; c++) begin : gen_wide_wo_resp_bundle_c
      assign floo_tcdm_rsp_o[i].floo_tcdm_wide_wo_resp[c].resp =
          floo_tcdm_wide_wo_resp_out_trans[c][i];
      assign floo_tcdm_rsp_o[i].floo_tcdm_wide_wo_resp[c].valid =
          floo_tcdm_wide_wo_resp_valid_out_trans[c][i];
      assign floo_tcdm_rsp_o[i].floo_tcdm_wide_wo_resp[c].ready =
          floo_tcdm_wide_wo_resp_ready_out_trans[c][i];
      assign floo_tcdm_wide_wo_resp_in_trans[c][i] =
          floo_tcdm_rsp_i[i].floo_tcdm_wide_wo_resp[c].resp;
      assign floo_tcdm_wide_wo_resp_valid_in_trans[c][i] =
          floo_tcdm_rsp_i[i].floo_tcdm_wide_wo_resp[c].valid;
      assign floo_tcdm_wide_wo_resp_ready_in_trans[c][i] =
          floo_tcdm_rsp_i[i].floo_tcdm_wide_wo_resp[c].ready;
    end
  end

  for (genvar c = 0; c < NumWideWoRespChannelsPerGroup; c++) begin : gen_wide_wo_resp_router_c
    if (NocRoutingAlgorithm == 1) begin: gen_id_table_routing
      floo_router #(
        .NumRoutes       (mempool_pkg::NumDirections                  ),
        .NumVirtChannels (1                                           ),
        .flit_t          (floo_tcdm_wide_wr_resp_t                    ),
        .InFifoDepth     (mempool_pkg::NumWideRespRouterInFifoDepth   ),
        .OutFifoDepth    (mempool_pkg::NumWideRespRouterOutFifoDepth  ),
        .RouteAlgo       (IdTable                                     ),
        .VcImpl          (VcPreemptValid                              ),
        .id_t            (routing_table_pkg::routing_rule_addr_t      ),
        .NumAddrRules    (NumGroups                                   ),
        .addr_rule_t     (routing_table_pkg::routing_rule_t           )
      ) i_floo_tcdm_wide_wo_resp_router (
        .clk_i,
        .rst_ni,
        .test_enable_i  (1'b0),
        .xy_id_i        ('0),
        .id_route_map_i (routing_table_pkg::RoutingTables[group_xy_id.x][group_xy_id.y]),
        .valid_i        ({floo_tcdm_wide_wo_resp_to_router_valid[c],   floo_tcdm_wide_wo_resp_valid_in_trans[c]}),
        .ready_o        ({floo_tcdm_wide_wo_resp_to_router_ready[c],   floo_tcdm_wide_wo_resp_ready_out_trans[c]}),
        .data_i         ({floo_tcdm_wide_wo_resp_to_router[c],         floo_tcdm_wide_wo_resp_in_trans[c]}),
        .valid_o        ({floo_tcdm_wide_wo_resp_from_router_valid[c], floo_tcdm_wide_wo_resp_valid_out_trans[c]}),
        .ready_i        ({floo_tcdm_wide_wo_resp_from_router_ready[c], floo_tcdm_wide_wo_resp_ready_in_trans[c]}),
        .data_o         ({floo_tcdm_wide_wo_resp_from_router[c],       floo_tcdm_wide_wo_resp_out_trans[c]}),
        .credit_o       (),
        .credit_i       ('0),
        .offload_req_o  (),
        .offload_rsp_i  ('0)
      );
    end else begin: gen_xy_routing
      floo_router #(
        .NumRoutes       (mempool_pkg::NumDirections                  ),
        .NumVirtChannels (1                                           ),
        .flit_t          (floo_tcdm_wide_wr_resp_t                    ),
        .InFifoDepth     (mempool_pkg::NumWideRespRouterInFifoDepth   ),
        .OutFifoDepth    (mempool_pkg::NumWideRespRouterOutFifoDepth  ),
        .RouteAlgo       (XYRouting                                   ),
        .VcImpl          (VcPreemptValid                              ),
        .id_t            (group_xy_id_t                               ),
        .NumAddrRules    (1                                           ),
        .addr_rule_t     (logic                                       )
      ) i_floo_tcdm_wide_wo_resp_router (
        .clk_i,
        .rst_ni,
        .test_enable_i  (1'b0),
        .xy_id_i        (group_xy_id),
        .id_route_map_i ('0),
        .valid_i        ({floo_tcdm_wide_wo_resp_to_router_valid[c],   floo_tcdm_wide_wo_resp_valid_in_trans[c]}),
        .ready_o        ({floo_tcdm_wide_wo_resp_to_router_ready[c],   floo_tcdm_wide_wo_resp_ready_out_trans[c]}),
        .data_i         ({floo_tcdm_wide_wo_resp_to_router[c],         floo_tcdm_wide_wo_resp_in_trans[c]}),
        .valid_o        ({floo_tcdm_wide_wo_resp_from_router_valid[c], floo_tcdm_wide_wo_resp_valid_out_trans[c]}),
        .ready_i        ({floo_tcdm_wide_wo_resp_from_router_ready[c], floo_tcdm_wide_wo_resp_ready_in_trans[c]}),
        .data_o         ({floo_tcdm_wide_wo_resp_from_router[c],       floo_tcdm_wide_wo_resp_out_trans[c]}),
        .credit_o       (),
        .credit_i       ('0),
        .offload_req_o  (),
        .offload_rsp_i  ('0)
      );
    end
  end

  // A write ack arriving on channel c goes to the requesting tile's port
  // c % ports.
  floo_tcdm_wide_wr_resp_t [NumTilesPerGroup*NumWideWoRespPortsPerTile-1:0] floo_tcdm_wide_wo_resp_after_xbar;
  logic [NumTilesPerGroup*NumWideWoRespPortsPerTile-1:0] floo_tcdm_wide_wo_resp_after_xbar_valid, floo_tcdm_wide_wo_resp_after_xbar_ready;
  logic [NumWideWoRespChannelsPerGroup-1:0]
      [idx_width(NumTilesPerGroup*NumWideWoRespPortsPerTile)-1:0] floo_tcdm_wide_wo_resp_xbar_sel;
  for (genvar c = 0; c < NumWideWoRespChannelsPerGroup; c++) begin : gen_wide_wo_resp_xbar_sel
    assign floo_tcdm_wide_wo_resp_xbar_sel[c] = idx_width(NumTilesPerGroup*NumWideWoRespPortsPerTile)'(
        ((NumTilesPerGroup == 1) ? 0 :
         WideTileSelW'(floo_tcdm_wide_wo_resp_from_router[c].hdr.tile_id)) *
        NumWideWoRespPortsPerTile + c % NumWideWoRespPortsPerTile);
  end

  stream_xbar #(
    .NumInp   (NumWideWoRespChannelsPerGroup                ),
    .NumOut   (NumTilesPerGroup * NumWideWoRespPortsPerTile ),
    .payload_t(floo_tcdm_wide_wr_resp_t                     )
  ) i_wide_wo_resp_xbar (
    .clk_i  (clk_i                                    ),
    .rst_ni (rst_ni                                   ),
    .flush_i(1'b0                                     ),
    .rr_i   ('0                                       ),
    .data_i (floo_tcdm_wide_wo_resp_from_router       ),
    .valid_i(floo_tcdm_wide_wo_resp_from_router_valid ),
    .ready_o(floo_tcdm_wide_wo_resp_from_router_ready ),
    .sel_i  (floo_tcdm_wide_wo_resp_xbar_sel                                 ),
    .data_o (floo_tcdm_wide_wo_resp_after_xbar                                 ),
    .valid_o(floo_tcdm_wide_wo_resp_after_xbar_valid                           ),
    .ready_i(floo_tcdm_wide_wo_resp_after_xbar_ready                           ),
    .idx_o  (/* unused */                             )
  );

  for (genvar t = 0; t < NumTilesPerGroup; t++) begin : gen_wide_wo_resp_master_t
    for (genvar w = 0; w < NumWideWoRespPortsPerTile; w++) begin : gen_wide_wo_resp_master_w
      localparam int unsigned Oup = t * NumWideWoRespPortsPerTile + w;
      // A write ack carries no data.
      assign tcdm_wide_master_resp[t][WideWoRespBase+w] = '{
        rdata: '{
          meta_id: floo_tcdm_wide_wo_resp_after_xbar[Oup].hdr.meta_id,
          core_id: floo_tcdm_wide_wo_resp_after_xbar[Oup].hdr.core_id,
          amo:     '0,
          data:    '0
        },
        tile_id: floo_tcdm_wide_wo_resp_after_xbar[Oup].hdr.tile_id
      };
      assign tcdm_wide_master_resp_valid[t][WideWoRespBase+w] = floo_tcdm_wide_wo_resp_after_xbar_valid[Oup];
      assign floo_tcdm_wide_wo_resp_after_xbar_ready[Oup] = tcdm_wide_master_resp_ready[t][WideWoRespBase+w];
    end
  end
`endif

// ----------------------------- //
// AXI FlooNoC Network Interface //
// ----------------------------- //

floo_req_t  [Eject:North] floo_axi_req_out, floo_axi_req_in;
floo_rsp_t  [Eject:North] floo_axi_rsp_out, floo_axi_rsp_in;
floo_wide_t [Eject:North] floo_axi_wide_out, floo_axi_wide_in;

// This group's source-routing table: row floo_id_i of the FlooGen table
// (floo_id_i is the group's endpoint index). Selected here from the ID rather
// than passed in on NumEndpoints*$bits(route_t) input pins, so every group
// is still the same block.
route_t [NumEndpoints-1:0] route_table;
assign route_table = RoutingTables[floo_id_i];

floo_nw_chimney #(
  .AxiCfgN             (AxiCfgN                                 ),
  .AxiCfgW             (AxiCfgW                                 ),
  .ChimneyCfgN         (set_ports(ChimneyDefaultCfg, 1'b0, 1'b0)),
  .ChimneyCfgW         (set_ports(ChimneyDefaultCfg, 1'b0, 1'b1)),
  .AtopSupport         ('0                                      ),
  .RouteCfg            (RouteCfg                                ),
  .id_t                (id_t                                    ),
  .rob_idx_t           (rob_idx_t                               ),
  .route_t             (route_t                                 ),
  .dst_t               (route_t                                 ),
  .hdr_t               (hdr_t                                   ),
  .sam_rule_t          (sam_rule_t                              ),
  .Sam                 (Sam                                     ),
  .axi_narrow_in_req_t (axi_narrow_in_req_t                     ),
  .axi_narrow_in_rsp_t (axi_narrow_in_rsp_t                     ),
  .axi_narrow_out_req_t(axi_narrow_out_req_t                    ),
  .axi_narrow_out_rsp_t(axi_narrow_out_rsp_t                    ),
  .axi_wide_in_req_t   (axi_wide_in_req_t                       ),
  .axi_wide_in_rsp_t   (axi_wide_in_rsp_t                       ),
  .axi_wide_out_req_t  (axi_wide_out_req_t                      ),
  .axi_wide_out_rsp_t  (axi_wide_out_rsp_t                      ),
  .floo_req_t          (floo_req_t                              ),
  .floo_rsp_t          (floo_rsp_t                              ),
  .floo_wide_t         (floo_wide_t                             )
) i_floo_narrow_wide_chimney (
  .clk_i,
  .rst_ni,
  .test_enable_i        (testmode_i                             ),
  .sram_cfg_i           ('0                                     ),
  .axi_narrow_in_req_i  ('0                                     ),
  .axi_narrow_in_rsp_o  (                                       ),
  .axi_narrow_out_req_o (                                       ),
  .axi_narrow_out_rsp_i ('0                                     ),
  .axi_wide_in_req_i    (axi_mst_req                            ),
  .axi_wide_in_rsp_o    (axi_mst_resp                           ),
  .axi_wide_out_req_o   (                                       ),
  .axi_wide_out_rsp_i   ('0                                     ),
  .id_i                 (floo_id_i                              ),
  .route_table_i        (route_table                            ),
  .floo_req_o           (floo_axi_req_in[Eject]                 ),
  .floo_rsp_o           (floo_axi_rsp_in[Eject]                 ),
  .floo_wide_o          (floo_axi_wide_in[Eject]                ),
  .floo_req_i           (floo_axi_req_out[Eject]                ),
  .floo_rsp_i           (floo_axi_rsp_out[Eject]                ),
  .floo_wide_i          (floo_axi_wide_out[Eject]               )
);

// ----------------- //
// AXI FlooNoC Rouer //
// ----------------- //

floo_nw_router #(
  .AxiCfgN      ( AxiCfgN                     ),
  .AxiCfgW      ( AxiCfgW                     ),
  .RouteAlgo    ( RouteCfg.RouteAlgo          ),
  .NumRoutes    ( mempool_pkg::NumDirections  ),
  .InFifoDepth  ( 2                           ),
  .OutFifoDepth ( 2                           ),
  .id_t         ( id_t                        ),
  .hdr_t        ( hdr_t                       ),
  .floo_req_t   ( floo_req_t                  ),
  .floo_rsp_t   ( floo_rsp_t                  ),
  .floo_wide_t  ( floo_wide_t                 ),
  .NumAddrRules (1                            )
) i_floo_narrow_wide_router (
  .clk_i,
  .rst_ni,
  .test_enable_i        ( testmode_i        ),
  .id_i                 ( '0                ),
  .id_route_map_i       ( '0                ),
  .floo_req_i           ( floo_axi_req_in   ),
  .floo_rsp_i           ( floo_axi_rsp_in   ),
  .floo_req_o           ( floo_axi_req_out  ),
  .floo_rsp_o           ( floo_axi_rsp_out  ),
  .floo_wide_i          ( floo_axi_wide_in  ),
  .floo_wide_o          ( floo_axi_wide_out ),
  .offload_wide_req_o   (                   ),
  .offload_wide_rsp_i   ( '0                ),
  .offload_narrow_req_o (                   ),
  .offload_narrow_rsp_i ( '0                )
);

assign floo_axi_req_o  = floo_axi_req_out[West:North];
assign floo_axi_rsp_o  = floo_axi_rsp_out[West:North];
assign floo_axi_wide_o = floo_axi_wide_out[West:North];
assign floo_axi_req_in[West:North]  = floo_axi_req_i;
assign floo_axi_rsp_in[West:North]  = floo_axi_rsp_i;
assign floo_axi_wide_in[West:North] = floo_axi_wide_i;

endmodule
