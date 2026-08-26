// Copyright 2025 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Author: Yichao Zhang <yiczhang@iis.ee.ethz.ch>

`include "common_cells/registers.svh"

module terapool_cluster_floonoc_wrapper
  import mempool_pkg::*;
  import cf_math_pkg::idx_width;
  import floo_pkg::*;
  import floo_terapool_noc_pkg::*;
#(
  // TCDM
  parameter addr_t        TCDMBaseAddr   = 32'b0,
  // Boot address
  parameter logic [31:0]  BootAddr       = 32'h0000_0000,
  // Dependant parameters. DO NOT CHANGE!
  parameter int unsigned  NumDMAReq      = NumGroups * NumDmasPerGroup,
  // One AXI channel per L2/HBM bank, attached to the mesh perimeter. This is
  // NOT the per-group master count: a NxN mesh offers 4N perimeter ports for
  // N^2 groups, and the two only coincide at N=4.
  parameter int unsigned  NumAXIMasters  = l2_bank_map_pkg::NumL2BankPorts
) (
  // Clock and reset
  input  logic                          clk_i,
  input  logic                          rst_ni,
  input  logic                          testmode_i,

  // Scan chain
  input  logic                          scan_enable_i,
  input  logic                          scan_data_i,
  output logic                          scan_data_o,

  // Wake-up signal
  input  logic         [NumCores-1:0]   wake_up_i,

  // RO-Cache configuration
  input  ro_cache_ctrl_t                ro_cache_ctrl_i,

  // DMA request
  input  dma_req_t     [NumGroups-1:0]  dma_req_i,
  input  logic         [NumGroups-1:0]  dma_req_valid_i,
  output logic         [NumGroups-1:0]  dma_req_ready_o,

  // DMA status
  output dma_meta_t    [NumGroups-1:0]  dma_meta_o,

  // AXI Interface
  input  floo_req_t    [NumAXIMasters-1:0] floo_axi_req_i,
  input  floo_rsp_t    [NumAXIMasters-1:0] floo_axi_rsp_i,
  input  floo_wide_t   [NumAXIMasters-1:0] floo_axi_wide_i,
  output floo_req_t    [NumAXIMasters-1:0] floo_axi_req_o,
  output floo_rsp_t    [NumAXIMasters-1:0] floo_axi_rsp_o,
  output floo_wide_t   [NumAXIMasters-1:0] floo_axi_wide_o
);

  /*********************
   *  Control Signals  *
   *********************/
  logic [NumCores-1:0] wake_up_q;
  `FF(wake_up_q, wake_up_i, '0, clk_i, rst_ni);

  // fractal_sync leaves, gathered per group for the cluster tree. Responses are
  // driven to a DEFINED value until the tree is instantiated: the tile shim ORs
  // the response wake into every core's wake_up, so an undriven input puts X on
  // the wake of every core and the cluster never boots.
  fsync_req_t [NumGroups-1:0][NumTilesPerGroup-1:0] fsync_h_req, fsync_v_req;
  fsync_rsp_t [NumGroups-1:0][NumTilesPerGroup-1:0] fsync_h_rsp, fsync_v_rsp;
  for (genvar g = 0; unsigned'(g) < NumGroups; g++) begin : gen_fsync_tieoff
    for (genvar t = 0; unsigned'(t) < NumTilesPerGroup; t++) begin
      assign fsync_h_rsp[g][t] = '0;
      assign fsync_v_rsp[g][t] = '0;
    end
  end

  ro_cache_ctrl_t [NumGroups-1:0] ro_cache_ctrl_q;
  for (genvar g = 0; unsigned'(g) < NumGroups; g++) begin : gen_ro_cache_ctrl_q
    `FF(ro_cache_ctrl_q[g], ro_cache_ctrl_i, ro_cache_ctrl_default, clk_i, rst_ni);
  end : gen_ro_cache_ctrl_q

  /*********
   *  DMA  *
   *********/
  dma_req_t  [NumGroups-1:0] dma_req_group, dma_req_group_q;
  logic      [NumGroups-1:0] dma_req_group_valid, dma_req_group_q_valid;
  logic      [NumGroups-1:0] dma_req_group_ready, dma_req_group_q_ready;
  dma_meta_t [NumGroups-1:0] dma_meta, dma_meta_q;

  assign dma_req_group = dma_req_i;
  assign dma_req_group_valid = dma_req_valid_i;
  assign dma_req_ready_o = dma_req_group_ready;
  assign dma_meta_o = dma_meta_q;

  `FF(dma_meta_q, dma_meta, '0, clk_i, rst_ni);

  for (genvar g = 0; unsigned'(g) < NumGroups; g++) begin: gen_dma_req_group_register
    spill_register #(
      .T(dma_req_t)
    ) i_dma_req_group_register (
      .clk_i  (clk_i                   ),
      .rst_ni (rst_ni                  ),
      .data_i (dma_req_group[g]        ),
      .valid_i(dma_req_group_valid[g]  ),
      .ready_o(dma_req_group_ready[g]  ),
      .data_o (dma_req_group_q[g]      ),
      .valid_o(dma_req_group_q_valid[g]),
      .ready_i(dma_req_group_q_ready[g])
    );
  end : gen_dma_req_group_register

  /************
   *  Groups  *
   ************/
  // FlooNoC TCDM interfaces
  floo_tcdm_req_if_t [NumX-1:0][NumY-1:0][West:North] floo_tcdm_req_out, floo_tcdm_req_in;
  floo_tcdm_rsp_if_t [NumX-1:0][NumY-1:0][West:North] floo_tcdm_rsp_out, floo_tcdm_rsp_in;

  // FlooNoC AXI interfaces
  floo_terapool_noc_pkg::floo_req_t  [NumX-1:0][NumY-1:0][West:North] floo_axi_req_out,  floo_axi_req_in;
  floo_terapool_noc_pkg::floo_rsp_t  [NumX-1:0][NumY-1:0][West:North] floo_axi_rsp_out,  floo_axi_rsp_in;
  floo_terapool_noc_pkg::floo_wide_t [NumX-1:0][NumY-1:0][West:North] floo_axi_wide_out, floo_axi_wide_in;

  // Chimney configuration
  localparam floo_pkg::chimney_cfg_t ChimneyCfgN = floo_pkg::set_ports(floo_pkg::ChimneyDefaultCfg, 1'b0, 1'b0);
  localparam floo_pkg::chimney_cfg_t ChimneyCfgW = floo_pkg::set_ports(floo_pkg::ChimneyDefaultCfg, 1'b1, 1'b0);

  for (genvar x = 0; x < NumX; x++) begin : gen_groups_x
    for (genvar y = 0; y < NumY; y++) begin : gen_groups_y
      group_xy_id_t group_id;
      assign group_id = '{x:x, y:y, port_id:1'b0};
      // Elaboration-time group id for this mesh position. A localparam, not the
      // group_id signal: indexing per-group arrays with a signal would build a
      // NumGroups-way mux per group instead of a static wire.
      localparam int unsigned GId =
          gid_of_xy(group_x_id_t'(x), group_y_id_t'(y));

      if (x == 0) begin : gen_hbm_chimney_west
        // West
        if (NocTopology == 1) begin
          assign floo_tcdm_req_in[x][y][West] = floo_tcdm_req_out[NumX-1][y][East];
          assign floo_tcdm_rsp_in[x][y][West] = floo_tcdm_rsp_out[NumX-1][y][East];
        end else begin
          assign floo_tcdm_req_in[x][y][West] = '0;
          assign floo_tcdm_rsp_in[x][y][West] = '0;
        end

        // East
        assign floo_tcdm_req_in[x][y][East] = floo_tcdm_req_out[x+1][y][West];
        assign floo_tcdm_rsp_in[x][y][East] = floo_tcdm_rsp_out[x+1][y][West];

        // AXI East
        assign floo_axi_req_in[x][y][East]  = floo_axi_req_out[x+1][y][West];
        assign floo_axi_rsp_in[x][y][East]  = floo_axi_rsp_out[x+1][y][West];
        assign floo_axi_wide_in[x][y][East] = floo_axi_wide_out[x+1][y][West];
        // AXI West is a perimeter chimney: driven by gen_hbm_chimney below.

      end else if (x == NumX-1) begin : gen_hbm_chimney_east
        // East
        if (NocTopology == 1) begin
          assign floo_tcdm_req_in[x][y][East] = floo_tcdm_req_out[0][y][West];
          assign floo_tcdm_rsp_in[x][y][East] = floo_tcdm_rsp_out[0][y][West];
        end else begin
          assign floo_tcdm_req_in[x][y][East] = '0;
          assign floo_tcdm_rsp_in[x][y][East] = '0;
        end

        // West
        assign floo_tcdm_req_in[x][y][West] = floo_tcdm_req_out[x-1][y][East];
        assign floo_tcdm_rsp_in[x][y][West] = floo_tcdm_rsp_out[x-1][y][East];

        // AXI West
        assign floo_axi_req_in[x][y][West]  = floo_axi_req_out[x-1][y][East];
        assign floo_axi_rsp_in[x][y][West]  = floo_axi_rsp_out[x-1][y][East];
        assign floo_axi_wide_in[x][y][West] = floo_axi_wide_out[x-1][y][East];

        // AXI East is a perimeter chimney: driven by gen_hbm_chimney below.

      end else begin : gen_hor_connections
        // East
        assign floo_tcdm_req_in[x][y][East] = floo_tcdm_req_out[x+1][y][West];
        assign floo_tcdm_rsp_in[x][y][East] = floo_tcdm_rsp_out[x+1][y][West];
        assign floo_axi_req_in[x][y][East]  = floo_axi_req_out[x+1][y][West];
        assign floo_axi_rsp_in[x][y][East]  = floo_axi_rsp_out[x+1][y][West];
        assign floo_axi_wide_in[x][y][East] = floo_axi_wide_out[x+1][y][West];

        // West
        assign floo_tcdm_req_in[x][y][West] = floo_tcdm_req_out[x-1][y][East];
        assign floo_tcdm_rsp_in[x][y][West] = floo_tcdm_rsp_out[x-1][y][East];
        assign floo_axi_req_in[x][y][West]  = floo_axi_req_out[x-1][y][East];
        assign floo_axi_rsp_in[x][y][West]  = floo_axi_rsp_out[x-1][y][East];
        assign floo_axi_wide_in[x][y][West] = floo_axi_wide_out[x-1][y][East];
      end

      if (y == 0) begin : gen_hbm_chimney_south
        // South
        if (NocTopology == 1) begin
          assign floo_tcdm_req_in[x][y][South] = floo_tcdm_req_out[x][NumY-1][North];
          assign floo_tcdm_rsp_in[x][y][South] = floo_tcdm_rsp_out[x][NumY-1][North];
        end else begin
          assign floo_tcdm_req_in[x][y][South] = '0;
          assign floo_tcdm_rsp_in[x][y][South] = '0;
        end

        // North
        assign floo_tcdm_req_in [x][y][North] = floo_tcdm_req_out [x][y+1][South];
        assign floo_tcdm_rsp_in [x][y][North] = floo_tcdm_rsp_out [x][y+1][South];

        // AXI North
        assign floo_axi_req_in  [x][y][North] = floo_axi_req_out  [x][y+1][South];
        assign floo_axi_rsp_in  [x][y][North] = floo_axi_rsp_out  [x][y+1][South];
        assign floo_axi_wide_in [x][y][North] = floo_axi_wide_out [x][y+1][South];

        // AXI South is a perimeter chimney: driven by gen_hbm_chimney below.

      end else if (y == NumY-1) begin
        // TCDM North
        if (NocTopology == 1) begin
          assign floo_tcdm_req_in[x][y][North] = floo_tcdm_req_out[x][0][South];
          assign floo_tcdm_rsp_in[x][y][North] = floo_tcdm_rsp_out[x][0][South];
        end else begin
          assign floo_tcdm_req_in[x][y][North] = '0;
          assign floo_tcdm_rsp_in[x][y][North] = '0;
        end

        // TCDM South
        assign floo_tcdm_req_in[x][y][South] = floo_tcdm_req_out[x][y-1][North];
        assign floo_tcdm_rsp_in[x][y][South] = floo_tcdm_rsp_out[x][y-1][North];

        // AXI South
        assign floo_axi_req_in [x][y][South] = floo_axi_req_out [x][y-1][North];
        assign floo_axi_rsp_in [x][y][South] = floo_axi_rsp_out [x][y-1][North];
        assign floo_axi_wide_in[x][y][South] = floo_axi_wide_out[x][y-1][North];

        // AXI North is a perimeter chimney: driven by gen_hbm_chimney below.

      end else begin
        // North
        assign floo_tcdm_req_in [x][y][North] = floo_tcdm_req_out [x][y+1][South];
        assign floo_tcdm_rsp_in [x][y][North] = floo_tcdm_rsp_out [x][y+1][South];
        assign floo_axi_req_in  [x][y][North] = floo_axi_req_out  [x][y+1][South];
        assign floo_axi_rsp_in  [x][y][North] = floo_axi_rsp_out  [x][y+1][South];
        assign floo_axi_wide_in [x][y][North] = floo_axi_wide_out [x][y+1][South];

        // South
        assign floo_tcdm_req_in [x][y][South] = floo_tcdm_req_out [x][y-1][North];
        assign floo_tcdm_rsp_in [x][y][South] = floo_tcdm_rsp_out [x][y-1][North];
        assign floo_axi_req_in  [x][y][South] = floo_axi_req_out  [x][y-1][North];
        assign floo_axi_rsp_in  [x][y][South] = floo_axi_rsp_out  [x][y-1][North];
        assign floo_axi_wide_in [x][y][South] = floo_axi_wide_out [x][y-1][North];
      end

      if (PostLayoutGr & (x == 0) & (y == 0)) begin : gen_postly_group
        mempool_group_floonoc_wrapper_postlayout i_group (
          .clk_i              (clk_i),
          .rst_ni             (rst_ni),
          .testmode_i         (testmode_i),
          .scan_enable_i      (scan_enable_i),
          .scan_data_i        (1'b0), // stitched by DFT insertion
          .scan_data_o        (/* Unconnected */),
          .group_id_i         (gid_of_xy(group_id.x, group_id.y)),
          .floo_id_i          (id_t'(GroupX0Y0 + x*NumY + y)),
          // TCDM Router interface
          .floo_tcdm_req_o    (floo_tcdm_req_out[x][y]),
          .floo_tcdm_rsp_o    (floo_tcdm_rsp_out[x][y]),
          .floo_tcdm_req_i    (floo_tcdm_req_in[x][y]),
          .floo_tcdm_rsp_i    (floo_tcdm_rsp_in[x][y]),
      // Indexed by GROUP ID, not by mesh position. mempool_group.sv:153 builds
      // tile ids from group_id_i (= gid_of_xy, Morton) and mempool_tile.sv:260
      // builds hart_id from those, so a raster index here permutes which
      // physical group receives which group's signals. Invisible while wake is
      // a broadcast and every group gets the same DMA descriptor; a wrong-group
      // fault the moment P3 gives each group its own.
          .wake_up_i          (wake_up_q[GId*NumCoresPerGroup +: NumCoresPerGroup]),
          .fsync_h_req_o      (fsync_h_req[GId]),
          .fsync_h_rsp_i      (fsync_h_rsp[GId]),
          .fsync_v_req_o      (fsync_v_req[GId]),
          .fsync_v_rsp_i      (fsync_v_rsp[GId]),
          .ro_cache_ctrl_i    (ro_cache_ctrl_q[GId]),
          // DMA request
          .dma_req_i          (dma_req_group_q[GId]),
          .dma_req_valid_i    (dma_req_group_q_valid[GId]),
          .dma_req_ready_o    (dma_req_group_q_ready[GId]),
          // DMA status
          .dma_meta_o_backend_idle_   (dma_meta[GId][1]),
          .dma_meta_o_trans_complete_ (dma_meta[GId][0]),
          // AXI Router interface
          .floo_axi_req_o     (floo_axi_req_out[x][y]),
          .floo_axi_rsp_o     (floo_axi_rsp_out[x][y]),
          .floo_axi_wide_o    (floo_axi_wide_out[x][y]),
          .floo_axi_req_i     (floo_axi_req_in[x][y]),
          .floo_axi_rsp_i     (floo_axi_rsp_in[x][y]),
          .floo_axi_wide_i    (floo_axi_wide_in[x][y])
        );

      end else begin : gen_rtl_group
        mempool_group_floonoc_wrapper #(
          .TCDMBaseAddr       (TCDMBaseAddr),
          .BootAddr           (BootAddr)
        ) i_group (
          .clk_i              (clk_i),
          .rst_ni             (rst_ni),
          .testmode_i         (testmode_i),
          .scan_enable_i      (scan_enable_i),
          .scan_data_i        (1'b0), // stitched by DFT insertion
          .scan_data_o        (/* Unconnected */),
          .group_id_i         (gid_of_xy(group_id.x, group_id.y)),
          .floo_id_i          (id_t'(GroupX0Y0 + x*NumY + y)),
          // TCDM Router interface
          .floo_tcdm_req_o    (floo_tcdm_req_out[x][y]),
          .floo_tcdm_rsp_o    (floo_tcdm_rsp_out[x][y]),
          .floo_tcdm_req_i    (floo_tcdm_req_in[x][y]),
          .floo_tcdm_rsp_i    (floo_tcdm_rsp_in[x][y]),
          .wake_up_i          (wake_up_q[GId*NumCoresPerGroup +: NumCoresPerGroup]),
          .fsync_h_req_o      (fsync_h_req[GId]),
          .fsync_h_rsp_i      (fsync_h_rsp[GId]),
          .fsync_v_req_o      (fsync_v_req[GId]),
          .fsync_v_rsp_i      (fsync_v_rsp[GId]),
          .ro_cache_ctrl_i    (ro_cache_ctrl_q[GId]),
          // DMA request
          .dma_req_i          (dma_req_group_q[GId]),
          .dma_req_valid_i    (dma_req_group_q_valid[GId]),
          .dma_req_ready_o    (dma_req_group_q_ready[GId]),
          // DMA status
          .dma_meta_o         (dma_meta[GId]),
          // AXI Router interface
          .floo_axi_req_o     (floo_axi_req_out[x][y]),
          .floo_axi_rsp_o     (floo_axi_rsp_out[x][y]),
          .floo_axi_wide_o    (floo_axi_wide_out[x][y]),
          .floo_axi_req_i     (floo_axi_req_in[x][y]),
          .floo_axi_rsp_i     (floo_axi_rsp_in[x][y]),
          .floo_axi_wide_i    (floo_axi_wide_in[x][y])
        );
      end
    end : gen_groups_y
  end : gen_groups_x

  /************************
   *  L2 / HBM chimneys   *
   ************************/
  // Every perimeter router port carries one L2/HBM channel. The bank -> port
  // map is generated by hardware/scripts/gen_l2_bank_map.py, which minimises
  // the mesh distance from each group to the bank holding its DMA stripe, and
  // is checked against the connections in the floogen YAML. Driving it from a
  // table rather than per-edge index arithmetic keeps this correct for any
  // mesh size: iterating over banks makes it structurally impossible for two
  // edges to claim the same channel.
  for (genvar b = 0; b < l2_bank_map_pkg::NumL2BankPorts; b++) begin : gen_hbm_chimney
    localparam int unsigned Bx = l2_bank_map_pkg::L2BankPortX[b];
    localparam int unsigned By = l2_bank_map_pkg::L2BankPortY[b];
    localparam int unsigned Bd = l2_bank_map_pkg::L2BankPortDir[b];

    assign floo_axi_req_in [Bx][By][Bd] = floo_axi_req_i[b];
    assign floo_axi_rsp_in [Bx][By][Bd] = floo_axi_rsp_i[b];
    assign floo_axi_wide_in[Bx][By][Bd] = floo_axi_wide_i[b];
    assign floo_axi_req_o[b]            = floo_axi_req_out [Bx][By][Bd];
    assign floo_axi_rsp_o[b]            = floo_axi_rsp_out [Bx][By][Bd];
    assign floo_axi_wide_o[b]           = floo_axi_wide_out[Bx][By][Bd];
  end : gen_hbm_chimney

  /****************
   *  Assertions  *
   ****************/

  if (l2_bank_map_pkg::NumL2BankPorts > 2*(NumX + NumY))
    $fatal(1, "[terapool] More L2 banks than mesh perimeter ports.");

  if (l2_bank_map_pkg::NumL2BankPorts != NumL2Banks)
    $fatal(1, "[terapool] The generated L2 bank map is stale: regenerate it.");

  if (NumCores > 1024)
    $fatal(1, "[mempool] MemPool is currently limited to 1024 cores.");

  if (NumTiles < NumGroups)
    $fatal(1, "[mempool] MemPool requires more tiles than groups.");

  if (NumCores != NumTiles * NumCoresPerTile)
    $fatal(1, "[mempool] The number of cores is not divisible by the number of cores per tile.");

  if (BankingFactor < 1)
    $fatal(1, "[mempool] The banking factor must be a positive integer.");

  if (BankingFactor != 2**$clog2(BankingFactor))
    $fatal(1, "[mempool] The banking factor must be a power of two.");

endmodule : terapool_cluster_floonoc_wrapper
