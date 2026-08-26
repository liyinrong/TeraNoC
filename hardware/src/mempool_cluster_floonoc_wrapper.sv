// Copyright 2025 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Author: Yichao Zhang <yiczhang@iis.ee.ethz.ch>

`include "common_cells/registers.svh"

module mempool_cluster_floonoc_wrapper
  import mempool_pkg::*;
  import cf_math_pkg::idx_width;
  import floo_pkg::*;
  import floo_terapool_noc_pkg::*;
#(
  // TCDM
  parameter addr_t         TCDMBaseAddr  = 32'b0,
  // Boot address
  parameter logic [31:0]   BootAddr      = 32'h0000_0000,
  // Dependent parameters. DO NOT CHANGE!
  parameter int unsigned   NumDMAReq     = NumGroups * NumDmasPerGroup,
  parameter int unsigned   NumAXIMasters = NumGroups * NumAXIMastersPerGroup
) (
  // Clock and reset
  input  logic                         clk_i,
  input  logic                         rst_ni,
  input  logic                         testmode_i,

  // Scan chain
  input  logic                         scan_enable_i,
  input  logic                         scan_data_i,
  output logic                         scan_data_o,

  // Wake up signal
  input  logic         [NumCores-1:0]  wake_up_i,

  // RO-Cache configuration
  input  ro_cache_ctrl_t               ro_cache_ctrl_i,

  // DMA request
  input  dma_req_t     [NumGroups-1:0] dma_req_i,
  input  logic         [NumGroups-1:0] dma_req_valid_i,
  output logic         [NumGroups-1:0] dma_req_ready_o,

  // DMA status
  output dma_meta_t    [NumGroups-1:0] dma_meta_o,

  // AXI Interface
  input  floo_req_t    [NumAXIMasters:0] floo_axi_req_i,
  input  floo_rsp_t    [NumAXIMasters:0] floo_axi_rsp_i,
  input  floo_wide_t   [NumAXIMasters:0] floo_axi_wide_i,
  output floo_req_t    [NumAXIMasters:0] floo_axi_req_o,
  output floo_rsp_t    [NumAXIMasters:0] floo_axi_rsp_o,
  output floo_wide_t   [NumAXIMasters:0] floo_axi_wide_o
);

  /*********************
   *  Control Signals  *
   *********************/
  logic [NumCores-1:0]            wake_up_q;
  `FF(wake_up_q, wake_up_i, '0, clk_i, rst_ni);

  // fractal_sync leaves, gathered per group for the cluster tree.
  fsync_req_t [NumGroups-1:0][NumTilesPerGroup-1:0] fsync_h_req, fsync_v_req;
  fsync_rsp_t [NumGroups-1:0][NumTilesPerGroup-1:0] fsync_h_rsp, fsync_v_rsp;

  /*****************************
   *  fractal_sync barrier     *
   *****************************/
  // Leaf = TILE. A group's tiles occupy a FSyncTpgSide-square block of the leaf
  // grid, so a level-k VC (2^k groups per axis) is exactly a
  // (2^k * FSyncTpgSide)-square block of leaves -- invariant #9. Verified for
  // every flavour and every level before this was written, WITH the negative
  // control that matters: under RASTER group ids a level-1 VC degenerates to a
  // 2x8 strip instead of a 4x4 block, so Morton is load-bearing here and not
  // merely a memory-map convenience.
  //
  // GROUP ids are Morton; LEAF ports are ROW-MAJOR (the tree computes
  // offset = global_row*NET_COLS + global_col). Mixing the two is the WP3 trap
  // one level down, so the bijection is asserted rather than trusted.
  localparam int unsigned FSyncLeaves = FSyncTileSide * FSyncTileSide;

  function automatic int unsigned fsync_leaf_of(input int unsigned gid,
                                                input int unsigned t);
    group_xy_id_t xy;
    int unsigned  tx, ty;
    xy = group_xy_of(group_id_t'(gid));
    tx = unsigned'(xy.x) * FSyncTpgSide + (t % FSyncTpgSide);
    ty = unsigned'(xy.y) * FSyncTpgSide + (t / FSyncTpgSide);
    return ty * FSyncTileSide + tx;
  endfunction

  fsync_req_t     fsync_h_leaf     [FSyncLeaves][1];
  fsync_rsp_t     fsync_h_leaf_rsp [FSyncLeaves][1];
  fsync_req_t     fsync_v_leaf     [FSyncLeaves][1];
  fsync_rsp_t     fsync_v_leaf_rsp [FSyncLeaves][1];
  fsync_nbr_req_t fsync_h_nbr_req  [FSyncLeaves];
  fsync_nbr_rsp_t fsync_h_nbr_rsp  [FSyncLeaves];
  fsync_nbr_req_t fsync_v_nbr_req  [FSyncLeaves];
  fsync_nbr_rsp_t fsync_v_nbr_rsp  [FSyncLeaves];
  fsync_out_req_t fsync_h_root_req [1][1];
  fsync_rsp_t     fsync_h_root_rsp [1][1];
  fsync_out_req_t fsync_v_root_req [1][1];
  fsync_rsp_t     fsync_v_root_rsp [1][1];

  for (genvar g = 0; unsigned'(g) < NumGroups; g++) begin : gen_fsync_leaf_map
    for (genvar t = 0; unsigned'(t) < NumTilesPerGroup; t++) begin
      localparam int unsigned L = fsync_leaf_of(g, t);
      assign fsync_h_leaf[L][0] = fsync_h_req[g][t];
      assign fsync_v_leaf[L][0] = fsync_v_req[g][t];
      assign fsync_h_rsp[g][t]  = fsync_h_leaf_rsp[L][0];
      assign fsync_v_rsp[g][t]  = fsync_v_leaf_rsp[L][0];
    end
  end

  // Root and neighbour ports tied off: this tree is the whole barrier domain.
  // The neighbour links carry id 2/3 (adjacent tiles under DIFFERENT sibling
  // nodes), which software does not request yet. Tying them off is safe --
  // an unmatched request never completes rather than mis-routing.
  for (genvar l = 0; unsigned'(l) < FSyncLeaves; l++) begin : gen_fsync_nbr_tie
    assign fsync_h_nbr_req[l] = '0;
    assign fsync_v_nbr_req[l] = '0;
  end
  assign fsync_h_root_rsp[0][0] = '0;
  assign fsync_v_root_rsp[0][0] = '0;

  if (FSyncTileSide == 4) begin : gen_fsync_4x4
    fractal_sync_4x4 i_fsync_tree (
      .clk_i, .rst_ni,
      .h_1d_fsync_req_i (fsync_h_leaf    ), .h_1d_fsync_rsp_o (fsync_h_leaf_rsp),
      .v_1d_fsync_req_i (fsync_v_leaf    ), .v_1d_fsync_rsp_o (fsync_v_leaf_rsp),
      .h_nbr_fsycn_req_i(fsync_h_nbr_req ), .h_nbr_fsycn_rsp_o(fsync_h_nbr_rsp ),
      .v_nbr_fsycn_req_i(fsync_v_nbr_req ), .v_nbr_fsycn_rsp_o(fsync_v_nbr_rsp ),
      .h_2d_fsync_req_o (fsync_h_root_req), .h_2d_fsync_rsp_i (fsync_h_root_rsp),
      .v_2d_fsync_req_o (fsync_v_root_req), .v_2d_fsync_rsp_i (fsync_v_root_rsp)
    );
  end else if (FSyncTileSide == 8) begin : gen_fsync_8x8
    fractal_sync_8x8 i_fsync_tree (
      .clk_i, .rst_ni,
      .h_1d_fsync_req_i (fsync_h_leaf    ), .h_1d_fsync_rsp_o (fsync_h_leaf_rsp),
      .v_1d_fsync_req_i (fsync_v_leaf    ), .v_1d_fsync_rsp_o (fsync_v_leaf_rsp),
      .h_nbr_fsycn_req_i(fsync_h_nbr_req ), .h_nbr_fsycn_rsp_o(fsync_h_nbr_rsp ),
      .v_nbr_fsycn_req_i(fsync_v_nbr_req ), .v_nbr_fsycn_rsp_o(fsync_v_nbr_rsp ),
      .h_2d_fsync_req_o (fsync_h_root_req), .h_2d_fsync_rsp_i (fsync_h_root_rsp),
      .v_2d_fsync_req_o (fsync_v_root_req), .v_2d_fsync_rsp_i (fsync_v_root_rsp)
    );
  end else if (FSyncTileSide == 16) begin : gen_fsync_16x16
    fractal_sync_16x16 i_fsync_tree (
      .clk_i, .rst_ni,
      .h_1d_fsync_req_i (fsync_h_leaf    ), .h_1d_fsync_rsp_o (fsync_h_leaf_rsp),
      .v_1d_fsync_req_i (fsync_v_leaf    ), .v_1d_fsync_rsp_o (fsync_v_leaf_rsp),
      .h_nbr_fsycn_req_i(fsync_h_nbr_req ), .h_nbr_fsycn_rsp_o(fsync_h_nbr_rsp ),
      .v_nbr_fsycn_req_i(fsync_v_nbr_req ), .v_nbr_fsycn_rsp_o(fsync_v_nbr_rsp ),
      .h_2d_fsync_req_o (fsync_h_root_req), .h_2d_fsync_rsp_i (fsync_h_root_rsp),
      .v_2d_fsync_req_o (fsync_v_root_req), .v_2d_fsync_rsp_i (fsync_v_root_rsp)
    );
  end else begin : gen_fsync_no_tree
    // The library ships only 2x2/4x4/8x8/16x16/32x32 and has NO generator, so
    // an unsupported geometry must fail loudly at elaboration rather than
    // leave every barrier silently unanswered.
    $fatal(1, "fractal_sync: no hand-written tree for a %0dx%0d tile grid",
           FSyncTileSide, FSyncTileSide);
  end

  // The leaf map must be a BIJECTION. A collision makes two tiles share a leaf
  // (barriers silently merge); a gap leaves a leaf unconnected (its scope never
  // completes). Both are far cheaper to catch here than from a frozen trace.
  `ifndef SYNTHESIS
  initial begin : check_fsync_leaf_bijection
    automatic int unsigned seen[] = new[FSyncLeaves];
    for (int unsigned l = 0; l < FSyncLeaves; l++) seen[l] = 0;
    for (int unsigned g = 0; g < NumGroups; g++)
      for (int unsigned t = 0; t < NumTilesPerGroup; t++)
        seen[fsync_leaf_of(g, t)]++;
    for (int unsigned l = 0; l < FSyncLeaves; l++)
      if (seen[l] != 1)
        $fatal(1, "fsync leaf %0d claimed %0d times (expected exactly 1)", l,
               seen[l]);
  end
  `endif


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
  floo_tcdm_req_if_t     [NumX-1:0][NumY-1:0][West:North] floo_tcdm_req_out,  floo_tcdm_req_in;
  floo_tcdm_rsp_if_t     [NumX-1:0][NumY-1:0][West:North] floo_tcdm_rsp_out,  floo_tcdm_rsp_in;

  // FlooNoC AXI interfaces
  floo_terapool_noc_pkg::floo_req_t   [NumX-1:0][NumY-1:0][West:North] floo_axi_req_out,   floo_axi_req_in;
  floo_terapool_noc_pkg::floo_rsp_t   [NumX-1:0][NumY-1:0][West:North] floo_axi_rsp_out,   floo_axi_rsp_in;
  floo_terapool_noc_pkg::floo_wide_t  [NumX-1:0][NumY-1:0][West:North] floo_axi_wide_out,  floo_axi_wide_in;

  // Chimney configuration
  localparam floo_pkg::chimney_cfg_t ChimneyCfgN = floo_pkg::set_ports(floo_pkg::ChimneyDefaultCfg, 1'b0, 1'b0);
  localparam floo_pkg::chimney_cfg_t ChimneyCfgW = floo_pkg::set_ports(floo_pkg::ChimneyDefaultCfg, 1'b1, 1'b0);

  for (genvar x = 0; x < NumX; x++) begin : gen_groups_x
    for (genvar y = 0; y < NumY; y++) begin : gen_groups_y
      group_xy_id_t group_id;
      assign group_id = '{x: x, y: y, port_id: 1'b0};
      // Elaboration-time group id for this mesh position. A localparam, not the
      // group_id signal: indexing per-group arrays with a signal would build a
      // NumGroups-way mux per group instead of a static wire.
      localparam int unsigned GId =
          gid_of_xy(group_x_id_t'(x), group_y_id_t'(y));

      if (x == 0) begin : gen_hbm_chimney_west
        // TCDM West
        if (NocTopology == 1) begin
          assign floo_tcdm_req_in[x][y][West] = floo_tcdm_req_out[NumX-1][y][East];
          assign floo_tcdm_rsp_in[x][y][West] = floo_tcdm_rsp_out[NumX-1][y][East];
        end else begin
          assign floo_tcdm_req_in[x][y][West] = '0;
          assign floo_tcdm_rsp_in[x][y][West] = '0;
        end

        // TCDM East
        assign floo_tcdm_req_in[x][y][East] = floo_tcdm_req_out[x+1][y][West];
        assign floo_tcdm_rsp_in[x][y][East] = floo_tcdm_rsp_out[x+1][y][West];

        // AXI East
        assign floo_axi_req_in [x][y][East] = floo_axi_req_out [x+1][y][West];
        assign floo_axi_rsp_in [x][y][East] = floo_axi_rsp_out [x+1][y][West];
        assign floo_axi_wide_in[x][y][East] = floo_axi_wide_out[x+1][y][West];

        // AXI West
        assign floo_axi_req_in [x][y][West] = floo_axi_req_i [y];
        assign floo_axi_rsp_in [x][y][West] = floo_axi_rsp_i [y];
        assign floo_axi_wide_in[x][y][West] = floo_axi_wide_i[y];
        assign floo_axi_req_o [y] = floo_axi_req_out [x][y][West];
        assign floo_axi_rsp_o [y] = floo_axi_rsp_out [x][y][West];
        assign floo_axi_wide_o[y] = floo_axi_wide_out[x][y][West];
      end : gen_hbm_chimney_west

      else if (x == NumX-1) begin : gen_hbm_chimney_east
        // TCDM East
        if (NocTopology == 1) begin
          assign floo_tcdm_req_in[x][y][East] = floo_tcdm_req_out[0][y][West];
          assign floo_tcdm_rsp_in[x][y][East] = floo_tcdm_rsp_out[0][y][West];
        end else begin
          assign floo_tcdm_req_in[x][y][East] = '0;
          assign floo_tcdm_rsp_in[x][y][East] = '0;
        end

        // TCDM West
        assign floo_tcdm_req_in[x][y][West] = floo_tcdm_req_out[x-1][y][East];
        assign floo_tcdm_rsp_in[x][y][West] = floo_tcdm_rsp_out[x-1][y][East];

        // AXI West
        assign floo_axi_req_in [x][y][West] = floo_axi_req_out [x-1][y][East];
        assign floo_axi_rsp_in [x][y][West] = floo_axi_rsp_out [x-1][y][East];
        assign floo_axi_wide_in[x][y][West] = floo_axi_wide_out[x-1][y][East];

        // AXI East
        assign floo_axi_req_in [x][y][East] = floo_axi_req_i [y+2];
        assign floo_axi_rsp_in [x][y][East] = floo_axi_rsp_i [y+2];
        assign floo_axi_wide_in[x][y][East] = floo_axi_wide_i[y+2];
        assign floo_axi_req_o [y+2] = floo_axi_req_out [x][y][East];
        assign floo_axi_rsp_o [y+2] = floo_axi_rsp_out [x][y][East];
        assign floo_axi_wide_o[y+2] = floo_axi_wide_out[x][y][East];
      end : gen_hbm_chimney_east

      else begin
        $fatal(1, "[topology] The current mesh grid is not 2x2, please check cluster mesh topology");
      end

      if (y == 0) begin
        // TCDM South
        if (NocTopology == 1) begin
          assign floo_tcdm_req_in[x][y][South] = floo_tcdm_req_out[x][NumY-1][North];
          assign floo_tcdm_rsp_in[x][y][South] = floo_tcdm_rsp_out[x][NumY-1][North];
        end else begin
          assign floo_tcdm_req_in[x][y][South] = '0;
          assign floo_tcdm_rsp_in[x][y][South] = '0;
        end

        // TCDM North
        assign floo_tcdm_req_in[x][y][North] = floo_tcdm_req_out[x][y+1][South];
        assign floo_tcdm_rsp_in[x][y][North] = floo_tcdm_rsp_out[x][y+1][South];

        // AXI North
        assign floo_axi_req_in [x][y][North] = floo_axi_req_out [x][y+1][South];
        assign floo_axi_rsp_in [x][y][North] = floo_axi_rsp_out [x][y+1][South];
        assign floo_axi_wide_in[x][y][North] = floo_axi_wide_out[x][y+1][South];

        if (x == 0) begin : gen_periph_chimney_south
          // AXI South
          assign floo_axi_req_in [x][y][South] = floo_axi_req_i [4];
          assign floo_axi_rsp_in [x][y][South] = floo_axi_rsp_i [4];
          assign floo_axi_wide_in[x][y][South] = floo_axi_wide_i[4];
          assign floo_axi_req_o [4] = floo_axi_req_out [x][y][South];
          assign floo_axi_rsp_o [4] = floo_axi_rsp_out [x][y][South];
          assign floo_axi_wide_o[4] = floo_axi_wide_out[x][y][South];
        end : gen_periph_chimney_south

        else begin
          assign floo_axi_req_in [x][y][South] = '0;
          assign floo_axi_rsp_in [x][y][South] = '0;
          assign floo_axi_wide_in[x][y][South] = '0;
        end
      end

      else if (y == NumY-1) begin
        // TCDM North
        if (NocTopology == 1) begin
          assign floo_tcdm_req_in[x][y][North] = floo_tcdm_req_out[x][0][South];
          assign floo_tcdm_rsp_in[x][y][North] = floo_tcdm_rsp_out[x][0][South];
        end else begin
          assign floo_tcdm_req_in[x][y][North] = '0;
          assign floo_tcdm_rsp_in[x][y][North] = '0;
        end

        // AXI North
        assign floo_axi_req_in [x][y][North] = '0;
        assign floo_axi_rsp_in [x][y][North] = '0;
        assign floo_axi_wide_in[x][y][North] = '0;

        // TCDM South
        assign floo_tcdm_req_in[x][y][South] = floo_tcdm_req_out[x][y-1][North];
        assign floo_tcdm_rsp_in[x][y][South] = floo_tcdm_rsp_out[x][y-1][North];

        // AXI South
        assign floo_axi_req_in [x][y][South] = floo_axi_req_out [x][y-1][North];
        assign floo_axi_rsp_in [x][y][South] = floo_axi_rsp_out [x][y-1][North];
        assign floo_axi_wide_in[x][y][South] = floo_axi_wide_out[x][y-1][North];
      end else begin
        $fatal(1, "[Topology] The current mesh grid is not 2x2, please check cluster mesh topology");
      end

      if (PostLayoutGr & (x == 0) & (y == 0)) begin : gen_postly_group
        mempool_group_floonoc_wrapper_postlayout i_group (
          .clk_i                      (clk_i),
          .rst_ni                     (rst_ni),
          .testmode_i                 (testmode_i),
          .scan_enable_i              (scan_enable_i),
          .scan_data_i                (1'b0), // stitched by DFT insertion
          .scan_data_o                (/* Unconnected */),
          .group_id_i                 (gid_of_xy(group_id.x, group_id.y)),
          .floo_id_i                  (id_t'(GroupX0Y0 + x*NumY + y)),
          // TCDM Router interface
          .floo_tcdm_req_o            (floo_tcdm_req_out[x][y]),
          .floo_tcdm_rsp_o            (floo_tcdm_rsp_out[x][y]),
          .floo_tcdm_req_i            (floo_tcdm_req_in[x][y]),
          .floo_tcdm_rsp_i            (floo_tcdm_rsp_in[x][y]),
      // Indexed by GROUP ID, not by mesh position. mempool_group.sv:153 builds
      // tile ids from group_id_i (= gid_of_xy, Morton) and mempool_tile.sv:260
      // builds hart_id from those, so a raster index here permutes which
      // physical group receives which group's signals. Invisible while wake is
      // a broadcast and every group gets the same DMA descriptor; a wrong-group
      // fault the moment P3 gives each group its own.
          .wake_up_i                  (wake_up_q[GId*NumCoresPerGroup +: NumCoresPerGroup]),
          .fsync_h_req_o              (fsync_h_req[GId]),
          .fsync_h_rsp_i              (fsync_h_rsp[GId]),
          .fsync_v_req_o              (fsync_v_req[GId]),
          .fsync_v_rsp_i              (fsync_v_rsp[GId]),
          .ro_cache_ctrl_i            (ro_cache_ctrl_q[GId]),
          // DMA request
          .dma_req_i                  (dma_req_group_q[GId]),
          .dma_req_valid_i            (dma_req_group_q_valid[GId]),
          .dma_req_ready_o            (dma_req_group_q_ready[GId]),
          // DMA status
          .dma_meta_o_backend_idle_   (dma_meta[GId][1]),
          .dma_meta_o_trans_complete_ (dma_meta[GId][0]),
          // AXI Router interfac
          .floo_axi_req_o             (floo_axi_req_out[x][y]),
          .floo_axi_rsp_o             (floo_axi_rsp_out[x][y]),
          .floo_axi_wide_o            (floo_axi_wide_out[x][y]),
          .floo_axi_req_i             (floo_axi_req_in[x][y]),
          .floo_axi_rsp_i             (floo_axi_rsp_in[x][y]),
          .floo_axi_wide_i            (floo_axi_wide_in[x][y])
        );
      end else begin : gen_rtl_group
        mempool_group_floonoc_wrapper #(
          .TCDMBaseAddr               (TCDMBaseAddr),
          .BootAddr                   (BootAddr)
        ) i_group (
          .clk_i                      (clk_i),
          .rst_ni                     (rst_ni),
          .testmode_i                 (testmode_i),
          .scan_enable_i              (scan_enable_i),
          .scan_data_i                (1'b0), // stitched by DFT insertion
          .scan_data_o                (/* Unconnected */),
          .group_id_i                 (gid_of_xy(group_id.x, group_id.y)),
          .floo_id_i                  (id_t'(GroupX0Y0 + x*NumY + y)),
          // TCDM Router interface
          .floo_tcdm_req_o            (floo_tcdm_req_out[x][y]),
          .floo_tcdm_rsp_o            (floo_tcdm_rsp_out[x][y]),
          .floo_tcdm_req_i            (floo_tcdm_req_in[x][y]),
          .floo_tcdm_rsp_i            (floo_tcdm_rsp_in[x][y]),
          .wake_up_i                  (wake_up_q[GId*NumCoresPerGroup +: NumCoresPerGroup]),
          .fsync_h_req_o              (fsync_h_req[GId]),
          .fsync_h_rsp_i              (fsync_h_rsp[GId]),
          .fsync_v_req_o              (fsync_v_req[GId]),
          .fsync_v_rsp_i              (fsync_v_rsp[GId]),
          .ro_cache_ctrl_i            (ro_cache_ctrl_q[GId]),
          // DMA request
          .dma_req_i                  (dma_req_group_q[GId]),
          .dma_req_valid_i            (dma_req_group_q_valid[GId]),
          .dma_req_ready_o            (dma_req_group_q_ready[GId]),
          // DMA status
          .dma_meta_o                 (dma_meta[GId]),
          // AXI Router interfac
          .floo_axi_req_o             (floo_axi_req_out[x][y]),
          .floo_axi_rsp_o             (floo_axi_rsp_out[x][y]),
          .floo_axi_wide_o            (floo_axi_wide_out[x][y]),
          .floo_axi_req_i             (floo_axi_req_in[x][y]),
          .floo_axi_rsp_i             (floo_axi_rsp_in[x][y]),
          .floo_axi_wide_i            (floo_axi_wide_in[x][y])
        );
      end
    end : gen_groups_y
  end : gen_groups_x

  /****************
   *  Assertions  *
   ****************/

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

endmodule : mempool_cluster_floonoc_wrapper
