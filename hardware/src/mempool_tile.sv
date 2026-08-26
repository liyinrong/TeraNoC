// Copyright 2024 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

`include "mempool/mempool.svh"
`include "hci_helpers.svh"

/* verilator lint_off DECLFILENAME */
module mempool_tile
  import mempool_pkg::*;
  import cf_math_pkg::idx_width;
  import hci_package::*;
#(
  // TCDM
  parameter addr_t       TCDMBaseAddr = 32'b0,
  // Boot address
  parameter logic [31:0] BootAddr     = 32'h0000_1000,
  // Enable a RedMulE tensor core: widens the local interconnect by RMNarrowPorts
  // (its narrow ports without the wide plane; none with it).
  parameter logic        RedMulE      = 1'b0,
  parameter int unsigned NumLocalPorts = RedMulE ? RMNarrowPorts + NumCorePortsPerTile : NumCorePortsPerTile,
  // Wide engines have their own local-port axis: one slot per wide client (only
  // RedMulE's has an own-tile bank-interconnect port). Derived in mempool_pkg
  // from which clients the build instantiates (RedMulE, Spatz burst adapter).
  parameter int unsigned NumWideLocalPorts = mempool_pkg::NumWideLocalPorts,
  // Dependent parameters. DO NOT CHANGE.
  parameter int unsigned NumCaches    = NumCoresPerTile / NumCoresPerCache,
  // If NumDivsqrtPerTile is set, otherwise the parameter defaults to 1.
  parameter int unsigned NumCoresPerDivsqrt = |NumDivsqrtPerTile ? (NumCoresPerTile/NumDivsqrtPerTile) : NumCoresPerTile,
  // fractal_sync leaf payload types, supplied by the cluster that owns the tree
  // instance and therefore the aggregate/id widths.
  parameter int unsigned FSyncAggrWidth = mempool_pkg::FSyncAggrWidth,
  parameter int unsigned FSyncIdWidth   = mempool_pkg::FSyncIdWidth,
  parameter int unsigned FSyncLvlWidth  = mempool_pkg::FSyncLvlWidth,
  parameter type         fsync_req_t    = mempool_pkg::fsync_req_t,
  parameter type         fsync_rsp_t    = mempool_pkg::fsync_rsp_t
) (
  // Clock and reset
  input  logic                                                                    clk_i,
  input  logic                                                                    rst_ni,
  // Scan chain
  input  logic                                                                    scan_enable_i,
  input  logic                                                                    scan_data_i,
  output logic                                                                    scan_data_o,
  // Tile ID
  input  logic                            [idx_width(NumTiles)-1:0]               tile_id_i,
  // TCDM Master interfaces
  output `STRUCT_VECT(tcdm_master_req_t,  [NumNarrowRemoteReqPortsPerTile-1:0])         tcdm_master_req_o,
  output logic                            [NumNarrowRemoteReqPortsPerTile-1:0]          tcdm_master_req_valid_o,
  input  logic                            [NumNarrowRemoteReqPortsPerTile-1:0]          tcdm_master_req_ready_i,
  input  `STRUCT_VECT(tcdm_master_resp_t, [NumNarrowRemoteRespPortsPerTile-1:0])        tcdm_master_resp_i,
  input  logic                            [NumNarrowRemoteRespPortsPerTile-1:0]         tcdm_master_resp_valid_i,
  output logic                            [NumNarrowRemoteRespPortsPerTile-1:0]         tcdm_master_resp_ready_o,
  // TCDM slave interfaces
  input  `STRUCT_VECT(tcdm_slave_req_t,   [NumNarrowRemoteReqPortsPerTile-1:0])         tcdm_slave_req_i,
  input  logic                            [NumNarrowRemoteReqPortsPerTile-1:0]          tcdm_slave_req_valid_i,
  output logic                            [NumNarrowRemoteReqPortsPerTile-1:0]          tcdm_slave_req_ready_o,
  output `STRUCT_VECT(tcdm_slave_resp_t,  [NumNarrowRemoteRespPortsPerTile-1:0])        tcdm_slave_resp_o,
  output logic                            [NumNarrowRemoteRespPortsPerTile-1:0]         tcdm_slave_resp_valid_o,
  input  logic                            [NumNarrowRemoteRespPortsPerTile-1:0]         tcdm_slave_resp_ready_i,
  // TCDM Wide Master interfaces
  `ifdef USE_WIDE_TCDM_PLANE
  output `STRUCT_VECT(tcdm_wide_master_req_t,  [NumWideRemoteReqPortsPerTile-1:0])       tcdm_wide_master_req_o,
  output logic                                 [NumWideRemoteReqPortsPerTile-1:0]        tcdm_wide_master_req_valid_o,
  input  logic                                 [NumWideRemoteReqPortsPerTile-1:0]        tcdm_wide_master_req_ready_i,
  input  `STRUCT_VECT(tcdm_wide_master_resp_t, [NumWideRemoteRespPortsPerTile-1:0])      tcdm_wide_master_resp_i,
  input  logic                                 [NumWideRemoteRespPortsPerTile-1:0]       tcdm_wide_master_resp_valid_i,
  output logic                                 [NumWideRemoteRespPortsPerTile-1:0]       tcdm_wide_master_resp_ready_o,
  // TCDM Wide Slave interfaces
  input  `STRUCT_VECT(tcdm_wide_slave_req_t,   [NumWideRemoteReqPortsPerTile-1:0])       tcdm_wide_slave_req_i,
  input  logic                                 [NumWideRemoteReqPortsPerTile-1:0]        tcdm_wide_slave_req_valid_i,
  output logic                                 [NumWideRemoteReqPortsPerTile-1:0]        tcdm_wide_slave_req_ready_o,
  output `STRUCT_VECT(tcdm_wide_slave_resp_t,  [NumWideRemoteRespPortsPerTile-1:0])      tcdm_wide_slave_resp_o,
  output logic                                 [NumWideRemoteRespPortsPerTile-1:0]       tcdm_wide_slave_resp_valid_o,
  input  logic                                 [NumWideRemoteRespPortsPerTile-1:0]       tcdm_wide_slave_resp_ready_i,
  `endif
  // TCDM DMA interfaces
  input  `STRUCT_PORT(tcdm_dma_req_t)                                             tcdm_dma_req_i,
  input  logic                                                                    tcdm_dma_req_valid_i,
  output logic                                                                    tcdm_dma_req_ready_o,
  output `STRUCT_PORT(tcdm_dma_resp_t)                                            tcdm_dma_resp_o,
  output logic                                                                    tcdm_dma_resp_valid_o,
  input  logic                                                                    tcdm_dma_resp_ready_i,
  // AXI Interface
  output `STRUCT_PORT(axi_tile_req_t)                                             axi_mst_req_o,
  input  `STRUCT_PORT(axi_tile_resp_t)                                            axi_mst_resp_i,
  // Wake up interface
  input  logic                            [NumCoresPerTile-1:0]                   wake_up_i,
  // fractal_sync leaf: one pair per TILE, h and v, selected by id[0]
  output fsync_req_t                                                              fsync_h_req_o,
  input  fsync_rsp_t                                                              fsync_h_rsp_i,
  output fsync_req_t                                                              fsync_v_req_o,
  input  fsync_rsp_t                                                              fsync_v_rsp_i
);

  /****************
   *   Includes   *
   ****************/

  `include "common_cells/registers.svh"

  /*****************
   *  Definitions  *
   *****************/

  import snitch_pkg::dreq_t;
  import snitch_pkg::dresp_t;

  typedef logic [idx_width(NumNarrowRemoteReqPortsPerTile)-1:0] remote_ports_index_t;
  typedef logic [idx_width(NumWideRemoteReqPortsPerTile)-1:0] wide_remote_ports_index_t;
  // The wide RESPONSE select indexes the tile's LOCAL wide clients -- the demux outputs --
  // which is a different space from the remote request ports. They coincide only while
  // there are at most two clients; with RedMulE at slot 0 and a Spatz core at each of
  // 1..NumCoresPerTile, reusing the remote type truncated core_id and made slots 2 and
  // above unreachable, so those cores never received a wide response.
  typedef logic [idx_width(NumWideLocalPorts)-1:0] wide_local_ports_index_t;

  typedef logic [idx_width(NumLocalPorts + NumNarrowRemoteReqPortsPerTile)-1:0] local_req_interco_addr_t;
  typedef logic [idx_width(NumLocalPorts + NumNarrowRemoteRespPortsPerTile)-1:0] local_resp_interco_addr_t;

  `ifdef USE_WIDE_TCDM_PLANE
  typedef enum int unsigned {
    TCDM_WIDE_EXTERNAL = 0, TCDM_WIDE_LOCAL
  } tcdm_wide_addr_map_slave_t;

  localparam int unsigned NumWideShimPorts = TCDM_WIDE_LOCAL + 1;
  `endif

  /*********************
   *  Control Signals  *
   *********************/
  // wake_up mixes RedMulE's EOC (redmule_evt[0]) into bit 0; see gen_redmule.
  logic [NumCoresPerTile-1:0] wake_up_q, wake_up;
  `FF(wake_up_q, wake_up_i, '0, clk_i, rst_ni);

  // Group ID
  // ---- WP3 virtual-cluster remap enables --------------------------------
  // Deliberately THREE named constants, one per COMPUTE scrambler, with an
  // elaboration check that they agree. Enabling VC at only some of the three
  // is exactly the regression this guards: forgetting the wide RedMulE site
  // (i_rm_wide_scrambler) gives an accelerator that bypasses the map only
  // when the wide plane is on, which is invisible on every narrow build.
  // The DMA scrambler in mempool_group.sv is deliberately NOT in this set --
  // it stays virtual-cluster blind so a descriptor can never carry a VC
  // address (roadmap §T2.1, and DmaRegionStart/End is the global window only).
  localparam bit VirtualClusterCore     = VirtualCluster;
  localparam bit VirtualClusterRmWide   = VirtualCluster;
  localparam bit VirtualClusterRmNarrow = VirtualCluster;
  if (!((VirtualClusterCore == VirtualClusterRmWide) && (VirtualClusterRmWide == VirtualClusterRmNarrow)))
    $fatal(1, "The three compute scramblers disagree on EnableVirtualCluster (core=%0d wide=%0d narrow=%0d); all three must see the same map.",
           VirtualClusterCore, VirtualClusterRmWide, VirtualClusterRmNarrow);

  group_id_t group_id;
  if (NumGroups != 1) begin: gen_group_id
    assign group_id = tile_id_i[$clog2(NumTiles)-1 -: $clog2(NumGroups)];
  end else begin: gen_group_id
    assign group_id = '0;
  end: gen_group_id

  /***********
   *  Cores  *
   ***********/

  // Instruction interfaces
  addr_t                     [NumCaches-1:0][NumCoresPerCache-1:0] snitch_inst_addr;
  data_t                     [NumCaches-1:0][NumCoresPerCache-1:0] snitch_inst_data;
  logic                      [NumCaches-1:0][NumCoresPerCache-1:0] snitch_inst_valid;
  logic                      [NumCaches-1:0][NumCoresPerCache-1:0] snitch_inst_ready;

  // Shared operational units interfaces
  // Shared-accelerator (sh_acc) interface to the tile-level divsqrt, always
  // declared; a Spatz core ties its ports off and the tile divsqrt is scalar-only.
  logic                      [NumCoresPerTile-1:0]                 sh_acc_req_valid;
  logic                      [NumCoresPerTile-1:0]                 sh_acc_req_ready;
  logic                      [NumCoresPerTile-1:0]                 sh_acc_resp_valid;
  logic                      [NumCoresPerTile-1:0]                 sh_acc_resp_ready;
  snitch_pkg::acc_req_t      [NumCoresPerTile-1:0]                 acc_req;
  snitch_pkg::acc_resp_t     [NumCoresPerTile-1:0]                 acc_resp;
  snitch_pkg::sh_acc_req_t   [NumCoresPerTile-1:0]                 sh_acc_req;
  snitch_pkg::sh_acc_resp_t  [NumCoresPerTile-1:0]                 sh_acc_resp;

  // Data interfaces. The cores' TCDM ports are flattened onto one axis (port p
  // of core c = core_port_base(c) + p): one port per Snitch-only core, then
  // NumDataPortsPerCore per Snitch-Spatz core; in scalar builds each index is
  // the core id. meta_id_t is cluster-width; mempool_cc zero-extends the snitch
  // id onto it.
  localparam int unsigned NumCorePorts = NumCorePortsPerTile;
  addr_t    [NumCorePorts-1:0] snitch_data_qaddr;
  addr_t    [NumCorePorts-1:0] snitch_data_qaddr_scrambled;
  logic     [NumCorePorts-1:0] snitch_data_qwrite;
  amo_t     [NumCorePorts-1:0] snitch_data_qamo;
  data_t    [NumCorePorts-1:0] snitch_data_qdata;
  strb_t    [NumCorePorts-1:0] snitch_data_qstrb;
  meta_id_t [NumCorePorts-1:0] snitch_data_qid;
  logic     [NumCorePorts-1:0] snitch_data_qvalid;
  logic     [NumCorePorts-1:0] snitch_data_qready;
  data_t    [NumCorePorts-1:0] snitch_data_pdata;
  logic     [NumCorePorts-1:0] snitch_data_perror;
  meta_id_t [NumCorePorts-1:0] snitch_data_pid;
  logic     [NumCorePorts-1:0] snitch_data_pvalid;
  logic     [NumCorePorts-1:0] snitch_data_pready;

  // Divert layer between the cores and the narrow path. A Spatz burst leaves the
  // core on its vector port 0 as ONE request with burst_len > 1; it must go to the
  // wide plane instead of the narrow shim, and its beats come back from the wide
  // adapter. Everything else passes straight through, so with the burst path off
  // these are wires.
  logic [NumCorePorts-1:0][spatz_pkg::BurstLenWidth-1:0] cc_data_qburst_len;
  // The core's request fields. The divert layer passes them to snitch_data_q*,
  // except on the lanes an own-tile burst is being expanded on.
  addr_t    [NumCorePorts-1:0] cc_data_qaddr;
  logic     [NumCorePorts-1:0] cc_data_qwrite;
  amo_t     [NumCorePorts-1:0] cc_data_qamo;
  data_t    [NumCorePorts-1:0] cc_data_qdata;
  strb_t    [NumCorePorts-1:0] cc_data_qstrb;
  meta_id_t [NumCorePorts-1:0] cc_data_qid;
  // Per-core, per-lane ROB base ids for the in-flight burst (mempool_cc.burst_base_id_o).
  logic     [NumCorePorts-1:0] cc_data_qvalid;
  logic     [NumCorePorts-1:0] cc_data_qready;
  data_t    [NumCorePorts-1:0] cc_data_pdata;
  logic     [NumCorePorts-1:0] cc_data_perror;
  meta_id_t [NumCorePorts-1:0] cc_data_pid;
  logic     [NumCorePorts-1:0] cc_data_pvalid;
  logic     [NumCorePorts-1:0] cc_data_pready;

  // Tile-level shared divsqrt serves scalar cores only (Spatz has its own); !RVV.
  if (snitch_pkg::XDIVSQRT && !TrafficGeneration && !mempool_pkg::RVV) begin: gen_divsqrt
    for (genvar c = 0; unsigned'(c) < NumDivsqrtPerTile; c++) begin: gen_divsqrt
      logic                     divsqrt_req_valid;
      logic                     divsqrt_req_ready;
      logic                     divsqrt_resp_valid;
      logic                     divsqrt_resp_ready;
      snitch_pkg::sh_acc_req_t  divsqrt_req;
      snitch_pkg::sh_acc_resp_t divsqrt_resp;

      // Assign output to shared response
      for (genvar i = 0; unsigned'(i) < NumCoresPerDivsqrt; i++) begin
        assign sh_acc_resp[c*NumCoresPerDivsqrt + i] = divsqrt_resp;
      end

      // Shared accelerator arbiter
      stream_arbiter #(
        .DATA_T      ( snitch_pkg::sh_acc_req_t                                               ),
        .N_INP       ( NumCoresPerDivsqrt                                                     ),
        .ARBITER     ( "rr"                                                                   )
      ) i_stream_arbiter_offload (
        .clk_i       ( clk_i                                                                  ),
        .rst_ni      ( rst_ni                                                                 ),
        .inp_data_i  ( sh_acc_req[((c+1)*NumCoresPerDivsqrt)-1:(c*NumCoresPerDivsqrt)]        ),
        .inp_valid_i ( sh_acc_req_valid[((c+1)*NumCoresPerDivsqrt)-1:(c*NumCoresPerDivsqrt)]  ),
        .inp_ready_o ( sh_acc_req_ready[((c+1)*NumCoresPerDivsqrt)-1:(c*NumCoresPerDivsqrt)]  ),
        .oup_data_o  ( divsqrt_req                                                            ),
        .oup_valid_o ( divsqrt_req_valid                                                      ),
        .oup_ready_i ( divsqrt_req_ready                                                      )
      );

      // Shared accelerator output demux
      stream_demux #(
        .N_OUP ( NumCoresPerDivsqrt                                                               )
      ) i_stream_demux_offload (
        .inp_valid_i  ( divsqrt_resp_valid                                                        ),
        .inp_ready_o  ( divsqrt_resp_ready                                                        ),
        .oup_sel_i    ( sh_acc_resp[c*NumCoresPerDivsqrt].hart_id[$clog2(NumCoresPerDivsqrt)-1:0] ),
        .oup_valid_o  ( sh_acc_resp_valid[((c+1)*NumCoresPerDivsqrt)-1:(c*NumCoresPerDivsqrt)]    ),
        .oup_ready_i  ( sh_acc_resp_ready[((c+1)*NumCoresPerDivsqrt)-1:(c*NumCoresPerDivsqrt)]    )
      );

      // Tile shared divsqrt unit
      snitch_fp_divsqrt #(
        .FPUImplementation       (snitch_pkg::DIVSQRT_IMPLEMENTATION)
      ) i_snitch_divsqrt (
        .clk_i,
        .rst_i                   (!rst_ni             ),
        // pragma translate_off
        .trace_port_o            (                    ),
        // pragma translate_on
        .acc_req_i               ( divsqrt_req        ),
        .acc_req_valid_i         ( divsqrt_req_valid  ),
        .acc_req_ready_o         ( divsqrt_req_ready  ),
        .acc_resp_o              ( divsqrt_resp       ),
        .acc_resp_valid_o        ( divsqrt_resp_valid ),
        .acc_resp_ready_i        ( divsqrt_resp_ready ),
        .divsqrt_rnd_mode_i      ( fpnew_pkg::RNE     ),
        .divsqrt_status_o        (                    ),
        .core_events_o           (                    )
      );
    end
  end

  for (genvar c = 0; unsigned'(c) < NumCoresPerTile; c++) begin: gen_cores
    logic [31:0] hart_id;
    assign hart_id = unsigned'(tile_id_i) * NumCoresPerTile + unsigned'(c);

    if (!TrafficGeneration) begin: gen_mempool_cc
      // Unified core complex: Snitch-Spatz, or Snitch-only for the tile's first
      // NumScalarCoresPerTile cores. sh_acc ports are always present (a Spatz
      // core ties them off). Core c owns the flat-axis slice
      // [core_port_base(c) +: CorePorts] (= [c] for scalar builds).
      localparam bit          CoreSpatz = mempool_pkg::RVV && (c >= NumScalarCoresPerTile);
      localparam int unsigned CorePorts = CoreSpatz ? NumDataPortsPerCore : 1;
      mempool_cc #(
        .BootAddr            (BootAddr                        ),
        .RVE                 (1'b0                            ),
        .RVM                 (1'b1                            ),
        .SpatzEn             (CoreSpatz                       ),
        .RVV                 (CoreSpatz                       ),
        .XFVEC               (mempool_pkg::XFVEC              ),
        .XFDOTP              (mempool_pkg::XFDOTP             ),
        .XFAUX               (mempool_pkg::XFAUX              ),
        .RVF                 (mempool_pkg::RVF                ),
        .RVD                 (mempool_pkg::RVD                ),
        .XF16                (mempool_pkg::XF16               ),
        .XF16ALT             (mempool_pkg::XF16ALT            ),
        .XF8                 (mempool_pkg::XF8                ),
        .XF8ALT              (mempool_pkg::XF8ALT             ),
        .XDivSqrt            (mempool_pkg::XDivSqrt           ),
        .NumMemPortsPerSpatz (mempool_pkg::NumMemPortsPerSpatz),
        // Drive the full cluster meta-id width on the TCDM ports (wider than the
        // snitch id when a RedMulE shares the tile); upper bits are zero.
        .meta_id_t           (meta_id_t                       )
      ) riscv_core (
        .clk_i         (clk_i                                                    ),
        .rst_ni        (rst_ni                                                   ),
        .hart_id_i     (hart_id                                                  ),
        // IMEM Port
        .inst_addr_o   (snitch_inst_addr[c/NumCoresPerCache][c%NumCoresPerCache] ),
        .inst_data_i   (snitch_inst_data[c/NumCoresPerCache][c%NumCoresPerCache] ),
        .inst_valid_o  (snitch_inst_valid[c/NumCoresPerCache][c%NumCoresPerCache]),
        .inst_ready_i  (snitch_inst_ready[c/NumCoresPerCache][c%NumCoresPerCache]),
        // Shared operational-units ports (a Spatz core ties these off internally)
        .sh_acc_req_o         (acc_req[c]                                        ),
        .sh_acc_req_valid_o   (sh_acc_req_valid[c]                               ),
        .sh_acc_req_ready_i   (sh_acc_req_ready[c]                               ),
        .sh_acc_resp_i        (acc_resp[c]                                       ),
        .sh_acc_resp_valid_i  (sh_acc_resp_valid[c]                              ),
        .sh_acc_resp_ready_o  (sh_acc_resp_ready[c]                              ),
        // Data Ports -- slice [core_port_base(c) +: CorePorts]
        .data_qaddr_o  (cc_data_qaddr     [core_port_base(c) +: CorePorts]),
        .data_qwrite_o (cc_data_qwrite    [core_port_base(c) +: CorePorts]),
        .data_qamo_o   (cc_data_qamo      [core_port_base(c) +: CorePorts]),
        .data_qdata_o  (cc_data_qdata     [core_port_base(c) +: CorePorts]),
        .data_qstrb_o  (cc_data_qstrb     [core_port_base(c) +: CorePorts]),
        .data_qid_o    (cc_data_qid       [core_port_base(c) +: CorePorts]),
        .data_qburst_len_o(cc_data_qburst_len[core_port_base(c) +: CorePorts]),
        .data_qvalid_o (cc_data_qvalid    [core_port_base(c) +: CorePorts]),
        .data_qready_i (cc_data_qready    [core_port_base(c) +: CorePorts]),
        .data_pdata_i  (cc_data_pdata     [core_port_base(c) +: CorePorts]),
        // Store write-ack is carried in a meta_id bit (Spatz); data_pwrite_i unused.
        .data_pwrite_i ('0                                                              ),
        .data_perror_i (cc_data_perror    [core_port_base(c) +: CorePorts]),
        .data_pid_i    (cc_data_pid       [core_port_base(c) +: CorePorts]),
        .data_pvalid_i (cc_data_pvalid    [core_port_base(c) +: CorePorts]),
        .data_pready_o (cc_data_pready    [core_port_base(c) +: CorePorts]),
        .wake_up_sync_i(wake_up[c]                                               ),
        // Core Events
        .core_events_o (/* Unused */                                             )
      );
      if (snitch_pkg::XDIVSQRT && !mempool_pkg::RVV) begin: gen_sh_acc_interface
        // Assign the cores' hart_id to the corresponding shared accelerator
        assign sh_acc_req[c].addr      = acc_req[c].addr;
        assign sh_acc_req[c].id        = acc_req[c].id;
        assign sh_acc_req[c].hart_id   = hart_id[$clog2(NumCoresPerDivsqrt)-1:0];
        assign sh_acc_req[c].data_op   = acc_req[c].data_op;
        assign sh_acc_req[c].data_arga = acc_req[c].data_arga;
        assign sh_acc_req[c].data_argb = acc_req[c].data_argb;
        assign sh_acc_req[c].data_argc = acc_req[c].data_argc;
        // Redistribute shared response to cores
        assign acc_resp[c].id     = sh_acc_resp[c].id;
        assign acc_resp[c].error  = sh_acc_resp[c].error;
        assign acc_resp[c].data   = sh_acc_resp[c].data;
      end else begin: silence_sh_acc_interface
        assign acc_resp[c]          = '0;
        assign sh_acc_req[c]        = '0;
        assign sh_acc_req_ready[c]  = '0;
        assign sh_acc_resp[c]       = '0;
        assign sh_acc_resp_valid[c] = '0;
      end
    end else begin
      // Silence the (always-present) shared-accelerator interface
      assign acc_req[c]                                                = '0;
      assign sh_acc_req[c]                                             = '0;
      assign sh_acc_req_valid[c]                                       = '0;
      assign sh_acc_req_ready[c]                                       = '0;
      assign sh_acc_resp[c]                                            = '0;
      assign sh_acc_resp_valid[c]                                      = '0;
      assign sh_acc_resp_ready[c]                                      = '0;
      // Silence memory interfaces
      assign cc_data_qaddr[c]                                          = '0;
      assign cc_data_qwrite[c]                                         = '0;
      assign cc_data_qamo[c]                                           = '0;
      assign cc_data_qdata[c]                                          = '0;
      assign cc_data_qstrb[c]                                          = '0;
      assign cc_data_qid[c]                                            = '0;
      assign snitch_data_qvalid[c]                                     = '0;
      assign snitch_data_pready[c]                                     = '0;
      assign snitch_inst_addr[c/NumCoresPerCache][c%NumCoresPerCache]  = '0;
      assign snitch_inst_valid[c/NumCoresPerCache][c%NumCoresPerCache] = '0;
    end
  end

  /***********************
   *  Instruction Cache  *
   ***********************/
  // Instruction interface
  axi_core_req_t  [NumCaches-1:0] axi_cache_req_d, axi_cache_req_q;
  axi_core_resp_t [NumCaches-1:0] axi_cache_resp_d, axi_cache_resp_q;

  for (genvar c = 0; unsigned'(c) < NumCaches; c++) begin: gen_caches
    snitch_icache_pkg::icache_l0_events_t [NumCoresPerCache-1:0] icache_l0_events;
    snitch_icache_pkg::icache_l1_events_t                        icache_l1_events;

    snitch_icache #(
      .NR_FETCH_PORTS     (NumCoresPerCache                                    ),
      /// Cache Line Width
      .L0_LINE_COUNT      (ICacheL0Lines                                       ),
      .LINE_WIDTH         (ICacheLineWidth                                     ),
      .LINE_COUNT         (ICacheSizeByte / (ICacheWays * ICacheLineWidth / 8) ),
      .WAY_COUNT          (ICacheWays                                          ),
      .FETCH_AW           (AddrWidth                                           ),
      .FETCH_DW           (DataWidth                                           ),
      .FILL_AW            (AddrWidth                                           ),
      .FILL_DW            (AxiDataWidth                                        ),
      .FETCH_PRIORITY     (1                                                   ),
      .MERGE_FETCHES      (1                                                   ),
      .SERIAL_LOOKUP      (1                                                   ),
      .L1_TAG_SCM         (1                                                   ),
      .L1_DATA_SCM        (0                                                   ),
      .NUM_AXI_OUTSTANDING(8                                                   ),
      /// Make the early cache latch-based. This reduces latency at the cost of
      /// increased combinatorial path lengths and the hassle of having latches in
      /// the design.
      .EARLY_LATCH        (1                                                   ),
      .L0_EARLY_TAG_WIDTH (11                                                  ),
      .ISO_CROSSING       (0                                                   ),
      .axi_req_t          (axi_core_req_t                                      ),
      .axi_rsp_t          (axi_core_resp_t                                     )
    ) i_snitch_icache (
      .clk_i                (clk_i                   ),
      .clk_d2_i             (clk_i                   ),
      .rst_ni               (rst_ni                  ),
      .enable_prefetching_i (1'b1                    ),
      .enable_branch_pred_i (1'b1                    ),
      .icache_l0_events_o   (icache_l0_events        ),
      .icache_l1_events_o   (icache_l1_events        ),
      .flush_valid_i        ({NumCoresPerCache{1'b0}}),
      .flush_ready_o        (/* Unused */            ),
      .inst_addr_i          (snitch_inst_addr[c]     ),
      .inst_data_o          (snitch_inst_data[c]     ),
      .inst_cacheable_i     ({NumCoresPerCache{1'b1}}),
      .inst_valid_i         (snitch_inst_valid[c]    ),
      .inst_ready_o         (snitch_inst_ready[c]    ),
      .inst_error_o         (/* Unused */            ),
      .sram_cfg_data_i      ('0                      ),
      .sram_cfg_tag_i       ('0                      ),
      .sram_cfg_out_data_o  (/* Unused */            ),
      .sram_cfg_out_tag_o   (/* Unused */            ),
      .axi_req_o            (axi_cache_req_d[c]      ),
      .axi_rsp_i            (axi_cache_resp_q[c]     )
    );

`ifndef SYNTHESIS
    // pragma translate_off
    // +icache_stats: event counts per cache. An L0 miss holds until its refill
    // issues, so count its rising edge; L1 events are one per lookup.
    longint unsigned l0_miss, l0_hit, l0_prefetch, l0_stall, l1_miss, l1_hit, l1_stall;
    logic [NumCoresPerCache-1:0] ev_miss, ev_hit, ev_prefetch, ev_stall;
    logic [NumCoresPerCache-1:0] ev_miss_q, ev_prefetch_q;
    for (genvar i = 0; i < NumCoresPerCache; i++) begin : gen_l0_events
      assign ev_miss[i]     = icache_l0_events[i].l0_miss;
      assign ev_hit[i]      = icache_l0_events[i].l0_hit;
      assign ev_prefetch[i] = icache_l0_events[i].l0_prefetch;
      assign ev_stall[i]    = icache_l0_events[i].l0_stall;
    end : gen_l0_events
    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        {l0_miss, l0_hit, l0_prefetch, l0_stall, l1_miss, l1_hit, l1_stall} <= '0;
        ev_miss_q     <= '0;
        ev_prefetch_q <= '0;
      end else begin
        l0_miss       <= l0_miss + $countones(ev_miss & ~ev_miss_q);
        l0_prefetch   <= l0_prefetch + $countones(ev_prefetch & ~ev_prefetch_q);
        l0_hit        <= l0_hit + $countones(ev_hit);
        l0_stall      <= l0_stall + $countones(ev_stall);
        ev_miss_q     <= ev_miss;
        ev_prefetch_q <= ev_prefetch;
        l1_miss       <= l1_miss + icache_l1_events.l1_miss;
        l1_hit        <= l1_hit + icache_l1_events.l1_hit;
        l1_stall      <= l1_stall + icache_l1_events.l1_stall;
      end
    end
    final begin
      if ($test$plusargs("icache_stats"))
        $display("[ICACHE] %m l0_miss=%0d l0_hit=%0d l0_prefetch=%0d l0_stall=%0d l1_miss=%0d ",
                 l0_miss, l0_hit, l0_prefetch, l0_stall, l1_miss,
                 "l1_hit=%0d l1_stall=%0d", l1_hit, l1_stall);
    end
    // pragma translate_on
`endif
    axi_cut #(
      .aw_chan_t (axi_core_aw_t  ),
      .w_chan_t  (axi_core_w_t   ),
      .b_chan_t  (axi_core_b_t   ),
      .ar_chan_t (axi_core_ar_t  ),
      .r_chan_t  (axi_core_r_t   ),
      .axi_req_t (axi_core_req_t ),
      .axi_resp_t(axi_core_resp_t)
    ) axi_cache_slice (
      .clk_i     (clk_i              ),
      .rst_ni    (rst_ni             ),
      .slv_req_i (axi_cache_req_d[c] ),
      .slv_resp_o(axi_cache_resp_q[c]),
      .mst_req_o (axi_cache_req_q[c] ),
      .mst_resp_i(axi_cache_resp_d[c])
    );
  end

  /******************
   *  Memory Banks  *
   ******************/

  // Bank metadata
  typedef struct packed {
    local_req_interco_addr_t ini_addr;
    meta_id_t meta_id;
    group_id_t group_id;
    tile_group_id_t tile_id;
    tile_core_id_t core_id;
    logic wide;
  } bank_metadata_t;

  // Memory interfaces
  local_resp_interco_addr_t[NumBanksPerTile-1:0] superbank_resp_ini_addr;

  logic                    [NumBanksPerTile-1:0] bank_req_valid;
  logic                    [NumBanksPerTile-1:0] bank_req_ready;
  local_req_interco_addr_t [NumBanksPerTile-1:0] bank_req_ini_addr;
  logic                    [NumBanksPerTile-1:0] bank_req_wide;
  tcdm_slave_req_t         [NumBanksPerTile-1:0] bank_req_payload;
  logic                    [NumBanksPerTile-1:0] bank_resp_valid;
  logic                    [NumBanksPerTile-1:0] bank_resp_ready;
  tcdm_slave_resp_t        [NumBanksPerTile-1:0] bank_resp_payload;
  logic                    [NumBanksPerTile-1:0] bank_resp_wide;
  local_resp_interco_addr_t[NumBanksPerTile-1:0] bank_resp_ini_addr;

  // Condition bank_resp_ini_addr into the resp xbar's sel, handling
  // req/resp port-count asymmetry; feeds i_tcdm_bank_interco.mst_resp_ini_addr_i.
  for (genvar b = 0; unsigned'(b) < NumBanksPerTile; b++) begin: gen_superbank_resp_ini_addr
    if(NumNarrowRemoteReqPortsPerTile > NumNarrowRemoteRespPortsPerTile ) begin: gen_superbank_resp_ini_addr_req_gt_resp
      always_comb begin
        superbank_resp_ini_addr[b] = '0;
        superbank_resp_ini_addr[b] = bank_resp_ini_addr[b];
        if(bank_resp_ini_addr[b] > (NumLocalPorts + NumNarrowRemoteRespPortsPerTile - 1)) begin
          superbank_resp_ini_addr[b] = bank_resp_ini_addr[b] - (NumNarrowRemoteReqPortsPerTile - NumNarrowRemoteRespPortsPerTile);
        end
      end
    end else if (NumNarrowRemoteReqPortsPerTile == NumNarrowRemoteRespPortsPerTile) begin: gen_superbank_resp_ini_addr_req_eq_resp
      always_comb begin
        superbank_resp_ini_addr[b] = '0;
        superbank_resp_ini_addr[b] = bank_resp_ini_addr[b];
      end
    end
    else begin: gen_superbank_resp_ini_addr_req_lt_resp
      localparam int unsigned bank_resp_payload_width = $bits(tcdm_slave_resp_t);
      localparam int unsigned hash_width = $clog2(NumNarrowRemoteRespPortsPerTile - 1) + 2;
      localparam int unsigned hash_binning_step = (1 << hash_width) / (NumNarrowRemoteRespPortsPerTile - 1);
      logic [bank_resp_payload_width-1:0] bank_resp_payload_raw;
      logic [hash_width-1:0] hash_src0, hash_src1, hash;
      assign bank_resp_payload_raw = bank_resp_payload[b];
      if(hash_width == 4) begin: gen_customed_hash_4
        assign hash_src0 = {bank_resp_payload[b].rdata.core_id[0],
                            bank_resp_payload[b].rdata.core_id[1],
                            bank_resp_payload[b].src_group_id[0],
                            bank_resp_payload[b].src_group_id[2]
                            };
        assign hash_src1 = {
                            bank_resp_payload[b].ini_addr[2],
                            bank_resp_payload[b].src_group_id[3],
                            bank_resp_payload[b].ini_addr[0],
                            bank_resp_payload[b].ini_addr[1]
                            };
      end else begin: gen_general_hash
        $warning("Customed hashing is not implemented for hash_width = %0d, ",
                  "using general congif, the performance may degrade.", hash_width);
        assign hash_src0 = bank_resp_payload_raw[0 +: hash_width];
        assign hash_src1 = bank_resp_payload_raw[hash_width +: hash_width];
      end
      assign hash = hash_src0 ^ hash_src1;
      always_comb begin
        superbank_resp_ini_addr[b] = '0;
        superbank_resp_ini_addr[b] = bank_resp_ini_addr[b];
        if(bank_resp_ini_addr[b] > NumLocalPorts) begin
          superbank_resp_ini_addr[b] = NumLocalPorts + 1 + 0;
          for (int i = 1; i < (NumNarrowRemoteRespPortsPerTile-1); i++) begin
            if (hash >= i * hash_binning_step) begin
              superbank_resp_ini_addr[b] = NumLocalPorts + 1 + i;
            end
          end
        end
      end
    end
  end

  // mempool_tcdm_bank_interco is instantiated further below, after its
  // connected signal declarations.

  for (genvar b = 0; unsigned'(b) < NumBanksPerTile; b++) begin: gen_banks
    bank_metadata_t meta_in;
    bank_metadata_t meta_out;
    logic req_valid;
    logic req_write;
    bank_addr_t req_addr;
    data_t req_wdata;
    data_t resp_rdata;
    strb_t req_be;

    // Un/Pack metadata
    assign meta_in = '{
      ini_addr  : bank_req_ini_addr[b],
      meta_id   : bank_req_payload[b].wdata.meta_id,
      core_id   : bank_req_payload[b].wdata.core_id,
      tile_id   : bank_req_payload[b].ini_addr,
      group_id  : bank_req_payload[b].src_group_id,
      wide      : bank_req_wide[b]
    };
    always_comb begin
      bank_resp_ini_addr[b] = '0;
      bank_resp_ini_addr[b] = meta_out.ini_addr;
    end
    assign bank_resp_payload[b].rdata.meta_id = meta_out.meta_id;
    assign bank_resp_payload[b].ini_addr      = meta_out.tile_id;
    assign bank_resp_payload[b].rdata.core_id = meta_out.core_id;
    assign bank_resp_payload[b].src_group_id  = meta_out.group_id;
    assign bank_resp_payload[b].rdata.amo     = '0; // Don't care
    assign bank_resp_wide[b]                  = meta_out.wide;

    tcdm_adapter #(
      .AddrWidth  (TCDMAddrMemWidth),
      .DataWidth  (DataWidth       ),
      .metadata_t (bank_metadata_t ),
      .LrScEnable (LrScEnable      ),
      .RegisterAmo(1'b0            )
    ) i_tcdm_adapter (
      .clk_i       (clk_i                                                                       ),
      .rst_ni      (rst_ni                                                                      ),
      .in_valid_i  (bank_req_valid[b]                                                           ),
      .in_ready_o  (bank_req_ready[b]                                                           ),
      .in_address_i(bank_req_payload[b].tgt_addr[idx_width(NumBanksPerTile) +: TCDMAddrMemWidth]),
      .in_amo_i    (bank_req_payload[b].wdata.amo                                               ),
      .in_write_i  (bank_req_payload[b].wen                                                     ),
      .in_wdata_i  (bank_req_payload[b].wdata.data                                              ),
      .in_meta_i   (meta_in                                                                     ),
      .in_be_i     (bank_req_payload[b].be                                                      ),
      .in_valid_o  (bank_resp_valid[b]                                                          ),
      .in_ready_i  (bank_resp_ready[b]                                                          ),
      .in_rdata_o  (bank_resp_payload[b].rdata.data                                             ),
      .in_meta_o   (meta_out                                                                    ),
      .out_req_o   (req_valid                                                                   ),
      .out_add_o   (req_addr                                                                    ),
      .out_write_o (req_write                                                                   ),
      .out_wdata_o (req_wdata                                                                   ),
      .out_be_o    (req_be                                                                      ),
      .out_rdata_i (resp_rdata                                                                  )
    );

    // Bank
    tc_sram #(
      .DataWidth(DataWidth          ),
      .NumWords (2**TCDMAddrMemWidth),
      .NumPorts (1                  )
    ) mem_bank (
      .clk_i  (clk_i     ),
      .rst_ni (rst_ni    ),
      .req_i  (req_valid ),
      .we_i   (req_write ),
      .addr_i (req_addr  ),
      .wdata_i(req_wdata ),
      .be_i   (req_be    ),
      .rdata_o(resp_rdata)
    );

  end

  /***************
   *  Registers  *
   ***************/

  // These are required to break dependencies between request and response, establishing a correct
  // valid/ready handshake.
  tcdm_master_req_t  [NumNarrowRemoteReqPortsPerTile-1:0] prereg_tcdm_master_req;
  logic              [NumNarrowRemoteReqPortsPerTile-1:0] prereg_tcdm_master_req_valid;
  logic              [NumNarrowRemoteReqPortsPerTile-1:0] prereg_tcdm_master_req_ready;
  tcdm_slave_req_t   [NumNarrowRemoteReqPortsPerTile-1:0] postreg_tcdm_slave_req;
  logic              [NumNarrowRemoteReqPortsPerTile-1:0] postreg_tcdm_slave_req_valid;
  logic              [NumNarrowRemoteReqPortsPerTile-1:0] postreg_tcdm_slave_req_ready;
  tcdm_slave_resp_t  [NumNarrowRemoteRespPortsPerTile-1:0] prereg_tcdm_slave_resp;
  logic              [NumNarrowRemoteRespPortsPerTile-1:0] prereg_tcdm_slave_resp_valid;
  logic              [NumNarrowRemoteRespPortsPerTile-1:0] prereg_tcdm_slave_resp_ready;
  tcdm_master_resp_t [NumNarrowRemoteRespPortsPerTile-1:0] postreg_tcdm_master_resp;
  tile_core_id_t     [NumNarrowRemoteRespPortsPerTile-1:0] postreg_tcdm_master_resp_ini_sel;
  logic              [NumNarrowRemoteRespPortsPerTile-1:0] postreg_tcdm_master_resp_valid;
  logic              [NumNarrowRemoteRespPortsPerTile-1:0] postreg_tcdm_master_resp_ready;

  `ifdef USE_WIDE_TCDM_PLANE
  tcdm_wide_master_req_t  [NumWideRemoteReqPortsPerTile-1:0] prereg_tcdm_wide_master_req;
  logic                   [NumWideRemoteReqPortsPerTile-1:0] prereg_tcdm_wide_master_req_valid;
  logic                   [NumWideRemoteReqPortsPerTile-1:0] prereg_tcdm_wide_master_req_ready;
  tcdm_wide_slave_req_t   [NumWideRemoteReqPortsPerTile-1:0] postreg_tcdm_wide_slave_req;
  logic                   [NumWideRemoteReqPortsPerTile-1:0] postreg_tcdm_wide_slave_req_valid;
  logic                   [NumWideRemoteReqPortsPerTile-1:0] postreg_tcdm_wide_slave_req_ready;
  tcdm_wide_slave_resp_t  [NumWideRemoteRespPortsPerTile-1:0] prereg_tcdm_wide_slave_resp;
  logic                   [NumWideRemoteRespPortsPerTile-1:0] prereg_tcdm_wide_slave_resp_valid;
  logic                   [NumWideRemoteRespPortsPerTile-1:0] prereg_tcdm_wide_slave_resp_ready;
  tcdm_wide_master_resp_t [NumWideRemoteRespPortsPerTile-1:0] postreg_tcdm_wide_master_resp;
  logic                   [NumWideRemoteRespPortsPerTile-1:0] postreg_tcdm_wide_master_resp_valid;
  logic                   [NumWideRemoteRespPortsPerTile-1:0] postreg_tcdm_wide_master_resp_ready;
  `endif

  // Break paths between request and response with registers
  for (genvar h = 0; unsigned'(h) < NumNarrowRemoteReqPortsPerTile; h++) begin: gen_tcdm_registers_req
    spill_register #(
      .T(tcdm_master_req_t)
    ) i_tcdm_master_req_register (
      .clk_i  (clk_i                          ),
      .rst_ni (rst_ni                         ),
      .data_i (prereg_tcdm_master_req[h]      ),
      .valid_i(prereg_tcdm_master_req_valid[h]),
      .ready_o(prereg_tcdm_master_req_ready[h]),
      .data_o (tcdm_master_req_o[h]           ),
      .valid_o(tcdm_master_req_valid_o[h]     ),
      .ready_i(tcdm_master_req_ready_i[h]     )
    );

    fall_through_register #(
      .T(tcdm_slave_req_t)
    ) i_tcdm_slave_req_register (
      .clk_i     (clk_i                          ),
      .rst_ni    (rst_ni                         ),
      .clr_i     (1'b0                           ),
      .testmode_i(1'b0                           ),
      .data_i    (tcdm_slave_req_i[h]            ),
      .valid_i   (tcdm_slave_req_valid_i[h]      ),
      .ready_o   (tcdm_slave_req_ready_o[h]      ),
      .data_o    (postreg_tcdm_slave_req[h]      ),
      .valid_o   (postreg_tcdm_slave_req_valid[h]),
      .ready_i   (postreg_tcdm_slave_req_ready[h])
    );
  end: gen_tcdm_registers_req

  for (genvar h = 0; unsigned'(h) < NumNarrowRemoteRespPortsPerTile; h++) begin: gen_tcdm_registers_resp
    fall_through_register #(
      .T(tcdm_master_resp_t)
    ) i_tcdm_master_resp_register (
      .clk_i     (clk_i                            ),
      .rst_ni    (rst_ni                           ),
      .clr_i     (1'b0                             ),
      .testmode_i(1'b0                             ),
      .data_i    (tcdm_master_resp_i[h]            ),
      .valid_i   (tcdm_master_resp_valid_i[h]      ),
      .ready_o   (tcdm_master_resp_ready_o[h]      ),
      .data_o    (postreg_tcdm_master_resp[h]      ),
      .valid_o   (postreg_tcdm_master_resp_valid[h]),
      .ready_i   (postreg_tcdm_master_resp_ready[h])
    );

    // Helper signal to drive the remote response interconnect
    assign postreg_tcdm_master_resp_ini_sel[h] = postreg_tcdm_master_resp[h].rdata.core_id;

    spill_register #(
      .T(tcdm_slave_resp_t)
    ) i_tcdm_slave_resp_register (
      .clk_i  (clk_i                          ),
      .rst_ni (rst_ni                         ),
      .data_i (prereg_tcdm_slave_resp[h]      ),
      .valid_i(prereg_tcdm_slave_resp_valid[h]),
      .ready_o(prereg_tcdm_slave_resp_ready[h]),
      .data_o (tcdm_slave_resp_o[h]           ),
      .valid_o(tcdm_slave_resp_valid_o[h]     ),
      .ready_i(tcdm_slave_resp_ready_i[h]     )
    );
  end: gen_tcdm_registers_resp

  `ifdef USE_WIDE_TCDM_PLANE
  for (genvar h = 0; unsigned'(h) < NumWideRemoteReqPortsPerTile; h++) begin:
      gen_tcdm_wide_registers_req
    spill_register #(
      .T(tcdm_wide_master_req_t)
    ) i_tcdm_wide_master_req_register (
      .clk_i  (clk_i                               ),
      .rst_ni (rst_ni                              ),
      .data_i (prereg_tcdm_wide_master_req[h]      ),
      .valid_i(prereg_tcdm_wide_master_req_valid[h]),
      .ready_o(prereg_tcdm_wide_master_req_ready[h]),
      .data_o (tcdm_wide_master_req_o[h]           ),
      .valid_o(tcdm_wide_master_req_valid_o[h]     ),
      .ready_i(tcdm_wide_master_req_ready_i[h]     )
    );

    fall_through_register #(
      .T(tcdm_wide_slave_req_t)
    ) i_tcdm_wide_slave_req_register (
      .clk_i     (clk_i                               ),
      .rst_ni    (rst_ni                              ),
      .clr_i     (1'b0                                ),
      .testmode_i(1'b0                                ),
      .data_i    (tcdm_wide_slave_req_i[h]            ),
      .valid_i   (tcdm_wide_slave_req_valid_i[h]      ),
      .ready_o   (tcdm_wide_slave_req_ready_o[h]      ),
      .data_o    (postreg_tcdm_wide_slave_req[h]      ),
      .valid_o   (postreg_tcdm_wide_slave_req_valid[h]),
      .ready_i   (postreg_tcdm_wide_slave_req_ready[h])
    );
  end: gen_tcdm_wide_registers_req

  for (genvar h = 0; unsigned'(h) < NumWideRemoteRespPortsPerTile; h++) begin:
      gen_tcdm_wide_registers_resp
    fall_through_register #(
      .T(tcdm_wide_master_resp_t)
    ) i_tcdm_wide_master_resp_register (
      .clk_i     (clk_i                                ),
      .rst_ni    (rst_ni                               ),
      .clr_i     (1'b0                                 ),
      .testmode_i(1'b0                                 ),
      .data_i    (tcdm_wide_master_resp_i[h]           ),
      .valid_i   (tcdm_wide_master_resp_valid_i[h]     ),
      .ready_o   (tcdm_wide_master_resp_ready_o[h]     ),
      .data_o    (postreg_tcdm_wide_master_resp[h]     ),
      .valid_o   (postreg_tcdm_wide_master_resp_valid[h]),
      .ready_i   (postreg_tcdm_wide_master_resp_ready[h])
    );

    spill_register #(
      .T(tcdm_wide_slave_resp_t)
    ) i_tcdm_wide_slave_resp_register (
      .clk_i  (clk_i                                ),
      .rst_ni (rst_ni                               ),
      .data_i (prereg_tcdm_wide_slave_resp[h]       ),
      .valid_i(prereg_tcdm_wide_slave_resp_valid[h] ),
      .ready_o(prereg_tcdm_wide_slave_resp_ready[h] ),
      .data_o (tcdm_wide_slave_resp_o[h]            ),
      .valid_o(tcdm_wide_slave_resp_valid_o[h]      ),
      .ready_i(tcdm_wide_slave_resp_ready_i[h]      )
    );
  end: gen_tcdm_wide_registers_resp
  `endif

  /****************************
   *   Remote Interconnects   *
   ****************************/

  // Slots [0..NumCoresPerTile-1] = cores; redmule tiles add [NumCoresPerTile..NumLocalPorts-1].
  tcdm_master_req_t    [NumLocalPorts-1:0] remote_req_interco;
  logic                [NumLocalPorts-1:0] remote_req_interco_valid;
  logic                [NumLocalPorts-1:0] remote_req_interco_ready;
  addr_t               [NumLocalPorts-1:0] remote_req_interco_addr_int;
  logic                [NumLocalPorts-1:0] remote_req_interco_wen;
  logic                [NumLocalPorts-1:0] remote_req_interco_amoen;
  tcdm_addr_t          [NumLocalPorts-1:0] remote_req_tgt_addr;
  // Set by the compute scramblers when a request was VC-remapped; used only
  // by the confinement assertion below, which needs to tell VC traffic from
  // ordinary global traffic (stacks, barriers, the sequential region).
  logic [NumLocalPorts-1:0]      vc_hit_port;
  logic [NumLocalPorts-1:0][2:0] vc_level_port;
  logic                          vc_hit_wide_port;
  logic [2:0]                    vc_level_wide_port;

  group_id_t           [NumLocalPorts-1:0] remote_req_tgt_group_id;
  remote_ports_index_t [NumLocalPorts-1:0] remote_req_interco_tgt_sel;
  remote_ports_index_t [NumLocalPorts-1:0] remote_req_tgt_sel;

  tcdm_master_resp_t   [NumLocalPorts-1:0] remote_resp_interco;
  logic                [NumLocalPorts-1:0] remote_resp_interco_valid;
  logic                [NumLocalPorts-1:0] remote_resp_interco_ready;

  `ifdef USE_WIDE_TCDM_PLANE
  // Other-tile wide traffic, mirroring remote_req_interco above. tgt_sel spans
  // ALL remote ports: the low ones are the same-group links, the high ones the
  // inter-group links, exactly as the narrow remote req router numbers them.
  tcdm_wide_master_req_t    [NumWideLocalPorts-1:0] remote_wide_req_interco;
  logic                     [NumWideLocalPorts-1:0] remote_wide_req_interco_valid;
  logic                     [NumWideLocalPorts-1:0] remote_wide_req_interco_ready;
  addr_t                    [NumWideLocalPorts-1:0] remote_wide_req_interco_addr_int;
  logic                     [NumWideLocalPorts-1:0] remote_wide_req_interco_wen;
  group_id_t                [NumWideLocalPorts-1:0] remote_wide_req_tgt_group_id;
  tcdm_addr_t               [NumWideLocalPorts-1:0] remote_wide_req_tgt_addr;
  wide_remote_ports_index_t [NumWideLocalPorts-1:0] remote_wide_req_tgt_sel;
  wide_remote_ports_index_t [NumWideLocalPorts-1:0] remote_wide_req_interco_tgt_sel;

  tcdm_wide_master_resp_t   [NumWideLocalPorts-1:0] remote_wide_resp_interco;
  logic                     [NumWideLocalPorts-1:0] remote_wide_resp_interco_valid;
  logic                     [NumWideLocalPorts-1:0] remote_wide_resp_interco_ready;

`ifdef USE_WIDE_TCDM_PLANE
  // ------------------------------------------------------------------
  //  Shared wide remote link
  // ------------------------------------------------------------------
  // Every wide client has its own slot (RedMulE also its own-tile bank ingress), but
  // the tile has ONE physical wide master link, so these two crossbars -- and only
  // these -- are shared. They live at module scope rather than inside a client's generate
  // so a second client can be added without moving them again.
  wide_local_ports_index_t [NumWideRemoteRespPortsPerTile-1:0] remote_wide_resp_sel;
  for (genvar r = 0; r < NumWideRemoteRespPortsPerTile; r++) begin : gen_wide_resp_sel
    assign remote_wide_resp_sel[r] =
        wide_local_ports_index_t'(postreg_tcdm_wide_master_resp[r].rdata.core_id);
  end : gen_wide_resp_sel

  // The remote port select sends reads and writes to different port classes.
  for (genvar w = 0; w < NumWideLocalPorts; w++) begin : gen_wide_req_wen
    assign remote_wide_req_interco_wen[w] = remote_wide_req_interco[w].wen;
  end : gen_wide_req_wen

  stream_xbar #(
    .NumInp   (NumWideLocalPorts             ),
    .NumOut   (NumWideRemoteReqPortsPerTile  ),
    .payload_t(tcdm_wide_master_req_t        ),
    .AxiVldRdy(0                             ), // as i_remote_req_interco: sel_i may
    .LockIn   (0                             )  // change before the handshake
  ) i_remote_wide_req_interco (
    .clk_i  (clk_i                          ),
    .rst_ni (rst_ni                         ),
    .flush_i(1'b0                           ),
    // External priority flag
    .rr_i   ('0                             ),
    // Master
    .data_i (remote_wide_req_interco        ),
    .valid_i(remote_wide_req_interco_valid  ),
    .ready_o(remote_wide_req_interco_ready  ),
    .sel_i  (remote_wide_req_interco_tgt_sel),
    // Slave
    .data_o (prereg_tcdm_wide_master_req      ),
    .valid_o(prereg_tcdm_wide_master_req_valid),
    .ready_i(prereg_tcdm_wide_master_req_ready),
    .idx_o  (/* unused */                     )
  );

  // Every group response -- same-group or inter-group -- merges onto the
  // shim's remote port, mirroring i_remote_resp_interco on the narrow side.
  // The shim then merges it with the own-tile response from the bank
  // interconnect, so RedMulE still sees a single response stream.
  stream_xbar #(
    .NumInp   (NumWideRemoteRespPortsPerTile),
    .NumOut   (NumWideLocalPorts            ),
    .payload_t(tcdm_wide_master_resp_t      )
  ) i_remote_wide_resp_interco (
    .clk_i  (clk_i                              ),
    .rst_ni (rst_ni                             ),
    .flush_i(1'b0                               ),
    // External priority flag
    .rr_i   ('0                                 ),
    // Master
    .data_i (postreg_tcdm_wide_master_resp      ),
    .valid_i(postreg_tcdm_wide_master_resp_valid),
    .ready_o(postreg_tcdm_wide_master_resp_ready),
    // Route each remote response back to the client that issued it. core_id is
  // the wide slot index, set per requester and round-tripped by the bank meta.
  .sel_i  (remote_wide_resp_sel                ),
    // Slave
    .data_o (remote_wide_resp_interco           ),
    .valid_o(remote_wide_resp_interco_valid     ),
    .ready_i(remote_wide_resp_interco_ready     ),
    .idx_o  (/* unused */                       )
  );


`endif

  `endif

  stream_xbar #(
    .NumInp   (NumLocalPorts                 ), // cores + RedMulE clients
    .NumOut   (NumNarrowRemoteReqPortsPerTile),
    .payload_t(tcdm_master_req_t             ),
    .AxiVldRdy(0                             ), // the sel_i can be changed before the hsk happen, as the priority of cores can be different
    .LockIn   (0                             )
  ) i_remote_req_interco (
    .clk_i  (clk_i                           ),
    .rst_ni (rst_ni                          ),
    .flush_i(1'b0                            ),
    // External priority flag
    .rr_i   ('0                              ),
    // Master
    .data_i (remote_req_interco              ),
    .valid_i(remote_req_interco_valid        ),
    .ready_o(remote_req_interco_ready        ),
    .sel_i  (remote_req_interco_tgt_sel      ),
    // Slave
    .data_o (prereg_tcdm_master_req          ),
    .valid_o(prereg_tcdm_master_req_valid    ),
    .ready_i(prereg_tcdm_master_req_ready    ),
    .idx_o  (/* Unused */                    )
  );

  // Slice sel per slot: resp_ini_sel is cluster-width but sel_i is
  // idx_width(NumLocalPorts); a flat connect misroutes (trips invalid_resp_id).
  logic [NumNarrowRemoteRespPortsPerTile-1:0][idx_width(NumLocalPorts)-1:0] remote_resp_sel;
  for (genvar h = 0; h < NumNarrowRemoteRespPortsPerTile; h++) begin: gen_remote_resp_sel
    assign remote_resp_sel[h] = postreg_tcdm_master_resp_ini_sel[h][idx_width(NumLocalPorts)-1:0];
  end

  stream_xbar #(
    .NumInp   (NumNarrowRemoteRespPortsPerTile     ),
    .NumOut   (NumLocalPorts                 ), // cores + RedMulE clients
    .payload_t(tcdm_master_resp_t            )
  ) i_remote_resp_interco (
    .clk_i  (clk_i                           ),
    .rst_ni (rst_ni                          ),
    .flush_i(1'b0                            ),
    // External priority flag
    .rr_i   ('0                              ),
    // Master
    .data_i (postreg_tcdm_master_resp        ),
    .valid_i(postreg_tcdm_master_resp_valid  ),
    .ready_o(postreg_tcdm_master_resp_ready  ),
    .sel_i  (remote_resp_sel                 ),
    // Slave
    .data_o (remote_resp_interco             ),
    .valid_o(remote_resp_interco_valid       ),
    .ready_i(remote_resp_interco_ready       ),
    .idx_o  (/* Unused */                    )
  );

  /**********************
   *   Local Intercos   *
   **********************/

  logic             [NumLocalPorts-1:0] local_req_interco_valid;
  logic             [NumLocalPorts-1:0] local_req_interco_ready;
  tcdm_slave_req_t  [NumLocalPorts-1:0] local_req_interco_payload;
  logic             [NumLocalPorts-1:0] local_resp_interco_valid;
  logic             [NumLocalPorts-1:0] local_resp_interco_ready;
  tcdm_slave_resp_t [NumLocalPorts-1:0] local_resp_interco_payload;
  addr_t            [NumLocalPorts-1:0] local_req_interco_addr_int;

  `ifdef USE_WIDE_TCDM_PLANE
  // Own-tile wide clients, mirroring the narrow local interco above: the wide
  // shim's local port feeds the bank interconnect directly and never reaches the
  // group. Payload is the bank-interconnect ingress type, so tgt_addr is the
  // {row, bank} tile address with the group and tile fields already consumed.
  logic            [NumWideLocalPorts-1:0] local_wide_req_interco_valid;
  logic            [NumWideLocalPorts-1:0] local_wide_req_interco_ready;
  tcdm_wide_req_t  [NumWideLocalPorts-1:0] local_wide_req_interco_payload;
  logic            [NumWideLocalPorts-1:0] local_wide_resp_interco_valid;
  logic            [NumWideLocalPorts-1:0] local_wide_resp_interco_ready;
  tcdm_wide_resp_t [NumWideLocalPorts-1:0] local_wide_resp_interco_payload;
  addr_t           [NumWideLocalPorts-1:0] local_wide_req_interco_addr_int;

  // Only the first NumWideBankLocalPorts slots (RedMulE) have an own-tile wide
  // ingress. The Spatz slots above are remote-only (an own-tile burst is issued
  // on the core's narrow vector ports), so they have no bank side: tie it off.
  for (genvar s = NumWideBankLocalPorts; s < NumWideLocalPorts; s++) begin : gen_wide_remote_only_slot
    assign local_wide_req_interco_ready[s]    = 1'b0;
    assign local_wide_resp_interco_valid[s]   = 1'b0;
    assign local_wide_resp_interco_payload[s] = '0;
  end : gen_wide_remote_only_slot
  `endif

  // ------------------------------------------------------------------------
  // B2 wide landing: a raw wide slave request becomes an engine access, and the
  // engine response returns through the raw slave response. Carried source routing
  // (ini_addr/src_group_id) mirrors the narrow slave path.
  // ------------------------------------------------------------------------
  `ifdef USE_WIDE_TCDM_PLANE
  // Bank-interconnect wide ingresses, ordered exactly like the narrow ones:
  // this tile's own-tile clients at the low indices, the group's slave ports
  // above. Keeping own-tile, same-group and inter-group traffic on separate
  // ingresses keeps them independent through superbank selection.
  localparam int unsigned NumWideEngIngress = NumWideBankLocalPorts + NumWideRemoteReqPortsPerTile;

  // Group slave ports retyped to the bank-interconnect ingress: the wide
  // intra-group xbar leaves the within-group tile field in tgt_addr (unlike the
  // narrow variable_latency_interconnect, which consumes it), so drop it here.
  tcdm_wide_req_t [NumWideRemoteReqPortsPerTile-1:0] postreg_tcdm_wide_slave_req_bank;

  // Wide RedMulE shim payloads. Declared at module scope because some simulators
  // reject generate-local types used as module type parameters.
  typedef struct packed {
    addr_t          addr;
    meta_id_t       id;
    amo_t           amo;
    logic           write;
    wide_tcdm_data_t data;
    wide_tcdm_strb_t strb;
  } tcdm_wide_rm_req_t;

  typedef struct packed {
    wide_tcdm_data_t data;
    meta_id_t        id;
    logic            write;
    logic            error;
  } tcdm_wide_rm_resp_t;

  // This tile's within-group ID, carried with requests so responses self-route.
  tile_group_id_t tcdm_wide_tile_id;
  assign tcdm_wide_tile_id = tile_group_id_t'(tile_id_i);

  localparam int unsigned WideTileOff =
      (NumTilesPerGroup == 1) ? 0 : idx_width(NumTilesPerGroup);

  for (genvar p = 0; p < NumWideRemoteReqPortsPerTile; p++) begin :
      gen_wide_group_ingress
    assign postreg_tcdm_wide_slave_req_bank[p] = '{
      wdata:        postreg_tcdm_wide_slave_req[p].wdata,
      wen:          postreg_tcdm_wide_slave_req[p].wen,
      be:           postreg_tcdm_wide_slave_req[p].be,
      tgt_addr:     postreg_tcdm_wide_slave_req[p].tgt_addr[WideTileOff +: idx_width(NumBanksPerTile) + TCDMAddrMemWidth],
      ini_addr:     postreg_tcdm_wide_slave_req[p].ini_addr,
      src_group_id: postreg_tcdm_wide_slave_req[p].src_group_id
    };
  end

  // The group's slave ports answer on the response ports by class: a same-group
  // port on its own link, RO port k on RW port k % NumWideRwRespPortsPerTile and
  // WO port k on WO port k. Ports that share a response port meet in the xbar.
  tcdm_wide_slave_resp_t [NumWideRemoteReqPortsPerTile-1:0] ingress_tcdm_wide_slave_resp;
  logic                  [NumWideRemoteReqPortsPerTile-1:0] ingress_tcdm_wide_slave_resp_valid;
  logic                  [NumWideRemoteReqPortsPerTile-1:0] ingress_tcdm_wide_slave_resp_ready;
  logic [NumWideRemoteReqPortsPerTile-1:0][idx_width(NumWideRemoteRespPortsPerTile)-1:0]
      ingress_tcdm_wide_slave_resp_sel;

  for (genvar p = 0; p < NumWideRemoteReqPortsPerTile; p++) begin : gen_wide_resp_port_sel
    localparam int unsigned RoBase = NumWideIntraGroupPortsPerTile;
    localparam int unsigned WoBase = RoBase + NumWideRoReqPortsPerTile;
    assign ingress_tcdm_wide_slave_resp_sel[p] = idx_width(NumWideRemoteRespPortsPerTile)'(
        (p < RoBase) ? p :
        (p < WoBase) ? RoBase + (p - RoBase) % NumWideRwRespPortsPerTile :
                       RoBase + NumWideRwRespPortsPerTile + (p - WoBase) % NumWideWoRespPortsPerTile);
  end

  stream_xbar #(
    .NumInp   (NumWideRemoteReqPortsPerTile ),
    .NumOut   (NumWideRemoteRespPortsPerTile),
    .payload_t(tcdm_wide_slave_resp_t       ),
    .AxiVldRdy(1'b1                         )
  ) i_wide_slave_resp_port_xbar (
    .clk_i  (clk_i                              ),
    .rst_ni (rst_ni                             ),
    .flush_i(1'b0                               ),
    .rr_i   ('0                                 ),
    .data_i (ingress_tcdm_wide_slave_resp       ),
    .valid_i(ingress_tcdm_wide_slave_resp_valid ),
    .ready_o(ingress_tcdm_wide_slave_resp_ready ),
    .sel_i  (ingress_tcdm_wide_slave_resp_sel   ),
    .data_o (prereg_tcdm_wide_slave_resp        ),
    .valid_o(prereg_tcdm_wide_slave_resp_valid  ),
    .ready_i(prereg_tcdm_wide_slave_resp_ready  ),
    .idx_o  (/* unused */                       )
  );
  `endif

  // Bank-side TCDM interconnect: narrow req/resp routing, wide superbank
  // demux/mux, per-bank wide-narrow arbitration, and within-superbank bank-id remap.
  mempool_tcdm_bank_interco #(
    .NumNarrowReq   (NumLocalPorts + NumNarrowRemoteReqPortsPerTile  ),
    .NumNarrowResp  (NumLocalPorts + NumNarrowRemoteRespPortsPerTile ),
    `ifdef USE_WIDE_TCDM_PLANE
    .NumWidePorts(NumWideEngIngress),
    `endif
    .NumBanksPerTile(NumBanksPerTile  ),
    .NumSuperbanks  (NumSuperbanks    ),
    .NarrowDataWidth(DataWidth        ),
    .DmaWideDataWidth(DmaDataWidth    ),
    .ByteOffset     (ByteOffset       ),
    .SpmBankIdRemap (SpmBankIdRemap   ),
    .narrow_req_t   (tcdm_slave_req_t ),
    .narrow_resp_t  (tcdm_slave_resp_t),
    .dma_wide_req_t (tcdm_dma_req_t   ),
    .dma_wide_resp_t(tcdm_dma_resp_t  ),
    .group_id_t     (group_id_t       ),
    .resp_idx_t     (local_resp_interco_addr_t)
    `ifdef USE_WIDE_TCDM_PLANE
    ,.WideDataWidth(WideTcdmDataWidth   )
    ,.wide_req_t   (tcdm_wide_req_t )
    ,.wide_resp_t  (tcdm_wide_resp_t)
    `endif
  ) i_tcdm_bank_interco (
    .clk_i                  (clk_i),
    .rst_ni                 (rst_ni),
    .group_id_i             (group_id),
    // Narrow req inputs: cores at low indices, remote slaves above; concat
    // order must match the legacy i_local_req_interco order.
    .slv_narrow_req_i       ({postreg_tcdm_slave_req,       local_req_interco_payload}),
    .slv_narrow_req_valid_i ({postreg_tcdm_slave_req_valid, local_req_interco_valid  }),
    .slv_narrow_req_ready_o ({postreg_tcdm_slave_req_ready, local_req_interco_ready  }),
    // Narrow resp outputs: cores at low indices, remote resp ports above.
    .slv_narrow_resp_o      ({prereg_tcdm_slave_resp,       local_resp_interco_payload}),
    .slv_narrow_resp_valid_o({prereg_tcdm_slave_resp_valid, local_resp_interco_valid  }),
    .slv_narrow_resp_ready_i({prereg_tcdm_slave_resp_ready, local_resp_interco_ready  }),
    .slv_dma_req_i          (tcdm_dma_req_i       ),
    .slv_dma_req_valid_i    (tcdm_dma_req_valid_i ),
    .slv_dma_req_ready_o    (tcdm_dma_req_ready_o ),
    .slv_dma_resp_o         (tcdm_dma_resp_o      ),
    .slv_dma_resp_valid_o   (tcdm_dma_resp_valid_o),
    .slv_dma_resp_ready_i   (tcdm_dma_resp_ready_i),
    `ifdef USE_WIDE_TCDM_PLANE
    // Wide req inputs: this tile's own-tile wide clients at low indices, the
    // group's wide slave ports above -- same ordering as the narrow ports.
    // Only the bank-local slots (RedMulE) have an own-tile ingress.
    .slv_wide_req_i         ({postreg_tcdm_wide_slave_req_bank,  local_wide_req_interco_payload [NumWideBankLocalPorts-1:0]}),
    .slv_wide_req_valid_i   ({postreg_tcdm_wide_slave_req_valid, local_wide_req_interco_valid   [NumWideBankLocalPorts-1:0]}),
    .slv_wide_req_ready_o   ({postreg_tcdm_wide_slave_req_ready, local_wide_req_interco_ready   [NumWideBankLocalPorts-1:0]}),
    // Wide resp outputs: own-tile clients at low indices, slave resp above.
    .slv_wide_resp_o        ({ingress_tcdm_wide_slave_resp,       local_wide_resp_interco_payload[NumWideBankLocalPorts-1:0]}),
    .slv_wide_resp_valid_o  ({ingress_tcdm_wide_slave_resp_valid, local_wide_resp_interco_valid  [NumWideBankLocalPorts-1:0]}),
    .slv_wide_resp_ready_i  ({ingress_tcdm_wide_slave_resp_ready, local_wide_resp_interco_ready  [NumWideBankLocalPorts-1:0]}),
    `endif
    .mst_req_o              (bank_req_payload    ),
    .mst_req_valid_o        (bank_req_valid      ),
    .mst_req_ready_i        (bank_req_ready      ),
    .mst_req_wide_o         (bank_req_wide       ),
    .mst_req_ini_addr_o     (bank_req_ini_addr   ),
    .mst_resp_i             (bank_resp_payload   ),
    .mst_resp_valid_i       (bank_resp_valid     ),
    .mst_resp_ready_o       (bank_resp_ready     ),
    .mst_resp_wide_i        (bank_resp_wide      ),
    // Conditioned resp xbar sel (see gen_superbank_resp_ini_addr above).
    .mst_resp_ini_addr_i    (superbank_resp_ini_addr)
  );

  /*******************
   *   Core De/mux   *
   *******************/

   // Pass NumLocalPorts (= local requesters): leaving the upper RedMulE slots
   // undriven bleeds X into the remote interco valid/sel and deadlocks.
   mempool_tile_remote_req_router #(
    .NumCoresPerTile                      (NumLocalPorts                       ),
    .NumNarrowRemoteReqPortsPerTile       (NumNarrowRemoteReqPortsPerTile      ),
    .NumNarrowRoReqPortsPerTile           (NumNarrowRoReqPortsPerTile          ),
    .NumNarrowWoReqPortsPerTile           (NumNarrowWoReqPortsPerTile          ),
    .NumNarrowRwReqPortsPerTile           (NumNarrowRwReqPortsPerTile          ),
    .NumBanksPerTile                      (NumBanksPerTile                     ),
    .NumTilesPerGroup                     (NumTilesPerGroup                    ),
    .NumGroups                            (NumGroups                           ),
    .ByteOffset                           (ByteOffset                          )
    `ifdef USE_WIDE_TCDM_PLANE
    ,.NumWideLocalPorts                   (NumWideLocalPorts                   )
    ,.NumWideIntraGroupPortsPerTile       (NumWideIntraGroupPortsPerTile       )
    ,.NumWideRoReqPortsPerTile            (NumWideRoReqPortsPerTile            )
    ,.NumWideWoReqPortsPerTile            (NumWideWoReqPortsPerTile            )
    ,.WideBeatBytes                       (WideTcdmDataWidth/8                 )
    `endif
  ) i_mempool_tile_remote_req_router (
    .group_id_i                           (group_id                            ),
    .remote_req_interco_wen_i             (remote_req_interco_wen              ),
    .remote_req_interco_amoen_i           (remote_req_interco_amoen            ),
    .remote_req_interco_addr_int_i        (remote_req_interco_addr_int         ),
    .remote_req_tgt_addr_o                (remote_req_tgt_addr                 ),
    .remote_req_tgt_group_id_o            (remote_req_tgt_group_id             ),
    .remote_req_tgt_sel_o                 (remote_req_tgt_sel                  )
    `ifdef USE_WIDE_TCDM_PLANE
    ,.remote_wide_req_interco_wen_i       (remote_wide_req_interco_wen         )
    ,.remote_wide_req_interco_addr_int_i  (remote_wide_req_interco_addr_int    )
    ,.remote_wide_req_tgt_addr_o          (remote_wide_req_tgt_addr            )
    ,.remote_wide_req_tgt_group_id_o      (remote_wide_req_tgt_group_id        )
    ,.remote_wide_req_tgt_sel_o           (remote_wide_req_tgt_sel             )
    `endif
  );

`ifndef SYNTHESIS
  // GATE item 3 (roadmap §T2.3), network-side half. The scrambler already
  // asserts confinement where the address is produced; this repeats it on the
  // ROUTER's decoded target group, so anything that rewrites the address
  // between the two is caught as well. Gated on vc_hit so ordinary global
  // traffic -- stacks, barriers, the sequential region -- is not flagged, and
  // parameterized by the level the request matched so the rectangle is exact.
  function automatic bit vc_in_rect(input int unsigned own,
                                    input int unsigned tgt,
                                    input int unsigned k);
    // Morton group ids: a level-k rectangle is a contiguous id range, so
    // membership is "same base", one mask and one compare. The per-axis form
    // this replaces was the raster rectangle test and would reject legal
    // Morton neighbours -- it flagged group 0 reaching group 2 at k=1, which
    // is inside group 0's own rectangle {0,1,2,3}.
    automatic int unsigned ngroups = 1 << (2*k);
    vc_in_rect = (own / ngroups) == (tgt / ngroups);
  endfunction

  if (VirtualCluster && (NumGroups > 1)) begin : gen_vc_router_confinement
    // CLOCKED, not combinational: an immediate assertion in always_comb fires
    // on delta-cycle transients while the address decode settles, which
    // reports escapes that never exist at any clock edge.
    for (genvar i = 0; i < NumLocalPorts; i++) begin : gen_vc_conf_narrow
      always_ff @(posedge clk_i) begin
        if (rst_ni && vc_hit_port[i] && remote_req_interco_valid[i]) begin
          assert (vc_in_rect(group_id, remote_req_tgt_group_id[i],
                             vc_level_port[i]))
            else $fatal(1,
                "VC confinement escape, narrow port %0d level %0d: group %0d reached %0d",
                i, vc_level_port[i], group_id, remote_req_tgt_group_id[i]);
        end
      end
    end
  end
`endif

  // SoC requests (one per core data port; Spatz vector ports never hit SoC
  // addresses, so their slots are inert). NumDataPortsPerCore==1 for non-Spatz.
  localparam int unsigned NumSoCPorts = NumCorePorts;
  dreq_t  [NumSoCPorts-1:0] soc_data_q;
  logic   [NumSoCPorts-1:0] soc_data_qvalid;
  logic   [NumSoCPorts-1:0] soc_data_qready;
  dresp_t [NumSoCPorts-1:0] soc_data_p;
  logic   [NumSoCPorts-1:0] soc_data_pvalid;
  logic   [NumSoCPorts-1:0] soc_data_pready;
  // Post-demux SoC channels (gen_redmule may divert core 0's to redmule_rmcfg).
  // Post-fsync remainder: everything that is NOT a tile-local barrier access.
  // The fsync window is split off BEFORE the RedMulE/passthrough logic so that
  // EVERY core reaches it, not just core 0 -- which core is the tile leader is
  // a software choice and must not be wired in.
  dreq_t  [NumSoCPorts-1:0] soc_nofs_q;
  logic   [NumSoCPorts-1:0] soc_nofs_qvalid;
  logic   [NumSoCPorts-1:0] soc_nofs_qready;
  dresp_t [NumSoCPorts-1:0] soc_nofs_p;
  logic   [NumSoCPorts-1:0] soc_nofs_pvalid;
  logic   [NumSoCPorts-1:0] soc_nofs_pready;

  dreq_t  [NumSoCPorts-1:0] soc_mux_q;
  logic   [NumSoCPorts-1:0] soc_mux_qvalid;
  logic   [NumSoCPorts-1:0] soc_mux_qready;
  dresp_t [NumSoCPorts-1:0] soc_mux_p;
  logic   [NumSoCPorts-1:0] soc_mux_pvalid;
  logic   [NumSoCPorts-1:0] soc_mux_pready;

  // Address map
  typedef enum int unsigned {
    TCDM_EXTERNAL = 0, TCDM_LOCAL, SOC
  } addr_map_slave_t;

  address_map_t [2:0] mask_map;
  assign mask_map = '{
    // Lowest priority: send request through the SoC port
    '{slave_idx: SOC,
      mask     : '0,
      value    : '0
    },
    // Send request through the external TCDM port
    '{slave_idx: TCDM_EXTERNAL,
      mask     : TCDMMask,
      value    : TCDMBaseAddr
    },
    // Highest priority: send request through the local TCDM port
    '{slave_idx: TCDM_LOCAL,
      mask     : TCDMMask | ({idx_width(NumTiles){1'b1}} << (ByteOffset + $clog2(NumBanksPerTile))),
      value    : TCDMBaseAddr | (tile_id_i << (ByteOffset + $clog2(NumBanksPerTile)))
    }
  };

  // TCDM-only clients use the same local/external address routing without
  // instantiating the scalar SoC response-ID path.
  address_map_t [1:0] tcdm_mask_map;
  assign tcdm_mask_map = '{
    '{slave_idx: TCDM_EXTERNAL,
      mask     : TCDMMask,
      value    : TCDMBaseAddr
    },
    '{slave_idx: TCDM_LOCAL,
      mask     : TCDMMask | ({idx_width(NumTiles){1'b1}} <<
                 (ByteOffset + $clog2(NumBanksPerTile))),
      value    : TCDMBaseAddr |
                 (tile_id_i << (ByteOffset + $clog2(NumBanksPerTile)))
    }
  };

  // The cores' TCDM master ports are flattened onto the local-port axis: port p
  // of core c is index core_port_base(c) + p (= i). Scalar builds have one port
  // per core, so i == core id and this reduces to the original per-core loop;
  // the Spatz vector ports extend the same axis.
  // ------------------------------------------------------------------
  //  Spatz burst -> wide plane divert
  // ------------------------------------------------------------------
  // A burst leaves the core on its VECTOR PORT 0 (index core_port_base(c) + 1;
  // port 0 of the slice is the scalar/FP port) as a single request with
  // burst_len > 1, and its beats come back on the first RecvPorts vector ports of
  // the same core. Off => every signal below is a wire and the netlist is unchanged.
`ifdef USE_WIDE_TCDM_PLANE
  // The wide client stream types exist only with the plane compiled in.
  tcdm_wide_rm_req_t  spatz_wide_req;
  logic               spatz_wide_req_valid, spatz_wide_req_ready;
  tcdm_wide_rm_resp_t spatz_wide_resp;
  logic               spatz_wide_resp_valid, spatz_wide_resp_ready;
`endif

  logic     [NumCorePorts-1:0] core_divert;
  logic     [NumCorePorts-1:0] adp_qready;
  logic     [NumCorePorts-1:0] adp_pvalid;
  data_t    [NumCorePorts-1:0] adp_pdata;
  meta_id_t [NumCorePorts-1:0] adp_pid;
  // Own-tile burst expansion (spatz_local_burst_expander): while exp_own[i], the
  // expander drives port i's narrow request and the core sees exp_qready[i].
  logic     [NumCorePorts-1:0] exp_own;
  logic     [NumCorePorts-1:0] exp_qvalid;
  addr_t    [NumCorePorts-1:0] exp_qaddr;
  meta_id_t [NumCorePorts-1:0] exp_qid;
  logic     [NumCorePorts-1:0] exp_qready;

`ifdef USE_WIDE_TCDM_PLANE
  if (SpatzWideBurst) begin : gen_spatz_wide_burst
    // One lane per vector memory port: a word's lane is fixed by its address, so
    // the VLSU consumes a burst through its ordinary 4-ROB commit path.
    localparam int unsigned RecvPorts = NumDataPortsPerCore - 1;
    // Vector port 0 of a core: the only port a burst is ever issued on.
    localparam int unsigned BurstPortOff = 1;

    // One expander + one adapter per core, each adapter on its own wide slot. No
    // arbitration here: the cores contend only at the tile's single wide remote
    // link, which the mux after the slots resolves, together with RedMulE.
    for (genvar c = NumScalarCoresPerTile; c < NumCoresPerTile; c++) begin : gen_spatz_wide_core
      localparam int unsigned P0   = core_port_base(c) + BurstPortOff;
      localparam int unsigned WIdx = WideSpatzPortIdx + c - NumScalarCoresPerTile;

      logic     [RecvPorts-1:0] rsp_valid;
      data_t    [RecvPorts-1:0] rsp_data;
      meta_id_t [RecvPorts-1:0] rsp_id;
      logic                     req_ready;
      logic [RecvPorts-1:0]     drain_stall;
      tcdm_wide_rm_req_t        wreq;
      logic                     wreq_valid, wreq_ready;
      tcdm_wide_rm_resp_t       wresp;
      logic                     wresp_valid, wresp_ready;

      // The narrow path owns these ports whenever it is answering on them, so hold
      // the drain rather than presenting two beats on one port in one cycle.
      // PER-LANE, not an OR across lanes. A vector store's acks come back on every
      // vector port, so an all-lanes stall blocks the whole drain for as long as any
      // ack is in flight -- and the burst's reserved ROB entries keep rob_empty low,
      // which is what the store is waiting on. Stall only the lane that is busy.
      for (genvar l = 0; l < RecvPorts; l++) begin : gen_drain_stall
        assign drain_stall[l] = snitch_data_pvalid[P0 + l];
      end : gen_drain_stall

      // ---- own-tile or remote burst ----
      // The burst base, scrambled like every narrow and wide address; its own
      // scrambler because snitch_data_qaddr[P0] carries expanded words while an
      // own-tile burst is being issued. A burst is aligned to its span, so it
      // lies in one tile and the base alone decides.
      addr_t burst_addr_scrambled;
      addr_t own_tile_mask, own_tile_value;
      logic  burst_req, burst_local;

      mempool_addr_scrambler #(
        .AddrWidth               (AddrWidth        ),
        .ByteOffset              (ByteOffset       ),
        .NumTiles                (NumTiles         ),
        .NumTilesPerDma          (NumTilesPerDma   ),
        .NumBanksPerTile         (NumBanksPerTile  ),
        .Bypass                  (0                ),
        .SeqMemSizePerTile       (SeqMemSizePerTile),
        .TCDMBaseAddr            (TCDMBaseAddr     ),
        .TCDMMask                (TCDMMask         ),
        .EnableSeqInterleaveSwap (1'b1             ),
        .EnableTileIdRemap       (TileIdRemap      )
      ) i_burst_addr_scrambler (
        .address_i (cc_data_qaddr[P0]    ),
        .address_o (burst_addr_scrambled )
      );

      // Same own-tile rule as RedMulE's wide demux (TCDM_WIDE_LOCAL entry).
      assign own_tile_mask  = TCDMMask | ({idx_width(NumTiles){1'b1}} << (ByteOffset + $clog2(NumBanksPerTile)));
      assign own_tile_value = TCDMBaseAddr | (tile_id_i << (ByteOffset + $clog2(NumBanksPerTile)));
      assign burst_req      = cc_data_qvalid[P0] &&
                              (cc_data_qburst_len[P0] > spatz_pkg::BurstLenWidth'(1));
      assign burst_local    = ((burst_addr_scrambled & own_tile_mask) == own_tile_value);

      // ---- own-tile burst: narrow words on the vector ports ----
      logic                     exp_busy, exp_burst_ready;
      logic     [RecvPorts-1:0] exp_lane_valid;
      addr_t    [RecvPorts-1:0] exp_lane_addr;
      meta_id_t [RecvPorts-1:0] exp_lane_id;

      spatz_local_burst_expander #(
        .RecvPorts    (RecvPorts                 ),
        .MaxBurstWords(spatz_pkg::MaxBurstWords  ),
        .BurstLenWidth(spatz_pkg::BurstLenWidth  ),
        .RobIdWidth   (spatz_pkg::MemRspIdWidth  ),
        .WordBytes    (DataWidth/8               ),
        .addr_t       (addr_t                    ),
        .meta_id_t    (meta_id_t                 )
      ) i_local_burst_expander (
        .clk_i        (clk_i                            ),
        .rst_ni       (rst_ni                           ),
        .burst_valid_i(burst_req && burst_local         ),
        .burst_addr_i (cc_data_qaddr[P0]                ),
        .burst_id_i   (cc_data_qid[P0]                  ),
        .burst_len_i  (cc_data_qburst_len[P0]           ),
        .burst_ready_o(exp_burst_ready                  ),
        .busy_o       (exp_busy                         ),
        .lane_valid_o (exp_lane_valid                   ),
        .lane_addr_o  (exp_lane_addr                    ),
        .lane_id_o    (exp_lane_id                      ),
        .lane_ready_i (snitch_data_qready[P0 +: RecvPorts])
      );

      // The expander owns all of this core's lanes while busy: the VLSU's own
      // requests on them wait. On vector port 0 the core sees the burst
      // handshake; a remote burst still goes to the adapter (core_divert wins).
      for (genvar l = 0; l < RecvPorts; l++) begin : gen_exp_lane
        assign exp_own[P0 + l]    = exp_busy;
        assign exp_qvalid[P0 + l] = exp_lane_valid[l];
        assign exp_qaddr[P0 + l]  = exp_lane_addr[l];
        assign exp_qid[P0 + l]    = exp_lane_id[l];
        if (l == 0) begin : gen_burst_port
          assign exp_qready[P0 + l] = burst_req && burst_local && exp_burst_ready;
        end else begin : gen_other_lane
          assign exp_qready[P0 + l] = 1'b0;
        end
      end : gen_exp_lane

      // ---- remote burst: one wide access ----
      spatz_wide_burst_adapter #(
        .RecvPorts  (RecvPorts               ),
        .RobIdWidth (spatz_pkg::MemRspIdWidth),
        // Jointly determined with Spatz, not chosen freely: every in-flight burst holds
        // MaxBurstWords/NrMemPorts ids in every lane, so the reorder buffers can back
        // exactly RobDepth*NrMemPorts/MaxBurstWords of them. More slots could never be
        // filled; fewer would leave issue capacity Spatz already has unused.
        .NumOutstanding (((2**spatz_pkg::MemRspIdWidth) * NumMemPortsPerSpatz >
                          spatz_pkg::MaxBurstWords)
                         ? ((2**spatz_pkg::MemRspIdWidth) * NumMemPortsPerSpatz /
                            spatz_pkg::MaxBurstWords)
                         : 1),
        .wide_req_t (tcdm_wide_rm_req_t      ),
        .wide_resp_t(tcdm_wide_rm_resp_t     )
      ) i_spatz_wide_burst_adapter (
        .clk_i                (clk_i                    ),
        .rst_ni               (rst_ni                   ),
        // Only remote bursts reach the adapter (its wide slot is remote-only).
        .spatz_req_addr_i     (burst_addr_scrambled     ),
        .spatz_req_id_i       (cc_data_qid[P0]          ),
        .spatz_req_burst_len_i(cc_data_qburst_len[P0]   ),
        .spatz_req_valid_i    (cc_data_qvalid[P0] && !burst_local),
        .spatz_req_ready_o    (req_ready                ),
        .spatz_resp_valid_o   (rsp_valid                ),
        .spatz_resp_data_o    (rsp_data                 ),
        .spatz_resp_id_o      (rsp_id                   ),
        .drain_stall_i        (drain_stall              ),
        .wide_req_o           (wreq                     ),
        .wide_req_valid_o     (wreq_valid               ),
        .wide_req_ready_i     (wreq_ready               ),
        .wide_resp_i          (wresp                    ),
        .wide_resp_valid_i    (wresp_valid              ),
        .wide_resp_ready_o    (wresp_ready              )
      );

      // Divert this core's vector port 0 off the narrow shim for a remote burst,
      // and merge the adapter's beats back onto its receive lanes.
      assign core_divert[P0] = burst_req && !burst_local;

`ifndef SYNTHESIS
      // pragma translate_off
      // +spatz_burst_stats: report the remote bursts sent over the wide plane.
      longint unsigned stat_remote_bursts;
      always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni)                        stat_remote_bursts <= 0;
        else if (core_divert[P0] && req_ready) stat_remote_bursts <= stat_remote_bursts + 1;
      end
      final begin
        if ($test$plusargs("spatz_burst_stats") && (stat_remote_bursts != 0))
          $display("[BURST] %m remote_bursts=%0d", stat_remote_bursts);
      end
      // pragma translate_on
`endif
      assign adp_qready[P0]  = req_ready;
      for (genvar l = 0; l < RecvPorts; l++) begin : gen_lane
        assign adp_pvalid[P0 + l] = rsp_valid[l];
        assign adp_pdata[P0 + l]  = rsp_data[l];
        assign adp_pid[P0 + l]    = rsp_id[l];
      end : gen_lane

      // ---- remote wide link ----
      // Only remote bursts reach the adapter (own-tile ones are expanded above),
      // so its wide access goes straight to the tile's remote wide port; the
      // remote request router derives tgt_addr / tgt_group_id / tgt_sel from
      // the address, as for every other wide client.
      assign remote_wide_req_interco_valid[WIdx]         = wreq_valid;
      assign wreq_ready                                  = remote_wide_req_interco_ready[WIdx];
      assign remote_wide_req_interco_addr_int[WIdx]      = wreq.addr;
      assign remote_wide_req_interco[WIdx].tgt_addr      = remote_wide_req_tgt_addr[WIdx];
      assign remote_wide_req_interco[WIdx].tgt_group_id  = remote_wide_req_tgt_group_id[WIdx];
      assign remote_wide_req_interco[WIdx].src_tile_id   = tcdm_wide_tile_id;
      assign remote_wide_req_interco[WIdx].wen           = wreq.write;
      assign remote_wide_req_interco[WIdx].be            = wreq.strb;
      assign remote_wide_req_interco[WIdx].wdata.data    = wreq.data;
      assign remote_wide_req_interco[WIdx].wdata.amo     = wreq.amo;
      assign remote_wide_req_interco[WIdx].wdata.meta_id = wreq.id;
      // core_id is the wide slot index: what the shared response crossbar
      // selects on to bring the response back to this core's adapter.
      assign remote_wide_req_interco[WIdx].wdata.core_id = tile_core_id_t'(WIdx);
      assign remote_wide_req_interco_tgt_sel[WIdx]       = remote_wide_req_tgt_sel[WIdx];

      assign wresp = '{
        data:  remote_wide_resp_interco[WIdx].rdata.data,
        id:    remote_wide_resp_interco[WIdx].rdata.meta_id,
        write: 1'b0,
        error: 1'b0
      };
      assign wresp_valid                          = remote_wide_resp_interco_valid[WIdx];
      assign remote_wide_resp_interco_ready[WIdx] = wresp_ready;

      // No own-tile side on this slot.
      assign local_wide_req_interco_valid[WIdx]    = 1'b0;
      assign local_wide_req_interco_addr_int[WIdx] = '0;
      assign local_wide_req_interco_payload[WIdx]  = '0;
      assign local_wide_resp_interco_ready[WIdx]   = 1'b1;
    end : gen_spatz_wide_core

    // Ports that are not a Spatz core's vector port 0 are never diverted.
    for (genvar i = 0; i < NumCorePorts; i++) begin : gen_plain_ports
      localparam bit          Scalar = i < NumScalarCoresPerTile;
      localparam int unsigned Port   =
          Scalar ? 0 : (i - NumScalarCoresPerTile) % NumDataPortsPerCore;
      if (Scalar || (Port != BurstPortOff)) begin : gen_plain
        assign core_divert[i] = 1'b0;
        assign adp_qready[i]  = 1'b0;
      end
      if (Scalar || (Port < BurstPortOff) || (Port >= BurstPortOff + RecvPorts)) begin : gen_no_lane
        assign adp_pvalid[i] = 1'b0;
        assign adp_pdata[i]  = '0;
        assign adp_pid[i]    = '0;
        assign exp_own[i]    = 1'b0;
        assign exp_qvalid[i] = 1'b0;
        assign exp_qaddr[i]  = '0;
        assign exp_qid[i]    = '0;
        assign exp_qready[i] = 1'b0;
      end
    end : gen_plain_ports

  end else begin : gen_no_spatz_wide_burst
    assign core_divert = '0;
    assign adp_qready  = '0;
    assign adp_pvalid  = '0;
    assign adp_pdata   = '0;
    assign adp_pid     = '0;
    assign exp_own     = '0;
    assign exp_qvalid  = '0;
    assign exp_qaddr   = '0;
    assign exp_qid     = '0;
    assign exp_qready  = '0;
  end
`else
  assign core_divert = '0;
  assign adp_qready  = '0;
  assign adp_pvalid  = '0;
  assign adp_pdata   = '0;
  assign adp_pid     = '0;
  assign exp_own     = '0;
  assign exp_qvalid  = '0;
  assign exp_qaddr   = '0;
  assign exp_qid     = '0;
  assign exp_qready  = '0;
`endif

  // The divert layer proper: mask the narrow request, merge the two response
  // sources. A diverted port's ready is driven in gen_adp_gnt above. While the
  // own-tile burst expander owns a port, it drives the narrow request (a load
  // word: no write, no AMO, full strobe) and the core sees exp_qready.
  for (genvar i = 0; i < NumCorePorts; i++) begin : gen_divert_layer
    assign snitch_data_qvalid[i] = exp_own[i] ? exp_qvalid[i] : (cc_data_qvalid[i] && !core_divert[i]);
    assign snitch_data_qaddr[i]  = exp_own[i] ? exp_qaddr[i]  : cc_data_qaddr[i];
    assign snitch_data_qwrite[i] = exp_own[i] ? 1'b0          : cc_data_qwrite[i];
    assign snitch_data_qamo[i]   = exp_own[i] ? '0            : cc_data_qamo[i];
    assign snitch_data_qdata[i]  = exp_own[i] ? '0            : cc_data_qdata[i];
    assign snitch_data_qstrb[i]  = exp_own[i] ? '1            : cc_data_qstrb[i];
    assign snitch_data_qid[i]    = exp_own[i] ? exp_qid[i]    : cc_data_qid[i];
    assign cc_data_qready[i]     = core_divert[i] ? adp_qready[i] :
                                   exp_own[i]     ? exp_qready[i] : snitch_data_qready[i];
    assign cc_data_pvalid[i]     = snitch_data_pvalid[i] || adp_pvalid[i];
    assign cc_data_pdata[i]      = adp_pvalid[i] ? adp_pdata[i] : snitch_data_pdata[i];
    assign cc_data_pid[i]        = adp_pvalid[i] ? adp_pid[i]   : snitch_data_pid[i];
    assign cc_data_perror[i]     = snitch_data_perror[i];
    assign snitch_data_pready[i] = cc_data_pready[i];
  end : gen_divert_layer

  for (genvar i = 0; i < NumCorePorts; i++) begin: gen_core_mux
      assign local_req_interco_payload[i].tgt_addr =
       tcdm_addr_t'({local_req_interco_addr_int[i][ByteOffset + idx_width(NumBanksPerTile) + $clog2(NumTiles) +: TCDMAddrMemWidth], // Bank address
               local_req_interco_addr_int[i][ByteOffset +: idx_width(NumBanksPerTile)]}); // Bank

      // Remote routing metadata (tgt_addr, tgt_group_id) comes from
      // i_mempool_tile_remote_req_router; wire it into the request payload.
      assign remote_req_interco[i].tgt_addr     = remote_req_tgt_addr[i];
      assign remote_req_interco[i].tgt_group_id = remote_req_tgt_group_id[i];
      if (NumGroups == 1) begin : gen_remote_req_interco_tgt_sel
        assign remote_req_interco_tgt_sel[i] = 1'b0;
      end else begin : gen_remote_req_interco_tgt_sel
        assign remote_req_interco_tgt_sel[i] = remote_req_tgt_sel[i];
      end

    // We don't care about these
    assign local_req_interco_payload[i].wdata.core_id = '0;
    assign local_req_interco_payload[i].ini_addr      = '0;
    assign local_req_interco_payload[i].src_group_id  = '0;
    assign soc_data_q[i].id                           = '0;

    // Requester local-port id (core id for scalar builds; flat port index for Spatz)
    assign remote_req_interco[i].wdata.core_id = i[idx_width(NumLocalPorts)-1:0];

    // The wen of the req
    assign remote_req_interco_wen   [i] = remote_req_interco[i].wen;
    assign remote_req_interco_amoen [i] = |remote_req_interco[i].wdata.amo;

    // Scramble + remap address before the TCDM shim: stage 1 seq->interleave
    // swap always on; stage 2 tile-id remap gated by TileIdRemap.
    mempool_addr_scrambler #(
      .AddrWidth               (AddrWidth        ),
      .ByteOffset              (ByteOffset       ),
      .NumTiles                (NumTiles         ),
      .NumTilesPerDma          (NumTilesPerDma   ),
      .NumBanksPerTile         (NumBanksPerTile  ),
      .Bypass                  (0                ),
      .SeqMemSizePerTile       (SeqMemSizePerTile),
      .TCDMBaseAddr            (TCDMBaseAddr     ),
      .TCDMMask                (TCDMMask         ),
      .EnableSeqInterleaveSwap (1'b1             ),
      .EnableTileIdRemap       (TileIdRemap      ),
      .TCDMSize                (TCDMSize         ),
      .NumTilesPerGroup        (NumTilesPerGroup ),
      .NumX                    (NumX             ),
      .NumY                    (NumY             ),
      .EnableVirtualCluster           (VirtualClusterCore      ),
      .VcRegionEndChk          (VcRegionEnd      )
    ) i_addr_scrambler (
      .address_i  (snitch_data_qaddr[i]          ),
      .group_id_i (group_id                      ),
      .vc_hit_o   (vc_hit_port[i]                ),
      .vc_level_o (vc_level_port[i]              ),
      .address_o  (snitch_data_qaddr_scrambled[i])
    );

    if (!TrafficGeneration) begin: gen_tcdm_shim
      tcdm_shim #(
        .AddrWidth           (AddrWidth                         ),
        .DataWidth           (DataWidth                         ),
        .MaxOutStandingTrans (snitch_pkg::NumIntOutstandingLoads),
        .NrTCDM              (2                                 ),
        .NrSoC               (1                                 ),
        .NumRules            (3                                 ),
        .req_t               (snitch_pkg::dreq_t                ),
        .resp_t              (snitch_pkg::dresp_t               )
      ) i_tcdm_shim (
        .clk_i              (clk_i                                                                              ),
        .rst_ni             (rst_ni                                                                             ),
        // to TCDM --> FF Connection to outside of tile
        .tcdm_req_valid_o   ({local_req_interco_valid[i],                  remote_req_interco_valid[i]}         ),
        .tcdm_req_tgt_addr_o({local_req_interco_addr_int[i],               remote_req_interco_addr_int[i]}      ),
        .tcdm_req_wen_o     ({local_req_interco_payload[i].wen,            remote_req_interco[i].wen}           ),
        .tcdm_req_wdata_o   ({local_req_interco_payload[i].wdata.data,     remote_req_interco[i].wdata.data}    ),
        .tcdm_req_amo_o     ({local_req_interco_payload[i].wdata.amo,      remote_req_interco[i].wdata.amo}     ),
        .tcdm_req_id_o      ({local_req_interco_payload[i].wdata.meta_id,  remote_req_interco[i].wdata.meta_id} ),
        .tcdm_req_be_o      ({local_req_interco_payload[i].be,             remote_req_interco[i].be}            ),
        .tcdm_req_ready_i   ({local_req_interco_ready[i],                  remote_req_interco_ready[i]}         ),
        .tcdm_resp_valid_i  ({local_resp_interco_valid[i],                 remote_resp_interco_valid[i]}        ),
        .tcdm_resp_ready_o  ({local_resp_interco_ready[i],                 remote_resp_interco_ready[i]}        ),
        .tcdm_resp_rdata_i  ({local_resp_interco_payload[i].rdata.data,    remote_resp_interco[i].rdata.data}   ),
        .tcdm_resp_id_i     ({local_resp_interco_payload[i].rdata.meta_id, remote_resp_interco[i].rdata.meta_id}),
        // to SoC
        .soc_qaddr_o        (soc_data_q[i].addr                                                                 ),
        .soc_qwrite_o       (soc_data_q[i].write                                                                ),
        .soc_qamo_o         (soc_data_q[i].amo                                                                  ),
        .soc_qdata_o        (soc_data_q[i].data                                                                 ),
        .soc_qstrb_o        (soc_data_q[i].strb                                                                 ),
        .soc_qvalid_o       (soc_data_qvalid[i]                                                                 ),
        .soc_qready_i       (soc_data_qready[i]                                                                 ),
        .soc_pdata_i        (soc_data_p[i].data                                                                 ),
        .soc_pwrite_i       (soc_data_p[i].write                                                                ),
        .soc_perror_i       (soc_data_p[i].error                                                                ),
        .soc_pvalid_i       (soc_data_pvalid[i]                                                                 ),
        .soc_pready_o       (soc_data_pready[i]                                                                 ),
        // from core
        .data_qaddr_i       (snitch_data_qaddr_scrambled[i]                                                     ),
        .data_qwrite_i      (snitch_data_qwrite[i]                                                              ),
        .data_qamo_i        (snitch_data_qamo[i]                                                                ),
        .data_qdata_i       (snitch_data_qdata[i]                                                               ),
        .data_qstrb_i       (snitch_data_qstrb[i]                                                               ),
        .data_qid_i         (snitch_data_qid[i]                                                                 ),
        .data_qvalid_i      (snitch_data_qvalid[i]                                                              ),
        .data_qready_o      (snitch_data_qready[i]                                                              ),
        .data_pdata_o       (snitch_data_pdata[i]                                                               ),
        .data_perror_o      (snitch_data_perror[i]                                                              ),
        .data_pid_o         (snitch_data_pid[i]                                                                 ),
        .data_pvalid_o      (snitch_data_pvalid[i]                                                              ),
        .data_pready_i      (snitch_data_pready[i]                                                              ),
        .address_map_i      (mask_map                                                                           )
      );
    end else begin: gen_traffic_generator
      logic [31:0] core_id;
      assign core_id = unsigned'(tile_id_i) * NumCoresPerTile + unsigned'(i);
      traffic_generator #(
        .NumRules           (3                                 ),
        .TCDMBaseAddr       (TCDMBaseAddr                      ),
        .MaxOutStandingReads(snitch_pkg::NumIntOutstandingLoads)
      ) i_traffic_gen (
        .clk_i              (clk_i                                                        ),
        .rst_ni             (rst_ni                                                       ),
        .core_id_i          (core_id                                                      ),
        // Address map
        .address_map_i      (mask_map                                                     ),
        // To TCDM
        .tcdm_req_valid_o   ({local_req_interco_valid[i],                  remote_req_interco_valid[i]}         ),
        .tcdm_req_tgt_addr_o({local_req_interco_addr_int[i],               remote_req_interco_addr_int[i]}      ),
        .tcdm_req_wen_o     ({local_req_interco_payload[i].wen,            remote_req_interco[i].wen}           ),
        .tcdm_req_wdata_o   ({local_req_interco_payload[i].wdata.data,     remote_req_interco[i].wdata.data}    ),
        .tcdm_req_amo_o     ({local_req_interco_payload[i].wdata.amo,      remote_req_interco[i].wdata.amo}     ),
        .tcdm_req_id_o      ({local_req_interco_payload[i].wdata.meta_id,  remote_req_interco[i].wdata.meta_id} ),
        .tcdm_req_be_o      ({local_req_interco_payload[i].be,             remote_req_interco[i].be}            ),
        .tcdm_req_ready_i   ({local_req_interco_ready[i],                  remote_req_interco_ready[i]}         ),
        .tcdm_resp_valid_i  ({local_resp_interco_valid[i],                 remote_resp_interco_valid[i]}        ),
        .tcdm_resp_ready_o  ({local_resp_interco_ready[i],                 remote_resp_interco_ready[i]}        ),
        .tcdm_resp_rdata_i  ({local_resp_interco_payload[i].rdata.data,    remote_resp_interco[i].rdata.data}   ),
        .tcdm_resp_id_i     ({local_resp_interco_payload[i].rdata.meta_id, remote_resp_interco[i].rdata.meta_id})
      );

      // Tie unused signals
      assign soc_data_q[i].addr    = '0;
      assign soc_data_q[i].write   = '0;
      assign soc_data_q[i].amo     = '0;
      assign soc_data_q[i].data    = '0;
      assign soc_data_q[i].strb    = '0;
      assign soc_data_qvalid[i]    = '0;
      assign soc_data_pready[i]    = '0;
      assign snitch_data_qready[i] = '0;
      assign snitch_data_pdata[i]  = '0;
      assign snitch_data_perror[i] = '0;
      assign snitch_data_pid[i]    = '0;
      assign snitch_data_pvalid[i] = '0;
    end
  end

  /****************
   *   AXI Plug   *
   ****************/

  snitch_pkg::dreq_t soc_req_o;
  snitch_pkg::dresp_t soc_resp_i;

  logic soc_qvalid;
  logic soc_qready;
  logic soc_pvalid;
  logic soc_pready;

  // Drive '0 (not 'x): SoC resp has no meta-id, and 'x would propagate into the resp arb.
  assign soc_resp_i.id = '0;

  snitch_demux #(
    .NrPorts (NumSoCPorts        ),
    .req_t   (snitch_pkg::dreq_t ),
    .resp_t  (snitch_pkg::dresp_t)
  ) i_snitch_demux_data (
    .clk_i         (clk_i          ),
    .rst_ni        (rst_ni         ),
    .req_payload_i (soc_mux_q      ),
    .req_valid_i   (soc_mux_qvalid ),
    .req_ready_o   (soc_mux_qready ),
    .resp_payload_o(soc_mux_p      ),
    .resp_last_o   (/* Unused */   ),
    .resp_valid_o  (soc_mux_pvalid ),
    .resp_ready_i  (soc_mux_pready ),
    // Output
    .req_payload_o (soc_req_o      ),
    .req_valid_o   (soc_qvalid     ),
    .req_ready_i   (soc_qready     ),
    .resp_payload_i(soc_resp_i     ),
    .resp_last_i   (1'b1           ),
    .resp_valid_i  (soc_pvalid     ),
    .resp_ready_o  (soc_pready     )
  );

  // Core request
  axi_core_req_t  axi_cores_req_d, axi_cores_req_q;
  axi_core_resp_t axi_cores_resp_d, axi_cores_resp_q;

  snitch_axi_adapter #(
    .addr_t         (snitch_pkg::addr_t),
    .data_t         (snitch_pkg::data_t),
    .strb_t         (snitch_pkg::strb_t),
    .axi_mst_req_t  (axi_core_req_t    ),
    .axi_mst_resp_t (axi_core_resp_t   )
  ) i_snitch_core_axi_adapter (
    .clk_i       (clk_i           ),
    .rst_ni      (rst_ni          ),
    .slv_qaddr_i (soc_req_o.addr  ),
    .slv_qwrite_i(soc_req_o.write ),
    .slv_qamo_i  (soc_req_o.amo   ),
    .slv_qdata_i (soc_req_o.data  ),
    .slv_qsize_i (3'b010          ),
    .slv_qstrb_i (soc_req_o.strb  ),
    .slv_qrlen_i ('0              ),
    .slv_qvalid_i(soc_qvalid      ),
    .slv_qready_o(soc_qready      ),
    .slv_pdata_o (soc_resp_i.data ),
    .slv_pwrite_o(soc_resp_i.write),
    .slv_perror_o(soc_resp_i.error),
    .slv_plast_o (/* Unused */    ),
    .slv_pvalid_o(soc_pvalid      ),
    .slv_pready_i(soc_pready      ),
    .axi_req_o   (axi_cores_req_d ),
    .axi_resp_i  (axi_cores_resp_q)
  );

  axi_cut #(
    .aw_chan_t (axi_core_aw_t  ),
    .w_chan_t  (axi_core_w_t   ),
    .b_chan_t  (axi_core_b_t   ),
    .ar_chan_t (axi_core_ar_t  ),
    .r_chan_t  (axi_core_r_t   ),
    .axi_req_t (axi_core_req_t ),
    .axi_resp_t(axi_core_resp_t)
  ) axi_core_slice (
    .clk_i     (clk_i           ),
    .rst_ni    (rst_ni          ),
    .slv_req_i (axi_cores_req_d ),
    .slv_resp_o(axi_cores_resp_q),
    .mst_req_o (axi_cores_req_q ),
    .mst_resp_i(axi_cores_resp_d)
  );

  axi_mux #(
    .SlvAxiIDWidth (AxiCoreIdWidth ),
    .slv_aw_chan_t (axi_core_aw_t  ),
    .mst_aw_chan_t (axi_tile_aw_t  ),
    .w_chan_t      (axi_tile_w_t   ),
    .slv_b_chan_t  (axi_core_b_t   ),
    .mst_b_chan_t  (axi_tile_b_t   ),
    .slv_ar_chan_t (axi_core_ar_t  ),
    .mst_ar_chan_t (axi_tile_ar_t  ),
    .slv_r_chan_t  (axi_core_r_t   ),
    .mst_r_chan_t  (axi_tile_r_t   ),
    .slv_req_t     (axi_core_req_t ),
    .slv_resp_t    (axi_core_resp_t),
    .mst_req_t     (axi_tile_req_t ),
    .mst_resp_t    (axi_tile_resp_t),
    .NoSlvPorts    (1+NumCaches    ),
    .MaxWTrans     (8              ),
    .FallThrough   (1              )
  ) i_axi_mux (
    .clk_i      (clk_i                               ),
    .rst_ni     (rst_ni                              ),
    .test_i     (1'b0                                ),
    .slv_reqs_i ({axi_cores_req_q, axi_cache_req_q}  ),
    .slv_resps_o({axi_cores_resp_d, axi_cache_resp_d}),
    .mst_req_o  (axi_mst_req_o                       ),
    .mst_resp_i (axi_mst_resp_i                      )
  );

  /*****************************
   *  fractal_sync tile shim   *
   *****************************/
  // Carve the tile-local barrier window off EVERY SoC port. MAGIA reaches its
  // per-tile peripherals from every manager via an OBI crossbar rule table;
  // this is the same idea in the dialect we have -- a per-port address demux
  // feeding one arbiter. It deliberately does NOT extend the RedMulE demux,
  // which only ever saw core 0: which core is the tile leader is a software
  // choice and must not be wired in.
  dreq_t  [NumSoCPorts-1:0] fs_q;
  logic   [NumSoCPorts-1:0] fs_qvalid, fs_qready;
  dresp_t [NumSoCPorts-1:0] fs_p;
  logic   [NumSoCPorts-1:0] fs_pvalid, fs_pready;

  for (genvar c = 0; c < NumSoCPorts; c++) begin: gen_fsync_demux
    address_map_t [1:0] fs_map;
    assign fs_map = '{
      '{slave_idx: 32'd1, mask: '0,                  value: '0                     },
      '{slave_idx: 32'd0, mask: addr_t'(FSyncMask),  value: addr_t'(FSyncBaseAddr) }
    };

    snitch_addr_demux #(
      .NrOutput     (2                   ),
      .AddressWidth (AddrWidth           ),
      .NumRules     (2                   ),
      .req_t        (snitch_pkg::dreq_t  ),
      .resp_t       (snitch_pkg::dresp_t )
    ) i_fsync_demux (
      .clk_i,
      .rst_ni,
      .req_addr_i     (soc_data_q[c].addr                  ),
      .req_payload_i  (soc_data_q[c]                       ),
      .req_valid_i    (soc_data_qvalid[c]                  ),
      .req_ready_o    (soc_data_qready[c]                  ),
      .resp_payload_o (soc_data_p[c]                       ),
      .resp_valid_o   (soc_data_pvalid[c]                  ),
      .resp_ready_i   (soc_data_pready[c]                  ),
      .req_payload_o  ({soc_nofs_q[c],      fs_q[c]      } ),
      .req_valid_o    ({soc_nofs_qvalid[c], fs_qvalid[c] } ),
      .req_ready_i    ({soc_nofs_qready[c], fs_qready[c] } ),
      .resp_payload_i ({soc_nofs_p[c],      fs_p[c]      } ),
      .resp_valid_i   ({soc_nofs_pvalid[c], fs_pvalid[c] } ),
      .resp_ready_o   ({soc_nofs_pready[c], fs_pready[c] } ),
      .address_map_i  (fs_map                              )
    );
  end

  dreq_t                             fs_arb_q;
  logic                              fs_arb_valid;
  logic [idx_width(NumSoCPorts)-1:0] fs_arb_idx;
  logic [31:0]                       fs_rdata;
  logic                              fs_wake;

  rr_arb_tree #(
    .NumIn     (NumSoCPorts        ),
    .DataType  (snitch_pkg::dreq_t ),
    .AxiVldRdy (1'b1               )
  ) i_fsync_arb (
    .clk_i,
    .rst_ni,
    .flush_i (1'b0         ),
    .rr_i    ('0           ),
    .req_i   (fs_qvalid    ),
    .gnt_o   (fs_qready    ),
    .data_i  (fs_q         ),
    .req_o   (fs_arb_valid ),
    // A register access retires in the cycle it is granted, so the shim never
    // backpressures the bus. Its one-outstanding rule governs BARRIERS, not bus
    // accesses: a trigger arriving while busy is ignored, and STATUS.BUSY is
    // how software sees that.
    .gnt_i   (1'b1         ),
    .data_o  (fs_arb_q     ),
    .idx_o   (fs_arb_idx   )
  );

  logic                              fs_resp_valid_q;
  logic [idx_width(NumSoCPorts)-1:0] fs_resp_idx_q;
  logic [31:0]                       fs_resp_data_q;
  logic                              fs_resp_write_q;
  meta_id_t                          fs_resp_id_q;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      fs_resp_valid_q <= 1'b0;
      fs_resp_idx_q   <= '0;
      fs_resp_data_q  <= '0;
      fs_resp_write_q <= 1'b0;
      fs_resp_id_q    <= '0;
    end else begin
      fs_resp_valid_q <= fs_arb_valid;
      fs_resp_idx_q   <= fs_arb_idx;
      fs_resp_data_q  <= fs_rdata;
      fs_resp_write_q <= fs_arb_q.write;
      fs_resp_id_q    <= fs_arb_q.id;
    end
  end

  for (genvar c = 0; c < NumSoCPorts; c++) begin: gen_fsync_resp
    assign fs_pvalid[c] = fs_resp_valid_q & (fs_resp_idx_q == c);
    assign fs_p[c]      = '{data  : fs_resp_data_q,
                            id    : fs_resp_id_q,
                            write : fs_resp_write_q,
                            error : 1'b0};
  end

  fsync_shim #(
    .AggrWidth   (FSyncAggrWidth ),
    .IdWidth     (FSyncIdWidth   ),
    .LvlWidth    (FSyncLvlWidth  ),
    .fsync_req_t (fsync_req_t    ),
    .fsync_rsp_t (fsync_rsp_t    )
  ) i_fsync_shim (
    .clk_i,
    .rst_ni,
    .reg_req_i    (fs_arb_valid        ),
    .reg_we_i     (fs_arb_q.write      ),
    .reg_addr_i   (fs_arb_q.addr[3:0]  ),
    .reg_wdata_i  (fs_arb_q.data[31:0] ),
    .reg_rdata_o  (fs_rdata            ),
    // Unused: the bus is the only requester today. Kept so an in-tile
    // requester with no bus in front of it can arm a barrier later.
    .trig_valid_i (1'b0                ),
    .trig_aggr_i  ('0                  ),
    .trig_id_i    ('0                  ),
    .trig_ready_o (                    ),
    .h_req_o      (fsync_h_req_o       ),
    .h_rsp_i      (fsync_h_rsp_i       ),
    .v_req_o      (fsync_v_req_o       ),
    .v_rsp_i      (fsync_v_rsp_i       ),
    .wake_o       (fs_wake             ),
    .error_o      (                    ),
    .busy_o       (                    ),
    .lvl_o        (                    ),
    .rsp_id_o     (                    )
  );

  /*****************************
   *   RedMulE Tensor Core     *
   *****************************/
  // RedMulE on local slots [NumCoresPerTile..NumLocalPorts-1]; core 0 owns the
  // HWPE control path (MMIO [RMBaseAddr, +RMRegSize)) and the EOC wake-up.
  if (RedMulE) begin: gen_redmule

    logic redmule_evt; // end-of-computation (wakes core 0)

    // ID_WIDTH must be non-zero (VCS rejects the [ID_WIDTH-1:0] underflow); match redmule_top's 8.
    hwpe_ctrl_intf_periph #( .ID_WIDTH(8) ) redmule_rmcfg ( .clk( clk_i ) );

    rm_dreq_t  [RMMasterPorts-1:0] redmule_req;
    logic      [RMMasterPorts-1:0] redmule_req_valid;
    logic      [RMMasterPorts-1:0] redmule_req_ready;
    rm_dresp_t [RMMasterPorts-1:0] redmule_resp;
    logic      [RMMasterPorts-1:0] redmule_resp_valid;
    logic      [RMMasterPorts-1:0] redmule_resp_ready;

    localparam hci_size_parameter_t `HCI_SIZE_PARAM(tcdm) = '{
      DW:  RMDataWidth,
      AW:  AddrWidth,
      BW:  BeWidth,
      UW:  idx_width(RMOutstandingTransactions),
      IW:  idx_width(RMNumStreams),
      EW:  0,
      EHW: 0
    };
    // The engine speaks classic HCI (req/gnt + r_valid/r_ready). The TCDM side of
    // this tile is variable-latency and reorders responses; the streamer's
    // per-channel hci_core_rob (`user`) plus hci_core_mux_ooo (`id`) absorb both,
    // so `EnableReordering` is mandatory here.
    hci_core_intf #(
      .DW (RMDataWidth                         ),
      .UW (idx_width(RMOutstandingTransactions)),
      .IW (idx_width(RMNumStreams)             )
    ) tcdm ( .clk ( clk_i ) );

    // No ECC on this path (EW = EHW = 0); tie the target-driven side off.
    assign tcdm.egnt     = '0;
    assign tcdm.r_evalid = '0;
    assign tcdm.r_ecc    = '0;
    assign tcdm.r_opc    = '0;

    redmule_mm_wrap #(
      .DataW                   (RMDataWidth                         ),
      .MisalignedAccessSupport (0                                   ),
      .EnableReordering        (1'b1                                ),
      .FpFormat                (fpnew_pkg::FP16                     ),
      .Height                  (ARRAY_HEIGHT                        ),
      .Width                   (ARRAY_WIDTH                         ),
      .NumPipeRegs             (PIPE_REGS                           ),
      .`HCI_SIZE_PARAM(tcdm) (`HCI_SIZE_PARAM(tcdm))
    ) i_redmule_wrap (
      .clk_i              (clk_i                     ),
      .rst_ni             (rst_ni                    ),
      .test_mode_i        ('0                        ),
      .evt_o              (redmule_evt               ),
      .busy_o             (/* Unused */              ),
      .sync_o             (/* Unused */              ),
      .sync_i             (1'b0                      ),
      .tcdm               (tcdm                      ),
      .target             (redmule_rmcfg             )
    );

    assign wake_up = wake_up_q | {{(NumCoresPerTile-1){1'b0}}, redmule_evt} |
                     {NumCoresPerTile{fs_wake}};

    // ----------------------------------------------------------------------
    // WP2 wide source path: issue each RedMulE HCI beat as one raw wide access.
    // Own-tile accesses bypass locally; the engine's TCDM shim separates
    // same-group and inter-group traffic. Each access carries ini_addr/src_group_id,
    // and RedMulE's hci_core_rob reorders by meta_id, so out-of-order is fine.
    // No 16x32b unpack,
    // no transactions_table — the superbank gathers per request.
    // ----------------------------------------------------------------------
    `ifdef USE_WIDE_TCDM_PLANE
    localparam int unsigned RobW = idx_width(RMOutstandingTransactions);
    localparam int unsigned WidePortIdx = 0;

    // The demux speaks tcdm_wide_rm_req_t/_resp_t; the HCI interface is translated
    // into them in exactly one place below, as the narrow path unpacks into
    // redmule_req/redmule_resp before its own shim.
    tcdm_wide_rm_req_t  tcdm_wide_rm_req;
    logic               tcdm_wide_rm_req_valid, tcdm_wide_rm_req_ready;
    tcdm_wide_rm_resp_t tcdm_wide_rm_resp;
    logic               tcdm_wide_rm_resp_valid, tcdm_wide_rm_resp_ready;

    // Scramble the wide base address exactly like the per-core/RedMulE scrambler.
    mempool_addr_scrambler #(
      .AddrWidth               (AddrWidth        ),
      .ByteOffset              (ByteOffset       ),
      .NumTiles                (NumTiles         ),
      .NumTilesPerDma          (NumTilesPerDma   ),
      .NumBanksPerTile         (NumBanksPerTile  ),
      .Bypass                  (0                ),
      .SeqMemSizePerTile       (SeqMemSizePerTile),
      .TCDMBaseAddr            (TCDMBaseAddr     ),
      .TCDMMask                (TCDMMask         ),
      .EnableSeqInterleaveSwap (1'b1             ),
      .EnableTileIdRemap       (TileIdRemap      ),
      .TCDMSize                (TCDMSize         ),
      .NumTilesPerGroup        (NumTilesPerGroup ),
      .NumX                    (NumX             ),
      .NumY                    (NumY             ),
      .EnableVirtualCluster           (VirtualClusterRmWide      ),
      .VcRegionEndChk          (VcRegionEnd      )
    ) i_rm_wide_scrambler (
      .address_i  (tcdm.add             ),
      .group_id_i (group_id             ),
      .vc_hit_o   (vc_hit_wide_port     ),
      .vc_level_o (vc_level_wide_port   ),
      .address_o  (tcdm_wide_rm_req.addr)
    );

    // ----- RedMulE HCI <-> wide demux, the ONLY translation point -----
    // HCI convention: wen==1 is a read, so write is its inverse. meta_id
    // packs the stream id + ROB id exactly as the narrow unpack does, so
    // RedMulE's hci_core_rob still matches responses by id.
    assign tcdm_wide_rm_req.write                  = ~tcdm.wen;
    assign tcdm_wide_rm_req.amo                    = '0;
    assign tcdm_wide_rm_req.data                   = tcdm.data;
    assign tcdm_wide_rm_req.strb                   = tcdm.be;
    assign tcdm_wide_rm_req.id[MetaIdWidth-1:RobW] = tcdm.id;
    assign tcdm_wide_rm_req.id[RobW-1:0]           = tcdm.user;
    assign tcdm_wide_rm_req_valid                  = tcdm.req;
    assign tcdm.gnt                                = tcdm_wide_rm_req_ready;

    assign tcdm.r_data             = tcdm_wide_rm_resp.data;
    assign tcdm.r_id               = tcdm_wide_rm_resp.id[MetaIdWidth-1:RobW];
    assign tcdm.r_user             = tcdm_wide_rm_resp.id[RobW-1:0];
    assign tcdm.r_valid            = tcdm_wide_rm_resp_valid;
    assign tcdm_wide_rm_resp_ready = tcdm.r_ready;

    // The demux splits own-tile from every other tile; the same-group /
    // other-group decision is made by i_mempool_tile_remote_req_router, which
    // produces tgt_addr / tgt_group_id / tgt_sel for the wide port just as it
    // does for the narrow cores.
    address_map_t [NumWideShimPorts-1:0] tcdm_wide_mask_map;

    // Own-tile request: {row, bank}, group and tile fields consumed -- the same
    // layout the narrow local_req_interco_payload uses.
    assign local_wide_req_interco_payload[WidePortIdx].tgt_addr = tile_addr_t'({
        local_wide_req_interco_addr_int[WidePortIdx][ByteOffset +
            idx_width(NumBanksPerTile) + $clog2(NumTiles) +: TCDMAddrMemWidth],
        local_wide_req_interco_addr_int[WidePortIdx][
            ByteOffset +: idx_width(NumBanksPerTile)]});

    assign remote_wide_req_interco[WidePortIdx].tgt_addr = remote_wide_req_tgt_addr[WidePortIdx];

`ifndef SYNTHESIS
    // GATE item 3 (roadmap §T2.3), wide half. Same clocked check as the
    // narrow ports; it lives here because WidePortIdx is scoped to this
    // block. Without it a RedMulE using the wide plane could leave its
    // rectangle on exactly the configs where the wide plane is enabled.
    if (VirtualCluster && (NumGroups > 1)) begin : gen_vc_conf_wide
      always_ff @(posedge clk_i) begin
        if (rst_ni && vc_hit_wide_port &&
            remote_wide_req_interco_valid[WidePortIdx]) begin
          assert (vc_in_rect(group_id,
                             remote_wide_req_tgt_group_id[WidePortIdx],
                             vc_level_wide_port))
            else $fatal(1,
                "VC confinement escape, WIDE port level %0d: group %0d reached %0d",
                vc_level_wide_port, group_id,
                remote_wide_req_tgt_group_id[WidePortIdx]);
        end
      end
    end
`endif

    // Same split as the narrow mask_map: LOCAL is THIS TILE (the tile-id field
    // pinned to tile_id_i), EXTERNAL is every other TCDM address. The
    // same-group / other-group decision happens after the shim, on the remote
    // stream, exactly as the narrow remote req router does it.
    assign tcdm_wide_mask_map = '{
      '{slave_idx: TCDM_WIDE_EXTERNAL,
        mask : TCDMMask,
        value: TCDMBaseAddr},
      '{slave_idx: TCDM_WIDE_LOCAL,
        mask : TCDMMask | ({idx_width(NumTiles){1'b1}} << (ByteOffset + $clog2(NumBanksPerTile))),
        value: TCDMBaseAddr | (tile_id_i << (ByteOffset + $clog2(NumBanksPerTile)))}
    };

    // The request/response structs already are the demux payloads: split the
    // request by address (own tile vs every other TCDM address) and merge the
    // two response streams back.
    tcdm_wide_rm_req_t  [NumWideShimPorts-1:0] rm_wide_req_split;
    tcdm_wide_rm_resp_t [NumWideShimPorts-1:0] rm_wide_resp_merge;

    snitch_addr_demux #(
      .NrOutput    (NumWideShimPorts   ),
      .AddressWidth(AddrWidth          ),
      .NumRules    (NumWideShimPorts   ),
      .req_t       (tcdm_wide_rm_req_t ),
      .resp_t      (tcdm_wide_rm_resp_t)
    ) i_rm_wide_demux (
      .clk_i         (clk_i                                                                                  ),
      .rst_ni        (rst_ni                                                                                 ),
      .req_addr_i    (tcdm_wide_rm_req.addr                                                                  ),
      .req_payload_i (tcdm_wide_rm_req                                                                       ),
      .req_valid_i   (tcdm_wide_rm_req_valid                                                                 ),
      .req_ready_o   (tcdm_wide_rm_req_ready                                                                 ),
      .resp_payload_o(tcdm_wide_rm_resp                                                                      ),
      .resp_valid_o  (tcdm_wide_rm_resp_valid                                                                ),
      .resp_ready_i  (tcdm_wide_rm_resp_ready                                                                ),
      // {local, remote}: index TCDM_WIDE_LOCAL = own tile, TCDM_WIDE_EXTERNAL = the rest.
      .req_payload_o (rm_wide_req_split                                                                      ),
      .req_valid_o   ({local_wide_req_interco_valid[WidePortIdx],  remote_wide_req_interco_valid[WidePortIdx]} ),
      .req_ready_i   ({local_wide_req_interco_ready[WidePortIdx],  remote_wide_req_interco_ready[WidePortIdx]} ),
      .resp_payload_i(rm_wide_resp_merge                                                                     ),
      .resp_valid_i  ({local_wide_resp_interco_valid[WidePortIdx], remote_wide_resp_interco_valid[WidePortIdx]}),
      .resp_ready_o  ({local_wide_resp_interco_ready[WidePortIdx], remote_wide_resp_interco_ready[WidePortIdx]}),
      .address_map_i (tcdm_wide_mask_map                                                                     )
    );

    assign local_wide_req_interco_addr_int[WidePortIdx]              = rm_wide_req_split[TCDM_WIDE_LOCAL].addr;
    assign local_wide_req_interco_payload[WidePortIdx].wen           = rm_wide_req_split[TCDM_WIDE_LOCAL].write;
    assign local_wide_req_interco_payload[WidePortIdx].wdata.data    = rm_wide_req_split[TCDM_WIDE_LOCAL].data;
    assign local_wide_req_interco_payload[WidePortIdx].wdata.amo     = rm_wide_req_split[TCDM_WIDE_LOCAL].amo;
    assign local_wide_req_interco_payload[WidePortIdx].wdata.meta_id = rm_wide_req_split[TCDM_WIDE_LOCAL].id;
    assign local_wide_req_interco_payload[WidePortIdx].be            = rm_wide_req_split[TCDM_WIDE_LOCAL].strb;

    assign remote_wide_req_interco_addr_int[WidePortIdx]             = rm_wide_req_split[TCDM_WIDE_EXTERNAL].addr;
    assign remote_wide_req_interco[WidePortIdx].wen                  = rm_wide_req_split[TCDM_WIDE_EXTERNAL].write;
    assign remote_wide_req_interco[WidePortIdx].wdata.data           = rm_wide_req_split[TCDM_WIDE_EXTERNAL].data;
    assign remote_wide_req_interco[WidePortIdx].wdata.amo            = rm_wide_req_split[TCDM_WIDE_EXTERNAL].amo;
    assign remote_wide_req_interco[WidePortIdx].wdata.meta_id        = rm_wide_req_split[TCDM_WIDE_EXTERNAL].id;
    assign remote_wide_req_interco[WidePortIdx].be                   = rm_wide_req_split[TCDM_WIDE_EXTERNAL].strb;

    assign rm_wide_resp_merge[TCDM_WIDE_LOCAL] = '{
      data:  local_wide_resp_interco_payload[WidePortIdx].rdata.data,
      id:    local_wide_resp_interco_payload[WidePortIdx].rdata.meta_id,
      write: 1'b0,
      error: 1'b0
    };
    assign rm_wide_resp_merge[TCDM_WIDE_EXTERNAL] = '{
      data:  remote_wide_resp_interco[WidePortIdx].rdata.data,
      id:    remote_wide_resp_interco[WidePortIdx].rdata.meta_id,
      write: 1'b0,
      error: 1'b0
    };

    // Own-tile ingress carries the same self-routing fields the group slave
    // ports do, so the bank interconnect returns its response the same way.
    assign local_wide_req_interco_payload[WidePortIdx].wdata.core_id = '0;
    assign local_wide_req_interco_payload[WidePortIdx].ini_addr      = tcdm_wide_tile_id;
    assign local_wide_req_interco_payload[WidePortIdx].src_group_id  = group_id;
    assign remote_wide_req_interco[WidePortIdx].wdata.core_id = '0;
    assign remote_wide_req_interco[WidePortIdx].tgt_group_id = remote_wide_req_tgt_group_id[WidePortIdx];
    assign remote_wide_req_interco[WidePortIdx].src_tile_id   = tcdm_wide_tile_id;

    assign remote_wide_req_interco_tgt_sel[WidePortIdx] =
        remote_wide_req_tgt_sel[WidePortIdx];

    // No narrow interco slots for RedMulE in the wide plane (RMNarrowPorts = 0).

`ifndef SYNTHESIS
    // WP2 assumes WIDE_W-aligned RedMulE accesses: one access lands in one
    // superbank and never crosses a tile.
    localparam int unsigned RmWideAlignW = $clog2(WideTcdmDataWidth/DataWidth);
    always_ff @(posedge clk_i) begin
      if (rst_ni && tcdm_wide_rm_req_valid && tcdm_wide_rm_req_ready &&
          (tcdm_wide_rm_req.addr[ByteOffset +: RmWideAlignW] != '0))
        $warning("[mempool_tile] RedMulE wide access misaligned (low bank bits=%0d): crosses a superbank/tile.",
                 tcdm_wide_rm_req.addr[ByteOffset +: RmWideAlignW]);
    end
`endif

    `else
    // Unpack the wide HCI tcdm port into RMMasterPorts narrow ports.
    for (genvar p = 0; p < RMMasterPorts; p++) begin: gen_redmule_tcdm_unpack
      assign redmule_req[p].addr  = tcdm.add + p*4;
      assign redmule_req[p].write = ~tcdm.wen;
      assign redmule_req[p].strb  = tcdm.be[(p+1)*4-1:p*4];
      assign redmule_req[p].data  = tcdm.data[(p+1)*DataWidth-1:p*DataWidth];
      assign redmule_req[p].amo   = '0;
      assign redmule_req[p].id[mempool_pkg::MetaIdWidth-1:idx_width(RMOutstandingTransactions)] = tcdm.id;
      assign redmule_req[p].id[idx_width(RMOutstandingTransactions)-1:0] = tcdm.user;
      assign tcdm.r_data[(p+1)*DataWidth-1:p*DataWidth] = redmule_resp[p].data;
    end : gen_redmule_tcdm_unpack
    assign redmule_req_valid  = {RMMasterPorts{tcdm.req}};
    assign tcdm.gnt           = &(redmule_req_ready);
    assign tcdm.r_valid       = &(redmule_resp_valid);
    assign redmule_resp_ready = {RMMasterPorts{tcdm.r_ready}};
    assign tcdm.r_id          = redmule_resp[0].id[mempool_pkg::MetaIdWidth-1:idx_width(RMOutstandingTransactions)];
    assign tcdm.r_user        = redmule_resp[0].id[idx_width(RMOutstandingTransactions)-1:0];

    /************************************
     *  Handshake glue (stream regs)    *
     ************************************/
    // Stream regs + the handshake gate below make the wide HCI port advance atomically.
    rm_dreq_t  [RMMasterPorts-1:0] redmule_req_q;
    logic      [RMMasterPorts-1:0] redmule_req_qvalid;
    logic      [RMMasterPorts-1:0] redmule_req_qready;
    rm_dresp_t [RMMasterPorts-1:0] redmule_resp_q;
    logic      [RMMasterPorts-1:0] redmule_resp_qvalid;
    logic      [RMMasterPorts-1:0] redmule_resp_qready;

    rm_dreq_t  [RMMasterPorts-1:0] redmule_tcdm_req;
    logic      [RMMasterPorts-1:0] redmule_tcdm_req_valid;
    logic      [RMMasterPorts-1:0] redmule_tcdm_req_ready;
    rm_dresp_t [RMMasterPorts-1:0] redmule_tcdm_resp;
    logic      [RMMasterPorts-1:0] redmule_tcdm_resp_valid;
    logic      [RMMasterPorts-1:0] redmule_tcdm_resp_ready;

    logic      [RMMasterPorts-1:0] redmule_handshake_p;
    logic      [RMMasterPorts-1:0] redmule_handshake_q;

    for (genvar p = 0; p < RMMasterPorts; p++) begin: gen_redmule_regs
      stream_register #(.T(rm_dreq_t)) i_redmule_req_register (
        .clk_i      ( clk_i                 ),
        .rst_ni     ( rst_ni                ),
        .clr_i      ( 1'b0                  ),
        .testmode_i ( 1'b0                  ),
        .valid_i    ( redmule_req_valid[p]  ),
        .ready_o    ( redmule_req_ready[p]  ),
        .data_i     ( redmule_req[p]        ),
        .valid_o    ( redmule_req_qvalid[p] ),
        .ready_i    ( redmule_req_qready[p] ),
        .data_o     ( redmule_req_q[p]      )
      );
      stream_register #(.T(rm_dresp_t)) i_redmule_resp_register (
        .clk_i      ( clk_i                  ),
        .rst_ni     ( rst_ni                 ),
        .clr_i      ( 1'b0                   ),
        .testmode_i ( 1'b0                   ),
        .valid_o    ( redmule_resp_valid[p]  ),
        .ready_i    ( redmule_resp_ready[p]  ),
        .data_o     ( redmule_resp[p]        ),
        .valid_i    ( redmule_resp_qvalid[p] ),
        .ready_o    ( redmule_resp_qready[p] ),
        .data_i     ( redmule_resp_q[p]      )
      );
    end : gen_redmule_regs

    // Per-port "already-fired" flag; clears once every port has fired this beat.
    assign redmule_handshake_p = (&redmule_req_qready) ? '0 : redmule_req_qready;
    `FF(redmule_handshake_q, redmule_handshake_p, '0, clk_i, rst_ni);
    assign redmule_tcdm_req       = redmule_req_q;
    assign redmule_tcdm_req_valid = ~redmule_handshake_q & redmule_req_qvalid;
    assign redmule_req_qready     = redmule_handshake_q
                                  | (redmule_tcdm_req_valid & redmule_tcdm_req_ready);

    // redmule_top expects all per-port responses in one beat: realign via the txn table.
    transactions_table #(
      .NumPorts       (RMMasterPorts                                  ),
      .NumTransactions((RMNumStreams - 1) * RMOutstandingTransactions ),
      .resp_t         (rm_dresp_t                                     )
    ) i_redmule_transactions_table (
      .clk_i         (clk_i                   ),
      .rst_ni        (rst_ni                  ),
      .resp_payload_i(redmule_tcdm_resp       ),
      .resp_valid_i  (redmule_tcdm_resp_valid ),
      .resp_ready_o  (redmule_tcdm_resp_ready ),
      .resp_payload_o(redmule_resp_q          ),
      .resp_valid_o  (redmule_resp_qvalid     ),
      .resp_ready_i  (redmule_resp_qready     )
    );

    /************************************
     *  Per-port TCDM shim chain        *
     ************************************/
    // RedMulE master ports occupy the local-port slots after the core ports:
    // slot c = p + NumSoCPorts. NumSoCPorts already folds in the Spatz vector
    // ports (NumCorePortsPerTile), so this places RedMulE after
    // them without collision (== NumCoresPerTile for scalar builds). Scramble →
    // shim → local/remote, exactly as gen_core_mux.
    addr_t [RMMasterPorts-1:0] redmule_addr_scrambled;

    for (genvar p = 0; p < RMMasterPorts; p++) begin: gen_redmule_mux
      localparam int unsigned c = p + NumSoCPorts;

      mempool_addr_scrambler #(
        .AddrWidth               (AddrWidth        ),
        .ByteOffset              (ByteOffset       ),
        .NumTiles                (NumTiles         ),
        .NumTilesPerDma          (NumTilesPerDma   ),
        .NumBanksPerTile         (NumBanksPerTile  ),
        .Bypass                  (0                ),
        .SeqMemSizePerTile       (SeqMemSizePerTile),
        .TCDMBaseAddr            (TCDMBaseAddr     ),
        .TCDMMask                (TCDMMask         ),
        .EnableSeqInterleaveSwap (1'b1             ),
        .EnableTileIdRemap       (TileIdRemap      ),
        .TCDMSize                (TCDMSize         ),
        .NumTilesPerGroup        (NumTilesPerGroup ),
        .NumX                    (NumX             ),
        .NumY                    (NumY             ),
        .EnableVirtualCluster           (VirtualClusterRmNarrow      ),
        .VcRegionEndChk          (VcRegionEnd      )
      ) i_addr_scrambler (
        .address_i  (redmule_tcdm_req[p].addr  ),
        .group_id_i (group_id                  ),
        .vc_hit_o   (vc_hit_port[c]            ),
        .vc_level_o (vc_level_port[c]          ),
        .address_o  (redmule_addr_scrambled[p] )
      );

      // Same shim as the snitch path; req_t/resp_t carry redmule's cluster-width meta_id.
      tcdm_shim #(
        .AddrWidth           (AddrWidth                         ),
        .DataWidth           (DataWidth                         ),
        .MaxOutStandingTrans (snitch_pkg::NumIntOutstandingLoads),
        .NrTCDM              (2                                 ),
        .NrSoC               (1                                 ),
        .EnableSoC           (1'b0                              ),
        .NumRules            (2                                 ),
        .req_t               (rm_dreq_t                         ),
        .resp_t              (rm_dresp_t                        )
      ) i_tcdm_shim (
        .clk_i              (clk_i                                                                                ),
        .rst_ni             (rst_ni                                                                               ),
        // {local, remote}: idx 1 = local, idx 0 = remote (as gen_core_mux).
        .tcdm_req_valid_o   ({local_req_interco_valid[c],                 remote_req_interco_valid[c]}            ),
        .tcdm_req_tgt_addr_o({local_req_interco_addr_int[c],              remote_req_interco_addr_int[c]}         ),
        .tcdm_req_wen_o     ({local_req_interco_payload[c].wen,           remote_req_interco[c].wen}              ),
        .tcdm_req_wdata_o   ({local_req_interco_payload[c].wdata.data,    remote_req_interco[c].wdata.data}       ),
        .tcdm_req_amo_o     ({local_req_interco_payload[c].wdata.amo,     remote_req_interco[c].wdata.amo}        ),
        .tcdm_req_id_o      ({local_req_interco_payload[c].wdata.meta_id, remote_req_interco[c].wdata.meta_id}    ),
        .tcdm_req_be_o      ({local_req_interco_payload[c].be,            remote_req_interco[c].be}               ),
        .tcdm_req_ready_i   ({local_req_interco_ready[c],                 remote_req_interco_ready[c]}            ),
        .tcdm_resp_valid_i  ({local_resp_interco_valid[c],                remote_resp_interco_valid[c]}           ),
        .tcdm_resp_ready_o  ({local_resp_interco_ready[c],                remote_resp_interco_ready[c]}           ),
        .tcdm_resp_rdata_i  ({local_resp_interco_payload[c].rdata.data,   remote_resp_interco[c].rdata.data}      ),
        .tcdm_resp_id_i     ({local_resp_interco_payload[c].rdata.meta_id,remote_resp_interco[c].rdata.meta_id}   ),
        // SoC routing is disabled for RedMulE.
        .soc_qaddr_o        (/* unused */                                                                         ),
        .soc_qwrite_o       (/* unused */                                                                         ),
        .soc_qamo_o         (/* unused */                                                                         ),
        .soc_qdata_o        (/* unused */                                                                         ),
        .soc_qstrb_o        (/* unused */                                                                         ),
        .soc_qvalid_o       (/* unused */                                                                         ),
        .soc_qready_i       ('0                                                                                   ),
        .soc_pdata_i        ('0                                                                                   ),
        .soc_pwrite_i       ('0                                                                                   ),
        .soc_perror_i       ('0                                                                                   ),
        .soc_pvalid_i       ('0                                                                                   ),
        .soc_pready_o       (/* unused */                                                                         ),
        .data_qaddr_i       (redmule_addr_scrambled[p]                                                            ),
        .data_qwrite_i      (redmule_tcdm_req[p].write                                                            ),
        .data_qamo_i        (redmule_tcdm_req[p].amo                                                              ),
        .data_qdata_i       (redmule_tcdm_req[p].data                                                             ),
        .data_qstrb_i       (redmule_tcdm_req[p].strb                                                             ),
        .data_qid_i         (redmule_tcdm_req[p].id                                                               ),
        .data_qvalid_i      (redmule_tcdm_req_valid[p]                                                            ),
        .data_qready_o      (redmule_tcdm_req_ready[p]                                                            ),
        .data_pdata_o       (redmule_tcdm_resp[p].data                                                            ),
        .data_perror_o      (redmule_tcdm_resp[p].error                                                           ),
        .data_pid_o         (redmule_tcdm_resp[p].id                                                              ),
        .data_pvalid_o      (redmule_tcdm_resp_valid[p]                                                           ),
        .data_pready_i      (redmule_tcdm_resp_ready[p]                                                           ),
        .address_map_i      (tcdm_mask_map                                                                        )
      );
      assign redmule_tcdm_resp[p].write = 1'b0;

      // Slice post-shim addresses into local/remote layout (same bit math as gen_core_mux).
      assign local_req_interco_payload[c].tgt_addr =
        tcdm_addr_t'({local_req_interco_addr_int[c][ByteOffset + idx_width(NumBanksPerTile)
                                                    + $clog2(NumTiles) +: TCDMAddrMemWidth],
                      local_req_interco_addr_int[c][ByteOffset +: idx_width(NumBanksPerTile)]});

      // Remote routing metadata comes from i_mempool_tile_remote_req_router.
      assign remote_req_interco[c].tgt_addr = remote_req_tgt_addr[c];

      // Local-side don't-cares: the xbar tracks the initiator via idx_o.
      assign local_req_interco_payload[c].wdata.core_id = '0;
      assign local_req_interco_payload[c].ini_addr      = '0;
      assign local_req_interco_payload[c].src_group_id  = '0;

      // Encode slot index in core_id so the dest tile's resp xbar routes back (sel = rdata.core_id).
      assign remote_req_interco[c].wdata.core_id = tile_core_id_t'(c);
      assign remote_req_interco[c].tgt_group_id = remote_req_tgt_group_id[c];

      assign remote_req_interco_wen[c]     = remote_req_interco[c].wen;
      assign remote_req_interco_amoen[c]   = |remote_req_interco[c].wdata.amo;
      assign remote_req_interco_tgt_sel[c] = remote_req_tgt_sel[c];
    end : gen_redmule_mux
    `endif

    /************************************
     *  HWPE control demux on core 0    *
     ************************************/
    // Route core 0's SoC accesses to [RMBaseAddr, +RMRegSize) into redmule_rmcfg; rest to soc_mux.
    snitch_pkg::dreq_t  snitch_rmcfg_q;
    logic               snitch_rmcfg_qvalid;
    logic               snitch_rmcfg_qready;
    snitch_pkg::dresp_t snitch_rmcfg_p;
    logic               snitch_rmcfg_pvalid;
    logic               snitch_rmcfg_pready;

    address_map_t [1:0] rmcfg_map;
    assign rmcfg_map = '{
      // Lowest priority (port 1): catch-all → goes to soc_mux.
      '{slave_idx: 32'd1, mask: '0,     value: '0         },
      // Highest priority (port 0): RedMulE CSR region → goes to redmule_rmcfg.
      '{slave_idx: 32'd0, mask: RMMask, value: RMBaseAddr }
    };

    snitch_addr_demux #(
      .NrOutput     (2                   ),
      .AddressWidth (AddrWidth           ),
      .NumRules     (2                   ),
      .req_t        (snitch_pkg::dreq_t  ),
      .resp_t       (snitch_pkg::dresp_t )
    ) i_snitch_addr_demux (
      .clk_i,
      .rst_ni,
      .req_addr_i     (soc_nofs_q[0].addr                              ),
      .req_payload_i  (soc_nofs_q[0]                                   ),
      .req_valid_i    (soc_nofs_qvalid[0]                              ),
      .req_ready_o    (soc_nofs_qready[0]                              ),
      .resp_payload_o (soc_nofs_p[0]                                   ),
      .resp_valid_o   (soc_nofs_pvalid[0]                              ),
      .resp_ready_i   (soc_nofs_pready[0]                              ),
      // {port1, port0} = {soc_mux, rmcfg}
      .req_payload_o  ({soc_mux_q[0],      snitch_rmcfg_q            } ),
      .req_valid_o    ({soc_mux_qvalid[0], snitch_rmcfg_qvalid       } ),
      .req_ready_i    ({soc_mux_qready[0], snitch_rmcfg_qready       } ),
      .resp_payload_i ({soc_mux_p[0],      snitch_rmcfg_p            } ),
      .resp_valid_i   ({soc_mux_pvalid[0], snitch_rmcfg_pvalid       } ),
      .resp_ready_o   ({soc_mux_pready[0], snitch_rmcfg_pready       } ),
      .address_map_i  (rmcfg_map                                       )
    );

    // Pass every other SoC port (1..NumSoCPorts-1) straight through to the SoC
    // mux; only core 0's scalar port (slot 0) is split off above for the RedMulE
    // config region. For Spatz this also covers the per-core vector ports, whose
    // SoC channels stay idle (the vector LSU only targets TCDM) but must be driven.
    for (genvar c = 1; c < NumSoCPorts; c++) begin: gen_redmule_soc_bypass
      assign soc_mux_q[c]       = soc_nofs_q[c];
      assign soc_mux_qvalid[c]  = soc_nofs_qvalid[c];
      assign soc_nofs_qready[c] = soc_mux_qready[c];
      assign soc_nofs_p[c]      = soc_mux_p[c];
      assign soc_nofs_pvalid[c] = soc_mux_pvalid[c];
      assign soc_mux_pready[c]  = soc_nofs_pready[c];
    end : gen_redmule_soc_bypass

    snitch_hwpe_cfg_adapter #(
      .req_t  (snitch_pkg::dreq_t ),
      .resp_t (snitch_pkg::dresp_t)
    ) i_redmule_cfg_adapter (
      .clk_i,
      .rst_ni,
      .req_i        (snitch_rmcfg_q     ),
      .req_valid_i  (snitch_rmcfg_qvalid),
      .req_ready_o  (snitch_rmcfg_qready),
      .resp_o       (snitch_rmcfg_p     ),
      .resp_valid_o (snitch_rmcfg_pvalid),
      .resp_ready_i (snitch_rmcfg_pready),
      .periph       (redmule_rmcfg      )
    );

  end else begin: gen_redmule_tieoff

    // The barrier wake releases EVERY core of the tile, which is what makes one
    // arrival per tile sufficient.
    assign wake_up = wake_up_q | {NumCoresPerTile{fs_wake}};

    // No RedMulE here: this tile never sources a wide access from the RedMulE SLOT.
    // It still accepts raw wide slave requests through the module-scope B2 path, and
    // another client (the Spatz burst adapter) may still drive its own slot -- so tie
    // off this slot's crossbar inputs only, never the shared crossbar outputs.
    `ifdef USE_WIDE_TCDM_PLANE
    if (NumWideRmPorts > 0) begin : gen_rm_wide_tieoff
    assign local_wide_req_interco_payload[WideRmPortIdx]   = '0;
    assign local_wide_req_interco_valid[WideRmPortIdx]     = 1'b0;
    assign local_wide_req_interco_addr_int[WideRmPortIdx]  = '0;
    assign local_wide_resp_interco_ready[WideRmPortIdx]    = 1'b1;
    assign remote_wide_req_interco[WideRmPortIdx]          = '0;
    assign remote_wide_req_interco_valid[WideRmPortIdx]    = 1'b0;
    assign remote_wide_req_interco_addr_int[WideRmPortIdx] = '0;
    assign remote_wide_req_interco_tgt_sel[WideRmPortIdx]  = '0;
    assign remote_wide_resp_interco_ready[WideRmPortIdx]   = 1'b1;
    end : gen_rm_wide_tieoff
    `endif

    for (genvar c = 0; c < NumSoCPorts; c++) begin: gen_redmule_soc_passthrough
      assign soc_mux_q[c]       = soc_nofs_q[c];
      assign soc_mux_qvalid[c]  = soc_nofs_qvalid[c];
      assign soc_nofs_qready[c] = soc_mux_qready[c];
      assign soc_nofs_p[c]      = soc_mux_p[c];
      assign soc_nofs_pvalid[c] = soc_mux_pvalid[c];
      assign soc_mux_pready[c]  = soc_nofs_pready[c];
    end : gen_redmule_soc_passthrough

  end : gen_redmule_tieoff

  /******************
   *   Assertions   *
   ******************/

  // Check invariants.
  if (BootAddr[1:0] != 2'b00)
    $fatal(1, "[mempool_tile] Boot address should be aligned in a 4-byte boundary.");

  if (NumCoresPerTile != 2**$clog2(NumCoresPerTile))
    $fatal(1, "[mempool_tile] The number of cores per tile must be a power of two.");

  if (NumCores != unsigned'(2**$clog2(NumCores)))
    $fatal(1, "[mempool_tile] The number of cores must be a power of two.");

  if (NumBanksPerTile < 1)
    $fatal(1, "[mempool_tile] The number of banks per tile must be larger than one");

  if (NumCaches != 1)
    $error("NumCaches > 1 is not supported!");

  if (DataWidth > AxiDataWidth)
    $error("AxiDataWidth needs to be larger than DataWidth!");

  `ifdef USE_WIDE_TCDM_PLANE
  if (NumWideLocalPorts < 1)
    $fatal(1, "[mempool_tile] The wide plane requires at least one local engine port.");

  // RedMulE drives the plane at exactly its TCDM port width, so the two must agree
  // whenever it is a client. Without RedMulE the plane is sized by the other client:
  // a Spatz burst is MaxBurstWords 32-bit words, one superbank by construction.
  if ((NumWideRmPorts > 0) && (WideTcdmDataWidth != RMDataWidth))
    $fatal(1, "[mempool_tile] Wide width %0d must equal RedMulE TCDM width %0d.",
           WideTcdmDataWidth, RMDataWidth);

  if (SpatzWideBurst && (WideTcdmDataWidth != spatz_pkg::MaxBurstWords * DataWidth))
    $fatal(1, "[mempool_tile] Wide width %0d must equal one Spatz burst (%0d x %0d b).",
           WideTcdmDataWidth, spatz_pkg::MaxBurstWords, DataWidth);
  `endif

  // Wide TCDM plane (WP2): the raw master side is driven per tile by gen_redmule
  // or gen_redmule_tieoff.

endmodule : mempool_tile
