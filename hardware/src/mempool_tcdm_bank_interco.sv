// Copyright 2025 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

`include "mempool/mempool.svh"

// Tile-internal TCDM bank-side interconnect, organised by superbank: the
// narrow req/resp crossbars, one wide-word handler per ingress, and the
// per-superbank wide-over-narrow arbitration. Companion module:
// mempool_tcdm_wide_superbank.sv (the per-ingress split/join wide datapath).
//
// A superbank is NumBanksPerTile/NumSuperbanks banks (the DMA width). Every
// wide-word ingress -- the DMA (slv_dma_*, always present) and each WP2 wide
// plane port (slv_wide_*, only with USE_WIDE_TCDM_PLANE) -- lands on the same
// superbanks: a wide word wider than a superbank is split into superbank-wide
// parts by its handler. `wide` names the plane, never the DMA, even though both
// move a multi-bank word. Per superbank the ingresses share the banks through
// one arbiter (wide plane > DMA > narrow), granted SUPERBANK-ATOMICALLY so
// ingress paths cannot interleave partial accesses into the per-bank response
// queues (see gen_sb_arb).
//
// The narrow crossbar is split the same way: each narrow master first picks
// its superbank (bank_id_hi), then a per-superbank crossbar picks the bank.
// Per bank the arbitration is exactly the flat crossbar's; only the response
// merge per master is two-level. The bank-id remap rewrites bank_id_lo only.
module mempool_tcdm_bank_interco
  import mempool_pkg::*;
  import cf_math_pkg::idx_width;
#(
  parameter int unsigned NumNarrowReq      = 8,
  parameter int unsigned NumNarrowResp     = 8,
  parameter int unsigned NumBanksPerTile   = 16,
  parameter int unsigned NumSuperbanks     = 1,
  parameter int unsigned NarrowDataWidth   = 32,
  parameter int unsigned ByteOffset        = 2,
  parameter bit          SpmBankIdRemap    = 0,
  parameter type narrow_req_t  = tcdm_slave_req_t,
  parameter type narrow_resp_t = tcdm_slave_resp_t,
  parameter type group_id_t    = mempool_pkg::group_id_t,
  // Must be wide enough to address NumNarrowResp ports (logic [idx_width(NumNarrowResp)-1:0]).
  parameter type resp_idx_t    = logic [3:0],
  // DMA ingress. The `wide` qualifier stays on these because mempool_pkg already
  // owns DmaDataWidth and dma_req_t (the DMA frontend descriptor, unrelated).
  parameter int unsigned DmaWideDataWidth = 512,
  parameter type dma_wide_req_t  = tcdm_dma_req_t,
  parameter type dma_wide_resp_t = tcdm_dma_resp_t,
  // WP2 wide TCDM plane ingress. Defaults keep it inert when unused; a WP2 build
  // overrides these with WIDE_W and the wide req/resp types.
  parameter int unsigned NumWidePorts  = 1,
  parameter int unsigned WideDataWidth = 512,
  parameter type wide_req_t  = logic,
  parameter type wide_resp_t = logic,
  // derived
  localparam int unsigned NumBanksPerSB    = NumBanksPerTile / NumSuperbanks,
  localparam int unsigned BankOffsetBits   = $clog2(NumBanksPerTile),
  localparam int unsigned BankIdLoBits     = $clog2(NumBanksPerSB),
  localparam int unsigned NarrowBeWidth    = NarrowDataWidth / 8,
  // Bank-id remap offset width (== bank_id_lo width). NumBanksPerSB >= 2 is
  // required (asserted below); a single-bank superbank would zero-width the slices.
  localparam int unsigned BankRemapShiftAmt = BankIdLoBits,
  // Outstanding wide requests per superbank, per ingress.
  localparam int unsigned RotFifoDepth      = 2,
  // Per-bank class tags in flight. Size for every wide ingress to reach its
  // per-superbank outstanding-request limit without this queue becoming a
  // lower shared limit.
  localparam int unsigned ClassFifoDepth =
      (NumWidePorts + 1) * RotFifoDepth,
  localparam int unsigned SBSelBits = (NumSuperbanks > 1) ? $clog2(NumSuperbanks) : 1
) (
  input  logic                                          clk_i,
  input  logic                                          rst_ni,
  input  group_id_t                                     group_id_i,

  // ----- Narrow request inputs -----
  input  narrow_req_t  [NumNarrowReq-1:0]               slv_narrow_req_i,
  input  logic         [NumNarrowReq-1:0]               slv_narrow_req_valid_i,
  output logic         [NumNarrowReq-1:0]               slv_narrow_req_ready_o,

  // ----- Narrow response outputs -----
  output narrow_resp_t [NumNarrowResp-1:0]              slv_narrow_resp_o,
  output logic         [NumNarrowResp-1:0]              slv_narrow_resp_valid_o,
  input  logic         [NumNarrowResp-1:0]              slv_narrow_resp_ready_i,

  // ----- DMA request input (single, demuxed internally) -----
  input  dma_wide_req_t                                 slv_dma_req_i,
  input  logic                                          slv_dma_req_valid_i,
  output logic                                          slv_dma_req_ready_o,

  // ----- DMA response output (single, muxed internally) -----
  output dma_wide_resp_t                                slv_dma_resp_o,
  output logic                                          slv_dma_resp_valid_o,
  input  logic                                          slv_dma_resp_ready_i,

  // ----- Wide TCDM plane request/response ports (WP2 RedMulE) -----
  `ifdef USE_WIDE_TCDM_PLANE
  input  wide_req_t    [NumWidePorts-1:0]               slv_wide_req_i,
  input  logic         [NumWidePorts-1:0]               slv_wide_req_valid_i,
  output logic         [NumWidePorts-1:0]               slv_wide_req_ready_o,
  output wide_resp_t   [NumWidePorts-1:0]               slv_wide_resp_o,
  output logic         [NumWidePorts-1:0]               slv_wide_resp_valid_o,
  input  logic         [NumWidePorts-1:0]               slv_wide_resp_ready_i,
  `endif

  // ----- Bank-side request ports (per bank) -----
  output narrow_req_t  [NumBanksPerTile-1:0]            mst_req_o,
  output logic         [NumBanksPerTile-1:0]            mst_req_valid_o,
  input  logic         [NumBanksPerTile-1:0]            mst_req_ready_i,
  output logic         [NumBanksPerTile-1:0]            mst_req_wide_o,
  output resp_idx_t    [NumBanksPerTile-1:0]            mst_req_ini_addr_o,

  // ----- Bank-side response ports (per bank) -----
  input  narrow_resp_t [NumBanksPerTile-1:0]            mst_resp_i,
  input  logic         [NumBanksPerTile-1:0]            mst_resp_valid_i,
  output logic         [NumBanksPerTile-1:0]            mst_resp_ready_o,
  input  logic         [NumBanksPerTile-1:0]            mst_resp_wide_i,
  input  resp_idx_t    [NumBanksPerTile-1:0]            mst_resp_ini_addr_i
);

  // A single-bank superbank would zero-width the bank_id_lo slices.
  if (NumBanksPerSB < 2)
    $fatal(1, "mempool_tcdm_bank_interco: NumBanksPerSB (=%0d) must be >= 2.", NumBanksPerSB);
  if (NumBanksPerSB * NarrowDataWidth != DmaWideDataWidth)
    $fatal(1, "mempool_tcdm_bank_interco: DMA width %0d b != superbank width %0d b.",
           DmaWideDataWidth, NumBanksPerSB * NarrowDataWidth);
  `ifdef USE_WIDE_TCDM_PLANE
  if (NumWidePorts < 1)
    $fatal(1, "mempool_tcdm_bank_interco: at least one wide TCDM plane port is required.");
  `endif

  // ============================================================
  // Narrow request: bank-id-lo remap, superbank demux, per-superbank xbar.
  // ============================================================
  // Within-superbank constraint: only the low BankIdLoBits of bank_id are rewritten.
  logic [NumNarrowReq-1:0][BankIdLoBits-1:0] narrow_lo;
  logic [NumNarrowReq-1:0][SBSelBits-1:0]    narrow_sb;
  logic [NumSuperbanks-1:0][NumNarrowReq-1:0] narrow_sb_valid, narrow_sb_ready;

  for (genvar i = 0; i < NumNarrowReq; i++) begin : gen_narrow_sel
    logic [BankIdLoBits-1:0] raw_lo;
    assign raw_lo = slv_narrow_req_i[i].tgt_addr[0 +: BankIdLoBits];
    if (SpmBankIdRemap) begin : gen_remap_active
      // Offset comes from the row-id slice just above bank_id.
      logic [BankRemapShiftAmt-1:0] offset_lo;
      assign offset_lo = slv_narrow_req_i[i].tgt_addr[BankOffsetBits +: BankRemapShiftAmt];
      assign narrow_lo[i] = raw_lo + offset_lo;
    end else begin : gen_remap_passthrough
      assign narrow_lo[i] = raw_lo;
    end

    logic [NumSuperbanks-1:0] dmx_valid, dmx_ready;
    if (NumSuperbanks > 1) begin : gen_sb_demux
      // bank_id_hi (the superbank) passes through untouched.
      assign narrow_sb[i] = slv_narrow_req_i[i].tgt_addr[BankIdLoBits +: SBSelBits];
      stream_demux #(.N_OUP(NumSuperbanks)) i_sb_demux (
        .inp_valid_i(slv_narrow_req_valid_i[i]),
        .inp_ready_o(slv_narrow_req_ready_o[i]),
        .oup_sel_i  (narrow_sb[i]             ),
        .oup_valid_o(dmx_valid                ),
        .oup_ready_i(dmx_ready                )
      );
    end else begin : gen_no_sb_demux
      assign narrow_sb[i]              = '0;
      assign dmx_valid                 = slv_narrow_req_valid_i[i];
      assign slv_narrow_req_ready_o[i] = dmx_ready;
    end
    for (genvar s = 0; s < NumSuperbanks; s++) begin : gen_sb_valid
      assign narrow_sb_valid[s][i] = dmx_valid[s];
      assign dmx_ready[s]          = narrow_sb_ready[s][i];
    end
  end

  narrow_req_t [NumBanksPerTile-1:0] narrow_to_bank;
  logic        [NumBanksPerTile-1:0] narrow_to_bank_valid;
  logic        [NumBanksPerTile-1:0] narrow_to_bank_ready;
  resp_idx_t   [NumBanksPerTile-1:0] narrow_to_bank_idx;

  for (genvar s = 0; s < NumSuperbanks; s++) begin : gen_narrow_req_xbar
    stream_xbar #(
      .NumInp   (NumNarrowReq ),
      .NumOut   (NumBanksPerSB),
      .payload_t(narrow_req_t )
    ) i_narrow_req_xbar (
      .clk_i  (clk_i                                             ),
      .rst_ni (rst_ni                                            ),
      .flush_i(1'b0                                              ),
      .rr_i   ('0                                                ),
      .data_i (slv_narrow_req_i                                  ),
      .valid_i(narrow_sb_valid[s]                                ),
      .ready_o(narrow_sb_ready[s]                                ),
      .sel_i  (narrow_lo                                         ),
      .data_o (narrow_to_bank      [s*NumBanksPerSB +: NumBanksPerSB]),
      .valid_o(narrow_to_bank_valid[s*NumBanksPerSB +: NumBanksPerSB]),
      .ready_i(narrow_to_bank_ready[s*NumBanksPerSB +: NumBanksPerSB]),
      .idx_o  (narrow_to_bank_idx  [s*NumBanksPerSB +: NumBanksPerSB])
    );
  end

  // ============================================================
  // Wide-word ingress: one DMA handler plus one handler per wide plane port,
  // all on the same superbanks.
  // ============================================================
  narrow_req_t [NumBanksPerTile-1:0] dma_req_for_bank;
  logic        [NumSuperbanks-1:0]   dma_sb_valid, dma_sb_ready;
  logic        [NumBanksPerTile-1:0] dma_resp_valid, dma_resp_ready;
  narrow_req_t [NumWidePorts-1:0][NumBanksPerTile-1:0] wide_req_for_bank;
  logic [NumWidePorts-1:0][NumSuperbanks-1:0]   wide_sb_valid, wide_sb_ready;
  logic [NumWidePorts-1:0][NumBanksPerTile-1:0] wide_resp_valid, wide_resp_ready;

  // mempool_tcdm_wide_superbank is the generic per-ingress datapath, so ITS
  // `wide` prefix means "one wide word", not "the wide plane": the DMA drives
  // it too.
  mempool_tcdm_wide_superbank #(
    .NumBanksPerTile(NumBanksPerTile ),
    .NumBanksPerSB  (NumBanksPerSB   ),
    .NarrowDataWidth(NarrowDataWidth ),
    .WideDataWidth  (DmaWideDataWidth),
    .SpmBankIdRemap (SpmBankIdRemap  ),
    .RotFifoDepth   (RotFifoDepth    ),
    // Cut like the wide plane. The extra round-trip cycle is covered by 4 DMA
    // beats in flight (mempool_group.sv axi_to_reqrsp / reqrsp_demux depth 4);
    // with only 2 it cost ~30 % of the L1 DMA bandwidth.
    .IngressSpill   (1'b1            ),
    // axi_to_reqrsp matches DMA responses to requests in order.
    .InOrderResp    (1'b1            ),
    .wide_req_t     (dma_wide_req_t  ),
    .wide_resp_t    (dma_wide_resp_t ),
    .narrow_req_t   (narrow_req_t    ),
    .narrow_resp_t  (narrow_resp_t   ),
    .group_id_t     (group_id_t      ),
    .ini_addr_t     (tile_group_id_t )
  ) i_dma_superbank (
    .clk_i, .rst_ni,
    .slv_wide_req_i           (slv_dma_req_i        ),
    .slv_wide_req_valid_i     (slv_dma_req_valid_i  ),
    .slv_wide_req_ready_o     (slv_dma_req_ready_o  ),
    .slv_wide_resp_o          (slv_dma_resp_o       ),
    .slv_wide_resp_valid_o    (slv_dma_resp_valid_o ),
    .slv_wide_resp_ready_i    (slv_dma_resp_ready_i ),
    // The DMA response never leaves the tile: no source routing to carry, and the
    // DMA frontend does not need a read/write tag on its response.
    .slv_wide_ini_addr_i      ('0        ),
    .slv_wide_src_group_i     (group_id_i),
    .slv_wide_resp_ini_addr_o (/* unused */),
    .slv_wide_resp_src_group_o(/* unused */),
    .slv_wide_resp_is_write_o (/* unused */),
    .wide_req_for_bank_o(dma_req_for_bank), .sb_req_valid_o(dma_sb_valid),
    .sb_req_ready_i     (dma_sb_ready    ),
    .resp_i             (mst_resp_i      ), .resp_valid_i(dma_resp_valid),
    .resp_ready_o       (dma_resp_ready  )
  );

  // Per-bank class tag: routes each wide response back to the ingress that was
  // granted (0 = DMA, p+1 = wide port p). Bank responses are in order, so a
  // small FIFO per bank suffices.
  localparam int unsigned ClassTagWidth = idx_width(NumWidePorts + 1);
  typedef logic [ClassTagWidth-1:0] class_tag_t;
  logic       [NumBanksPerTile-1:0] ctag_push_valid, ctag_push_ready;
  logic       [NumBanksPerTile-1:0] ctag_pop_valid, ctag_pop_ready;
  class_tag_t [NumBanksPerTile-1:0] ctag_push_data, ctag_pop_data;

  `ifdef USE_WIDE_TCDM_PLANE
  // Each ingress path owns a handler. Paths targeting different superbanks
  // progress in parallel; arbitration below occurs only on a shared superbank.
  for (genvar p = 0; p < NumWidePorts; p++) begin : gen_wide_port
    wide_resp_t wide_submod_resp;
    tile_group_id_t wide_resp_ini_addr;
    group_id_t      wide_resp_src_group;
    logic           wide_resp_is_write;
    assign slv_wide_resp_o[p] = '{
      rdata:        wide_submod_resp.rdata,
      ini_addr:     wide_resp_ini_addr,
      src_group_id: wide_resp_src_group,
      is_write:     wide_resp_is_write
    };

    mempool_tcdm_wide_superbank #(
      .NumBanksPerTile(NumBanksPerTile ),
      .NumBanksPerSB  (NumBanksPerSB   ),
      .NarrowDataWidth(NarrowDataWidth ),
      .WideDataWidth  (WideDataWidth   ),
      .SpmBankIdRemap (SpmBankIdRemap  ),
      .RotFifoDepth   (RotFifoDepth    ),
      .wide_req_t     (wide_req_t      ),
      .wide_resp_t    (wide_resp_t     ),
      .narrow_req_t   (narrow_req_t    ),
      .narrow_resp_t  (narrow_resp_t   ),
      .group_id_t     (group_id_t      ),
      .ini_addr_t     (tile_group_id_t )
    ) i_wide_superbank (
      .clk_i,
      .rst_ni,
      .slv_wide_req_i           (slv_wide_req_i[p]             ),
      .slv_wide_req_valid_i     (slv_wide_req_valid_i[p]       ),
      .slv_wide_req_ready_o     (slv_wide_req_ready_o[p]       ),
      .slv_wide_resp_o          (wide_submod_resp              ),
      .slv_wide_resp_valid_o    (slv_wide_resp_valid_o[p]      ),
      .slv_wide_resp_ready_i    (slv_wide_resp_ready_i[p]      ),
      .slv_wide_ini_addr_i      (slv_wide_req_i[p].ini_addr    ),
      .slv_wide_src_group_i     (slv_wide_req_i[p].src_group_id),
      .slv_wide_resp_ini_addr_o (wide_resp_ini_addr            ),
      .slv_wide_resp_src_group_o(wide_resp_src_group           ),
      .slv_wide_resp_is_write_o (wide_resp_is_write            ),
      .wide_req_for_bank_o      (wide_req_for_bank[p]          ),
      .sb_req_valid_o           (wide_sb_valid[p]              ),
      .sb_req_ready_i           (wide_sb_ready[p]              ),
      .resp_i                   (mst_resp_i                    ),
      .resp_valid_i             (wide_resp_valid[p]            ),
      .resp_ready_o             (wide_resp_ready[p]            )
    );
  end

  for (genvar b = 0; b < NumBanksPerTile; b++) begin : gen_ctag_fifo
    stream_fifo_optimal_wrap #(
      .Depth (ClassFifoDepth),
      .type_t(class_tag_t    )
    ) i_ctag_fifo (
      .clk_i (clk_i), .rst_ni(rst_ni), .flush_i(1'b0), .testmode_i(1'b0), .usage_o(),
      .data_i(ctag_push_data[b]), .valid_i(ctag_push_valid[b]), .ready_o(ctag_push_ready[b]),
      .data_o(ctag_pop_data[b]),  .valid_o(ctag_pop_valid[b]),  .ready_i(ctag_pop_ready[b])
    );
  end
  `else
  // No wide plane: tie it inert so the arbitration degrades to DMA > narrow.
  // wide_resp_valid is NOT tied here -- gen_resp_split's always_comb already
  // drives it (to 0, since ctag_pop_data is tied low), and a continuous assign
  // on top of that is a vopt-12003 error.
  assign wide_req_for_bank = '0;
  assign wide_sb_valid     = '0;
  assign wide_resp_ready   = '0;
  assign ctag_push_ready  = '1;   // never blocks a (DMA-only) wide grant
  // The DMA owns every wide response, so the tag is permanently available: this
  // feeds dma_resp_valid in gen_resp_split. Tying it low instead would stall the
  // DMA join forever and wedge the tile.
  assign ctag_pop_valid   = '1;
  assign ctag_pop_data    = '0;   // route every wide response to the DMA
  `endif

  // ============================================================
  // Per-superbank arbitration: ATOMIC grant, wide plane > DMA > narrow.
  //
  // An ingress is granted a superbank only when EVERY bank of it can take the
  // access in the SAME cycle. This is load-bearing, not an optimisation: with
  // several wide paths sharing banks, a partial grant lets the per-bank response
  // queues acquire DIFFERENT class orders (bank i holds [wide, dma] while bank j
  // holds [dma, wide]). Each ingress's join then waits on an entry stuck behind
  // the other's, and the two joins deadlock -- which is exactly how rm_gemm
  // wedged at ~cyc 5564. Granting atomically keeps every bank of a superbank in
  // lockstep, so the class order is identical at every bank. Nothing spans two
  // superbanks: a split wide word's parts are granted independently and never
  // wait for each other at the banks (mempool_tcdm_wide_superbank).
  //
  // Priority is round-robin among the wide ports, then strict wide plane > DMA.
  // The residual risk is DMA STARVATION if the plane streams without gaps.
  // Acceptable today (RedMulE and DMA phases are disjoint in the dense-LLM
  // kernels) -- if a measurement ever shows a starved DMA, add a round-robin
  // toggle here rather than reintroducing per-bank aging.
  // ============================================================
  // A bank can take a wide access iff it accepts AND its class-tag FIFO has room
  // for the response routing.
  logic [NumBanksPerTile-1:0] bank_ingress_ok;
  assign bank_ingress_ok = mst_req_ready_i & ctag_push_ready;

  logic [NumSuperbanks-1:0][NumWidePorts-1:0] wide_sb_req, wide_sb_gnt;
  logic [NumSuperbanks-1:0] sb_ok, sb_wide_want, dma_sb_go, sb_wants;

  for (genvar s = 0; s < NumSuperbanks; s++) begin : gen_sb_arb
    assign sb_ok[s] = &bank_ingress_ok[s*NumBanksPerSB +: NumBanksPerSB];
    for (genvar p = 0; p < NumWidePorts; p++) begin : gen_wide_port_req
      assign wide_sb_req[s][p]   = wide_sb_valid[p][s];
      assign wide_sb_ready[p][s] = wide_sb_req[s][p] & wide_sb_gnt[s][p];
    end
    assign sb_wide_want[s] = |wide_sb_req[s];

    rr_arb_tree #(
      .NumIn    (NumWidePorts),
      .DataType (logic       ),
      .AxiVldRdy(1'b1        ),
      .LockIn   (1'b1        ),
      .FairArb  (1'b1        )
    ) i_wide_sb_arb (
      .clk_i  (clk_i         ),
      .rst_ni (rst_ni        ),
      .flush_i(1'b0          ),
      .rr_i   ('0            ),
      .req_i  (wide_sb_req[s]),
      .gnt_o  (wide_sb_gnt[s]),
      .data_i ('0            ),
      .req_o  (/* unused */  ),
      .gnt_i  (sb_ok[s]      ),
      .data_o (/* unused */  ),
      .idx_o  (/* unused */  )
    );

    // The plane outranks the DMA: block on WANT (not on grant).
    assign dma_sb_go[s]    = dma_sb_valid[s] & sb_ok[s] & ~sb_wide_want[s];
    assign dma_sb_ready[s] = dma_sb_go[s];
    // A wide access waiting for its whole superbank holds narrow off, so it
    // cannot be starved while it assembles.
    assign sb_wants[s]     = dma_sb_valid[s] | sb_wide_want[s];
  end

  logic [NumBanksPerTile-1:0] wide_grant_any;
  narrow_req_t [NumBanksPerTile-1:0] wide_req_granted;
  class_tag_t [NumBanksPerTile-1:0] wide_grant_tag;

  for (genvar b = 0; b < NumBanksPerTile; b++) begin : gen_bank_mux
    localparam int unsigned S = b / NumBanksPerSB;

    always_comb begin
      wide_grant_any[b]   = 1'b0;
      wide_req_granted[b] = '0;
      wide_grant_tag[b]   = '0;
      for (int unsigned p = 0; p < NumWidePorts; p++) begin
        if (wide_sb_ready[p][S]) begin
          wide_grant_any[b]   = 1'b1;
          wide_req_granted[b] = wide_req_for_bank[p][b];
          wide_grant_tag[b]   = class_tag_t'(p + 1);
        end
      end
    end

    always_comb begin
      mst_req_o[b]            = narrow_to_bank[b];
      mst_req_valid_o[b]      = 1'b0;
      mst_req_wide_o[b]       = 1'b0;
      mst_req_ini_addr_o[b]   = narrow_to_bank_idx[b];
      narrow_to_bank_ready[b] = 1'b0;
      if (wide_grant_any[b] | dma_sb_go[S]) begin
        mst_req_o[b]          = wide_grant_any[b] ? wide_req_granted[b] : dma_req_for_bank[b];
        mst_req_valid_o[b]    = 1'b1;
        mst_req_wide_o[b]     = 1'b1;
        mst_req_ini_addr_o[b] = '0;
      end else if (sb_wants[S]) begin
        // A wide access is waiting for this superbank but is not yet feasible
        // across all its banks: hold the bank idle rather than letting narrow
        // in, which would break wide-over-narrow and could starve it.
        mst_req_valid_o[b]    = 1'b0;
      end else begin
        mst_req_valid_o[b]      = narrow_to_bank_valid[b];
        narrow_to_bank_ready[b] = mst_req_ready_i[b];
      end
    end

    // Queue the granted class so its (in-order) response can be routed back.
    assign ctag_push_valid[b] = wide_grant_any[b] | dma_sb_go[S];
    assign ctag_push_data[b]  = wide_grant_any[b] ? wide_grant_tag[b] : '0;
  end

  // ============================================================
  // Per-bank response split: wide -> the granted class's join, narrow -> xbar.
  // ============================================================
  narrow_resp_t [NumBanksPerTile-1:0] narrow_resp_to_xbar;
  logic         [NumBanksPerTile-1:0] narrow_resp_to_xbar_valid;
  logic         [NumBanksPerTile-1:0] narrow_resp_to_xbar_ready;

  for (genvar b = 0; b < NumBanksPerTile; b++) begin : gen_resp_split
    assign narrow_resp_to_xbar[b] = mst_resp_i[b];
    for (genvar p = 0; p < NumWidePorts; p++) begin : gen_wide_resp_valid
      assign wide_resp_valid[p][b] = mst_resp_wide_i[b] && ctag_pop_valid[b] &&
          (ctag_pop_data[b] == class_tag_t'(p + 1)) && mst_resp_valid_i[b];
    end
    always_comb begin
      dma_resp_valid[b]            = 1'b0;
      narrow_resp_to_xbar_valid[b] = 1'b0;
      ctag_pop_ready[b]            = 1'b0;
      mst_resp_ready_o[b]          = 1'b0;
      if (mst_resp_wide_i[b]) begin
        if (ctag_pop_data[b] == '0) begin
          dma_resp_valid[b]   = ctag_pop_valid[b] & mst_resp_valid_i[b];
          mst_resp_ready_o[b] = dma_resp_ready[b];
        end else begin
          for (int unsigned p = 0; p < NumWidePorts; p++) begin
            if (ctag_pop_data[b] == class_tag_t'(p + 1)) begin
              mst_resp_ready_o[b]  = wide_resp_ready[p][b];
            end
          end
        end
        ctag_pop_ready[b] = mst_resp_valid_i[b] & mst_resp_ready_o[b];
      end else begin
        narrow_resp_to_xbar_valid[b] = mst_resp_valid_i[b];
        mst_resp_ready_o[b]          = narrow_resp_to_xbar_ready[b];
      end
    end
  end

  // ============================================================
  // Narrow response: per-superbank xbar (NumBanksPerSB -> NumNarrowResp), then
  // a per-master merge over the superbanks.
  // sel = caller-supplied resp idx (already conditioned for any
  // NumNarrowRemoteReqPortsPerTile vs NumNarrowRemoteRespPortsPerTile asymmetry).
  // ============================================================
  narrow_resp_t [NumSuperbanks-1:0][NumNarrowResp-1:0] sb_resp;
  logic         [NumSuperbanks-1:0][NumNarrowResp-1:0] sb_resp_valid, sb_resp_ready;

  for (genvar s = 0; s < NumSuperbanks; s++) begin : gen_narrow_resp_xbar
    stream_xbar #(
      .NumInp   (NumBanksPerSB ),
      .NumOut   (NumNarrowResp ),
      .payload_t(narrow_resp_t )
    ) i_narrow_resp_xbar (
      .clk_i  (clk_i                                                      ),
      .rst_ni (rst_ni                                                     ),
      .flush_i(1'b0                                                       ),
      .rr_i   ('0                                                         ),
      .data_i (narrow_resp_to_xbar      [s*NumBanksPerSB +: NumBanksPerSB]),
      .valid_i(narrow_resp_to_xbar_valid[s*NumBanksPerSB +: NumBanksPerSB]),
      .ready_o(narrow_resp_to_xbar_ready[s*NumBanksPerSB +: NumBanksPerSB]),
      .sel_i  (mst_resp_ini_addr_i      [s*NumBanksPerSB +: NumBanksPerSB]),
      .data_o (sb_resp[s]                                                 ),
      .valid_o(sb_resp_valid[s]                                           ),
      .ready_i(sb_resp_ready[s]                                           ),
      .idx_o  (/* unused */                                               )
    );
  end

  for (genvar m = 0; m < NumNarrowResp; m++) begin : gen_narrow_resp_merge
    narrow_resp_t [NumSuperbanks-1:0] merge_in;
    logic         [NumSuperbanks-1:0] merge_valid, merge_ready;
    for (genvar s = 0; s < NumSuperbanks; s++) begin : gen_merge_in
      assign merge_in[s]         = sb_resp[s][m];
      assign merge_valid[s]      = sb_resp_valid[s][m];
      assign sb_resp_ready[s][m] = merge_ready[s];
    end
    if (NumSuperbanks > 1) begin : gen_merge
      stream_xbar #(
        .NumInp   (NumSuperbanks),
        .NumOut   (1            ),
        .payload_t(narrow_resp_t)
      ) i_merge (
        .clk_i, .rst_ni, .flush_i(1'b0), .rr_i('0),
        .data_i (merge_in                  ),
        .valid_i(merge_valid               ),
        .ready_o(merge_ready               ),
        .sel_i  ('0                        ),
        .data_o (slv_narrow_resp_o[m]      ),
        .valid_o(slv_narrow_resp_valid_o[m]),
        .ready_i(slv_narrow_resp_ready_i[m]),
        .idx_o  (/* unused */              )
      );
    end else begin : gen_no_merge
      assign slv_narrow_resp_o[m]       = merge_in[0];
      assign slv_narrow_resp_valid_o[m] = merge_valid[0];
      assign merge_ready[0]             = slv_narrow_resp_ready_i[m];
    end
  end

endmodule
