// Copyright 2025 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

`include "mempool/mempool.svh"

// Tile-internal TCDM bank-side interconnect: the narrow req/resp crossbars, one
// wide superbank handler per wide ingress, and the per-bank wide-over-narrow
// priority. Companion modules: mempool_tcdm_wide_superbank.sv (the per-ingress
// fork/join wide datapath) and mempool_bank_id_remapper.sv.
//
// Wide ingress paths: the DMA (always present) and, with the WP2 wide TCDM
// plane, the RedMulE tile paths. Each owns its own superbank handler; WIDE_W
// need not equal the DMA width. They share the banks through the priority below
// (engine > DMA > narrow), granted SUPERBANK-ATOMICALLY so ingress paths cannot
// interleave partial forks into the per-bank response queues (see gen_bank_mux).
//
// The superbank index (the upper bank-id bits, i.e. which superbank a wide
// access lands on) passes through unchanged so a wide chunk lands in the same
// superbank a narrow access of its base address would pick; each class's fork
// then rotates within that superbank to stay coherent with the narrow path.
module mempool_tcdm_bank_interco
  import mempool_pkg::*;
  import cf_math_pkg::idx_width;
#(
  parameter int unsigned NumNarrowReq      = 8,
  parameter int unsigned NumNarrowResp     = 8,
  parameter int unsigned NumBanksPerTile   = 16,
  parameter int unsigned NumSuperbanks     = 1,
  parameter int unsigned NarrowDataWidth   = 32,
  parameter int unsigned WideDataWidth     = 512,
  parameter int unsigned ByteOffset        = 2,
  parameter bit          SpmBankIdRemap    = 0,
  parameter type narrow_req_t  = tcdm_slave_req_t,
  parameter type narrow_resp_t = tcdm_slave_resp_t,
  parameter type wide_req_t    = tcdm_dma_req_t,
  parameter type wide_resp_t   = tcdm_dma_resp_t,
  parameter type group_id_t    = mempool_pkg::group_id_t,
  // Must be wide enough to address NumNarrowResp ports (logic [idx_width(NumNarrowResp)-1:0]).
  parameter type resp_idx_t    = logic [3:0],
  // Wide-engine class (WP2 RedMulE). Defaults keep it inert when unused; a WP2
  // build overrides these (its WIDE_W width + wide-engine req/resp types).
  parameter int unsigned NumWideEngPorts = 1,
  parameter int unsigned WideEngDataWidth = 512,
  parameter type wide_eng_req_t  = logic,
  parameter type wide_eng_resp_t = logic,
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
      (NumWideEngPorts + 1) * RotFifoDepth,
  // Per-ingress superbank geometry. DMA and engine widths may differ.
  localparam int unsigned NumBanksPerDmaSB  = WideDataWidth / NarrowDataWidth,
  localparam int unsigned NumDmaSB          = NumBanksPerTile / NumBanksPerDmaSB,
  localparam int unsigned NumBanksPerEngSB  = WideEngDataWidth / NarrowDataWidth,
  localparam int unsigned NumEngSB          = NumBanksPerTile / NumBanksPerEngSB
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

  // ----- Wide DMA request input (single, demuxed internally) -----
  input  wide_req_t                                     slv_wide_req_i,
  input  logic                                          slv_wide_req_valid_i,
  output logic                                          slv_wide_req_ready_o,

  // ----- Wide DMA response output (single, muxed internally) -----
  output wide_resp_t                                    slv_wide_resp_o,
  output logic                                          slv_wide_resp_valid_o,
  input  logic                                          slv_wide_resp_ready_i,

  // ----- Wide-engine request/response ports (WP2 RedMulE) -----
  `ifdef USE_WIDE_TCDM_PLANE
  input  wide_eng_req_t [NumWideEngPorts-1:0]           slv_wide_eng_req_i,
  input  logic          [NumWideEngPorts-1:0]           slv_wide_eng_req_valid_i,
  output logic          [NumWideEngPorts-1:0]           slv_wide_eng_req_ready_o,
  output wide_eng_resp_t [NumWideEngPorts-1:0]          slv_wide_eng_resp_o,
  output logic           [NumWideEngPorts-1:0]          slv_wide_eng_resp_valid_o,
  input  logic           [NumWideEngPorts-1:0]          slv_wide_eng_resp_ready_i,
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
  // Every wide class must tile the banks exactly, or the atomic grant below would
  // index a partial superbank.
  if (NumDmaSB * NumBanksPerDmaSB != NumBanksPerTile)
    $fatal(1, "mempool_tcdm_bank_interco: DMA superbank (%0d banks) does not tile %0d banks.",
           NumBanksPerDmaSB, NumBanksPerTile);
  `ifdef USE_WIDE_TCDM_PLANE
  if (NumWideEngPorts < 1)
    $fatal(1, "mempool_tcdm_bank_interco: at least one wide-engine port is required.");
  if (NumEngSB * NumBanksPerEngSB != NumBanksPerTile)
    $fatal(1, "mempool_tcdm_bank_interco: engine superbank (%0d banks) does not tile %0d banks.",
           NumBanksPerEngSB, NumBanksPerTile);
  `endif

  // ============================================================
  // Narrow-request: bank-id-lo remap + xbar
  // ============================================================
  // Within-superbank constraint: only the low BankIdLoBits of bank_id are rewritten.
  logic [NumNarrowReq-1:0][BankOffsetBits-1:0] narrow_sel;
  for (genvar i = 0; i < NumNarrowReq; i++) begin : gen_narrow_sel
    logic [BankIdLoBits-1:0] raw_lo, new_lo;
    assign raw_lo = slv_narrow_req_i[i].tgt_addr[0 +: BankIdLoBits];
    if (SpmBankIdRemap) begin : gen_remap_active
      // Offset comes from the row-id slice just above bank_id.
      logic [BankRemapShiftAmt-1:0] offset_lo;
      assign offset_lo = slv_narrow_req_i[i].tgt_addr[BankOffsetBits +: BankRemapShiftAmt];
      assign new_lo = raw_lo + offset_lo;
    end else begin : gen_remap_passthrough
      assign new_lo = raw_lo;
    end
    if (NumSuperbanks > 1) begin : gen_hi_passthrough
      // bank_id_hi passes through untouched
      assign narrow_sel[i] =
          {slv_narrow_req_i[i].tgt_addr[BankIdLoBits +: (BankOffsetBits - BankIdLoBits)],
           new_lo};
    end else begin : gen_no_hi
      assign narrow_sel[i] = new_lo;
    end
  end

  narrow_req_t [NumBanksPerTile-1:0] narrow_to_bank;
  logic        [NumBanksPerTile-1:0] narrow_to_bank_valid;
  logic        [NumBanksPerTile-1:0] narrow_to_bank_ready;
  resp_idx_t   [NumBanksPerTile-1:0] narrow_to_bank_idx;

  stream_xbar #(
    .NumInp   (NumNarrowReq    ),
    .NumOut   (NumBanksPerTile ),
    .payload_t(narrow_req_t    )
  ) i_narrow_req_xbar (
    .clk_i  (clk_i                  ),
    .rst_ni (rst_ni                 ),
    .flush_i(1'b0                   ),
    .rr_i   ('0                     ),
    .data_i (slv_narrow_req_i       ),
    .valid_i(slv_narrow_req_valid_i ),
    .ready_o(slv_narrow_req_ready_o ),
    .sel_i  (narrow_sel             ),
    .data_o (narrow_to_bank         ),
    .valid_o(narrow_to_bank_valid   ),
    .ready_i(narrow_to_bank_ready   ),
    .idx_o  (narrow_to_bank_idx     )
  );

  // ============================================================
  // Wide ingress paths: one DMA handler plus one handler per engine tile path.
  // The per-bank priority below is shared across all paths and narrow.
  // ============================================================
  narrow_req_t [NumBanksPerTile-1:0] dma_req_for_bank;
  logic        [NumBanksPerTile-1:0] dma_fork_valid, dma_fork_ready;
  logic        [NumBanksPerTile-1:0] dma_resp_valid, dma_resp_ready;
  narrow_req_t [NumWideEngPorts-1:0][NumBanksPerTile-1:0] eng_req_for_bank;
  logic [NumWideEngPorts-1:0][NumBanksPerTile-1:0] eng_fork_valid, eng_fork_ready;
  logic [NumWideEngPorts-1:0][NumBanksPerTile-1:0] eng_resp_valid, eng_resp_ready;

  mempool_tcdm_wide_superbank #(
    .NumBanksPerTile(NumBanksPerTile),
    .NarrowDataWidth(NarrowDataWidth),
    .WideDataWidth  (WideDataWidth  ),
    .SpmBankIdRemap (SpmBankIdRemap ),
    .RotFifoDepth   (RotFifoDepth   ),
    .wide_req_t     (wide_req_t     ),
    .wide_resp_t    (wide_resp_t    ),
    .narrow_req_t   (narrow_req_t   ),
    .narrow_resp_t  (narrow_resp_t  ),
    .group_id_t     (group_id_t     ),
    .ini_addr_t     (tile_group_id_t)
  ) i_wide_dma (
    .clk_i, .rst_ni,
    .group_id_i               (group_id_i           ),
    .slv_wide_req_i           (slv_wide_req_i       ),
    .slv_wide_req_valid_i     (slv_wide_req_valid_i ),
    .slv_wide_req_ready_o     (slv_wide_req_ready_o ),
    .slv_wide_resp_o          (slv_wide_resp_o      ),
    .slv_wide_resp_valid_o    (slv_wide_resp_valid_o),
    .slv_wide_resp_ready_i    (slv_wide_resp_ready_i),
    // The DMA response never leaves the tile: no source routing to carry, and the
    // DMA frontend does not need a read/write tag on its response.
    .slv_wide_ini_addr_i      ('0        ),
    .slv_wide_src_group_i     (group_id_i),
    .slv_wide_resp_ini_addr_o (/* unused */),
    .slv_wide_resp_src_group_o(/* unused */),
    .slv_wide_resp_is_write_o (/* unused */),
    .wide_req_for_bank_o(dma_req_for_bank), .wide_fork_valid_o(dma_fork_valid),
    .wide_fork_ready_i  (dma_fork_ready  ),
    .resp_i             (mst_resp_i      ), .resp_valid_i(dma_resp_valid),
    .resp_ready_o       (dma_resp_ready  )
  );

  // Per-bank class tag: routes each wide response back to the class that was
  // granted. Bank responses are in order, so a small FIFO per bank suffices.
  localparam int unsigned ClassTagWidth = idx_width(NumWideEngPorts + 1);
  typedef logic [ClassTagWidth-1:0] class_tag_t;
  logic       [NumBanksPerTile-1:0] ctag_push_valid, ctag_push_ready;
  logic       [NumBanksPerTile-1:0] ctag_pop_valid, ctag_pop_ready;
  class_tag_t [NumBanksPerTile-1:0] ctag_push_data, ctag_pop_data;

  `ifdef USE_WIDE_TCDM_PLANE
  // Each ingress path owns a superbank handler. Paths targeting different
  // superbanks can therefore progress in parallel; arbitration below occurs
  // only when their bank sets overlap.
  for (genvar p = 0; p < NumWideEngPorts; p++) begin : gen_wide_eng_port
    wide_eng_resp_t eng_submod_resp;
    tile_group_id_t eng_resp_ini_addr;
    group_id_t      eng_resp_src_group;
    logic           eng_resp_is_write;
    assign slv_wide_eng_resp_o[p] = '{
      rdata:        eng_submod_resp.rdata,
      ini_addr:     eng_resp_ini_addr,
      src_group_id: eng_resp_src_group,
      is_write:     eng_resp_is_write
    };

    mempool_tcdm_wide_superbank #(
      .NumBanksPerTile(NumBanksPerTile ),
      .NarrowDataWidth(NarrowDataWidth ),
      .WideDataWidth  (WideEngDataWidth),
      .SpmBankIdRemap (SpmBankIdRemap  ),
      .RotFifoDepth   (RotFifoDepth    ),
      .wide_req_t     (wide_eng_req_t  ),
      .wide_resp_t    (wide_eng_resp_t ),
      .narrow_req_t   (narrow_req_t    ),
      .narrow_resp_t  (narrow_resp_t   ),
      .group_id_t     (group_id_t      ),
      .ini_addr_t     (tile_group_id_t )
    ) i_wide_eng (
      .clk_i,
      .rst_ni,
      .group_id_i               (group_id_i                       ),
      .slv_wide_req_i           (slv_wide_eng_req_i[p]            ),
      .slv_wide_req_valid_i     (slv_wide_eng_req_valid_i[p]      ),
      .slv_wide_req_ready_o     (slv_wide_eng_req_ready_o[p]      ),
      .slv_wide_resp_o          (eng_submod_resp                   ),
      .slv_wide_resp_valid_o    (slv_wide_eng_resp_valid_o[p]     ),
      .slv_wide_resp_ready_i    (slv_wide_eng_resp_ready_i[p]     ),
      .slv_wide_ini_addr_i      (slv_wide_eng_req_i[p].ini_addr   ),
      .slv_wide_src_group_i     (slv_wide_eng_req_i[p].src_group_id),
      .slv_wide_resp_ini_addr_o (eng_resp_ini_addr                ),
      .slv_wide_resp_src_group_o(eng_resp_src_group               ),
      .slv_wide_resp_is_write_o (eng_resp_is_write                ),
      .wide_req_for_bank_o      (eng_req_for_bank[p]              ),
      .wide_fork_valid_o        (eng_fork_valid[p]                ),
      .wide_fork_ready_i        (eng_fork_ready[p]                ),
      .resp_i                   (mst_resp_i                       ),
      .resp_valid_i             (eng_resp_valid[p]                ),
      .resp_ready_o             (eng_resp_ready[p]                )
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
  // No engine class: tie it inert so the priority degrades to DMA > narrow, i.e.
  // exactly the single-wide-class behaviour. eng_resp_valid is NOT tied here --
  // gen_resp_split's always_comb already drives it (to 0, since ctag_pop_data is
  // tied low), and a continuous assign on top of that is a vopt-12003 error.
  assign eng_req_for_bank = '0;
  assign eng_fork_valid   = '0;
  assign eng_resp_ready   = '0;
  assign ctag_push_ready  = '1;   // never blocks a (DMA-only) wide grant
  assign ctag_pop_valid   = '0;
  assign ctag_pop_data    = '0;   // route every wide response to the DMA
  `endif

  // ============================================================
  // Wide arbitration: SUPERBANK-ATOMIC grant, wide over narrow, engine over DMA.
  //
  // A wide class's fork is granted only when EVERY bank of its superbank can take
  // it in the SAME cycle. This is load-bearing, not an optimisation: with multiple
  // wide paths sharing banks, a partially granted fork lets the per-bank response
  // queues acquire DIFFERENT class orders (bank i holds [eng, dma] while bank j
  // holds [dma, eng]). Each class's stream_join then waits on an entry stuck
  // behind the other class's entry, and the two joins deadlock -- which is exactly
  // how rm_gemm wedged at ~cyc 5564. Granting atomically keeps every bank of a
  // superbank in lockstep, so the class order is identical at every bank and the
  // circular wait cannot form. (The original single-path design got this for free:
  // one path cannot deadlock against itself.)
  //
  // Priority is strict engine > DMA. Deadlock-free by the argument above; the
  // residual risk is DMA STARVATION if an engine streams without gaps. Acceptable
  // today (RedMulE and DMA phases are disjoint in the dense-LLM kernels) -- if a
  // measurement ever shows a starved DMA, add a round-robin toggle here rather
  // than reintroducing per-bank aging.
  // ============================================================
  // A bank can take a wide access iff it accepts AND its class-tag FIFO has room
  // for the response routing.
  logic [NumBanksPerTile-1:0] bank_wide_ok;
  assign bank_wide_ok = mst_req_ready_i & ctag_push_ready;

  logic [NumEngSB-1:0][NumWideEngPorts-1:0] eng_sb_req, eng_sb_gnt;
  logic [NumDmaSB-1:0] dma_sb_go;
  // Per bank: does a wide class want it (even if not yet feasible)? Used to hold
  // narrow off so a wide fork cannot be starved while it assembles.
  logic [NumBanksPerTile-1:0] wide_wants, eng_grant_any, dma_grant;
  logic [NumWideEngPorts-1:0][NumBanksPerTile-1:0] eng_grant;
  narrow_req_t [NumBanksPerTile-1:0] eng_req_granted;
  class_tag_t [NumBanksPerTile-1:0] eng_grant_tag;

  for (genvar d = 0; d < NumEngSB; d++) begin : gen_eng_sb_go
    logic [NumBanksPerEngSB-1:0] ok;
    logic eng_sb_ready;
    for (genvar b = 0; b < NumBanksPerEngSB; b++) begin : gen_eng_sb_ok
      assign ok[b] = bank_wide_ok[d*NumBanksPerEngSB + b];
    end
    assign eng_sb_ready = &ok;
    for (genvar p = 0; p < NumWideEngPorts; p++) begin : gen_eng_port_req
      logic [NumBanksPerEngSB-1:0] want;
      for (genvar b = 0; b < NumBanksPerEngSB; b++) begin : gen_eng_sb_want
        assign want[b] = eng_fork_valid[p][d*NumBanksPerEngSB + b];
      end
      assign eng_sb_req[d][p] = &want;
    end

    rr_arb_tree #(
      .NumIn    (NumWideEngPorts),
      .DataType (logic          ),
      .AxiVldRdy(1'b1           ),
      .LockIn   (1'b1           ),
      .FairArb  (1'b1           )
    ) i_eng_sb_arb (
      .clk_i  (clk_i        ),
      .rst_ni (rst_ni       ),
      .flush_i(1'b0         ),
      .rr_i   ('0           ),
      .req_i  (eng_sb_req[d]),
      .gnt_o  (eng_sb_gnt[d]),
      .data_i ('0           ),
      .req_o  (/* unused */ ),
      .gnt_i  (eng_sb_ready ),
      .data_o (/* unused */ ),
      .idx_o  (/* unused */ )
    );
  end

  for (genvar e = 0; e < NumDmaSB; e++) begin : gen_dma_sb_go
    logic [NumBanksPerDmaSB-1:0] want, ok, blocked;
    for (genvar b = 0; b < NumBanksPerDmaSB; b++) begin : gen_dma_sb_bit
      assign want[b]    = dma_fork_valid[e*NumBanksPerDmaSB + b];
      assign ok[b]      = bank_wide_ok  [e*NumBanksPerDmaSB + b];
      // The engine outranks the DMA: block on WANT (not on grant), so the DMA can
      // never slip in between the banks of an engine fork that is still assembling.
      always_comb begin
        blocked[b] = 1'b0;
        for (int unsigned p = 0; p < NumWideEngPorts; p++)
          blocked[b] |= eng_fork_valid[p][e*NumBanksPerDmaSB + b];
      end
    end
    assign dma_sb_go[e] = (&want) & (&ok) & ~(|blocked);
  end

  for (genvar b = 0; b < NumBanksPerTile; b++) begin : gen_bank_mux
    assign dma_grant[b]  = dma_sb_go[b / NumBanksPerDmaSB];
    for (genvar p = 0; p < NumWideEngPorts; p++) begin : gen_eng_grant
      assign eng_grant[p][b] =
          eng_sb_req[b / NumBanksPerEngSB][p] &
          eng_sb_gnt[b / NumBanksPerEngSB][p];
    end

    always_comb begin
      wide_wants[b]    = dma_fork_valid[b];
      eng_grant_any[b] = 1'b0;
      eng_req_granted[b] = '0;
      eng_grant_tag[b] = '0;
      for (int unsigned p = 0; p < NumWideEngPorts; p++) begin
        wide_wants[b] |= eng_fork_valid[p][b];
        if (eng_grant[p][b]) begin
          eng_grant_any[b] = 1'b1;
          eng_req_granted[b] = eng_req_for_bank[p][b];
          eng_grant_tag[b] = class_tag_t'(p + 1);
        end
      end
    end

    always_comb begin
      mst_req_o[b]            = narrow_to_bank[b];
      mst_req_valid_o[b]      = 1'b0;
      mst_req_wide_o[b]       = 1'b0;
      mst_req_ini_addr_o[b]   = narrow_to_bank_idx[b];
      narrow_to_bank_ready[b] = 1'b0;
      if (eng_grant_any[b] | dma_grant[b]) begin
        mst_req_o[b]          = eng_grant_any[b] ? eng_req_granted[b] : dma_req_for_bank[b];
        mst_req_valid_o[b]    = 1'b1;
        mst_req_wide_o[b]     = 1'b1;
        mst_req_ini_addr_o[b] = '0;
      end else if (wide_wants[b]) begin
        // A wide fork is assembling on this bank but is not yet feasible across
        // its whole superbank: hold the bank idle rather than letting narrow in,
        // which would break wide-over-narrow and could starve the fork.
        mst_req_valid_o[b]    = 1'b0;
      end else begin
        mst_req_valid_o[b]      = narrow_to_bank_valid[b];
        narrow_to_bank_ready[b] = mst_req_ready_i[b];
      end
    end

    // The engine arbiter and dma_sb_go include mst_req_ready_i for every bank
    // of the superbank, so the fork and bank handshake in the same cycle.
    for (genvar p = 0; p < NumWideEngPorts; p++) begin : gen_eng_ready
      assign eng_fork_ready[p][b] = eng_grant[p][b];
    end
    assign dma_fork_ready[b] = dma_grant[b];

    // Queue the granted class so its (in-order) response can be routed back.
    assign ctag_push_valid[b] = eng_grant_any[b] | dma_grant[b];
    assign ctag_push_data[b]  = eng_grant_any[b] ? eng_grant_tag[b] : '0;
  end

  // ============================================================
  // Per-bank response split: wide -> the granted class's join, narrow -> xbar.
  // ============================================================
  narrow_resp_t [NumBanksPerTile-1:0] narrow_resp_to_xbar;
  logic         [NumBanksPerTile-1:0] narrow_resp_to_xbar_valid;
  logic         [NumBanksPerTile-1:0] narrow_resp_to_xbar_ready;

  for (genvar b = 0; b < NumBanksPerTile; b++) begin : gen_resp_split
    assign narrow_resp_to_xbar[b] = mst_resp_i[b];
    for (genvar p = 0; p < NumWideEngPorts; p++) begin : gen_eng_resp_valid
      assign eng_resp_valid[p][b] = mst_resp_wide_i[b] && ctag_pop_valid[b] &&
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
          for (int unsigned p = 0; p < NumWideEngPorts; p++) begin
            if (ctag_pop_data[b] == class_tag_t'(p + 1)) begin
              mst_resp_ready_o[b]  = eng_resp_ready[p][b];
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
  // Narrow response xbar (NumBanksPerTile -> NumNarrowResp).
  // sel = caller-supplied resp idx (already conditioned for any
  // NumNarrowRemoteReqPortsPerTile vs NumNarrowRemoteRespPortsPerTile asymmetry).
  // ============================================================
  stream_xbar #(
    .NumInp   (NumBanksPerTile ),
    .NumOut   (NumNarrowResp   ),
    .payload_t(narrow_resp_t   )
  ) i_narrow_resp_xbar (
    .clk_i  (clk_i                      ),
    .rst_ni (rst_ni                     ),
    .flush_i(1'b0                       ),
    .rr_i   ('0                         ),
    .data_i (narrow_resp_to_xbar        ),
    .valid_i(narrow_resp_to_xbar_valid  ),
    .ready_o(narrow_resp_to_xbar_ready  ),
    .sel_i  (mst_resp_ini_addr_i        ),
    .data_o (slv_narrow_resp_o          ),
    .valid_o(slv_narrow_resp_valid_o    ),
    .ready_i(slv_narrow_resp_ready_i    ),
    .idx_o  (/* unused */               )
  );

endmodule
