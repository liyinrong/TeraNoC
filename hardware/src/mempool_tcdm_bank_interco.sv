// Copyright 2026 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

`include "mempool/mempool.svh"

// Tile-internal TCDM bank-side interconnect: the narrow req/resp crossbars, two
// wide "class" superbank handlers (DMA, and the WP2 RedMulE engine), a shared
// per-bank 3-way arbiter (narrow > eng > DMA with sticky aging), and per-bank
// class-tag routing of wide responses. The wide superbank scatter/gather and
// response pipelining live in mempool_tcdm_wide_superbank (instantiated per class).
//
// Bank-id remap is within-superbank only: only the low log2(NumBanksPerSB) bits
// of bank_id are rewritten (add the row-id slice); bank_id_hi (superbank select)
// passes through unchanged so a wide chunk lands in the same superbank a narrow
// access of its base address would pick.
//
// gen_superbank_resp_ini_addr (NumRemoteReqPortsPerTile vs NumRemoteRespPortsPerTile
// asymmetry) stays OUTSIDE this module: the caller passes the post-conditioning
// resp idx via mst_resp_ini_addr_i.

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
  localparam int unsigned WideOutstanding   = 3
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

  // ----- Wide-engine request/response (WP2 RedMulE, single, handled internally) -----
  `ifdef USE_WIDE_TCDM_PLANE
  input  wide_eng_req_t                                 slv_wide_eng_req_i,
  input  logic                                          slv_wide_eng_req_valid_i,
  output logic                                          slv_wide_eng_req_ready_o,
  output wide_eng_resp_t                                slv_wide_eng_resp_o,
  output logic                                          slv_wide_eng_resp_valid_o,
  input  logic                                          slv_wide_eng_resp_ready_i,
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
  // Wide classes: DMA (always) + engine (WP2). Each is its own WIDE_W superbank
  // handler; the per-bank arbiter below is shared across both + narrow.
  // ============================================================
  narrow_req_t [NumBanksPerTile-1:0]                    dma_req_for_bank, eng_req_for_bank;
  logic        [NumBanksPerTile-1:0]                    dma_present, eng_present;
  logic        [NumBanksPerTile-1:0]                    dma_served,  eng_served;
  logic        [NumBanksPerTile-1:0][NarrowDataWidth-1:0] dma_resp_data, eng_resp_data;
  logic        [NumBanksPerTile-1:0]                    dma_resp_valid, eng_resp_valid;
  logic        [NumBanksPerTile-1:0]                    dma_resp_ready, eng_resp_ready;

  mempool_tcdm_wide_superbank #(
    .NumBanksPerTile(NumBanksPerTile),
    .NarrowDataWidth(NarrowDataWidth),
    .WideDataWidth  (WideDataWidth  ),
    .SpmBankIdRemap (SpmBankIdRemap ),
    .WideOutstanding(WideOutstanding),
    .wide_req_t     (wide_req_t     ),
    .wide_resp_t    (wide_resp_t    ),
    .narrow_req_t   (narrow_req_t   ),
    .group_id_t     (group_id_t     )
  ) i_wide_dma (
    .clk_i (clk_i), .rst_ni(rst_ni), .group_id_i(group_id_i),
    .slv_wide_req_i (slv_wide_req_i), .slv_wide_req_valid_i(slv_wide_req_valid_i), .slv_wide_req_ready_o(slv_wide_req_ready_o),
    .slv_wide_resp_o(slv_wide_resp_o), .slv_wide_resp_valid_o(slv_wide_resp_valid_o), .slv_wide_resp_ready_i(slv_wide_resp_ready_i),
    .wide_req_for_bank_o(dma_req_for_bank), .wide_present_o(dma_present), .wide_served_i(dma_served),
    .resp_data_i(dma_resp_data), .resp_valid_i(dma_resp_valid), .resp_ready_o(dma_resp_ready)
  );

  // Per-bank class-tag FIFO: routes each wide response back to eng or dma
  // (0 = dma, 1 = eng). In-order per-bank service ⇒ the head is the next response.
  localparam int unsigned ClassFifoDepth = 8;
  logic [NumBanksPerTile-1:0] ctag_push_valid, ctag_push_ready, ctag_push_data;
  logic [NumBanksPerTile-1:0] ctag_pop_valid,  ctag_pop_ready,  ctag_pop_data;

  `ifdef USE_WIDE_TCDM_PLANE
  mempool_tcdm_wide_superbank #(
    .NumBanksPerTile(NumBanksPerTile ),
    .NarrowDataWidth(NarrowDataWidth ),
    .WideDataWidth  (WideEngDataWidth),
    .SpmBankIdRemap (SpmBankIdRemap  ),
    .WideOutstanding(WideOutstanding ),
    .wide_req_t     (wide_eng_req_t  ),
    .wide_resp_t    (wide_eng_resp_t ),
    .narrow_req_t   (narrow_req_t    ),
    .group_id_t     (group_id_t      )
  ) i_wide_eng (
    .clk_i (clk_i), .rst_ni(rst_ni), .group_id_i(group_id_i),
    .slv_wide_req_i (slv_wide_eng_req_i), .slv_wide_req_valid_i(slv_wide_eng_req_valid_i), .slv_wide_req_ready_o(slv_wide_eng_req_ready_o),
    .slv_wide_resp_o(slv_wide_eng_resp_o), .slv_wide_resp_valid_o(slv_wide_eng_resp_valid_o), .slv_wide_resp_ready_i(slv_wide_eng_resp_ready_i),
    .wide_req_for_bank_o(eng_req_for_bank), .wide_present_o(eng_present), .wide_served_i(eng_served),
    .resp_data_i(eng_resp_data), .resp_valid_i(eng_resp_valid), .resp_ready_o(eng_resp_ready)
  );
  for (genvar b = 0; b < NumBanksPerTile; b++) begin : gen_ctag_fifo
    stream_fifo_optimal_wrap #(.Depth(ClassFifoDepth), .type_t(logic)) i_ctag_fifo (
      .clk_i(clk_i), .rst_ni(rst_ni), .flush_i(1'b0), .testmode_i(1'b0), .usage_o(),
      .data_i(ctag_push_data[b]), .valid_i(ctag_push_valid[b]), .ready_o(ctag_push_ready[b]),
      .data_o(ctag_pop_data[b]),  .valid_o(ctag_pop_valid[b]),  .ready_i(ctag_pop_ready[b])
    );
  end
  `else
  // No engine class: tie it inert so the arbiter degrades to narrow > dma.
  assign eng_req_for_bank = '0;
  assign eng_present      = '0;
  assign eng_resp_ready   = '0;
  assign ctag_push_ready  = '1;   // never blocks a (dma-only) wide grant
  assign ctag_pop_valid   = '0;
  assign ctag_pop_data    = '0;   // route all wide responses to dma
  `endif

  // ============================================================
  // Per-bank 3-way arbiter: narrow > eng > DMA, with sticky aging per wide class
  // so neither wide class can be starved by narrow (INVARIANT #2). Priority
  // cascade: boosted-eng > boosted-dma > narrow > eng > dma.
  // ============================================================
  localparam int unsigned WideAgeThreshold = 8;
  localparam int unsigned AgeWidth = $clog2(WideAgeThreshold + 1);

  logic [NumBanksPerTile-1:0][AgeWidth-1:0] eng_age_q, eng_age_d, dma_age_q, dma_age_d;
  logic [NumBanksPerTile-1:0]               eng_boost_q, eng_boost_d, dma_boost_q, dma_boost_d;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      eng_age_q <= '0; dma_age_q <= '0; eng_boost_q <= '0; dma_boost_q <= '0;
    end else begin
      eng_age_q   <= eng_age_d;   dma_age_q   <= dma_age_d;
      eng_boost_q <= eng_boost_d; dma_boost_q <= dma_boost_d;
    end
  end

  for (genvar b = 0; b < NumBanksPerTile; b++) begin : gen_bank_mux
    logic narrow_present, eng_win, dma_win, narrow_win, wide_wins, wide_grant, wide_is_eng;
    logic eng_blocked, dma_blocked;
    assign narrow_present = narrow_to_bank_valid[b];

    always_comb begin
      eng_win = 1'b0; dma_win = 1'b0; narrow_win = 1'b0;
      if      (eng_present[b] & eng_boost_q[b]) eng_win    = 1'b1;
      else if (dma_present[b] & dma_boost_q[b]) dma_win    = 1'b1;
      else if (narrow_present)                  narrow_win = 1'b1;
      else if (eng_present[b])                  eng_win    = 1'b1;
      else if (dma_present[b])                  dma_win    = 1'b1;
    end

    assign wide_wins   = eng_win | dma_win;
    assign wide_is_eng = eng_win;
    // A wide grant needs class-tag room so its response can be routed later.
    assign wide_grant  = wide_wins & ctag_push_ready[b];

    always_comb begin
      mst_req_o[b]            = narrow_to_bank[b];
      mst_req_valid_o[b]      = 1'b0;
      mst_req_wide_o[b]       = 1'b0;
      mst_req_ini_addr_o[b]   = narrow_to_bank_idx[b];
      narrow_to_bank_ready[b] = 1'b0;
      if (wide_grant) begin
        mst_req_o[b]          = wide_is_eng ? eng_req_for_bank[b] : dma_req_for_bank[b];
        mst_req_valid_o[b]    = 1'b1;
        mst_req_wide_o[b]     = 1'b1;
        mst_req_ini_addr_o[b] = '0;
      end else if (wide_wins) begin
        // A wide class won but the class-tag FIFO is full: stall the bank (do NOT
        // let narrow through, which would violate priority / risk starvation).
        mst_req_valid_o[b]    = 1'b0;
      end else if (narrow_win) begin
        mst_req_valid_o[b]      = 1'b1;
        narrow_to_bank_ready[b] = mst_req_ready_i[b];
      end
    end

    assign eng_served[b] = wide_grant &  wide_is_eng & mst_req_ready_i[b];
    assign dma_served[b] = wide_grant & ~wide_is_eng & mst_req_ready_i[b];

    // Queue the granted class so the (in-order) response can be routed back.
    assign ctag_push_valid[b] = wide_grant & mst_req_ready_i[b];
    assign ctag_push_data[b]  = wide_is_eng;

    // Aging per wide class: climb while blocked by a higher priority; reset on served.
    assign eng_blocked = eng_present[b] & narrow_win;
    assign dma_blocked = dma_present[b] & (narrow_win | eng_win);
    always_comb begin
      eng_age_d[b] = eng_age_q[b]; eng_boost_d[b] = eng_boost_q[b];
      if (eng_served[b]) begin
        eng_age_d[b] = '0; eng_boost_d[b] = 1'b0;
      end else if (eng_blocked) begin
        if (eng_age_q[b] != AgeWidth'(WideAgeThreshold))     eng_age_d[b]   = eng_age_q[b] + 1'b1;
        if (eng_age_q[b] >= AgeWidth'(WideAgeThreshold - 1)) eng_boost_d[b] = 1'b1;
      end
      dma_age_d[b] = dma_age_q[b]; dma_boost_d[b] = dma_boost_q[b];
      if (dma_served[b]) begin
        dma_age_d[b] = '0; dma_boost_d[b] = 1'b0;
      end else if (dma_blocked) begin
        if (dma_age_q[b] != AgeWidth'(WideAgeThreshold))     dma_age_d[b]   = dma_age_q[b] + 1'b1;
        if (dma_age_q[b] >= AgeWidth'(WideAgeThreshold - 1)) dma_boost_d[b] = 1'b1;
      end
    end
  end

  // ============================================================
  // Per-bank response split: wide → eng/dma resp FIFO (by class tag), narrow → xbar.
  // ============================================================
  narrow_resp_t [NumBanksPerTile-1:0] narrow_resp_to_xbar;
  logic         [NumBanksPerTile-1:0] narrow_resp_to_xbar_valid;
  logic         [NumBanksPerTile-1:0] narrow_resp_to_xbar_ready;

  for (genvar b = 0; b < NumBanksPerTile; b++) begin : gen_resp_split
    assign narrow_resp_to_xbar[b] = mst_resp_i[b];
    assign eng_resp_data[b]       = mst_resp_i[b].rdata.data;
    assign dma_resp_data[b]       = mst_resp_i[b].rdata.data;
    logic resp_is_eng;
    assign resp_is_eng = ctag_pop_data[b];
    always_comb begin
      eng_resp_valid[b]            = 1'b0;
      dma_resp_valid[b]            = 1'b0;
      narrow_resp_to_xbar_valid[b] = 1'b0;
      ctag_pop_ready[b]            = 1'b0;
      if (mst_resp_wide_i[b]) begin
        if (resp_is_eng) begin
          eng_resp_valid[b]   = mst_resp_valid_i[b];
          mst_resp_ready_o[b] = eng_resp_ready[b];
        end else begin
          dma_resp_valid[b]   = mst_resp_valid_i[b];
          mst_resp_ready_o[b] = dma_resp_ready[b];
        end
        ctag_pop_ready[b] = mst_resp_valid_i[b] & mst_resp_ready_o[b];
      end else begin
        narrow_resp_to_xbar_valid[b] = mst_resp_valid_i[b];
        mst_resp_ready_o[b]          = narrow_resp_to_xbar_ready[b];
      end
    end
  end

  // ============================================================
  // Narrow response xbar (NumBanksPerTile → NumNarrowResp).
  // sel = caller-supplied resp idx (already conditioned for any
  // NumRemoteReqPortsPerTile vs NumRemoteRespPortsPerTile asymmetry).
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
