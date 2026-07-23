// Copyright 2026 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

`include "mempool/mempool.svh"

// Tile-internal TCDM bank-side interconnect: narrow req/resp crossbars, the
// wide DMA superbank demux/mux, per-bank wide-over-narrow priority, and the
// within-superbank bank-id remap. Replaces the legacy i_local_req_interco /
// i_local_resp_interco / i_dma_req_interco / i_dma_resp_interco stream_xbars,
// the per-superbank tcdm_wide_narrow_mux, and mempool_bank_id_remapper.sv.
//
// Bank-id remap is within-superbank only: only the low log2(NumBanksPerSB)
// bits of bank_id are rewritten (add the row-id slice). bank_id_hi (selects
// superbank for wide DMA) passes through unchanged so the wide chunk lands in
// the same superbank a narrow access of its base address would pick; the wide
// fork then rotates within that superbank to stay coherent with the narrow path.
//
// gen_superbank_resp_ini_addr (NumRemoteReqPortsPerTile vs
// NumRemoteRespPortsPerTile asymmetry) stays OUTSIDE this module: the caller
// passes the post-conditioning resp idx via mst_resp_ini_addr_i.

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
  // derived
  localparam int unsigned NumBanksPerSB    = NumBanksPerTile / NumSuperbanks,
  localparam int unsigned BankOffsetBits   = $clog2(NumBanksPerTile),
  localparam int unsigned BankIdLoBits     = $clog2(NumBanksPerSB),
  localparam int unsigned SBSelBits        = (NumSuperbanks > 1) ? $clog2(NumSuperbanks) : 1,
  localparam int unsigned NarrowBeWidth    = NarrowDataWidth / 8,
  localparam int unsigned RotBits          = (NumBanksPerSB > 1) ? $clog2(NumBanksPerSB) : 1,
  // Bank-id remap offset width (== bank_id_lo width). NumBanksPerSB >= 2 is
  // required (asserted below); a single-bank superbank would zero-width the slices.
  localparam int unsigned BankRemapShiftAmt = BankIdLoBits
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
  // Wide DMA: demux single wide req to per-superbank streams.
  // ============================================================
  wide_req_t [NumSuperbanks-1:0] wide_to_sb;
  logic      [NumSuperbanks-1:0] wide_to_sb_valid;
  logic      [NumSuperbanks-1:0] wide_to_sb_ready;

  if (NumSuperbanks > 1) begin : gen_wide_req_demux
    stream_xbar #(
      .NumInp   (1            ),
      .NumOut   (NumSuperbanks),
      .payload_t(wide_req_t   )
    ) i_wide_req_demux (
      .clk_i  (clk_i               ),
      .rst_ni (rst_ni              ),
      .flush_i(1'b0                ),
      .rr_i   ('0                  ),
      .data_i (slv_wide_req_i      ),
      .valid_i(slv_wide_req_valid_i),
      .ready_o(slv_wide_req_ready_o),
      // Superbank select uses the upper bits of bank_id (unchanged by remap).
      .sel_i  (slv_wide_req_i.tgt_addr[BankIdLoBits +: SBSelBits]),
      .data_o (wide_to_sb          ),
      .valid_o(wide_to_sb_valid    ),
      .ready_i(wide_to_sb_ready    ),
      .idx_o  (/* unused */        )
    );
  end else begin : gen_wide_req_bypass
    assign wide_to_sb[0]        = slv_wide_req_i;
    assign wide_to_sb_valid[0]  = slv_wide_req_valid_i;
    assign slv_wide_req_ready_o = wide_to_sb_ready[0];
  end

  // ============================================================
  // Per-superbank wide handling: scatter-with-completion on the req side and a
  // gather buffer on the resp side. Each bank is served independently (a
  // per-superbank `served` mask) instead of an all-or-nothing fork, so the
  // per-bank arbiter can prefer narrow (scalar) traffic without livelocking the
  // wide access (INVARIANT #2). The wide req is held (backpressured) for its
  // whole req+resp cycle, so its rotation is combinational (no rotation FIFO)
  // and one wide req is outstanding per superbank at a time.
  // ============================================================
  wide_resp_t [NumSuperbanks-1:0] sb_wide_resp;
  logic       [NumSuperbanks-1:0] sb_wide_resp_valid;
  logic       [NumSuperbanks-1:0] sb_wide_resp_ready;

  // Per-bank rotated wide request + arbiter cross-signals (global bank index).
  narrow_req_t [NumBanksPerTile-1:0] wide_req_for_bank;
  logic        [NumBanksPerTile-1:0] wide_present;     // wide wants this bank, unserved
  logic        [NumBanksPerTile-1:0] wide_grant;       // arbiter picked wide this cycle
  logic        [NumBanksPerTile-1:0] wide_served_now;  // wide granted AND bank ready

  // Per-superbank scatter/gather state.
  logic [NumSuperbanks-1:0][NumBanksPerSB-1:0]                      served_q, served_d;
  logic [NumSuperbanks-1:0][NumBanksPerSB-1:0]                      rgot_q,   rgot_d;
  logic [NumSuperbanks-1:0][NumBanksPerSB-1:0][NarrowDataWidth-1:0] rdata_q,  rdata_d;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      served_q <= '0;
      rgot_q   <= '0;
      rdata_q  <= '0;
    end else begin
      served_q <= served_d;
      rgot_q   <= rgot_d;
      rdata_q  <= rdata_d;
    end
  end

  for (genvar d = 0; d < NumSuperbanks; d++) begin : gen_sb
    // Rotation amount = (remapped) bank_id_lo of the held wide req's tgt_addr.
    logic [RotBits-1:0] req_rot;
    if (NumBanksPerSB > 1) begin : gen_rot
      // Same bank-id-lo remap arithmetic as the narrow xbar sel, so rotations match.
      logic [BankIdLoBits-1:0] raw_lo, post_lo;
      assign raw_lo = wide_to_sb[d].tgt_addr[0 +: BankIdLoBits];
      if (SpmBankIdRemap) begin : gen_remap_on
        logic [BankRemapShiftAmt-1:0] off_lo;
        assign off_lo = wide_to_sb[d].tgt_addr[BankOffsetBits +: BankRemapShiftAmt];
        assign post_lo = raw_lo + off_lo;
      end else begin : gen_remap_off
        assign post_lo = raw_lo;
      end
      assign req_rot = post_lo;
    end else begin : gen_no_rot
      assign req_rot = '0;
    end

    // Compose each bank's wide req (rotated word slice); present it until served.
    for (genvar b = 0; b < NumBanksPerSB; b++) begin : gen_sb_bank_req
      localparam int unsigned global_bank = d*NumBanksPerSB + b;
      always_comb begin
        // word j = (b - req_rot) mod NumBanksPerSB feeds bank port b.
        automatic int unsigned j = (b + NumBanksPerSB - req_rot) % NumBanksPerSB;
        wide_req_for_bank[global_bank] = '{
          wdata: '{
            meta_id: wide_to_sb[d].wdata.meta_id,
            core_id: wide_to_sb[d].wdata.core_id,
            amo:     wide_to_sb[d].wdata.amo,
            data:    wide_to_sb[d].wdata.data[j*NarrowDataWidth +: NarrowDataWidth]
          },
          wen:           wide_to_sb[d].wen,
          be:            wide_to_sb[d].be[j*NarrowBeWidth +: NarrowBeWidth],
          tgt_addr:      wide_to_sb[d].tgt_addr,
          ini_addr:      '0,
          src_group_id:  group_id_i
        };
      end
      assign wide_present[global_bank] = wide_to_sb_valid[d] & ~served_q[d][b];
    end

    // Per-bank served / response-arrival events for this superbank.
    logic [NumBanksPerSB-1:0] served_now, rgot_now;
    for (genvar b = 0; b < NumBanksPerSB; b++) begin : gen_sb_events
      localparam int unsigned global_bank = d*NumBanksPerSB + b;
      assign served_now[b] = wide_served_now[global_bank];
      assign rgot_now[b]   = mst_resp_valid_i[global_bank] & mst_resp_wide_i[global_bank];
    end

    // Emit the gathered wide response once every bank response is registered;
    // consume the held wide req from the demux only when it is fully done.
    logic complete;
    assign sb_wide_resp_valid[d] = &rgot_q[d];
    assign complete              = sb_wide_resp_valid[d] & sb_wide_resp_ready[d];
    assign wide_to_sb_ready[d]   = complete;

    always_comb begin
      served_d[d] = served_q[d] | served_now;
      rgot_d[d]   = rgot_q[d]   | rgot_now;
      rdata_d[d]  = rdata_q[d];
      for (int b = 0; b < NumBanksPerSB; b++)
        if (rgot_now[b])
          rdata_d[d][b] = mst_resp_i[d*NumBanksPerSB + b].rdata.data;
      if (complete) begin
        served_d[d] = '0;
        rgot_d[d]   = '0;
      end
    end

    // Reassemble wide rdata with inverse rotation from the gather buffer.
    always_comb begin
      sb_wide_resp[d] = '{rdata: '{default: '0}};
      for (int b = 0; b < NumBanksPerSB; b++) begin
        // Bank port b carries word j = (b - req_rot) mod NumBanksPerSB.
        automatic int unsigned j = (b + NumBanksPerSB - req_rot) % NumBanksPerSB;
        sb_wide_resp[d].rdata.data[j*NarrowDataWidth +: NarrowDataWidth] = rdata_q[d][b];
      end
      // Metadata from the held request — all banks of a wide chunk share it.
      sb_wide_resp[d].rdata.meta_id = wide_to_sb[d].wdata.meta_id;
      sb_wide_resp[d].rdata.core_id = wide_to_sb[d].wdata.core_id;
      sb_wide_resp[d].rdata.amo     = wide_to_sb[d].wdata.amo;
    end

`ifndef SYNTHESIS
    // Watchdog: a held wide access must complete; fires once on a suspected
    // livelock/deadlock (the failure mode the gather+aging rework guards against).
    int unsigned wd_cnt_q;
    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        wd_cnt_q <= 0;
      end else if (wide_to_sb_valid[d] && !complete) begin
        wd_cnt_q <= wd_cnt_q + 1;
        if (wd_cnt_q == 100000)
          $error("[mempool_tcdm_bank_interco] sb=%0d wide access stuck (>100k cyc) served=%b rgot=%b",
                 d, served_q[d], rgot_q[d]);
      end else begin
        wd_cnt_q <= 0;
      end
    end
`endif
  end

  // ============================================================
  // Per-bank arbiter: narrow (scalar) > wide, with sticky aging so a wide
  // access is never starved by a sustained narrow stream (INVARIANT #2). A wide
  // access blocked by narrow for WideAgeThreshold cycles becomes sticky-boosted
  // and preempts narrow at that bank until it is granted (bounded wait).
  // ============================================================
  localparam int unsigned WideAgeThreshold = 8;
  localparam int unsigned AgeWidth = $clog2(WideAgeThreshold + 1);

  logic [NumBanksPerTile-1:0][AgeWidth-1:0] age_q, age_d;
  logic [NumBanksPerTile-1:0]               boost_q, boost_d;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      age_q   <= '0;
      boost_q <= '0;
    end else begin
      age_q   <= age_d;
      boost_q <= boost_d;
    end
  end

  for (genvar b = 0; b < NumBanksPerTile; b++) begin : gen_bank_mux
    logic narrow_present, wide_win, narrow_win, blocked;
    assign narrow_present = narrow_to_bank_valid[b];
    // Wide wins if boosted or if no narrow contends; narrow wins otherwise.
    assign wide_win   = wide_present[b] & (boost_q[b] | ~narrow_present);
    assign narrow_win = narrow_present & ~wide_win;
    assign blocked    = wide_present[b] & narrow_win;

    assign wide_grant[b]      = wide_win;
    assign wide_served_now[b] = wide_win & mst_req_ready_i[b];

    always_comb begin
      if (wide_win) begin
        mst_req_o[b]            = wide_req_for_bank[b];
        mst_req_valid_o[b]      = 1'b1;
        mst_req_wide_o[b]       = 1'b1;
        mst_req_ini_addr_o[b]   = '0;
        narrow_to_bank_ready[b] = 1'b0;
      end else begin
        mst_req_o[b]            = narrow_to_bank[b];
        mst_req_valid_o[b]      = narrow_present;
        mst_req_wide_o[b]       = 1'b0;
        mst_req_ini_addr_o[b]   = narrow_to_bank_idx[b];
        narrow_to_bank_ready[b] = mst_req_ready_i[b];
      end
    end

    // Aging: climb while blocked by narrow; stick the boost at the threshold;
    // both reset once the wide access is granted at this bank.
    always_comb begin
      age_d[b]   = age_q[b];
      boost_d[b] = boost_q[b];
      if (wide_served_now[b]) begin
        age_d[b]   = '0;
        boost_d[b] = 1'b0;
      end else if (blocked) begin
        if (age_q[b] != AgeWidth'(WideAgeThreshold))
          age_d[b] = age_q[b] + 1'b1;
        if (age_q[b] >= AgeWidth'(WideAgeThreshold - 1))
          boost_d[b] = 1'b1;
      end
    end
  end

  // ============================================================
  // Per-bank response split: wide → gather (always accept), narrow → resp xbar.
  // ============================================================
  narrow_resp_t [NumBanksPerTile-1:0] narrow_resp_to_xbar;
  logic         [NumBanksPerTile-1:0] narrow_resp_to_xbar_valid;
  logic         [NumBanksPerTile-1:0] narrow_resp_to_xbar_ready;

  for (genvar b = 0; b < NumBanksPerTile; b++) begin : gen_resp_split
    assign narrow_resp_to_xbar[b] = mst_resp_i[b];
    always_comb begin
      if (mst_resp_wide_i[b]) begin
        mst_resp_ready_o[b]          = 1'b1;   // gather buffer always accepts
        narrow_resp_to_xbar_valid[b] = 1'b0;
      end else begin
        mst_resp_ready_o[b]          = narrow_resp_to_xbar_ready[b];
        narrow_resp_to_xbar_valid[b] = mst_resp_valid_i[b];
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

  // ============================================================
  // Wide response mux (NumSuperbanks → 1).
  // ============================================================
  if (NumSuperbanks > 1) begin : gen_wide_resp_mux
    stream_xbar #(
      .NumInp   (NumSuperbanks ),
      .NumOut   (1             ),
      .payload_t(wide_resp_t   )
    ) i_wide_resp_mux (
      .clk_i  (clk_i                 ),
      .rst_ni (rst_ni                ),
      .flush_i(1'b0                  ),
      .rr_i   ('0                    ),
      .data_i (sb_wide_resp          ),
      .valid_i(sb_wide_resp_valid    ),
      .ready_o(sb_wide_resp_ready    ),
      .sel_i  ('0                    ),
      .data_o (slv_wide_resp_o       ),
      .valid_o(slv_wide_resp_valid_o ),
      .ready_i(slv_wide_resp_ready_i ),
      .idx_o  (/* unused */          )
    );
  end else begin : gen_wide_resp_bypass
    assign slv_wide_resp_o        = sb_wide_resp[0];
    assign slv_wide_resp_valid_o  = sb_wide_resp_valid[0];
    assign sb_wide_resp_ready[0]  = slv_wide_resp_ready_i;
  end

endmodule
