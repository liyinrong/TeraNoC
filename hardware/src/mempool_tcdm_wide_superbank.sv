// Copyright 2026 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

`include "mempool/mempool.svh"

// One wide TCDM "class" (e.g. DMA, or the RedMulE engine). Handles a single wide
// port of width WideDataWidth: demux to per-superbank streams, scatter-with-
// completion across each superbank's banks (each bank served independently, so
// the caller's per-bank arbiter may prefer narrow traffic without livelocking),
// pipelined response gather (a wide req is accepted as soon as it is fully
// scattered; rotation+metadata are queued in a per-superbank context FIFO and
// each bank's response is consumed immediately into a per-bank response FIFO so
// early-served banks are freed), and reassembly of the wide response.
//
// The per-bank ARBITER lives in the caller (it is shared across wide classes and
// narrow): this module exposes, per bank, its rotated request word
// (wide_req_for_bank_o) and whether it still needs that bank (wide_present_o),
// and consumes the caller's grant (wide_served_i) and the bank's response for
// THIS class (resp_valid_i/resp_data_i, gated by resp_ready_o).
module mempool_tcdm_wide_superbank
  import cf_math_pkg::idx_width;
#(
  parameter int unsigned NumBanksPerTile = 16,
  parameter int unsigned NarrowDataWidth = 32,
  parameter int unsigned WideDataWidth   = 512,
  parameter bit          SpmBankIdRemap  = 0,
  parameter int unsigned WideOutstanding = 3,
  parameter type wide_req_t   = logic,
  parameter type wide_resp_t  = logic,
  parameter type narrow_req_t = logic,
  parameter type group_id_t   = logic,
  // derived
  localparam int unsigned NumBanksPerSB  = WideDataWidth / NarrowDataWidth,
  localparam int unsigned NumSuperbanks  = NumBanksPerTile / NumBanksPerSB,
  localparam int unsigned BankOffsetBits = $clog2(NumBanksPerTile),
  localparam int unsigned BankIdLoBits   = $clog2(NumBanksPerSB),
  localparam int unsigned NarrowBeWidth  = NarrowDataWidth / 8,
  localparam int unsigned RotBits        = (NumBanksPerSB > 1) ? $clog2(NumBanksPerSB) : 1,
  localparam int unsigned BankRemapShiftAmt = BankIdLoBits,
  localparam int unsigned WideCtxDepth   = WideOutstanding,
  localparam int unsigned WideRespDepth  = WideOutstanding + 1
) (
  input  logic                                          clk_i,
  input  logic                                          rst_ni,
  input  group_id_t                                     group_id_i,
  // Wide port (one class).
  input  wide_req_t                                     slv_wide_req_i,
  input  logic                                          slv_wide_req_valid_i,
  output logic                                          slv_wide_req_ready_o,
  output wide_resp_t                                    slv_wide_resp_o,
  output logic                                          slv_wide_resp_valid_o,
  input  logic                                          slv_wide_resp_ready_i,
  // Per-bank arbitration interface (arbiter is in the caller).
  output narrow_req_t [NumBanksPerTile-1:0]             wide_req_for_bank_o,
  output logic        [NumBanksPerTile-1:0]             wide_present_o,
  input  logic        [NumBanksPerTile-1:0]             wide_served_i,
  // Per-bank response for THIS class (routed to us by the caller).
  input  logic [NumBanksPerTile-1:0][NarrowDataWidth-1:0] resp_data_i,
  input  logic [NumBanksPerTile-1:0]                      resp_valid_i,
  output logic [NumBanksPerTile-1:0]                      resp_ready_o
);

  if (NumBanksPerSB < 2)
    $fatal(1, "mempool_tcdm_wide_superbank: NumBanksPerSB (=%0d) must be >= 2.", NumBanksPerSB);

  typedef struct packed {
    logic [RotBits-1:0] rot;
    logic [$bits(slv_wide_req_i.wdata.meta_id)-1:0] meta_id;
    logic [$bits(slv_wide_req_i.wdata.core_id)-1:0] core_id;
    logic [$bits(slv_wide_req_i.wdata.amo)-1:0]     amo;
  } wide_ctx_t;

  // Demux the single wide req to its target superbank.
  wide_req_t [NumSuperbanks-1:0] wide_to_sb;
  logic      [NumSuperbanks-1:0] wide_to_sb_valid;
  logic      [NumSuperbanks-1:0] wide_to_sb_ready;

  if (NumSuperbanks > 1) begin : gen_wide_req_demux
    logic [NumSuperbanks-1:0][idx_width(NumSuperbanks)-1:0] sb_sel;
    stream_xbar #(
      .NumInp   (1            ),
      .NumOut   (NumSuperbanks),
      .payload_t(wide_req_t   )
    ) i_wide_req_demux (
      .clk_i, .rst_ni, .flush_i(1'b0), .rr_i('0),
      .data_i (slv_wide_req_i      ),
      .valid_i(slv_wide_req_valid_i),
      .ready_o(slv_wide_req_ready_o),
      // Superbank select = upper bits of bank_id (unchanged by within-SB remap).
      .sel_i  (slv_wide_req_i.tgt_addr[BankIdLoBits +: idx_width(NumSuperbanks)]),
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

  wide_resp_t [NumSuperbanks-1:0] sb_wide_resp;
  logic       [NumSuperbanks-1:0] sb_wide_resp_valid;
  logic       [NumSuperbanks-1:0] sb_wide_resp_ready;

  // Per-superbank scatter mask of the currently-scattering req.
  logic [NumSuperbanks-1:0][NumBanksPerSB-1:0] served_q, served_d;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) served_q <= '0; else served_q <= served_d;
  end

  // Per-bank response FIFO (consume each response immediately, free the bank).
  logic [NumBanksPerTile-1:0]                      rfifo_pop_valid, rfifo_pop_ready;
  logic [NumBanksPerTile-1:0][NarrowDataWidth-1:0] rfifo_pop_data;
  for (genvar b = 0; b < NumBanksPerTile; b++) begin : gen_resp_fifo
    stream_fifo_optimal_wrap #(
      .Depth  (WideRespDepth              ),
      .type_t (logic [NarrowDataWidth-1:0])
    ) i_resp_fifo (
      .clk_i (clk_i), .rst_ni(rst_ni), .flush_i(1'b0), .testmode_i(1'b0), .usage_o(),
      .data_i (resp_data_i[b]  ), .valid_i(resp_valid_i[b]), .ready_o(resp_ready_o[b]),
      .data_o (rfifo_pop_data[b]), .valid_o(rfifo_pop_valid[b]), .ready_i(rfifo_pop_ready[b])
    );
  end

  // Per-superbank context FIFO: rotation + metadata, queued on accept.
  wide_ctx_t [NumSuperbanks-1:0] cfifo_push_data, cfifo_pop_data;
  logic      [NumSuperbanks-1:0] cfifo_push_valid, cfifo_push_ready;
  logic      [NumSuperbanks-1:0] cfifo_pop_valid,  cfifo_pop_ready;
  for (genvar d = 0; d < NumSuperbanks; d++) begin : gen_ctx_fifo
    stream_fifo_optimal_wrap #(
      .Depth (WideCtxDepth), .type_t (wide_ctx_t)
    ) i_ctx_fifo (
      .clk_i (clk_i), .rst_ni(rst_ni), .flush_i(1'b0), .testmode_i(1'b0), .usage_o(),
      .data_i (cfifo_push_data[d]), .valid_i(cfifo_push_valid[d]), .ready_o(cfifo_push_ready[d]),
      .data_o (cfifo_pop_data[d] ), .valid_o(cfifo_pop_valid[d] ), .ready_i(cfifo_pop_ready[d])
    );
  end

  for (genvar d = 0; d < NumSuperbanks; d++) begin : gen_sb
    // Rotation of the currently-scattering req.
    logic [RotBits-1:0] req_rot;
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

    // Compose each bank's wide req (rotated word slice); present until served.
    for (genvar b = 0; b < NumBanksPerSB; b++) begin : gen_sb_bank_req
      localparam int unsigned global_bank = d*NumBanksPerSB + b;
      always_comb begin
        automatic int unsigned j = (b + NumBanksPerSB - req_rot) % NumBanksPerSB;
        wide_req_for_bank_o[global_bank] = '{
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
      assign wide_present_o[global_bank] = wide_to_sb_valid[d] & ~served_q[d][b];
    end

    // Scatter completion: accept when fully scattered + context-FIFO room.
    logic [NumBanksPerSB-1:0] served_now, served_next;
    for (genvar b = 0; b < NumBanksPerSB; b++) begin : gen_sb_served
      assign served_now[b] = wide_served_i[d*NumBanksPerSB + b];
    end
    assign served_next = served_q[d] | served_now;

    logic all_served, accept;
    assign all_served          = &served_next;
    assign accept              = all_served & wide_to_sb_valid[d] & cfifo_push_ready[d];
    assign wide_to_sb_ready[d] = accept;
    assign served_d[d]         = accept ? '0 : served_next;

    assign cfifo_push_valid[d] = all_served & wide_to_sb_valid[d];
    assign cfifo_push_data[d]  = '{rot:     req_rot,
                                   meta_id: wide_to_sb[d].wdata.meta_id,
                                   core_id: wide_to_sb[d].wdata.core_id,
                                   amo:     wide_to_sb[d].wdata.amo};

    // Emit: join the per-bank response FIFO heads with the context FIFO head.
    logic [NumBanksPerSB-1:0] sb_rfifo_valid;
    for (genvar b = 0; b < NumBanksPerSB; b++) begin : gen_sb_rvalid
      assign sb_rfifo_valid[b] = rfifo_pop_valid[d*NumBanksPerSB + b];
    end
    logic emit;
    assign sb_wide_resp_valid[d] = cfifo_pop_valid[d] & (&sb_rfifo_valid);
    assign emit                  = sb_wide_resp_valid[d] & sb_wide_resp_ready[d];
    assign cfifo_pop_ready[d]    = emit;
    for (genvar b = 0; b < NumBanksPerSB; b++) begin : gen_sb_rpop
      assign rfifo_pop_ready[d*NumBanksPerSB + b] = emit;
    end

    always_comb begin
      sb_wide_resp[d] = '{rdata: '{default: '0}};
      for (int b = 0; b < NumBanksPerSB; b++) begin
        automatic int unsigned j = (b + NumBanksPerSB - cfifo_pop_data[d].rot) % NumBanksPerSB;
        sb_wide_resp[d].rdata.data[j*NarrowDataWidth +: NarrowDataWidth]
            = rfifo_pop_data[d*NumBanksPerSB + b];
      end
      sb_wide_resp[d].rdata.meta_id = cfifo_pop_data[d].meta_id;
      sb_wide_resp[d].rdata.core_id = cfifo_pop_data[d].core_id;
      sb_wide_resp[d].rdata.amo     = cfifo_pop_data[d].amo;
    end

`ifndef SYNTHESIS
    int unsigned wd_cnt_q;
    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        wd_cnt_q <= 0;
      end else if (wide_to_sb_valid[d] && !accept) begin
        wd_cnt_q <= wd_cnt_q + 1;
        if (wd_cnt_q == 100000)
          $error("[mempool_tcdm_wide_superbank] sb=%0d wide scatter stuck (>100k cyc) served=%b",
                 d, served_q[d]);
      end else begin
        wd_cnt_q <= 0;
      end
    end
`endif
  end

  // Wide response mux (NumSuperbanks -> 1).
  if (NumSuperbanks > 1) begin : gen_wide_resp_mux
    stream_xbar #(
      .NumInp   (NumSuperbanks ),
      .NumOut   (1             ),
      .payload_t(wide_resp_t   )
    ) i_wide_resp_mux (
      .clk_i, .rst_ni, .flush_i(1'b0), .rr_i('0),
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
    assign slv_wide_resp_o       = sb_wide_resp[0];
    assign slv_wide_resp_valid_o = sb_wide_resp_valid[0];
    assign sb_wide_resp_ready[0] = slv_wide_resp_ready_i;
  end

endmodule
