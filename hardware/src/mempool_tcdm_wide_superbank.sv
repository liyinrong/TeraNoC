// Copyright 2026 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

`include "mempool/mempool.svh"

// One wide TCDM "class" (e.g. the DMA, or the WP2 RedMulE engine) on a single
// wide port of width WideDataWidth. This is the original fork/join wide
// datapath, factored out so the bank interconnect can instantiate it once per
// class: each class keeps its OWN superbank geometry (NumBanksPerSB =
// WideDataWidth/NarrowDataWidth), which need not match the other's.
//
//   req:  demux the wide req to its superbank, then stream_fork it across that
//         superbank's banks with a rotation of bank_id_lo (so the wide chunk
//         stays coherent with where the narrow path would place each word).
//   resp: stream_join the superbank's bank responses and undo the rotation.
//         The per-request rotation (plus the source routing carried for the WP2
//         wide plane) is queued in a small per-superbank FIFO, so several wide
//         requests can be outstanding and still reassemble correctly.
//
// The per-bank ARBITER lives in the caller (it is shared across the wide classes
// and narrow): this module exposes each bank's rotated request word plus the
// fork handshake, and consumes the bank responses the caller routes to it.
module mempool_tcdm_wide_superbank
  import cf_math_pkg::idx_width;
#(
  parameter int unsigned NumBanksPerTile = 16,
  parameter int unsigned NarrowDataWidth = 32,
  parameter int unsigned WideDataWidth   = 512,
  parameter bit          SpmBankIdRemap  = 0,
  parameter int unsigned RotFifoDepth    = 2,
  parameter type wide_req_t    = logic,
  parameter type wide_resp_t   = logic,
  parameter type narrow_req_t  = logic,
  parameter type narrow_resp_t = logic,
  parameter type group_id_t    = logic,
  parameter type ini_addr_t    = logic,
  // derived
  localparam int unsigned NumBanksPerSB  = WideDataWidth / NarrowDataWidth,
  localparam int unsigned NumSuperbanks  = NumBanksPerTile / NumBanksPerSB,
  localparam int unsigned BankOffsetBits = $clog2(NumBanksPerTile),
  localparam int unsigned BankIdLoBits   = $clog2(NumBanksPerSB),
  localparam int unsigned SBSelBits      = (NumSuperbanks > 1) ? $clog2(NumSuperbanks) : 1,
  localparam int unsigned NarrowBeWidth  = NarrowDataWidth / 8,
  localparam int unsigned RotBits        = (NumBanksPerSB > 1) ? $clog2(NumBanksPerSB) : 1,
  localparam int unsigned BankRemapShiftAmt = BankIdLoBits
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
  // Source routing carried through the access (self-routing, mirrors the narrow
  // path). Tie inputs to '0/local for a class whose response never leaves the tile.
  input  ini_addr_t                                     slv_wide_ini_addr_i,
  input  group_id_t                                     slv_wide_src_group_i,
  output ini_addr_t                                     slv_wide_resp_ini_addr_o,
  output group_id_t                                     slv_wide_resp_src_group_o,
  // Replayed with the response so the caller can pick the return channel class
  // (read -> data response, write -> meta-only ack).
  output logic                                          slv_wide_resp_is_write_o,
  // Per-bank request side (the arbiter is in the caller).
  output narrow_req_t [NumBanksPerTile-1:0]             wide_req_for_bank_o,
  output logic        [NumBanksPerTile-1:0]             wide_fork_valid_o,
  input  logic        [NumBanksPerTile-1:0]             wide_fork_ready_i,
  // Per-bank response side for THIS class (routed to us by the caller). The FULL
  // bank response is needed, not just its data: the reassembled wide response has
  // to carry meta_id/core_id back too (see the join below).
  input  narrow_resp_t [NumBanksPerTile-1:0]              resp_i,
  input  logic [NumBanksPerTile-1:0]                      resp_valid_i,
  output logic [NumBanksPerTile-1:0]                      resp_ready_o
);

  if (NumBanksPerSB < 2)
    $fatal(1, "mempool_tcdm_wide_superbank: NumBanksPerSB (=%0d) must be >= 2.", NumBanksPerSB);

  // Queued per request: the rotation plus the source routing, replayed when the
  // response joins out.
  typedef struct packed {
    logic [RotBits-1:0] rot;
    ini_addr_t          ini_addr;
    group_id_t          src_group_id;
    logic               is_write;
  } wide_ctx_t;

  // ============================================================
  // Demux the single wide req to its target superbank.
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
      .clk_i, .rst_ni, .flush_i(1'b0), .rr_i('0),
      .data_i (slv_wide_req_i      ),
      .valid_i(slv_wide_req_valid_i),
      .ready_o(slv_wide_req_ready_o),
      // Superbank select = upper bits of bank_id (unchanged by within-SB remap).
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

  wide_resp_t [NumSuperbanks-1:0] sb_wide_resp;
  logic       [NumSuperbanks-1:0] sb_wide_resp_valid;
  logic       [NumSuperbanks-1:0] sb_wide_resp_ready;
  wide_ctx_t  [NumSuperbanks-1:0] sb_ctx;

  for (genvar d = 0; d < NumSuperbanks; d++) begin : gen_sb
    // Rotation amount = (remapped) bank_id_lo of the wide req's tgt_addr. Same
    // arithmetic as the narrow xbar select, so the rotations match.
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

    // Context FIFO: enqueued when the banks accept the req, dequeued when its
    // response joins out. The fork only fires when the FIFO has room, so it can
    // never overflow; RotFifoDepth is the knob for more outstanding wide reqs.
    wide_ctx_t ctx_in, ctx_out;
    logic      ctx_in_valid,  ctx_in_ready;
    logic      ctx_out_valid, ctx_out_ready;

    assign ctx_in = '{rot:          req_rot,
                      ini_addr:     slv_wide_ini_addr_i,
                      src_group_id: slv_wide_src_group_i,
                      is_write:     wide_to_sb[d].wen};

    // Fork the wide req across the superbank's banks, gated on FIFO room.
    logic [NumBanksPerSB-1:0] sb_fork_valid, sb_fork_ready;
    logic                     wide_acc_ready;
    stream_fork #(.N_OUP(NumBanksPerSB)) i_wide_fork (
      .clk_i  (clk_i                              ),
      .rst_ni (rst_ni                             ),
      .valid_i(wide_to_sb_valid[d] & ctx_in_ready ),
      .ready_o(wide_acc_ready                     ),
      .valid_o(sb_fork_valid                      ),
      .ready_i(sb_fork_ready                      )
    );
    assign wide_to_sb_ready[d] = wide_acc_ready & ctx_in_ready;
    // Enqueue when the banks accept; the FIFO masks the write when full itself.
    assign ctx_in_valid = wide_to_sb_valid[d] & wide_acc_ready;

    stream_fifo_optimal_wrap #(
      .Depth  (RotFifoDepth),
      .type_t (wide_ctx_t  )
    ) i_ctx_fifo (
      .clk_i (clk_i), .rst_ni(rst_ni), .flush_i(1'b0), .testmode_i(1'b0), .usage_o(),
      .data_i (ctx_in       ), .valid_i(ctx_in_valid ), .ready_o(ctx_in_ready ),
      .data_o (ctx_out      ), .valid_o(ctx_out_valid), .ready_i(ctx_out_ready)
    );
    assign sb_ctx[d] = ctx_out;

    // Compose each bank's wide req: rotated word slice + shared address.
    for (genvar b = 0; b < NumBanksPerSB; b++) begin : gen_sb_bank_req
      localparam int unsigned global_bank = d*NumBanksPerSB + b;
      always_comb begin
        // word j = (b - req_rot) mod NumBanksPerSB feeds bank port b.
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
      assign wide_fork_valid_o[global_bank] = sb_fork_valid[b];
      assign sb_fork_ready[b]               = wide_fork_ready_i[global_bank];
    end

    // Join the superbank's responses; the inverse rotation comes from the FIFO head.
    logic [NumBanksPerSB-1:0] sb_join_valid, sb_join_ready;
    for (genvar b = 0; b < NumBanksPerSB; b++) begin : gen_sb_bank_resp
      localparam int unsigned global_bank = d*NumBanksPerSB + b;
      assign sb_join_valid[b]           = resp_valid_i[global_bank];
      assign resp_ready_o[global_bank]  = sb_join_ready[b];
    end
    stream_join #(.N_INP(NumBanksPerSB)) i_wide_join (
      .inp_valid_i(sb_join_valid        ),
      .inp_ready_o(sb_join_ready        ),
      .oup_valid_o(sb_wide_resp_valid[d]),
      .oup_ready_i(sb_wide_resp_ready[d])
    );
    assign ctx_out_ready = sb_wide_resp_valid[d] & sb_wide_resp_ready[d];

    // The context must be queued before its response joins out; draining an empty
    // FIFO would replay a stale rotation and corrupt the reassembled rdata.
`ifndef SYNTHESIS
    assert property (@(posedge clk_i) disable iff (!rst_ni)
        ctx_out_ready |-> ctx_out_valid)
      else $fatal(1, "[mempool_tcdm_wide_superbank] sb=%0d context FIFO drained while empty", d);
`endif

    // Reassemble the wide rdata with the inverse rotation, then carry metadata.
    always_comb begin
      sb_wide_resp[d] = '0;
      for (int b = 0; b < NumBanksPerSB; b++) begin
        // Bank port b carries word j = (b - rot) mod NumBanksPerSB.
        automatic int unsigned j = (b + NumBanksPerSB - ctx_out.rot) % NumBanksPerSB;
        sb_wide_resp[d].rdata.data[j*NarrowDataWidth +: NarrowDataWidth]
            = resp_i[d*NumBanksPerSB + b].rdata.data;
      end
      // Metadata from the lowest bank -- all banks of a wide chunk share it.
      // LOAD-BEARING: RedMulE's hci_core_rob/hci_core_mux_ooo route a response by
      // meta_id. Leaving it at 0 sends every response to stream 0 slot 0, the ROBs
      // fill, and the engine stops issuing -- which is exactly the rm_gemm wedge.
      sb_wide_resp[d].rdata.meta_id = resp_i[d*NumBanksPerSB].rdata.meta_id;
      sb_wide_resp[d].rdata.core_id = resp_i[d*NumBanksPerSB].rdata.core_id;
      sb_wide_resp[d].rdata.amo     = resp_i[d*NumBanksPerSB].rdata.amo;
    end
  end

  // ============================================================
  // Wide response mux (NumSuperbanks -> 1), carrying the source routing so the
  // response self-routes at the caller (mirrors the narrow slave-resp path).
  // ============================================================
  typedef struct packed {
    wide_resp_t resp;
    ini_addr_t  ini_addr;
    group_id_t  src_group_id;
    logic       is_write;
  } sb_out_t;
  sb_out_t [NumSuperbanks-1:0] sb_out;
  sb_out_t                     mux_out;
  for (genvar d = 0; d < NumSuperbanks; d++) begin : gen_sb_out
    assign sb_out[d] = '{resp:         sb_wide_resp[d],
                         ini_addr:     sb_ctx[d].ini_addr,
                         src_group_id: sb_ctx[d].src_group_id,
                         is_write:     sb_ctx[d].is_write};
  end

  if (NumSuperbanks > 1) begin : gen_wide_resp_mux
    stream_xbar #(
      .NumInp   (NumSuperbanks),
      .NumOut   (1            ),
      .payload_t(sb_out_t     )
    ) i_wide_resp_mux (
      .clk_i, .rst_ni, .flush_i(1'b0), .rr_i('0),
      .data_i (sb_out                ),
      .valid_i(sb_wide_resp_valid    ),
      .ready_o(sb_wide_resp_ready    ),
      .sel_i  ('0                    ),
      .data_o (mux_out               ),
      .valid_o(slv_wide_resp_valid_o ),
      .ready_i(slv_wide_resp_ready_i ),
      .idx_o  (/* unused */          )
    );
  end else begin : gen_wide_resp_bypass
    assign mux_out               = sb_out[0];
    assign slv_wide_resp_valid_o = sb_wide_resp_valid[0];
    assign sb_wide_resp_ready[0] = slv_wide_resp_ready_i;
  end

  assign slv_wide_resp_o           = mux_out.resp;
  assign slv_wide_resp_ini_addr_o  = mux_out.ini_addr;
  assign slv_wide_resp_src_group_o = mux_out.src_group_id;
  assign slv_wide_resp_is_write_o  = mux_out.is_write;

endmodule
