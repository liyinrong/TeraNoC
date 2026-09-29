// Copyright 2026 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

`include "mempool/mempool.svh"

// One wide-word ingress (the DMA, or one WP2 wide-plane port) of width
// WideDataWidth onto the tile's superbanks of NumBanksPerSB banks each. Every
// ingress shares the same superbank geometry, so the caller arbitrates each
// superbank over all ingresses at once.
//
//   req:  split the wide word into NumParts superbank-wide parts (1024 b onto
//         512 b superbanks: part p goes to superbank 2i+p), register each part
//         in a spill register (IngressSpill: the only cut on the wide path; it
//         also lets the parts progress independently), then demux each part to
//         its superbank. The caller grants a superbank atomically: all its banks
//         take the part in the same cycle.
//   resp: join each superbank's bank responses, park them per part in a
//         holding FIFO, and zip the parts back into one wide response. The
//         holding FIFO always has room (it never holds more than the part's
//         outstanding limit), so a part response leaves its banks without
//         waiting for its sibling: two wide ingresses served in opposite orders
//         by the two superbanks of a pair cannot deadlock.
//
// The split, holding FIFOs and zip exist only when NumParts > 1; the rotation
// only when SpmBankIdRemap is set. Wide accesses must be aligned to the wide
// word (asserted); the only rotation left is the bank-id remap offset, which
// keeps each word in the bank the narrow path would pick.
module mempool_tcdm_wide_superbank
  import cf_math_pkg::idx_width;
#(
  parameter int unsigned NumBanksPerTile = 16,
  parameter int unsigned NumBanksPerSB   = 16,
  parameter int unsigned NarrowDataWidth = 32,
  parameter int unsigned WideDataWidth   = 512,
  parameter bit          SpmBankIdRemap  = 0,
  parameter int unsigned RotFifoDepth    = 2,
  // Register each part at the ingress (the wide-path cut). Off for an ingress
  // whose throughput is bound by its round trip (the DMA: 2 outstanding beats).
  parameter bit          IngressSpill    = 1'b1,
  parameter type wide_req_t    = logic,
  parameter type wide_resp_t   = logic,
  parameter type narrow_req_t  = logic,
  parameter type narrow_resp_t = logic,
  parameter type group_id_t    = logic,
  parameter type ini_addr_t    = logic,
  // derived
  localparam int unsigned NumSuperbanks  = NumBanksPerTile / NumBanksPerSB,
  localparam int unsigned SbDataWidth    = NumBanksPerSB * NarrowDataWidth,
  localparam int unsigned NumParts       = WideDataWidth / SbDataWidth,
  localparam int unsigned NumGroups      = NumSuperbanks / NumParts,
  localparam int unsigned BankOffsetBits = $clog2(NumBanksPerTile),
  localparam int unsigned BankIdLoBits   = $clog2(NumBanksPerSB),
  localparam int unsigned PartBits       = $clog2(NumParts),
  localparam int unsigned GroupSelBits   = (NumGroups > 1) ? $clog2(NumGroups) : 1,
  localparam int unsigned NarrowBeWidth  = NarrowDataWidth / 8
) (
  input  logic                                          clk_i,
  input  logic                                          rst_ni,
  // Wide port (one ingress).
  input  wide_req_t                                     slv_wide_req_i,
  input  logic                                          slv_wide_req_valid_i,
  output logic                                          slv_wide_req_ready_o,
  output wide_resp_t                                    slv_wide_resp_o,
  output logic                                          slv_wide_resp_valid_o,
  input  logic                                          slv_wide_resp_ready_i,
  // Source routing carried through the access (self-routing, mirrors the narrow
  // path). Tie inputs to '0/local for an ingress whose response never leaves the
  // tile.
  input  ini_addr_t                                     slv_wide_ini_addr_i,
  input  group_id_t                                     slv_wide_src_group_i,
  output ini_addr_t                                     slv_wide_resp_ini_addr_o,
  output group_id_t                                     slv_wide_resp_src_group_o,
  // Replayed with the response so the caller can pick the return channel class
  // (read -> data response, write -> meta-only ack).
  output logic                                          slv_wide_resp_is_write_o,
  // Per-superbank request side (the atomic arbiter is in the caller): one valid
  // per superbank, one word per bank.
  output narrow_req_t [NumBanksPerTile-1:0]             wide_req_for_bank_o,
  output logic        [NumSuperbanks-1:0]               sb_req_valid_o,
  input  logic        [NumSuperbanks-1:0]               sb_req_ready_i,
  // Per-bank response side for THIS ingress (routed to us by the caller). The
  // FULL bank response is needed, not just its data: the reassembled wide
  // response has to carry meta_id/core_id back too (see the zip below).
  input  narrow_resp_t [NumBanksPerTile-1:0]            resp_i,
  input  logic [NumBanksPerTile-1:0]                    resp_valid_i,
  output logic [NumBanksPerTile-1:0]                    resp_ready_o
);

  if (NumBanksPerSB < 2)
    $fatal(1, "mempool_tcdm_wide_superbank: NumBanksPerSB (=%0d) must be >= 2.", NumBanksPerSB);
  if (NumParts < 1 || NumParts * SbDataWidth != WideDataWidth ||
      NumGroups * NumParts != NumSuperbanks)
    $fatal(1, "mempool_tcdm_wide_superbank: %0d b wide word does not tile %0d superbanks of %0d b.",
           WideDataWidth, NumSuperbanks, SbDataWidth);

  typedef logic [NumBanksPerSB-1:0][NarrowDataWidth-1:0] sb_data_t;
  typedef logic [NumBanksPerSB-1:0][NarrowBeWidth-1:0]   sb_be_t;
  typedef logic [BankIdLoBits-1:0]                       rot_t;

  // One superbank-wide part. hdr carries the address, the metadata and the
  // source routing; its own data/be fields are unused (tied to 0).
  typedef struct packed {
    narrow_req_t hdr;
    sb_data_t    data;
    sb_be_t      be;
  } part_req_t;

  // Queued per granted part, replayed when its response zips out.
  typedef struct packed {
    rot_t      rot;
    ini_addr_t ini_addr;
    group_id_t src_group_id;
    logic      is_write;
  } wide_ctx_t;

  // A joined superbank response: the raw (still rotated) bank words plus the
  // metadata of the superbank's lowest bank.
  typedef struct packed {
    narrow_resp_t meta;
    sb_data_t     data;
  } part_resp_t;

  // ============================================================
  // Split into parts, one spill register per part.
  // ============================================================
  logic [NumParts-1:0] part_in_valid, part_in_ready;

  if (NumParts > 1) begin : gen_split
    stream_fork #(.N_OUP(NumParts)) i_part_fork (
      .clk_i  (clk_i               ),
      .rst_ni (rst_ni              ),
      .valid_i(slv_wide_req_valid_i),
      .ready_o(slv_wide_req_ready_o),
      .valid_o(part_in_valid       ),
      .ready_i(part_in_ready       )
    );
  end else begin : gen_no_split
    assign part_in_valid        = slv_wide_req_valid_i;
    assign slv_wide_req_ready_o = part_in_ready;
  end

  part_req_t [NumParts-1:0] part_q;
  logic      [NumParts-1:0] part_valid, part_ready;
  rot_t      [NumParts-1:0] part_rot;
  // Per part: its words as they land on the superbank's banks.
  sb_data_t  [NumParts-1:0] part_bank_data;
  sb_be_t    [NumParts-1:0] part_bank_be;

  for (genvar p = 0; p < NumParts; p++) begin : gen_part
    part_req_t part_d;
    always_comb begin
      part_d = '0;
      part_d.hdr.wdata.meta_id = slv_wide_req_i.wdata.meta_id;
      part_d.hdr.wdata.core_id = slv_wide_req_i.wdata.core_id;
      part_d.hdr.wdata.amo     = slv_wide_req_i.wdata.amo;
      part_d.hdr.wen           = slv_wide_req_i.wen;
      part_d.hdr.ini_addr      = slv_wide_ini_addr_i;
      part_d.hdr.src_group_id  = slv_wide_src_group_i;
      part_d.data = slv_wide_req_i.wdata.data[p*SbDataWidth +: SbDataWidth];
      part_d.be   = slv_wide_req_i.be[p*SbDataWidth/8 +: SbDataWidth/8];
      // The part lands on the next superbank of the aligned group.
      part_d.hdr.tgt_addr     = slv_wide_req_i.tgt_addr | (p << BankIdLoBits);
    end

    spill_register #(
      .T     (part_req_t   ),
      .Bypass(!IngressSpill)
    ) i_part_spill (
      .clk_i  (clk_i           ),
      .rst_ni (rst_ni          ),
      .valid_i(part_in_valid[p]),
      .ready_o(part_in_ready[p]),
      .data_i (part_d          ),
      .valid_o(part_valid[p]   ),
      .ready_i(part_ready[p]   ),
      .data_o (part_q[p]       )
    );

    // Rotation = the bank-id remap offset (the row-id slice just above
    // bank_id), the same arithmetic as the narrow select. Aligned wide words
    // start at bank_id_lo 0, so no base rotation is needed.
    if (SpmBankIdRemap) begin : gen_rot
      assign part_rot[p] = part_q[p].hdr.tgt_addr[BankOffsetBits +: BankIdLoBits];
      always_comb begin
        int unsigned j;
        for (int unsigned b = 0; b < NumBanksPerSB; b++) begin
          // word j = (b - rot) mod NumBanksPerSB feeds bank port b.
          j = (b + NumBanksPerSB - part_rot[p]) % NumBanksPerSB;
          part_bank_data[p][b] = part_q[p].data[j];
          part_bank_be[p][b]   = part_q[p].be[j];
        end
      end
    end else begin : gen_no_rot
      assign part_rot[p]       = '0;
      assign part_bank_data[p] = part_q[p].data;
      assign part_bank_be[p]   = part_q[p].be;
    end
  end

`ifndef SYNTHESIS
  // The split and the rotation both assume a wide word never straddles a
  // superbank boundary.
  assert property (@(posedge clk_i) disable iff (!rst_ni)
      slv_wide_req_valid_i |-> slv_wide_req_i.tgt_addr[0 +: BankIdLoBits + PartBits] == '0)
    else $fatal(1, "[mempool_tcdm_wide_superbank] unaligned wide access, tgt_addr=%h",
                slv_wide_req_i.tgt_addr);
`endif

  // ============================================================
  // Per superbank s = g*NumParts + p: request, context and response join.
  // ============================================================
  logic      [NumSuperbanks-1:0] sb_sel_valid;
  wide_ctx_t [NumSuperbanks-1:0] sb_ctx;
  logic      [NumSuperbanks-1:0] sb_ctx_valid, sb_ctx_pop;
  part_resp_t [NumSuperbanks-1:0] sb_hold;
  logic      [NumSuperbanks-1:0] sb_hold_valid, sb_hold_ready;

  for (genvar p = 0; p < NumParts; p++) begin : gen_part_demux
    logic [NumGroups-1:0] dmx_valid, dmx_ready;
    if (NumGroups > 1) begin : gen_demux
      stream_demux #(.N_OUP(NumGroups)) i_group_demux (
        .inp_valid_i(part_valid[p]),
        .inp_ready_o(part_ready[p]),
        .oup_sel_i  (part_q[p].hdr.tgt_addr[BankIdLoBits + PartBits +: GroupSelBits]),
        .oup_valid_o(dmx_valid    ),
        .oup_ready_i(dmx_ready    )
      );
    end else begin : gen_no_demux
      assign dmx_valid     = part_valid[p];
      assign part_ready[p] = dmx_ready;
    end
    for (genvar g = 0; g < NumGroups; g++) begin : gen_group
      localparam int unsigned S = g*NumParts + p;
      assign sb_sel_valid[S] = dmx_valid[g];
      assign dmx_ready[g]    = sb_req_ready_i[S];
    end
  end

  for (genvar s = 0; s < NumSuperbanks; s++) begin : gen_sb
    localparam int unsigned P = s % NumParts;

    // Context FIFO: pushed on grant, popped when the response zips out. A part
    // is offered only while it has room, so it bounds the outstanding parts
    // per superbank to RotFifoDepth (and the holding FIFO below with it).
    wide_ctx_t ctx_in;
    logic      ctx_in_ready;
    assign ctx_in = '{rot:          part_rot[P],
                      ini_addr:     part_q[P].hdr.ini_addr,
                      src_group_id: part_q[P].hdr.src_group_id,
                      is_write:     part_q[P].hdr.wen};
    // Stable until granted: ctx_in_ready can only rise while the part waits.
    assign sb_req_valid_o[s] = sb_sel_valid[s] & ctx_in_ready;

    stream_fifo_optimal_wrap #(
      .Depth  (RotFifoDepth),
      .type_t (wide_ctx_t  )
    ) i_ctx_fifo (
      .clk_i (clk_i), .rst_ni(rst_ni), .flush_i(1'b0), .testmode_i(1'b0), .usage_o(),
      .data_i (ctx_in                             ),
      .valid_i(sb_req_valid_o[s] & sb_req_ready_i[s]),
      .ready_o(ctx_in_ready                       ),
      .data_o (sb_ctx[s]      ), .valid_o(sb_ctx_valid[s]), .ready_i(sb_ctx_pop[s])
    );

    // Each bank of the superbank gets its word plus the shared header. The
    // source routing rides along only for tracing: the response is routed by
    // the context FIFO, never by the bank metadata.
    for (genvar b = 0; b < NumBanksPerSB; b++) begin : gen_bank_req
      always_comb begin
        wide_req_for_bank_o[s*NumBanksPerSB + b]            = part_q[P].hdr;
        wide_req_for_bank_o[s*NumBanksPerSB + b].wdata.data = part_bank_data[P][b];
        wide_req_for_bank_o[s*NumBanksPerSB + b].be         = part_bank_be[P][b];
      end
    end

    // Join the superbank's bank responses.
    part_resp_t joined;
    logic       join_valid, join_ready;
    stream_join #(.N_INP(NumBanksPerSB)) i_join (
      .inp_valid_i(resp_valid_i[s*NumBanksPerSB +: NumBanksPerSB]),
      .inp_ready_o(resp_ready_o[s*NumBanksPerSB +: NumBanksPerSB]),
      .oup_valid_o(join_valid),
      .oup_ready_i(join_ready)
    );
    always_comb begin
      joined.meta = resp_i[s*NumBanksPerSB];
      for (int unsigned b = 0; b < NumBanksPerSB; b++)
        joined.data[b] = resp_i[s*NumBanksPerSB + b].rdata.data;
    end

    if (NumParts > 1) begin : gen_hold
      // Holding FIFO: the part waits here for its siblings. Fall-through, so an
      // uncontended access pays no extra cycle.
      stream_fifo #(
        .FALL_THROUGH(1'b1        ),
        .DEPTH       (RotFifoDepth),
        .T           (part_resp_t )
      ) i_hold_fifo (
        .clk_i (clk_i), .rst_ni(rst_ni), .flush_i(1'b0), .testmode_i(1'b0), .usage_o(),
        .data_i (joined          ), .valid_i(join_valid      ), .ready_o(join_ready      ),
        .data_o (sb_hold[s]      ), .valid_o(sb_hold_valid[s]), .ready_i(sb_hold_ready[s])
      );
`ifndef SYNTHESIS
      // Load-bearing for deadlock freedom: a part response must never wait at
      // the banks for its sibling.
      assert property (@(posedge clk_i) disable iff (!rst_ni) join_valid |-> join_ready)
        else $fatal(1, "[mempool_tcdm_wide_superbank] sb=%0d holding FIFO full", s);
`endif
    end else begin : gen_no_hold
      assign sb_hold[s]       = joined;
      assign sb_hold_valid[s] = join_valid;
      assign join_ready       = sb_hold_ready[s];
    end

`ifndef SYNTHESIS
    // The context must be queued before its response zips out; draining an
    // empty FIFO would replay a stale rotation and routing.
    assert property (@(posedge clk_i) disable iff (!rst_ni) sb_ctx_pop[s] |-> sb_ctx_valid[s])
      else $fatal(1, "[mempool_tcdm_wide_superbank] sb=%0d context FIFO drained while empty", s);
`endif
  end

  // ============================================================
  // Per group: zip the parts into one wide response, then mux the groups.
  // Groups may answer out of order; responses carry their own meta_id/routing.
  // ============================================================
  typedef struct packed {
    wide_resp_t resp;
    ini_addr_t  ini_addr;
    group_id_t  src_group_id;
    logic       is_write;
  } grp_out_t;
  grp_out_t [NumGroups-1:0] grp_out;
  logic     [NumGroups-1:0] grp_valid, grp_ready;
  grp_out_t                 mux_out;

  for (genvar g = 0; g < NumGroups; g++) begin : gen_zip
    logic [NumParts-1:0] zip_in_valid;
    for (genvar p = 0; p < NumParts; p++) begin : gen_zip_in
      localparam int unsigned S = g*NumParts + p;
      assign zip_in_valid[p] = sb_hold_valid[S];
      assign sb_hold_ready[S] = grp_valid[g] & grp_ready[g];
      assign sb_ctx_pop[S]    = grp_valid[g] & grp_ready[g];
    end
    assign grp_valid[g] = &zip_in_valid;

    always_comb begin
      int unsigned s, j;
      grp_out[g] = '0;
      for (int unsigned p = 0; p < NumParts; p++) begin
        s = g*NumParts + p;
        for (int unsigned b = 0; b < NumBanksPerSB; b++) begin
          // Bank port b carries word j = (b - rot) mod NumBanksPerSB.
          j = SpmBankIdRemap ? (b + NumBanksPerSB - sb_ctx[s].rot) % NumBanksPerSB : b;
          grp_out[g].resp.rdata.data[(p*NumBanksPerSB + j)*NarrowDataWidth +: NarrowDataWidth]
              = sb_hold[s].data[b];
        end
      end
      // Metadata from the lowest bank -- all banks of a wide chunk share it.
      // LOAD-BEARING: RedMulE's hci_core_rob/hci_core_mux_ooo route a response
      // by meta_id. Leaving it at 0 sends every response to stream 0 slot 0,
      // the ROBs fill, and the engine stops issuing (the rm_gemm wedge).
      grp_out[g].resp.rdata.meta_id = sb_hold[g*NumParts].meta.rdata.meta_id;
      grp_out[g].resp.rdata.core_id = sb_hold[g*NumParts].meta.rdata.core_id;
      grp_out[g].resp.rdata.amo     = sb_hold[g*NumParts].meta.rdata.amo;
      grp_out[g].ini_addr           = sb_ctx[g*NumParts].ini_addr;
      grp_out[g].src_group_id       = sb_ctx[g*NumParts].src_group_id;
      grp_out[g].is_write           = sb_ctx[g*NumParts].is_write;
    end
  end

  if (NumGroups > 1) begin : gen_resp_mux
    stream_xbar #(
      .NumInp   (NumGroups),
      .NumOut   (1        ),
      .payload_t(grp_out_t)
    ) i_resp_mux (
      .clk_i, .rst_ni, .flush_i(1'b0), .rr_i('0),
      .data_i (grp_out              ),
      .valid_i(grp_valid            ),
      .ready_o(grp_ready            ),
      .sel_i  ('0                   ),
      .data_o (mux_out              ),
      .valid_o(slv_wide_resp_valid_o),
      .ready_i(slv_wide_resp_ready_i),
      .idx_o  (/* unused */         )
    );
  end else begin : gen_resp_bypass
    assign mux_out               = grp_out[0];
    assign slv_wide_resp_valid_o = grp_valid[0];
    assign grp_ready[0]          = slv_wide_resp_ready_i;
  end

  assign slv_wide_resp_o           = mux_out.resp;
  assign slv_wide_resp_ini_addr_o  = mux_out.ini_addr;
  assign slv_wide_resp_src_group_o = mux_out.src_group_id;
  assign slv_wide_resp_is_write_o  = mux_out.is_write;

endmodule
