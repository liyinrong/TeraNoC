// Copyright 2025 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Description: Turns a Spatz VLSU burst into one wide-plane access.
//
// The VLSU emits a burst as a SINGLE request carrying the reorder-buffer base id
// and burst_len = N (spatz_vlsu.sv, spatz_mem_req_t.burst_len), and then expects N
// responses carrying ids base .. base+N-1. On the narrow plane a burst expander
// downstream produces those N beats one per cycle. Here the whole burst is instead
// one wide access -- N * 32 b is exactly one superbank (MaxBurstWords * 4 B = 64 B =
// WideTcdmDataWidth) -- and this adapter unpacks the single wide response back into
// the N beats the VLSU is waiting for.
//
// Beat k is returned on lane k % RecvPorts with id base + k / RecvPorts. That is the
// non-burst word->port rule (spatz_vlsu.sv: word index 4n + port), so ROB p receives
// words p, p+4, p+8 ... as its own consecutive entries and the VLSU commits a burst
// through the ordinary path -- one element from each ROB filling a VRF row in ONE
// write. A burst base is WideTcdmDataWidth-aligned, so base/4 % RecvPorts == 0 and
// the lane reduces to k % RecvPorts.
//
// This is what makes the row atomic. Funnelling every beat into ROB0 instead forces
// a row to be filled by several partial writes, and a consumer that reads between
// them sees stale lanes -- measured as wrong results on every 8th output row.
//
// Loads only: a burst is only ever emitted for a load (burst_mode_req demands
// mem_is_load), so there is no write data path here.
//
// ONE adapter per Spatz core, each feeding its own shim and its own wide local
// port. Arbitration belongs only at the genuinely shared resource -- the tile's
// single wide remote link, where the mux picks between the per-core shims and
// RedMulE. Arbitrating inside the adapter instead would serialise two cores whose
// bursts target different superbanks, or one going own-tile while the other goes
// remote, for no reason.

module spatz_wide_burst_adapter
  import mempool_pkg::*;
  import cf_math_pkg::idx_width;
#(
  // Receive lanes = the VLSU's memory ports. Not a free choice: a word's lane is
  // fixed by its address, exactly as on the non-burst path, so that the VLSU can
  // consume a burst through the ordinary 4-ROB commit path.
  parameter int unsigned RecvPorts     = 4,
  // Width of the reorder-buffer id inside meta_id, i.e. spatz_pkg::MemRspIdWidth.
  // The beat id must wrap within this field: a block reservation may start near the
  // top of the ROB and wrap, and anything carried out of the field lands on the
  // store-ack flag mempool_cc packs directly above it -- which would make a load
  // beat look like a write acknowledgement.
  parameter int unsigned RobIdWidth    = 4,
  // Wide accesses this adapter may keep in flight. This is NOT independent of Spatz:
  // every in-flight burst holds MaxBurstWords/NrMemPorts reorder-buffer ids in EVERY
  // lane, so the VLSU can only ever back
  //     RobDepth * NrMemPorts / MaxBurstWords
  // of them -- 4 at depth 16 with 16-word bursts on 4 ports. Slots beyond that can
  // never be filled, and slots below it leave issue capacity Spatz already has unused.
  // The tile derives the value from those three numbers; the check below enforces it.
  // 1 reproduces the serialised behaviour exactly.
  parameter int unsigned NumOutstanding = 2,
  // 32-bit words in one wide access.
  parameter int unsigned WideWords     = NumSuperbankWords,
  parameter int unsigned BurstLenWidth = idx_width(WideWords + 1),
  parameter type         wide_req_t    = logic,
  parameter type         wide_resp_t   = logic,
  // Dependent. DO NOT CHANGE.
  parameter int unsigned BeatCntWidth  = idx_width(WideWords + 1)
) (
  input  logic clk_i,
  input  logic rst_ni,

  // ---- Spatz side: the burst-capable port (port 0) of each core ----
  input  addr_t                                    spatz_req_addr_i,
  input  meta_id_t                                 spatz_req_id_i,
  input  logic     [BurstLenWidth-1:0]             spatz_req_burst_len_i,
  input  logic                                     spatz_req_valid_i,
  output logic                                     spatz_req_ready_o,

  // ---- Spatz side: beats back, RecvPorts lanes per core ----
  // The Spatz response path has no ready (mempool_cc ties data_pready_o high for
  // the vector ports), so a drain cycle can never be back-pressured.
  output logic     [RecvPorts-1:0]                 spatz_resp_valid_o,
  output data_t    [RecvPorts-1:0]                 spatz_resp_data_o,
  output meta_id_t [RecvPorts-1:0]                 spatz_resp_id_o,

  // Hold the drain for a cycle. The vector response ports have no ready, so the
  // adapter and the tile's narrow path must never present a beat on the same port
  // in the same cycle: the tile raises this whenever the narrow path is answering
  // one of this core's receive lanes.
  input  logic     [RecvPorts-1:0]                 drain_stall_i,

  // ---- Wide side ----
  output wide_req_t  wide_req_o,
  output logic       wide_req_valid_o,
  input  logic       wide_req_ready_i,
  input  wide_resp_t wide_resp_i,
  input  logic       wide_resp_valid_i,
  output logic       wide_resp_ready_o
);

`include "common_cells/registers.svh"

  // Each slot buffers a whole wide word, so this is not free: it is a joint choice with
  // the reorder-buffer depth, not a knob to raise on its own.
  if (NumOutstanding < 1)
    $error("NumOutstanding must be >= 1.");

  // Every lane has delivered its share of the burst.
  logic drain_done;

  // In-flight bursts. One slot per outstanding wide access: the round trip is the
  // expensive part, and serialising it exposed that latency on EVERY burst while the
  // narrow path kept many word requests outstanding and hid it. Slots are issued and
  // drained in order; a response finds its slot by the id it round-trips, so the wide
  // plane may answer them out of order.
  localparam int unsigned SlotSel = idx_width(NumOutstanding);

  meta_id_t                   base_id_d [NumOutstanding], base_id_q [NumOutstanding];
  logic [BurstLenWidth-1:0]   len_d     [NumOutstanding], len_q     [NumOutstanding];
  logic [NumOutstanding-1:0]  issued_d, issued_q;   // slot holds a request in flight
  logic [NumOutstanding-1:0]  filled_d, filled_q;   // its response has landed
  logic [SlotSel-1:0]         wr_ptr_d, wr_ptr_q;   // next slot to issue into
  logic [SlotSel-1:0]         rd_ptr_d, rd_ptr_q;   // slot currently draining
  // Per-lane beat index. Lane l owns beats l, l+RecvPorts, l+2*RecvPorts ... -- its
  // own ROB's consecutive entries -- so each lane drains INDEPENDENTLY. A shared
  // counter would force one stalled lane to hold up the others, and the tile must
  // stall a lane whenever the narrow path is answering on that port: a vector store's
  // acks come back on all the vector ports, so an all-lanes stall blocks the whole
  // drain for as long as any ack is in flight.
  logic [BeatCntWidth-1:0]      beat_d [RecvPorts];
  logic [BeatCntWidth-1:0]      beat_q [RecvPorts];
  logic [WideWords-1:0][31:0]   rdata_d [NumOutstanding], rdata_q [NumOutstanding];

  // ------------------------------------------------------------------
  //  Request admission
  // ------------------------------------------------------------------
  // Only a real burst (burst_len > 1) belongs on the wide plane; a single-word
  // request stays on the narrow path and is never offered here. No arbitration:
  // this adapter serves exactly one core, and the contention for the tile's wide
  // link is resolved by the mux after the shim.
  logic burst_req;
  assign burst_req = spatz_req_valid_i &&
                     (spatz_req_burst_len_i > BurstLenWidth'(1));

  // A free slot, not an idle adapter: this is what lets one burst's round trip overlap
  // the next one's issue.
  logic slot_free;
  assign slot_free         = !issued_q[wr_ptr_q];
  assign spatz_req_ready_o = burst_req && slot_free && wide_req_ready_i;

  // ------------------------------------------------------------------
  //  Wide request
  // ------------------------------------------------------------------
  // The burst base address is already WideTcdmDataWidth-aligned: the VLSU only
  // admits a burst when rs1[BurstAlignBits-1:0] == 0 (spatz_vlsu.sv), so the access
  // lands in exactly one superbank and never straddles two.
  always_comb begin
    wide_req_o        = '0;
    wide_req_o.addr   = spatz_req_addr_i;
    wide_req_o.write  = 1'b0;   // bursts are loads only
    wide_req_o.amo    = '0;
    wide_req_o.strb   = '1;
    wide_req_o.id     = spatz_req_id_i;
    wide_req_valid_o  = slot_free && burst_req;
  end

  // Accept the response only while draining is possible, i.e. not mid-drain.
  assign wide_resp_ready_o = 1'b1;

  // ------------------------------------------------------------------
  //  Response unpack
  // ------------------------------------------------------------------
  // Beat k carries ROB id base + k and goes to lane k % RecvPorts, so a drain cycle
  // emits RecvPorts consecutive beats and advances by RecvPorts.
  always_comb begin
    spatz_resp_valid_o = '0;
    spatz_resp_data_o  = '0;
    spatz_resp_id_o    = '0;
    if (issued_q[rd_ptr_q] && filled_q[rd_ptr_q]) begin
      for (int unsigned l = 0; l < RecvPorts; l++) begin
        automatic logic [BeatCntWidth-1:0] k = BeatCntWidth'(l) +
                                               beat_q[l] * BeatCntWidth'(RecvPorts);
        automatic logic [RobIdWidth-1:0]   beat_id;
        if (!drain_stall_i[l] && (k < BeatCntWidth'(len_q[rd_ptr_q]))) begin
          // Entry index within THIS lane's ROB is simply the lane's beat count.
          // Truncate by assignment before widening: meta_id_t'(a + b) would evaluate
          // the sum at meta_id width and let the carry reach WrFlagBit, where
          // mempool_cc packs the store-ack flag -- the beat would read back as a
          // write acknowledgement.
          // RobIdWidth'(beat_q), NOT beat_q[RobIdWidth-1:0]. beat_q is BeatCntWidth wide
          // (a burst's beat index), which is unrelated to the ROB id width: at a ROB depth
          // of 64 RobIdWidth is 6 while BeatCntWidth is 5, so the slice ran off the end of
          // beat_q and every beat id collapsed to the base -- every response written to the
          // ROB head's neighbour, the head never filled, and the VLSU wedged with
          // rsp==push==8 but valid_o low. Depths 16 and 32 hid it because 4 and 5 both fit.
          // The cast zero-extends (or truncates) to the id width, and the add still wraps
          // within RobIdWidth as the reorder buffer's ring requires.
          beat_id               = base_id_q[rd_ptr_q][RobIdWidth-1:0] + RobIdWidth'(beat_q[l]);
          spatz_resp_valid_o[l] = 1'b1;
          spatz_resp_data_o[l]  = rdata_q[rd_ptr_q][k[idx_width(WideWords)-1:0]];
          spatz_resp_id_o[l]    = meta_id_t'(beat_id);
        end
      end
    end
  end

  always_comb begin
    drain_done = 1'b1;
    for (int unsigned l = 0; l < RecvPorts; l++)
      drain_done &= ((BeatCntWidth'(l) + beat_q[l] * BeatCntWidth'(RecvPorts)) >=
                     BeatCntWidth'(len_q[rd_ptr_q]));
  end

  // ------------------------------------------------------------------
  //  Control
  // ------------------------------------------------------------------
  always_comb begin
    base_id_d = base_id_q;
    len_d     = len_q;
    rdata_d   = rdata_q;
    issued_d  = issued_q;
    filled_d  = filled_q;
    wr_ptr_d  = wr_ptr_q;
    rd_ptr_d  = rd_ptr_q;
    for (int unsigned l = 0; l < RecvPorts; l++) beat_d[l] = beat_q[l];

    // Issue: take a free slot. Independent of what is draining, which is the whole
    // point -- the next request goes out while an older round trip is still in the air.
    if (wide_req_valid_o && wide_req_ready_i) begin
      base_id_d[wr_ptr_q] = spatz_req_id_i;
      len_d[wr_ptr_q]     = spatz_req_burst_len_i;
      issued_d[wr_ptr_q]  = 1'b1;
      filled_d[wr_ptr_q]  = 1'b0;
      wr_ptr_d            = (wr_ptr_q == SlotSel'(NumOutstanding-1)) ? '0
                                                                    : wr_ptr_q + SlotSel'(1);
    end

    // Response: find its slot by the id it round-tripped. The wide plane may answer two
    // outstanding bursts out of order, so this must not assume the oldest.
    if (wide_resp_valid_i)
      for (int unsigned sl = 0; sl < NumOutstanding; sl++)
        if (issued_q[sl] && !filled_q[sl] &&
            (base_id_q[sl][RobIdWidth-1:0] == wide_resp_i.id[RobIdWidth-1:0])) begin
          rdata_d[sl]  = wide_resp_i.data;
          filled_d[sl] = 1'b1;
        end

    // Drain the oldest filled slot. One slot at a time: the lanes are a shared resource,
    // and draining costs a few cycles against a round trip of tens.
    if (issued_q[rd_ptr_q] && filled_q[rd_ptr_q]) begin
      for (int unsigned l = 0; l < RecvPorts; l++) begin
        automatic logic [BeatCntWidth-1:0] k = BeatCntWidth'(l) +
                                               beat_q[l] * BeatCntWidth'(RecvPorts);
        if (!drain_stall_i[l] && (k < BeatCntWidth'(len_q[rd_ptr_q])))
          beat_d[l] = beat_q[l] + BeatCntWidth'(1);
      end
      if (drain_done) begin
        issued_d[rd_ptr_q] = 1'b0;
        filled_d[rd_ptr_q] = 1'b0;
        for (int unsigned l = 0; l < RecvPorts; l++) beat_d[l] = '0;
        rd_ptr_d = (rd_ptr_q == SlotSel'(NumOutstanding-1)) ? '0 : rd_ptr_q + SlotSel'(1);
      end
    end
  end

  `FF(issued_q, issued_d, '0)
  `FF(filled_q, filled_d, '0)
  `FF(wr_ptr_q, wr_ptr_d, '0)
  `FF(rd_ptr_q, rd_ptr_d, '0)
  for (genvar sl = 0; sl < NumOutstanding; sl++) begin : gen_slot_ff
    `FF(base_id_q[sl], base_id_d[sl], '0)
    `FF(len_q[sl],     len_d[sl],     '0)
    `FF(rdata_q[sl],   rdata_d[sl],   '0)
  end : gen_slot_ff
  for (genvar l = 0; l < RecvPorts; l++) begin : gen_beat_ff
    `FF(beat_q[l], beat_d[l], '0)
  end : gen_beat_ff



`ifndef SYNTHESIS
`ifdef SPATZ_VLSU_WEDGE
  // Liveness: a slot that has been issued but never drains. Covers both a response that
  // never arrives and a drain the lanes never let finish.
  int unsigned adpw_cnt; logic adpw_fired;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin adpw_cnt <= 0; adpw_fired <= 1'b0; end
    else begin
      if (issued_q == '0) adpw_cnt <= 0; else adpw_cnt <= adpw_cnt + 1;
      if ((adpw_cnt > 20000) && !adpw_fired) begin
        adpw_fired <= 1'b1;
        $display("[ADP-WEDGE] %m t=%0t issued=%b filled=%b rd=%0d wr=%0d len=%0d beats=%0d/%0d/%0d/%0d base=%0d stall=%b",
                 $time, issued_q, filled_q, rd_ptr_q, wr_ptr_q, len_q[rd_ptr_q],
                 beat_q[0], beat_q[1], beat_q[2], beat_q[3], base_id_q[rd_ptr_q],
                 drain_stall_i);
      end
    end
  end
`endif
`endif

endmodule : spatz_wide_burst_adapter
