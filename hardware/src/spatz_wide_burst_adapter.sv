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

  // Every lane has delivered its share of the burst.
  logic drain_done;

  typedef enum logic [1:0] {
    Idle,     // arbitrate, issue the wide request
    Await,    // wide access outstanding
    Drain     // unpack the wide response into beats
  } state_e;

  state_e state_d, state_q;

  meta_id_t                     base_id_d, base_id_q;  // ROB id of beat 0
  logic [BurstLenWidth-1:0]     len_d, len_q;          // beats to return
  // Per-lane beat index. Lane l owns beats l, l+RecvPorts, l+2*RecvPorts ... -- its
  // own ROB's consecutive entries -- so each lane drains INDEPENDENTLY. A shared
  // counter would force one stalled lane to hold up the others, and the tile must
  // stall a lane whenever the narrow path is answering on that port: a vector store's
  // acks come back on all the vector ports, so an all-lanes stall blocks the whole
  // drain for as long as any ack is in flight.
  logic [BeatCntWidth-1:0]      beat_d [RecvPorts];
  logic [BeatCntWidth-1:0]      beat_q [RecvPorts];
  logic [WideWords-1:0][31:0]   rdata_d, rdata_q;      // the captured wide beat

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

  assign spatz_req_ready_o = burst_req && (state_q == Idle) && wide_req_ready_i;

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
    wide_req_valid_o  = (state_q == Idle) && burst_req;
  end

  // Accept the response only while draining is possible, i.e. not mid-drain.
  assign wide_resp_ready_o = (state_q == Await);

  // ------------------------------------------------------------------
  //  Response unpack
  // ------------------------------------------------------------------
  // Beat k carries ROB id base + k and goes to lane k % RecvPorts, so a drain cycle
  // emits RecvPorts consecutive beats and advances by RecvPorts.
  always_comb begin
    spatz_resp_valid_o = '0;
    spatz_resp_data_o  = '0;
    spatz_resp_id_o    = '0;
    if (state_q == Drain) begin
      for (int unsigned l = 0; l < RecvPorts; l++) begin
        automatic logic [BeatCntWidth-1:0] k = BeatCntWidth'(l) +
                                               beat_q[l] * BeatCntWidth'(RecvPorts);
        automatic logic [RobIdWidth-1:0]   beat_id;
        if (!drain_stall_i[l] && (k < BeatCntWidth'(len_q))) begin
          // Entry index within THIS lane's ROB is simply the lane's beat count.
          // Truncate by assignment before widening: meta_id_t'(a + b) would evaluate
          // the sum at meta_id width and let the carry reach WrFlagBit, where
          // mempool_cc packs the store-ack flag -- the beat would read back as a
          // write acknowledgement.
          beat_id               = base_id_q[RobIdWidth-1:0] + beat_q[l][RobIdWidth-1:0];
          spatz_resp_valid_o[l] = 1'b1;
          spatz_resp_data_o[l]  = rdata_q[k[idx_width(WideWords)-1:0]];
          spatz_resp_id_o[l]    = meta_id_t'(beat_id);
        end
      end
    end
  end

  always_comb begin
    drain_done = 1'b1;
    for (int unsigned l = 0; l < RecvPorts; l++)
      drain_done &= ((BeatCntWidth'(l) + beat_q[l] * BeatCntWidth'(RecvPorts)) >=
                     BeatCntWidth'(len_q));
  end

  // ------------------------------------------------------------------
  //  Control
  // ------------------------------------------------------------------
  always_comb begin
    state_d   = state_q;
    for (int unsigned l = 0; l < RecvPorts; l++) beat_d[l] = beat_q[l];
    base_id_d = base_id_q;
    len_d     = len_q;
    beat_d    = beat_q;
    rdata_d   = rdata_q;

    unique case (state_q)
      Idle: begin
        if (wide_req_valid_o && wide_req_ready_i) begin
          base_id_d = spatz_req_id_i;
          len_d     = spatz_req_burst_len_i;
          for (int unsigned l = 0; l < RecvPorts; l++) beat_d[l] = '0;
          state_d   = Await;
        end
      end
      Await: begin
        if (wide_resp_valid_i) begin
          rdata_d = wide_resp_i.data;
          state_d = Drain;
        end
      end
      Drain: begin
        // Each lane advances only on a cycle whose beat it actually presented.
        for (int unsigned l = 0; l < RecvPorts; l++) begin
          automatic logic [BeatCntWidth-1:0] k = BeatCntWidth'(l) +
                                                 beat_q[l] * BeatCntWidth'(RecvPorts);
          if (!drain_stall_i[l] && (k < BeatCntWidth'(len_q)))
            beat_d[l] = beat_q[l] + BeatCntWidth'(1);
        end
        // Done when every lane has delivered its whole share.
        if (drain_done) state_d = Idle;
      end
      default: state_d = Idle;
    endcase
  end

  `FF(state_q,   state_d,   Idle)
  `FF(base_id_q, base_id_d, '0  )
  `FF(len_q,     len_d,     '0  )
  for (genvar l = 0; l < RecvPorts; l++) begin : gen_beat_ff
    `FF(beat_q[l], beat_d[l], '0)
  end : gen_beat_ff
  `FF(rdata_q,   rdata_d,   '0  )

`ifndef SYNTHESIS
  // pragma translate_off
  // ------------------------------------------------------------------
  //  End-to-end burst instrumentation (SPATZ_BURST_DEBUG)
  // ------------------------------------------------------------------
  // Self-checking rather than a dump: for each burst, record which beats were
  // presented and confirm at completion that all len_q of them went out exactly
  // once. That separates "the adapter mis-delivers" from "the VLSU mishandles a
  // correctly delivered beat" -- the latter matters because TwinROB0 (receive on
  // mem port 1) has never actually run in this build until now.
  localparam bit BurstDbgEn = `ifdef SPATZ_BURST_DEBUG `SPATZ_BURST_DEBUG `else 0 `endif;
  if (BurstDbgEn) begin : gen_burst_dbg
    logic [WideWords-1:0] seen_d, seen_q;
    int unsigned          dbg_bursts_q, dbg_stalls_q, dbg_dups_q, dbg_short_q;

    always_comb begin
      seen_d = seen_q;
      if ((state_q == Idle) && wide_req_valid_o && wide_req_ready_i) seen_d = '0;
      if ((state_q == Drain) && !drain_stall_i)
        for (int unsigned l = 0; l < RecvPorts; l++) begin
          automatic logic [BeatCntWidth-1:0] kk = beat_q + BeatCntWidth'(l);
          if (kk < BeatCntWidth'(len_q)) seen_d[kk[idx_width(WideWords)-1:0]] = 1'b1;
        end
    end

    always_ff @(posedge clk_i) begin
      if (!rst_ni) begin
        seen_q <= '0; dbg_bursts_q <= 0; dbg_stalls_q <= 0; dbg_dups_q <= 0; dbg_short_q <= 0;
      end else begin
        seen_q <= seen_d;
        if ((state_q == Drain) && drain_stall_i) dbg_stalls_q <= dbg_stalls_q + 1;
        // A beat presented onto a k already marked is a duplicate delivery.
        if ((state_q == Drain) && !drain_stall_i)
          for (int unsigned l = 0; l < RecvPorts; l++) begin
            automatic logic [BeatCntWidth-1:0] kk = beat_q + BeatCntWidth'(l);
            if ((kk < BeatCntWidth'(len_q)) && seen_q[kk[idx_width(WideWords)-1:0]])
              dbg_dups_q <= dbg_dups_q + 1;
          end
        // Burst completing: every beat below len_q must have been presented.
        if ((state_q == Drain) && (state_d == Idle)) begin
          dbg_bursts_q <= dbg_bursts_q + 1;
          for (int unsigned k = 0; k < WideWords; k++)
            if ((k < len_q) && !seen_d[k]) begin
              dbg_short_q <= dbg_short_q + 1;
              $display("[BURST-DBG] %m MISSING beat k=%0d of len=%0d (base_id=%0d)",
                       k, len_q, base_id_q);
            end
        end
        // First burst of this adapter: full picture, beat by beat.
        if ((dbg_bursts_q == 0) && (state_q == Drain) && !drain_stall_i) begin
          for (int unsigned l = 0; l < RecvPorts; l++)
            if ((beat_q + BeatCntWidth'(l)) < BeatCntWidth'(len_q))
              $display("[BURST-DBG] %m beat k=%0d lane=%0d id=%0d data=%08x",
                       beat_q + BeatCntWidth'(l), l,
                       spatz_resp_id_o[l], spatz_resp_data_o[l]);
        end
        if ((dbg_bursts_q == 0) && (state_q == Await) && wide_resp_valid_i)
          $display("[BURST-DBG] %m WIDE base_id=%0d len=%0d data=%0h",
                   base_id_q, len_q, wide_resp_i.data);
        if ((dbg_bursts_q == 0) && (state_q == Idle) && wide_req_valid_o && wide_req_ready_i)
          $display("[BURST-DBG] %m REQ addr=%08x id=%0d len=%0d",
                   spatz_req_addr_i, spatz_req_id_i,
                   spatz_req_burst_len_i);
        // Periodic census so a rare fault is still visible.
        if (dbg_bursts_q == 64)
          $display("[BURST-CENSUS] %m bursts=%0d stall_cycles=%0d dup_beats=%0d missing_beats=%0d",
                   dbg_bursts_q, dbg_stalls_q, dbg_dups_q, dbg_short_q);
      end
    end
  end : gen_burst_dbg

  // A burst longer than one wide access cannot be served: it would need a second
  // access this adapter never issues, and the VLSU would wait forever for the tail.
  always_ff @(posedge clk_i) begin
    if (rst_ni && wide_req_valid_o && wide_req_ready_i &&
        (spatz_req_burst_len_i > BurstLenWidth'(WideWords)))
      $fatal(1, "[spatz_wide_burst_adapter] burst_len %0d exceeds one wide access (%0d words).",
             spatz_req_burst_len_i, WideWords);
  end
  // pragma translate_on
`endif

endmodule : spatz_wide_burst_adapter
