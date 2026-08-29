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
// Beat k is returned on receive lane k % RecvPorts. That matches the VLSU's own
// parity rule: with RecvPorts = 2 its TwinROB0 expects the odd beats of a block on
// mem port 1 (burst_odd_alt in spatz_vlsu.sv: the ids inside the window whose LSB
// differs from the base's), and (base+k)[0] ^ base[0] == k[0]. RecvPorts is a
// parameter rather than a constant so this side is already right when the VLSU's
// receive datapath is generalised past the 2 it hardwires today.
//
// Loads only: a burst is only ever emitted for a load (burst_mode_req demands
// mem_is_load), so there is no write data path here.
//
// One wide slot serves all the Spatz cores of a tile, so requests are round-robin
// arbitrated. A tile's wide access is a whole superbank, so a grant is never a
// partial win: the winner gets its entire burst in one access.

module spatz_wide_burst_adapter
  import mempool_pkg::*;
  import cf_math_pkg::idx_width;
#(
  // Spatz cores sharing this tile's wide slot.
  parameter int unsigned NumSpatz      = 1,
  // Lanes used to return beats. Must match the VLSU's BurstRecvPorts.
  parameter int unsigned RecvPorts     = 2,
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
  parameter int unsigned SpatzIdxWidth = idx_width(NumSpatz),
  parameter int unsigned BeatCntWidth  = idx_width(WideWords + 1)
) (
  input  logic clk_i,
  input  logic rst_ni,

  // ---- Spatz side: the burst-capable port (port 0) of each core ----
  input  addr_t    [NumSpatz-1:0]                  spatz_req_addr_i,
  input  meta_id_t [NumSpatz-1:0]                  spatz_req_id_i,
  input  logic     [NumSpatz-1:0][BurstLenWidth-1:0] spatz_req_burst_len_i,
  input  logic     [NumSpatz-1:0]                  spatz_req_valid_i,
  output logic     [NumSpatz-1:0]                  spatz_req_ready_o,

  // ---- Spatz side: beats back, RecvPorts lanes per core ----
  // The Spatz response path has no ready (mempool_cc ties data_pready_o high for
  // the vector ports), so a drain cycle can never be back-pressured.
  output logic     [NumSpatz-1:0][RecvPorts-1:0]                 spatz_resp_valid_o,
  output data_t    [NumSpatz-1:0][RecvPorts-1:0]                 spatz_resp_data_o,
  output meta_id_t [NumSpatz-1:0][RecvPorts-1:0]                 spatz_resp_id_o,

  // Hold the drain for a cycle. The vector response ports have no ready, so the
  // adapter and the tile's narrow path must never present a beat on the same port
  // in the same cycle: the tile raises this whenever the narrow path is answering
  // one of this core's receive lanes.
  input  logic     [NumSpatz-1:0]                                drain_stall_i,

  // ---- Wide side ----
  output wide_req_t  wide_req_o,
  output logic       wide_req_valid_o,
  input  logic       wide_req_ready_i,
  input  wide_resp_t wide_resp_i,
  input  logic       wide_resp_valid_i,
  output logic       wide_resp_ready_o
);

`include "common_cells/registers.svh"

  typedef enum logic [1:0] {
    Idle,     // arbitrate, issue the wide request
    Await,    // wide access outstanding
    Drain     // unpack the wide response into beats
  } state_e;

  state_e state_d, state_q;

  logic [SpatzIdxWidth-1:0]     owner_d, owner_q;      // core the access belongs to
  meta_id_t                     base_id_d, base_id_q;  // ROB id of beat 0
  logic [BurstLenWidth-1:0]     len_d, len_q;          // beats to return
  logic [BeatCntWidth-1:0]      beat_d, beat_q;        // next beat to return
  logic [WideWords-1:0][31:0]   rdata_d, rdata_q;      // the captured wide beat

  // ------------------------------------------------------------------
  //  Request arbitration
  // ------------------------------------------------------------------
  // Only a real burst (burst_len > 1) belongs on the wide plane; a single-word
  // request stays on the narrow path and is never offered here.
  logic [NumSpatz-1:0]      arb_req;
  logic [NumSpatz-1:0]      arb_gnt;
  logic [SpatzIdxWidth-1:0] arb_idx;
  logic                     arb_valid;

  for (genvar c = 0; c < NumSpatz; c++) begin : gen_arb_req
    assign arb_req[c] = spatz_req_valid_i[c] &&
                        (spatz_req_burst_len_i[c] > BurstLenWidth'(1));
  end : gen_arb_req

  rr_arb_tree #(
    .NumIn    (NumSpatz),
    .DataWidth(1       ),
    .AxiVldRdy(1'b1    ),
    .LockIn   (1'b1    )
  ) i_arb (
    .clk_i  (clk_i                     ),
    .rst_ni (rst_ni                    ),
    .flush_i(1'b0                      ),
    .rr_i   ('0                        ),
    .req_i  (arb_req                   ),
    .gnt_o  (arb_gnt                   ),
    .data_i ('0                        ),
    .gnt_i  (state_q == Idle && wide_req_ready_i),
    .req_o  (arb_valid                 ),
    .data_o (/* unused */              ),
    .idx_o  (arb_idx                   )
  );

  // A core's port-0 request is consumed here only when it is a burst AND it wins.
  // Everything else keeps its narrow handshake, which the tile still drives.
  for (genvar c = 0; c < NumSpatz; c++) begin : gen_req_ready
    assign spatz_req_ready_o[c] = arb_gnt[c] && (state_q == Idle) && wide_req_ready_i;
  end : gen_req_ready

  // ------------------------------------------------------------------
  //  Wide request
  // ------------------------------------------------------------------
  // The burst base address is already WideTcdmDataWidth-aligned: the VLSU only
  // admits a burst when rs1[BurstAlignBits-1:0] == 0 (spatz_vlsu.sv), so the access
  // lands in exactly one superbank and never straddles two.
  always_comb begin
    wide_req_o        = '0;
    wide_req_o.addr   = spatz_req_addr_i[arb_idx];
    wide_req_o.write  = 1'b0;   // bursts are loads only
    wide_req_o.amo    = '0;
    wide_req_o.strb   = '1;
    wide_req_o.id     = spatz_req_id_i[arb_idx];
    wide_req_valid_o  = (state_q == Idle) && arb_valid;
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
    if ((state_q == Drain) && !drain_stall_i[owner_q]) begin
      for (int unsigned l = 0; l < RecvPorts; l++) begin
        automatic logic [BeatCntWidth-1:0] k = beat_q + BeatCntWidth'(l);
        if (k < BeatCntWidth'(len_q)) begin
          spatz_resp_valid_o[owner_q][l] = 1'b1;
          spatz_resp_data_o[owner_q][l]  = rdata_q[k[idx_width(WideWords)-1:0]];
          spatz_resp_id_o[owner_q][l]    =
              meta_id_t'((base_id_q[RobIdWidth-1:0] + k[RobIdWidth-1:0]));
        end
      end
    end
  end

  // ------------------------------------------------------------------
  //  Control
  // ------------------------------------------------------------------
  always_comb begin
    state_d   = state_q;
    owner_d   = owner_q;
    base_id_d = base_id_q;
    len_d     = len_q;
    beat_d    = beat_q;
    rdata_d   = rdata_q;

    unique case (state_q)
      Idle: begin
        if (wide_req_valid_o && wide_req_ready_i) begin
          owner_d   = arb_idx;
          base_id_d = spatz_req_id_i[arb_idx];
          len_d     = spatz_req_burst_len_i[arb_idx];
          beat_d    = '0;
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
        // Only advance on a cycle whose beats were actually presented.
        if (!drain_stall_i[owner_q]) begin
          beat_d = beat_q + BeatCntWidth'(RecvPorts);
          if ((beat_q + BeatCntWidth'(RecvPorts)) >= BeatCntWidth'(len_q))
            state_d = Idle;
        end
      end
      default: state_d = Idle;
    endcase
  end

  `FF(state_q,   state_d,   Idle)
  `FF(owner_q,   owner_d,   '0  )
  `FF(base_id_q, base_id_d, '0  )
  `FF(len_q,     len_d,     '0  )
  `FF(beat_q,    beat_d,    '0  )
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
      if ((state_q == Drain) && !drain_stall_i[owner_q])
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
        if ((state_q == Drain) && drain_stall_i[owner_q]) dbg_stalls_q <= dbg_stalls_q + 1;
        // A beat presented onto a k already marked is a duplicate delivery.
        if ((state_q == Drain) && !drain_stall_i[owner_q])
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
        if ((dbg_bursts_q == 0) && (state_q == Drain) && !drain_stall_i[owner_q]) begin
          for (int unsigned l = 0; l < RecvPorts; l++)
            if ((beat_q + BeatCntWidth'(l)) < BeatCntWidth'(len_q))
              $display("[BURST-DBG] %m beat k=%0d lane=%0d id=%0d data=%08x",
                       beat_q + BeatCntWidth'(l), l,
                       spatz_resp_id_o[owner_q][l], spatz_resp_data_o[owner_q][l]);
        end
        if ((dbg_bursts_q == 0) && (state_q == Await) && wide_resp_valid_i)
          $display("[BURST-DBG] %m WIDE base_id=%0d len=%0d data=%0h",
                   base_id_q, len_q, wide_resp_i.data);
        if ((dbg_bursts_q == 0) && (state_q == Idle) && wide_req_valid_o && wide_req_ready_i)
          $display("[BURST-DBG] %m REQ core=%0d addr=%08x id=%0d len=%0d",
                   arb_idx, spatz_req_addr_i[arb_idx], spatz_req_id_i[arb_idx],
                   spatz_req_burst_len_i[arb_idx]);
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
        (spatz_req_burst_len_i[arb_idx] > BurstLenWidth'(WideWords)))
      $fatal(1, "[spatz_wide_burst_adapter] burst_len %0d exceeds one wide access (%0d words).",
             spatz_req_burst_len_i[arb_idx], WideWords);
  end
  // pragma translate_on
`endif

endmodule : spatz_wide_burst_adapter
