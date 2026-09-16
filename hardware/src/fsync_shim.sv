// Copyright 2026 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Per-TILE shim between software and one fractal_sync leaf.
//
// One shim per tile, not per core. The barrier scope must be a COMPLETE square
// subtree or the members' (aggr, id) diverge and the O(1) closed form is lost,
// so exactly one arrival per tile is the invariant worth protecting: the tile
// leader arrives on behalf of the tile however many of its cores took part,
// which also lets a dedicated DMA/control core sit in the tile without
// participating. A per-core shim would need arbitration to re-establish the
// same property.
//
// A CU owns BOTH an h and a v leaf port and picks one by geometry. id bit 0
// selects which (0 = h 1D, 1 = v 1D, 2 = h neighbour, 3 = v neighbour -- even
// is horizontal, odd is vertical), so the request is steered on id[0] and the
// two responses are merged; only one network can be active for a given id.
//
// Requests have no backpressure, so the shim enforces one outstanding barrier
// per tile: trig_ready_o drops while a barrier is in flight. An `error` from
// the tree (RF/CAM overflow) is latched and released rather than dropped -- a
// silently swallowed error would wedge every core in the scope, which is the
// hardest failure to diagnose from a frozen trace.
//
// The register map is deliberately MAGIA-compatible (its obi_slave_fsync and
// sw/utils/fsync_mm_utils.h), because that costs nothing and keeps their
// software directly readable against ours:
//
//   +0x00 AGGR    W   aggregate bitmask for the next barrier
//   +0x04 ID      W   scope id, bit 0 selects h (even) or v (odd)
//   +0x08 CONTROL W   bit0 = trigger, bit1 = clear sticky error
//   +0x0C STATUS  R   {rsp_id, lvl, error, BUSY(bit2), done}
//   +0x20+4c SUBSCRIBE[c] RW  event sources that may wake core c of this tile
//
// E1, the event-unit core. A core is woken ONLY while it is asleep:
//
//     wake[c] = asleep[c] & |(pending[c])      pending = subscribed events
//
// asleep[c] needs no core change -- snitch.sv:287 already drives
// inst_valid_o = ~wfi_q and mempool_tile.sv already routes it for the icache.
//
// This event path does not bank awake-core wake credits: Snitch increments
// its counter only for a wake while awake. Independent legacy wake sources
// still use that counter and must be consumed before an event-based wait.
//
// pending is level-held, so events arriving while the core runs simply wait:
// Repeated events from a source coalesce into one pending bit. A wake consumes
// the pending bits, so nothing has to be acked. Writing SUBSCRIBE
// also drops the pending bits of every source it removes, so an event taken
// while awake cannot wake a later, unrelated sleep. SUBSCRIBE resets to the
// fsync bit, so existing code behaves exactly as before.
//
// A sleeping core is therefore woken only by an event it is subscribed to.
// The fsync barrier relies on that and sleeps once per barrier, so no core may
// receive a legacy wake_up while it waits in one.
//
// BUSY deliberately sits at bit 2, matching MAGIA's FSYNC_MM_STATUS_BUSY_MASK.
// The bus port is intentionally generic (req/we/addr/wdata/rdata) rather than
// OBI: our tile speaks the snitch reqrsp dialect, so the adapter belongs in
// the tile, not in the barrier engine.

module fsync_shim #(
  parameter int unsigned AggrWidth   = 1,
  parameter int unsigned IdWidth     = 1,
  parameter int unsigned LvlWidth    = 1,
  parameter int unsigned NumCores    = 1,
  parameter type         fsync_req_t = logic,
  parameter type         fsync_rsp_t = logic
) (
  input  logic                 clk_i,
  input  logic                 rst_ni,
  // One bit per core of the tile, high while that core is in WFI.
  input  logic [NumCores-1:0]  core_asleep_i,
  // Group engine completion, reduced across streams by the group. A PULSE, one
  // per retired transfer (mempool_idma_group.sv): the iDMA exposes no
  // interrupt, and this is the frontend's own view of its completed-ID counter.
  input  logic                 dma_done_evt_i,
  // Tile engine end-of-computation (RedMulE today). A PULSE, so it is taken as
  // is rather than edge-detected. Routing it here is what makes an engine
  // dependency expressible for ANY core, and for SEVERAL at once: every core
  // subscribed to EVT_ENGINE gets its own pending bit set from the same pulse,
  // so one producer can release many consumers with no software fan-out.
  input  logic                 engine_evt_i,
  // Register port. Bus-agnostic on purpose; the tile adapts its own dialect.
  input  logic                 reg_req_i,
  input  logic                 reg_we_i,
  input  logic [7:0]           reg_addr_i,   // byte offset inside the region
  input  logic [31:0]          reg_wdata_i,
  output logic [31:0]          reg_rdata_o,
  // Direct trigger, kept for the unit test and for a future in-tile requester
  // that has no bus in front of it.
  input  logic                 trig_valid_i,
  input  logic [AggrWidth-1:0] trig_aggr_i,
  input  logic [IdWidth-1:0]   trig_id_i,
  output logic                 trig_ready_o,
  // The tile's two leaf ports.
  output fsync_req_t           h_req_o,
  input  fsync_rsp_t           h_rsp_i,
  output fsync_req_t           v_req_o,
  input  fsync_rsp_t           v_rsp_i,
  // To the tile's cores.
  output logic                 wake_o,
  // Per-core wake pulses. wake_o stays scalar for the existing unit test and
  // for any consumer that wants the raw tree response.
  output logic [NumCores-1:0]  wake_vec_o,
  output logic                 error_o,
  output logic                 busy_o,
  output logic [LvlWidth-1:0]  lvl_o,
  output logic [IdWidth-1:0]   rsp_id_o
);

  logic                 busy_q;
  logic                 sync_q;      // the one-cycle request pulse
  logic [AggrWidth-1:0] aggr_q;
  logic [IdWidth-1:0]   id_q;
  logic                 error_q;
  logic [LvlWidth-1:0]  lvl_q;
  logic [IdWidth-1:0]   rsp_id_q;

  // Only one network can answer for a given id, so merging is safe.
  logic wake, err;
  assign wake = h_rsp_i.wake  | v_rsp_i.wake;
  assign err  = h_rsp_i.error | v_rsp_i.error;

  // Register file, MAGIA-compatible offsets.
  localparam logic [7:0] REG_AGGR      = 8'h00;
  localparam logic [7:0] REG_ID        = 8'h04;
  localparam logic [7:0] REG_CONTROL   = 8'h08;
  localparam logic [7:0] REG_STATUS    = 8'h0C;
  localparam logic [7:0] REG_SUBSCRIBE = 8'h20;  // +4 per core

  // Event sources. The tree response is edge-detected (it is a level); the two
  // engine completions arrive as pulses and are taken as they are.
  localparam int unsigned EvtWidth   = 3;
  localparam int unsigned EVT_FSYNC  = 0;
  localparam int unsigned EVT_DMA    = 1; // one group DMA transfer retired
  localparam int unsigned EVT_ENGINE = 2; // tile engine end-of-computation

  logic [AggrWidth-1:0] reg_aggr_q;
  logic [IdWidth-1:0]   reg_id_q;
  logic [EvtWidth-1:0]  subscribe_q [NumCores];
  logic [EvtWidth-1:0]  pending_q   [NumCores];
  logic [NumCores-1:0]  evt_wake;
  // The tree response may be asserted for more than one cycle. pending must be
  // set ONCE per response: setting it on every cycle of a held response re-arms
  // it right after the wake consumed it, and the leftover makes the NEXT wfi
  // return immediately -- which breaks mempool_barrier, whose sleeping path has
  // no re-check loop. Edge-detect instead.
  logic                 wake_q;
  logic                 wake_edge;
  logic                 reg_wr;
  logic                 reg_trigger, reg_clear;

  assign reg_wr      = reg_req_i & reg_we_i;
  assign reg_trigger = reg_wr & (reg_addr_i == REG_CONTROL) & reg_wdata_i[0];
  assign reg_clear   = reg_wr & (reg_addr_i == REG_CONTROL) & reg_wdata_i[1];

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      reg_aggr_q <= '0;
      reg_id_q   <= '0;
    end else if (reg_wr) begin
      if (reg_addr_i == REG_AGGR) reg_aggr_q <= reg_wdata_i[AggrWidth-1:0];
      if (reg_addr_i == REG_ID)   reg_id_q   <= reg_wdata_i[IdWidth-1:0];
    end
  end

  // STATUS: BUSY at bit 2 to match MAGIA's mask, so their polling loop works.
  always_comb begin
    reg_rdata_o = '0;
    if (reg_req_i && !reg_we_i && reg_addr_i == REG_STATUS) begin
      reg_rdata_o[0]  = ~busy_q;    // done: idle means the last barrier resolved
      reg_rdata_o[1]  = error_q;
      reg_rdata_o[2]  = busy_q;     // FSYNC_MM_STATUS_BUSY_MASK
      reg_rdata_o[7:4]                 = lvl_q;
      reg_rdata_o[8+IdWidth-1 -: IdWidth] = rsp_id_q;
    end
    for (int unsigned c = 0; c < NumCores; c++) begin
      if (reg_req_i && !reg_we_i && reg_addr_i == (REG_SUBSCRIBE + 8'(4*c)))
        reg_rdata_o[EvtWidth-1:0] = subscribe_q[c];
    end
  end

  logic accept;
  // Either source may arm a barrier; the register path is what software uses.
  logic                 src_valid;
  logic [AggrWidth-1:0] src_aggr;
  logic [IdWidth-1:0]   src_id;
  assign src_valid = reg_trigger | trig_valid_i;
  assign src_aggr  = reg_trigger ? reg_aggr_q : trig_aggr_i;
  assign src_id    = reg_trigger ? reg_id_q   : trig_id_i;

  assign accept       = src_valid & ~busy_q;
  assign trig_ready_o = ~busy_q;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      busy_q <= 1'b0;
      sync_q <= 1'b0;
      aggr_q <= '0;
      id_q   <= '0;
    end else begin
      // sync is a PULSE: the receiver samples on it (fractal_sync_rx.sv:96),
      // so holding it high would re-arm the same barrier every cycle.
      sync_q <= accept;
      if (accept) begin
        busy_q <= 1'b1;
        aggr_q <= src_aggr;
        id_q   <= src_id;
      end else if (wake || err) begin
        // Release on an error too. Holding busy would deadlock the tile with
        // no way back, and the trap is what tells software the barrier failed.
        busy_q <= 1'b0;
      end
    end
  end

  // Status. error is sticky so a trap handler can read it after the fact.
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      error_q  <= 1'b0;
      lvl_q    <= '0;
      rsp_id_q <= '0;
    end else begin
      if (err)            error_q <= 1'b1;
      else if (reg_clear) error_q <= 1'b0;
      if (wake || err) begin
        lvl_q    <= h_rsp_i.wake ? h_rsp_i.sig.lvl : v_rsp_i.sig.lvl;
        rsp_id_q <= h_rsp_i.wake ? h_rsp_i.sig.id  : v_rsp_i.sig.id;
      end
    end
  end

  // id[0] picks the network: even = horizontal, odd = vertical.
  always_comb begin
    h_req_o          = '0;
    v_req_o          = '0;
    h_req_o.sig.aggr = aggr_q;
    h_req_o.sig.id   = id_q;
    v_req_o.sig.aggr = aggr_q;
    v_req_o.sig.id   = id_q;
    h_req_o.sync     = sync_q & ~id_q[0];
    v_req_o.sync     = sync_q &  id_q[0];
  end

  // ---------------------------------------------------------------------
  // Event unit
  // ---------------------------------------------------------------------
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) wake_q <= 1'b0;
    else         wake_q <= wake;
  end
  assign wake_edge = wake & ~wake_q;

  // Every trigger, and whether it was ACCEPTED or silently dropped. accept is
  // gated on ~busy_q with no back-pressure to software -- fsync_trigger() is a
  // posted MMIO write -- so a trigger arriving while the tile's previous
  // barrier is still outstanding disappears and every participant sleeps for a
  // response that will never come.
  `ifdef FSYNC_WAKE_TRACE
  always_ff @(posedge clk_i) begin
    if (rst_ni && src_valid)
      $display("[%t] %m: TRIGGER %s id=%0d aggr=0x%0h", $time,
               accept ? "ACCEPT" : "DROP", src_id, src_aggr);
  end
  `endif

  // Every tree response this tile receives, and whether a barrier was actually
  // outstanding when it arrived. A response with busy_q low was not asked for
  // by this tile.
  `ifdef FSYNC_WAKE_TRACE
  always_ff @(posedge clk_i) begin
    if (rst_ni && wake_edge)
      $display("[%t] %m: TREE RESPONSE busy=%0b lvl=%0d id=%0d", $time, busy_q,
               h_rsp_i.wake ? h_rsp_i.sig.lvl : v_rsp_i.sig.lvl,
               h_rsp_i.wake ? h_rsp_i.sig.id : v_rsp_i.sig.id);
  end
  `endif

  for (genvar c = 0; c < NumCores; c++) begin : gen_events
    logic [EvtWidth-1:0] evt_set;

    // Only subscribed sources are recorded, so a core accumulates exactly what
    // it asked for.
    always_comb begin
      evt_set = '0;
      evt_set[EVT_FSYNC]  = wake_edge & subscribe_q[c][EVT_FSYNC];
      // Both engine events are PULSES, so they are taken as they are: an
      // engine edge detector would be a second, redundant filter.
      evt_set[EVT_DMA]    = dma_done_evt_i & subscribe_q[c][EVT_DMA];
      evt_set[EVT_ENGINE] = engine_evt_i & subscribe_q[c][EVT_ENGINE];
    end

    // Wake ONLY while the core is asleep. This is the whole safety argument:
    // an awake core is never pulsed, so wake_up_q never increments and the
    // counted-credit model cannot be violated.
    assign evt_wake[c] = core_asleep_i[c] & (|pending_q[c]);

    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        // Subscribed to the tree by default, so existing software that never
        // touches these registers behaves exactly as it did before.
        subscribe_q[c] <= (EvtWidth'(1) << EVT_FSYNC);
        pending_q[c]   <= '0;
      end else begin
        // Set beats clear: an event landing in the same cycle as the wake is
        // kept for the next sleep rather than lost.
        pending_q[c] <= (pending_q[c] & ~{EvtWidth{evt_wake[c]}}) | evt_set;
        // pending stays a subset of the subscription. An event recorded while
        // the core was awake would otherwise outlive its unsubscribe and wake
        // the next, unrelated sleep -- a barrier wait that then exits early.
        if (reg_wr && reg_addr_i == (REG_SUBSCRIBE + 8'(4*c))) begin
          subscribe_q[c] <= reg_wdata_i[EvtWidth-1:0];
          pending_q[c]   <= ((pending_q[c] & ~{EvtWidth{evt_wake[c]}}) | evt_set)
                            & reg_wdata_i[EvtWidth-1:0];
        end
      end
    end
  end

  assign wake_o     = wake;
  // Every event wake is gated on the core being asleep, so an awake core is
  // never pulsed and wake_up_q (snitch.sv) never increments from this path.
  // It is additive: the legacy wake (wake_up_q, from the control registers)
  // stays ungated in the tile, because mempool_barrier_init has core 0 pulse
  // itself and sleep to spend the credit, and boot and mempool_barrier rely on
  // the same banking.
  assign wake_vec_o = evt_wake;
  assign busy_o   = busy_q;
  assign error_o  = error_q;
  assign lvl_o    = lvl_q;
  assign rsp_id_o = rsp_id_q;

  // The tree's detected_error_o is DELIBERATELY not consumed.
  //
  // fractal_sync_cc declares it (:116) but the 1D and 2D nodes that instantiate
  // it tie it off (fractal_sync_1d.sv:351, fractal_sync_2d.sv:571) and the tree
  // wrappers have no error port at all, so surfacing it would mean patching a
  // dep that is pinned to the revision MAGIA also uses.
  //
  // It is worth giving up because of WHAT it reports: FIFO empty-on-pop /
  // full-on-push and RX/TX overflow (fractal_sync_cc.sv:686-696) -- queue
  // CAPACITY violations. This shim admits one outstanding barrier per tile
  // (accept = src_valid & ~busy_q), so the tree sees at most one request per
  // leaf, which our traffic cannot use to overrun it; sizing the tree for its
  // own leaf count is the dep's business.
  //
  // The FUNCTIONAL error path is wired and does reach software:
  // local_rsp.error -> error_q -> STATUS bit 1, and fsync_barrier_n traps on
  // it. What follows catches the other symptom worth having -- a lost or
  // duplicated message.

  // A barrier response with no barrier outstanding means a lost or duplicated
  // arrival somewhere in the tree; catching it here beats debugging it from a
  // frozen trace later.
  `ifndef SYNTHESIS
  always_ff @(posedge clk_i) begin
    if (rst_ni && (wake || err) && !busy_q)
      $display("[%t] %m: fsync response with no barrier outstanding", $time);
  end
  `endif

endmodule
