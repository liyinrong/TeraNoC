// Copyright 2026 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Per-TILE shim between software and one fractal_sync leaf (roadmap T3.1).
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

module fsync_shim #(
  parameter int unsigned AggrWidth   = 1,
  parameter int unsigned IdWidth     = 1,
  parameter int unsigned LvlWidth    = 1,
  parameter type         fsync_req_t = logic,
  parameter type         fsync_rsp_t = logic
) (
  input  logic                 clk_i,
  input  logic                 rst_ni,
  // Software trigger: one write per barrier, from the tile leader.
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

  logic accept;
  assign accept       = trig_valid_i & ~busy_q;
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
        aggr_q <= trig_aggr_i;
        id_q   <= trig_id_i;
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
      if (err) error_q <= 1'b1;
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

  assign wake_o   = wake;
  assign busy_o   = busy_q;
  assign error_o  = error_q;
  assign lvl_o    = lvl_q;
  assign rsp_id_o = rsp_id_q;

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
