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
//
// The register map is deliberately MAGIA-compatible (its obi_slave_fsync and
// sw/utils/fsync_mm_utils.h), because that costs nothing and keeps their
// software directly readable against ours:
//
//   +0x00 AGGR    W   aggregate bitmask for the next barrier
//   +0x04 ID      W   scope id, bit 0 selects h (even) or v (odd)
//   +0x08 CONTROL W   bit0 = trigger, bit1 = clear sticky error
//   +0x0C STATUS  R   {rsp_id, lvl, error, BUSY(bit2), done}
//   +0x10 WAKEMSK RW  one bit per core of the tile; 1 = this core receives the
//                     tree's wake. Resets to all-ones, which is the historical
//                     behaviour.
//
// WAKEMSK exists because fs_wake is broadcast to every core of the tile and
// wake_up_q is a saturating-free 3-bit counter: a core that receives the wake
// without ever sleeping for it banks a credit per barrier, and the eighth wrap
// destroys all eight. That is harmless while every core of the tile takes part
// in every barrier, and a latent hang as soon as one does not -- a dedicated
// DMA/control core in the tile being exactly that case.
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
  // Register port. Bus-agnostic on purpose; the tile adapts its own dialect.
  input  logic                 reg_req_i,
  input  logic                 reg_we_i,
  input  logic [4:0]           reg_addr_i,   // byte offset inside the region
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
  output logic [NumCores-1:0]  wake_mask_o,
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
  localparam logic [4:0] REG_AGGR     = 5'h00;
  localparam logic [4:0] REG_ID       = 5'h04;
  localparam logic [4:0] REG_CONTROL  = 5'h08;
  localparam logic [4:0] REG_STATUS   = 5'h0C;
  localparam logic [4:0] REG_WAKE_MSK = 5'h10;

  logic [AggrWidth-1:0] reg_aggr_q;
  logic [IdWidth-1:0]   reg_id_q;
  logic [NumCores-1:0]  reg_wake_mask_q;
  logic                 reg_wr;
  logic                 reg_trigger, reg_clear;

  assign reg_wr      = reg_req_i & reg_we_i;
  assign reg_trigger = reg_wr & (reg_addr_i == REG_CONTROL) & reg_wdata_i[0];
  assign reg_clear   = reg_wr & (reg_addr_i == REG_CONTROL) & reg_wdata_i[1];

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      reg_aggr_q      <= '0;
      reg_id_q        <= '0;
      // Every core wakes, so a design that never writes the mask behaves
      // exactly as before it existed.
      reg_wake_mask_q <= '1;
    end else if (reg_wr) begin
      if (reg_addr_i == REG_AGGR)     reg_aggr_q      <= reg_wdata_i[AggrWidth-1:0];
      if (reg_addr_i == REG_ID)       reg_id_q        <= reg_wdata_i[IdWidth-1:0];
      if (reg_addr_i == REG_WAKE_MSK) reg_wake_mask_q <= reg_wdata_i[NumCores-1:0];
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
    if (reg_req_i && !reg_we_i && reg_addr_i == REG_WAKE_MSK) begin
      reg_rdata_o[NumCores-1:0] = reg_wake_mask_q;
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

  assign wake_o     = wake;
  assign wake_mask_o = reg_wake_mask_q;
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
