// Copyright 2026 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Description: Issues an own-tile Spatz VLSU burst as narrow word requests on
// the core's vector ports, RecvPorts words per cycle.
//
// A burst is one request on vector port 0 carrying burst_len = N words from an
// aligned base; the VLSU then waits for beat k on lane k % RecvPorts with ROB id
// base + k / RecvPorts (spatz_wide_burst_adapter.sv). The VLSU can take at most
// RecvPorts beats per cycle, so fetching an own-tile burst as one wide superbank
// access buys nothing over RecvPorts narrow words per cycle, while costing a wide
// bank ingress and a superbank-atomic grant that holds 16 banks against the
// narrow path. This module therefore expands an own-tile burst in place: lane l
// issues words l, l + RecvPorts, l + 2*RecvPorts ... with exactly the ids the
// wide adapter would have returned, so the VLSU commits it unchanged. Remote
// bursts still go to the wide adapter.
//
// The burst is accepted in the cycle its first words issue, and the next one in
// the cycle the current one completes, so neither the first word nor a burst
// back-to-back with another pays a bubble. While the expander owns the lanes the
// tile must hold off the VLSU's own requests on them (busy_o). Only loads can
// follow a burst while its words are in flight -- the VLSU switches to stores
// only once every reorder buffer is empty (spatz_vlsu.sv, VLSU_RunningLoad) --
// so holding the lanes is a port conflict, not an ordering hazard.

module spatz_local_burst_expander #(
  // Receive lanes = the VLSU's vector memory ports.
  parameter int unsigned RecvPorts     = 4,
  // Longest burst, in words.
  parameter int unsigned MaxBurstWords = 16,
  parameter int unsigned BurstLenWidth = 5,
  // Width of the reorder-buffer id inside meta_id (spatz_pkg::MemRspIdWidth).
  // Beat ids wrap within this field, exactly as in the wide adapter.
  parameter int unsigned RobIdWidth    = 4,
  // Bytes per word.
  parameter int unsigned WordBytes     = 4,
  parameter type         addr_t        = logic [31:0],
  parameter type         meta_id_t     = logic
) (
  input  logic                           clk_i,
  input  logic                           rst_ni,

  // ---- An own-tile burst from the core's vector port 0 ----
  input  logic                           burst_valid_i,
  input  addr_t                          burst_addr_i,
  input  meta_id_t                       burst_id_i,
  input  logic     [BurstLenWidth-1:0]   burst_len_i,
  output logic                           burst_ready_o,

  // The expander drives the lanes this cycle.
  output logic                           busy_o,

  // ---- Narrow word requests, one per lane ----
  output logic     [RecvPorts-1:0]       lane_valid_o,
  output addr_t    [RecvPorts-1:0]       lane_addr_o,
  output meta_id_t [RecvPorts-1:0]       lane_id_o,
  input  logic     [RecvPorts-1:0]       lane_ready_i
);

  `include "common_cells/registers.svh"

  // Words one lane carries for the longest burst.
  localparam int unsigned MaxLaneWords = (MaxBurstWords + RecvPorts - 1) / RecvPorts;
  localparam int unsigned CntWidth     = $clog2(MaxLaneWords + 1);
  typedef logic [CntWidth-1:0] cnt_t;

  // The burst being issued: latched once accepted, else taken from the input.
  logic                     busy_d, busy_q;
  addr_t                    addr_d, addr_q;
  logic [RobIdWidth-1:0]    rob_d,  rob_q;
  logic [BurstLenWidth-1:0] len_d,  len_q;
  cnt_t [RecvPorts-1:0]     cnt_d,  cnt_q;

  addr_t                    cur_addr;
  logic [RobIdWidth-1:0]    cur_rob;
  logic [BurstLenWidth-1:0] cur_len;
  cnt_t [RecvPorts-1:0]     cur_cnt;
  logic                     active;

  always_comb begin
    if (busy_q) begin
      cur_addr = addr_q;
      cur_rob  = rob_q;
      cur_len  = len_q;
      cur_cnt  = cnt_q;
    end else begin
      cur_addr = burst_addr_i;
      cur_rob  = burst_id_i[RobIdWidth-1:0];
      cur_len  = burst_len_i;
      cur_cnt  = '0;
    end
  end
  assign active = busy_q || burst_valid_i;
  assign busy_o = active;

  // Lane l owns words l, l + RecvPorts, ...: share = ceil((len - l) / RecvPorts).
  cnt_t [RecvPorts-1:0] share;
  logic [RecvPorts-1:0] fire, lane_done;
  logic                 all_done;

  for (genvar l = 0; l < RecvPorts; l++) begin : gen_lane
    assign share[l] = (cur_len > BurstLenWidth'(l))
                    ? cnt_t'((cur_len - BurstLenWidth'(l) - 1) / RecvPorts + 1)
                    : '0;
    assign lane_valid_o[l] = active && (cur_cnt[l] < share[l]);
    // Word k = RecvPorts*cnt + l of the burst.
    assign lane_addr_o[l]  = cur_addr + addr_t'(WordBytes * (RecvPorts * cur_cnt[l] + l));
    // Truncate to the ROB field before widening so the carry cannot reach the
    // store-ack flag above it (as in spatz_wide_burst_adapter).
    assign lane_id_o[l]    = meta_id_t'(RobIdWidth'(cur_rob + RobIdWidth'(cur_cnt[l])));
    assign fire[l]         = lane_valid_o[l] && lane_ready_i[l];
    assign lane_done[l]    = (cur_cnt[l] + cnt_t'(fire[l])) >= share[l];
  end : gen_lane

  assign all_done = &lane_done;

  // Take the input burst when idle, or when the current one completes this cycle.
  assign burst_ready_o = burst_valid_i && (!busy_q || all_done);

  always_comb begin
    busy_d = busy_q;
    addr_d = addr_q;
    rob_d  = rob_q;
    len_d  = len_q;
    cnt_d  = cnt_q;
    if (!busy_q) begin
      // Idle: the input burst (if any) issued its first words this cycle.
      if (burst_valid_i && !all_done) begin
        busy_d = 1'b1;
        addr_d = burst_addr_i;
        rob_d  = burst_id_i[RobIdWidth-1:0];
        len_d  = burst_len_i;
        for (int unsigned l = 0; l < RecvPorts; l++) cnt_d[l] = cnt_t'(fire[l]);
      end
    end else if (all_done) begin
      // The latched burst completes; start the next one (if offered) next cycle.
      busy_d = burst_valid_i;
      addr_d = burst_addr_i;
      rob_d  = burst_id_i[RobIdWidth-1:0];
      len_d  = burst_len_i;
      cnt_d  = '0;
    end else begin
      for (int unsigned l = 0; l < RecvPorts; l++) cnt_d[l] = cnt_q[l] + cnt_t'(fire[l]);
    end
  end

  `FF(busy_q, busy_d, 1'b0)
  `FF(addr_q, addr_d, '0)
  `FF(rob_q,  rob_d,  '0)
  `FF(len_q,  len_d,  '0)
  `FF(cnt_q,  cnt_d,  '0)

`ifndef SYNTHESIS
  // pragma translate_off
  initial begin
    if (MaxBurstWords < 2)
      $fatal(1, "[spatz_local_burst_expander] MaxBurstWords must be >= 2.");
  end
  always_ff @(posedge clk_i) begin
    if (rst_ni && burst_valid_i && (burst_len_i > BurstLenWidth'(MaxBurstWords)))
      $error("[spatz_local_burst_expander] burst_len %0d exceeds MaxBurstWords %0d.",
             burst_len_i, MaxBurstWords);
  end

  // +spatz_burst_stats: report the own-tile bursts expanded here.
  // Cleared in the reset branch, not by a declaration initializer: VCS rejects
  // an initializer on a variable an always_ff also drives (ICPD_INIT).
  longint unsigned stat_bursts, stat_words;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      stat_bursts <= 0;
      stat_words  <= 0;
    end else begin
      if (burst_valid_i && burst_ready_o) stat_bursts <= stat_bursts + 1;
      stat_words <= stat_words + $countones(fire);
    end
  end
  final begin
    if ($test$plusargs("spatz_burst_stats") && (stat_bursts != 0))
      $display("[BURST] %m own_tile_bursts=%0d words=%0d", stat_bursts, stat_words);
  end
  // pragma translate_on
`endif

endmodule
