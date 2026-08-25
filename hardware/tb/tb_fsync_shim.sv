// Copyright 2026 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// fsync_shim against a real tree (roadmap T3.1).
//
// tb_fractal_sync drives the tree's leaf ports directly and so proves nothing
// about the shim. This wires one shim per CU into fractal_sync_4x4 and drives
// only the SOFTWARE-visible trigger, which is what a tile leader will do. It
// covers the three things the shim adds on top of the tree:
//
//   STEERING      id[0] selects the h or v network, so a vertical scope must
//                 resolve even though the h port is never driven;
//   BROADCAST     wake comes back as a one-cycle pulse per member;
//   ONE-OUTSTANDING trig_ready_o drops while a barrier is in flight, because
//                 the request path has no backpressure of its own.

module tb_fsync_shim;

  import fractal_sync_4x4_pkg::*;

  localparam int unsigned SIDE = 4;
  localparam int unsigned NCU  = SIDE * SIDE;
  localparam time         TCK  = 10ns;

  logic clk, rst_n;
  int unsigned errors = 0;

  function automatic int unsigned leaf_idx(input int unsigned x,
                                           input int unsigned y);
    return y * SIDE + x;
  endfunction

  function automatic logic [IN_AGGR_WIDTH-1:0] aggr_of(input int unsigned side);
    int unsigned lv = 0;
    for (int unsigned s = side; s > 1; s = s >> 1) lv += 2;
    return (({{(IN_AGGR_WIDTH){1'b0}}, 1'b1}) << lv) - 1;
  endfunction

  logic                      trig_valid [NCU];
  logic [IN_AGGR_WIDTH-1:0]  trig_aggr  [NCU];
  logic [ID_WIDTH-1:0]       trig_id    [NCU];
  logic                      trig_ready [NCU];
  logic                      wake       [NCU];
  logic                      error      [NCU];
  logic                      busy       [NCU];
  logic [LVL_WIDTH-1:0]      lvl        [NCU];
  logic [ID_WIDTH-1:0]       rsp_id     [NCU];

  fsync_in_req_t  h_req [NCU][N_LINKS_IN];
  fsync_rsp_t     h_rsp [NCU][N_LINKS_IN];
  fsync_in_req_t  v_req [NCU][N_LINKS_IN];
  fsync_rsp_t     v_rsp [NCU][N_LINKS_IN];
  fsync_nbr_req_t h_nbr_req [N_NBR_H_PORTS];
  fsync_nbr_rsp_t h_nbr_rsp [N_NBR_H_PORTS];
  fsync_nbr_req_t v_nbr_req [N_NBR_V_PORTS];
  fsync_nbr_rsp_t v_nbr_rsp [N_NBR_V_PORTS];
  fsync_out_req_t h2d_req [N_2D_H_PORTS][N_LINKS_OUT];
  fsync_rsp_t     h2d_rsp [N_2D_H_PORTS][N_LINKS_OUT];
  fsync_out_req_t v2d_req [N_2D_V_PORTS][N_LINKS_OUT];
  fsync_rsp_t     v2d_rsp [N_2D_V_PORTS][N_LINKS_OUT];

  logic [NCU-1:0] woke;
  logic           clr_woke;

  for (genvar c = 0; c < NCU; c++) begin : gen_shim
    fsync_shim #(
      .AggrWidth   (IN_AGGR_WIDTH  ),
      .IdWidth     (ID_WIDTH       ),
      .LvlWidth    (LVL_WIDTH      ),
      .fsync_req_t (fsync_in_req_t ),
      .fsync_rsp_t (fsync_rsp_t    )
    ) i_shim (
      .clk_i        (clk           ),
      .rst_ni       (rst_n         ),
      .trig_valid_i (trig_valid[c] ),
      .trig_aggr_i  (trig_aggr[c]  ),
      .trig_id_i    (trig_id[c]    ),
      .trig_ready_o (trig_ready[c] ),
      .h_req_o      (h_req[c][0]   ),
      .h_rsp_i      (h_rsp[c][0]   ),
      .v_req_o      (v_req[c][0]   ),
      .v_rsp_i      (v_rsp[c][0]   ),
      .wake_o       (wake[c]       ),
      .error_o      (error[c]      ),
      .busy_o       (busy[c]       ),
      .lvl_o        (lvl[c]        ),
      .rsp_id_o     (rsp_id[c]     )
    );
  end

  fractal_sync_4x4 i_dut (
    .clk_i             (clk      ),
    .rst_ni            (rst_n    ),
    .h_1d_fsync_req_i  (h_req    ),
    .h_1d_fsync_rsp_o  (h_rsp    ),
    .v_1d_fsync_req_i  (v_req    ),
    .v_1d_fsync_rsp_o  (v_rsp    ),
    .h_nbr_fsycn_req_i (h_nbr_req),
    .h_nbr_fsycn_rsp_o (h_nbr_rsp),
    .v_nbr_fsycn_req_i (v_nbr_req),
    .v_nbr_fsycn_rsp_o (v_nbr_rsp),
    .h_2d_fsync_req_o  (h2d_req  ),
    .h_2d_fsync_rsp_i  (h2d_rsp  ),
    .v_2d_fsync_req_o  (v2d_req  ),
    .v_2d_fsync_rsp_i  (v2d_rsp  )
  );

  always #(TCK/2) clk = ~clk;

  always_comb begin
    for (int unsigned p = 0; p < N_2D_H_PORTS; p++)
      for (int unsigned k = 0; k < N_LINKS_OUT; k++) h2d_rsp[p][k] = '0;
    for (int unsigned p = 0; p < N_2D_V_PORTS; p++)
      for (int unsigned k = 0; k < N_LINKS_OUT; k++) v2d_rsp[p][k] = '0;
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n || clr_woke) woke <= '0;
    else
      for (int unsigned c = 0; c < NCU; c++) if (wake[c]) woke[c] <= 1'b1;
  end

  task automatic clear_woke();
    @(negedge clk); clr_woke = 1'b1;
    @(posedge clk); @(negedge clk); clr_woke = 1'b0;
  endtask

  task automatic reset_dut();
    @(negedge clk);
    rst_n = 1'b0;
    for (int unsigned c = 0; c < NCU; c++) trig_valid[c] = 1'b0;
    repeat (4) @(posedge clk);
    @(negedge clk);
    rst_n = 1'b1;
    repeat (4) @(posedge clk);
  endtask

  // The software-visible action: one write, from one core of one tile.
  task automatic trigger(input int unsigned x, input int unsigned y,
                         input logic [IN_AGGR_WIDTH-1:0] aggr,
                         input logic [ID_WIDTH-1:0] id);
    int unsigned c = leaf_idx(x, y);
    @(negedge clk);
    trig_valid[c] = 1'b1;
    trig_aggr[c]  = aggr;
    trig_id[c]    = id;
    @(negedge clk);
    trig_valid[c] = 1'b0;
  endtask

  task automatic expect_woke(input logic [NCU-1:0] mask, input string what);
    if (woke !== mask) begin
      errors++;
      $display("[tb_fsync_shim] FAIL: %s -- woke=%b expected=%b", what, woke,
               mask);
    end
  endtask

  function automatic logic [NCU-1:0] rect_mask(input int unsigned bx,
                                               input int unsigned by,
                                               input int unsigned side);
    logic [NCU-1:0] m = '0;
    for (int unsigned dy = 0; dy < side; dy++)
      for (int unsigned dx = 0; dx < side; dx++)
        m[leaf_idx(bx + dx, by + dy)] = 1'b1;
    return m;
  endfunction

  initial begin
    clk      = 1'b0;
    rst_n    = 1'b0;
    clr_woke = 1'b0;
    for (int unsigned c = 0; c < NCU; c++) begin
      trig_valid[c] = 1'b0; trig_aggr[c] = '0; trig_id[c] = '0;
    end
    for (int unsigned p = 0; p < N_NBR_H_PORTS; p++) h_nbr_req[p] = '0;
    for (int unsigned p = 0; p < N_NBR_V_PORTS; p++) v_nbr_req[p] = '0;
    repeat (4) @(posedge clk);
    rst_n = 1'b1;
    repeat (4) @(posedge clk);

    // ------------------------------------------------------------- 1
    // A 2x2 scope driven ONLY through the software trigger.
    reset_dut(); clear_woke();
    trigger(0, 0, aggr_of(2), 0);
    trigger(1, 0, aggr_of(2), 0);
    trigger(0, 1, aggr_of(2), 0);
    repeat (20) @(posedge clk);
    expect_woke('0, "1: woke on 3 of 4 triggers");
    trigger(1, 1, aggr_of(2), 0);
    repeat (20) @(posedge clk);
    expect_woke(rect_mask(0, 0, 2), "1: 2x2 completion via shims");

    // ------------------------------------------------------------- 2
    // Vertical scope: id=1 must steer to the v network. The h port is never
    // driven, so this fails outright if the shim steers on the wrong bit.
    reset_dut(); clear_woke();
    trigger(0, 0, 1, 1);
    repeat (20) @(posedge clk);
    expect_woke('0, "2: v-pair woke on 1 of 2 triggers");
    trigger(0, 1, 1, 1);
    repeat (20) @(posedge clk);
    begin
      logic [NCU-1:0] m = '0;
      m[leaf_idx(0, 0)] = 1'b1;
      m[leaf_idx(0, 1)] = 1'b1;
      expect_woke(m, "2: v-pair completion (id steering)");
    end

    // ------------------------------------------------------------- 3
    // One outstanding barrier per tile: ready must drop while in flight and
    // come back once the barrier resolves.
    reset_dut(); clear_woke();
    if (!trig_ready[leaf_idx(0, 0)]) begin
      errors++; $display("[tb_fsync_shim] FAIL: 3: not ready when idle");
    end
    trigger(0, 0, aggr_of(2), 0);
    repeat (2) @(posedge clk);
    if (trig_ready[leaf_idx(0, 0)]) begin
      errors++; $display("[tb_fsync_shim] FAIL: 3: still ready while busy");
    end
    if (!busy[leaf_idx(0, 0)]) begin
      errors++; $display("[tb_fsync_shim] FAIL: 3: busy not asserted");
    end
    trigger(1, 0, aggr_of(2), 0);
    trigger(0, 1, aggr_of(2), 0);
    trigger(1, 1, aggr_of(2), 0);
    repeat (20) @(posedge clk);
    if (!trig_ready[leaf_idx(0, 0)]) begin
      errors++; $display("[tb_fsync_shim] FAIL: 3: not ready again after wake");
    end

    $display("[tb_fsync_shim] %0d errors", errors);
    if (errors == 0) $display("[EOC] Simulation ended at %t (retval = 0).", $time);
    else             $display("[EOC] Simulation ended at %t (retval = 1).", $time);
    $finish;
  end

endmodule
