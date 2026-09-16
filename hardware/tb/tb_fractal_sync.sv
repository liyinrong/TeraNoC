// Copyright 2026 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Standalone gate for the fractal_sync H-tree barrier (roadmap Phase 3).
//
// Drive the tree's leaf ports directly, independently of the tile shim and
// software runtime, and check barrier safety and isolation across groups:
//
//   SAFETY   a scope does not wake until its LAST member has arrived, and a
//            scope never wakes a non-member;
//   ISOLATION two DISJOINT scopes at the same level run concurrently without
//            either seeing the other's arrivals.
//
// The (aggr, id) values come from the closed form pinned in
// scripts/fsync/run_oracle.sh, which is checked against the reference
// generator: aggr = (1 << 2*log2(side)) - 1.
//
// Leaf index is ROW-MAJOR, not Morton: fractal_sync_4x4.sv computes
// offset = global_row*NET_COLS + global_col for its external ports. Getting
// that backwards is precisely the raster/Morton confusion that cost WP3 a
// day, so it is asserted here rather than assumed.

module tb_fractal_sync;

  import fractal_sync_4x4_pkg::*;

  localparam int unsigned SIDE = 4;             // CU grid side
  localparam int unsigned NCU  = SIDE * SIDE;   // 16 CUs
  localparam time         TCK  = 10ns;

  logic clk, rst_n;

  // Row-major leaf index, matching the tree's own offset arithmetic.
  function automatic int unsigned leaf_idx(input int unsigned x,
                                           input int unsigned y);
    return y * SIDE + x;
  endfunction

  // aggr for a complete side x side subtree: it climbs 2*log2(side) levels and
  // activates every one of them.
  function automatic logic [IN_AGGR_WIDTH-1:0] aggr_of(input int unsigned side);
    int unsigned lv = 0;
    for (int unsigned s = side; s > 1; s = s >> 1) lv += 2;
    return (({{(IN_AGGR_WIDTH){1'b0}}, 1'b1}) << lv) - 1;
  endfunction

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

  int unsigned errors = 0;
  // Sticky record of who woke, cleared between tests.
  logic [NCU-1:0] woke;
  logic           clr_woke;

  fractal_sync_4x4 i_dut (
    .clk_i             (clk),
    .rst_ni            (rst_n),
    .h_1d_fsync_req_i  (h_req),
    .h_1d_fsync_rsp_o  (h_rsp),
    .v_1d_fsync_req_i  (v_req),
    .v_1d_fsync_rsp_o  (v_rsp),
    .h_nbr_fsycn_req_i (h_nbr_req),   // sic: the dependency spells it fsycn
    .h_nbr_fsycn_rsp_o (h_nbr_rsp),
    .v_nbr_fsycn_req_i (v_nbr_req),
    .v_nbr_fsycn_rsp_o (v_nbr_rsp),
    .h_2d_fsync_req_o  (h2d_req),
    .h_2d_fsync_rsp_i  (h2d_rsp),
    .v_2d_fsync_req_o  (v2d_req),
    .v_2d_fsync_rsp_i  (v2d_rsp)
  );

  always #(TCK/2) clk = ~clk;

  // Root and neighbour ports are tied off: this tree is the whole barrier
  // domain, so nothing escapes above it (roadmap T3.2).
  always_comb begin
    for (int unsigned p = 0; p < N_2D_H_PORTS; p++)
      for (int unsigned k = 0; k < N_LINKS_OUT; k++) h2d_rsp[p][k] = '0;
    for (int unsigned p = 0; p < N_2D_V_PORTS; p++)
      for (int unsigned k = 0; k < N_LINKS_OUT; k++) v2d_rsp[p][k] = '0;
  end

  // Latch wakes as they appear; a wake is a one-cycle event.
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n || clr_woke) woke <= '0;
    else
      for (int unsigned c = 0; c < NCU; c++)
        if (h_rsp[c][0].wake || v_rsp[c][0].wake) woke[c] <= 1'b1;
  end

  // Reset the DUT between sections. clear_woke() only clears the OBSERVATION;
  // a barrier left incomplete keeps its arrivals pending inside the tree and
  // the next section's arrivals then complete it. Proven by negative control:
  // withholding one arrival in A made B report three cascaded failures that
  // had nothing to do with B. Independent sections keep a real failure local.
  task automatic reset_dut();
    @(negedge clk);
    rst_n = 1'b0;
    clear_reqs();
    repeat (4) @(posedge clk);
    @(negedge clk);
    rst_n = 1'b1;
    repeat (4) @(posedge clk);
  endtask

  task automatic clear_woke();
    @(negedge clk);
    clr_woke = 1'b1;
    @(posedge clk);
    @(negedge clk);
    clr_woke = 1'b0;
  endtask

  task automatic clear_reqs();
    for (int unsigned c = 0; c < NCU; c++) begin
      h_req[c][0] = '0;
      v_req[c][0] = '0;
    end
  endtask

  // One CU arrives at a barrier: a single-cycle sync pulse carrying (aggr,id).
  task automatic arrive(input int unsigned x, input int unsigned y,
                        input logic [IN_AGGR_WIDTH-1:0] aggr);
    int unsigned c = leaf_idx(x, y);
    @(negedge clk);
    h_req[c][0].sync     = 1'b1;
    h_req[c][0].sig.aggr = aggr;
    h_req[c][0].sig.id   = '0;   // h direction, per the reference generator
    @(negedge clk);
    h_req[c][0] = '0;
  endtask

  // Vertical pairing uses the CU's v leaf port and id=1. The direction is a
  // property of the GEOMETRY, not of a requested default: the reference
  // generator returns id=1 for a vertically-adjacent pair even when asked for
  // the horizontal direction, and id=0 for a horizontal pair asked vertically.
  task automatic arrive_v(input int unsigned x, input int unsigned y,
                          input logic [IN_AGGR_WIDTH-1:0] aggr);
    int unsigned c = leaf_idx(x, y);
    @(negedge clk);
    v_req[c][0].sync     = 1'b1;
    v_req[c][0].sig.aggr = aggr;
    v_req[c][0].sig.id   = 1;
    @(negedge clk);
    v_req[c][0] = '0;
  endtask

  task automatic arrive_id(input int unsigned x, input int unsigned y,
                           input logic [IN_AGGR_WIDTH-1:0] aggr,
                           input logic [ID_WIDTH-1:0] id);
    int unsigned c = leaf_idx(x, y);
    @(negedge clk);
    h_req[c][0].sync     = 1'b1;
    h_req[c][0].sig.aggr = aggr;
    h_req[c][0].sig.id   = id;
    @(negedge clk);
    h_req[c][0] = '0;
  endtask

  task automatic check(input logic cond, input string what);
    if (!cond) begin
      errors++;
      $display("[tb_fractal_sync] FAIL: %s", what);
    end
  endtask

  // Expect exactly `mask` to have woken.
  task automatic expect_woke(input logic [NCU-1:0] mask, input string what);
    if (woke !== mask) begin
      errors++;
      $display("[tb_fractal_sync] FAIL: %s -- woke=%b expected=%b", what, woke,
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
    clear_reqs();
    for (int unsigned p = 0; p < N_NBR_H_PORTS; p++) h_nbr_req[p] = '0;
    for (int unsigned p = 0; p < N_NBR_V_PORTS; p++) v_nbr_req[p] = '0;
    repeat (4) @(posedge clk);
    rst_n = 1'b1;
    repeat (4) @(posedge clk);

    // ---------------------------------------------------------------- A
    // SAFETY: a 2x2 scope must not wake on a partial set. Three of four
    // arrive; nothing may wake, however long we wait.
    reset_dut();
    clear_woke();
    arrive(0, 0, aggr_of(2));
    arrive(1, 0, aggr_of(2));
    arrive(0, 1, aggr_of(2));
    repeat (20) @(posedge clk);
    expect_woke('0, "A: 2x2 woke on 3 of 4 arrivals");

    // The last member arrives -- now exactly that rectangle wakes.
    arrive(1, 1, aggr_of(2));
    repeat (20) @(posedge clk);
    expect_woke(rect_mask(0, 0, 2), "A: 2x2 completion");

    // ---------------------------------------------------------------- B
    // ISOLATION: two disjoint 2x2 scopes, interleaved. Neither may complete
    // on the other's arrivals.
    reset_dut();
    clear_woke();
    arrive(0, 0, aggr_of(2));
    arrive(2, 2, aggr_of(2));
    arrive(1, 0, aggr_of(2));
    arrive(3, 2, aggr_of(2));
    arrive(0, 1, aggr_of(2));
    arrive(2, 3, aggr_of(2));
    repeat (20) @(posedge clk);
    expect_woke('0, "B: a scope completed on the other's arrivals");

    arrive(1, 1, aggr_of(2));       // completes rectangle (0,0)
    repeat (20) @(posedge clk);
    expect_woke(rect_mask(0, 0, 2), "B: only rectangle (0,0) should wake");

    arrive(3, 3, aggr_of(2));       // completes rectangle (2,2)
    repeat (20) @(posedge clk);
    expect_woke(rect_mask(0, 0, 2) | rect_mask(2, 2, 2), "B: both complete");

    // ---------------------------------------------------------------- C
    // CROSS-SCOPE: the level above. All 16 must arrive before any wakes.
    reset_dut();
    clear_woke();
    for (int unsigned y = 0; y < SIDE; y++)
      for (int unsigned x = 0; x < SIDE; x++)
        if (!(x == SIDE - 1 && y == SIDE - 1)) arrive(x, y, aggr_of(4));
    repeat (20) @(posedge clk);
    expect_woke('0, "C: 4x4 woke on 15 of 16 arrivals");

    arrive(SIDE - 1, SIDE - 1, aggr_of(4));
    repeat (30) @(posedge clk);
    expect_woke({NCU{1'b1}}, "C: 4x4 completion");

    // ---------------------------------------------------------------- D
    // PARTIAL SCOPE: a horizontal PAIR is the finest barrier the tree
    // supports (a single CU is refused by the generator). aggr=0x1, id=0.
    reset_dut();
    clear_woke();
    arrive(0, 0, 1);
    repeat (20) @(posedge clk);
    expect_woke('0, "D: h-pair woke on 1 of 2 arrivals");
    arrive(1, 0, 1);
    repeat (20) @(posedge clk);
    begin
      logic [NCU-1:0] m = '0;
      m[leaf_idx(0, 0)] = 1'b1;
      m[leaf_idx(1, 0)] = 1'b1;
      expect_woke(m, "D: h-pair completion");
    end

    // ---------------------------------------------------------------- E
    // The vertical direction, which tests A-D never exercise: same aggr, but
    // the v leaf port and id=1.
    reset_dut();
    clear_woke();
    arrive_v(0, 0, 1);
    repeat (20) @(posedge clk);
    expect_woke('0, "E: v-pair woke on 1 of 2 arrivals");
    arrive_v(0, 1, 1);
    repeat (20) @(posedge clk);
    begin
      logic [NCU-1:0] m = '0;
      m[leaf_idx(0, 0)] = 1'b1;
      m[leaf_idx(0, 1)] = 1'b1;
      expect_woke(m, "E: v-pair completion");
    end

    // ---------------------------------------------------------------- F
    // CONCURRENT RECTANGULAR SCOPES need distinct ids.
    //
    // A scope whose aggr has ZERO bits (any rectangle thinner than square)
    // PROPAGATES through the levels it skips instead of aggregating there, so
    // two such scopes meet at a shared node above. With equal ids that node
    // pairs them: driving row 0's LEFT half and row 1's RIGHT half -- neither
    // row complete -- wakes exactly those four tiles, a barrier that does not
    // exist. Measured, id=0 on both: woke=0000000011000011.
    //
    // Square scopes are immune: their aggr is contiguous all-ones, so they
    // aggregate at every level up to their own root and never pass through a
    // node another scope could match at. That is why test B passes with a
    // shared id and this one does not.
    //
    // Asserted here in its CORRECT form -- distinct ids -- because that is the
    // contract any software issuing such scopes concurrently must honour.
    reset_dut(); clear_woke();
    arrive_id(0, 0, 5, 0); arrive_id(1, 0, 5, 0);   // row 0, left,  id 0
    arrive_id(2, 1, 5, 2); arrive_id(3, 1, 5, 2);   // row 1, right, id 2
    repeat (40) @(posedge clk);
    expect_woke('0, "F: distinct-id rows cross-matched on half sets");
    arrive_id(2, 0, 5, 0); arrive_id(3, 0, 5, 0);
    arrive_id(0, 1, 5, 2); arrive_id(1, 1, 5, 2);
    repeat (40) @(posedge clk);
    expect_woke(16'h00ff, "F: both rows complete under distinct ids");

    $display("[tb_fractal_sync] %0d errors", errors);
    if (errors == 0) $display("[EOC] Simulation ended at %t (retval = 0).", $time);
    else             $display("[EOC] Simulation ended at %t (retval = 1).", $time);
    $finish;
  end

endmodule
