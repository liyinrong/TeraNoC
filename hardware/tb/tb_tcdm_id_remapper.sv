// Copyright 2026 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Standalone gate for the RESPONSE path of tcdm_id_remapper.
//
// The remapper merges the scalar LSU and the FP-sequence LSU onto one tile
// port, and their responses come back on one shared stream. The property under
// test is that an arm which stops accepting cannot stop the OTHER arm's
// responses. A plain demux presents the stalled arm's ready straight as the
// network accept, so the tile stops taking responses altogether and, on the
// single-VC mesh, every requester of the tile starves behind it -- the FFN
// decode wedge, where the FP arm waits for its own outstanding store whose ack
// is queued behind the response it is not taking.
//
// Two facts about the interface shape the script:
//   * an id is freed the moment the remapper TAKES a response, not when the
//     arm consumes it, so the id space recycles as soon as one is buffered
//     here -- the two arms can hold the same id at different times;
//   * the arm a response belongs to is looked up from the id it carries.
//
// The script:
//   1. arm 0 requests and then refuses its response: it must still be taken
//      from the network and held for it.
//   2. arm 1 requests and IS ready. Its response must be delivered while arm 0
//      is still refusing. This is the gate.
//   3. The documented limit: arm 0 issues a SECOND request, so this response
//      is legitimate, and its entry is still full -- so it cannot be taken.
//      One entry covers the wedge, where the stalled arm has exactly one
//      response in flight.
//   4. Drain: arm 0 gets its own response, with its own data and own id, and
//      the response that was refused goes through once the entry is free.

module tb_tcdm_id_remapper;

  timeunit 1ns;
  timeunit 1ps;

  localparam int unsigned NumIn     = 2;
  localparam int unsigned DataWidth = 32;

  logic clk = 1'b0;
  logic rst_n = 1'b0;
  always #1ns clk = ~clk;

  snitch_pkg::dreq_t  [NumIn-1:0] req_i;
  logic               [NumIn-1:0] req_valid_i, req_ready_o;
  snitch_pkg::dresp_t [NumIn-1:0] resp_o;
  logic               [NumIn-1:0] resp_valid_o, resp_ready_i;
  snitch_pkg::dreq_t              req_o;
  logic                           req_valid_o, req_ready_i;
  snitch_pkg::dresp_t             resp_i;
  logic                           resp_valid_i, resp_ready_o;

  tcdm_id_remapper #(
    .NumIn (NumIn)
  ) i_dut (
    .clk_i        (clk          ),
    .rst_ni       (rst_n        ),
    .req_i        (req_i        ),
    .req_valid_i  (req_valid_i  ),
    .req_ready_o  (req_ready_o  ),
    .resp_o       (resp_o       ),
    .resp_valid_o (resp_valid_o ),
    .resp_ready_i (resp_ready_i ),
    .req_o        (req_o        ),
    .req_valid_o  (req_valid_o  ),
    .req_ready_i  (req_ready_i  ),
    .resp_i       (resp_i       ),
    .resp_valid_i (resp_valid_i ),
    .resp_ready_o (resp_ready_o )
  );

  int unsigned errors = 0;

  task automatic check(input logic ok, input string what);
    if (!ok) begin
      errors++;
      $display("[tb_tcdm_id_remapper] FAIL: %s", what);
    end
  endtask

  // Step half a clock: end on a negedge, so every check reads a settled state.
  task automatic tick();
    @(negedge clk);
    #0;
  endtask

  // Clocked sampler. At a posedge the right-hand side is still the PRE-edge
  // value, which is exactly the id the request was accepted with; reading it
  // any other way races, because out_id is combinational and moves to the next
  // free id on the very edge that completes the handshake.
  logic [31:0] req_id_q;
  always_ff @(posedge clk)
    if (req_valid_o && req_ready_i) req_id_q <= req_o.id;

  // Present one request on `arm` for exactly one clock, and report its id.
  task automatic send_req(input int unsigned arm, input logic [DataWidth-1:0] tag,
                          input logic [1:0] own_id, output logic [31:0] out_id);
    req_i[arm].addr  = '0;
    req_i[arm].data  = tag;
    req_i[arm].id    = own_id;
    req_i[arm].amo   = '0;
    req_i[arm].write = 1'b0;
    req_i[arm].strb  = '0;
    // The remapper's slave ready is tied high here, so it accepts as soon as
    // it is offered: valid must not survive a second edge, or it would
    // allocate a second id to this arm.
    req_valid_i[arm] = 1'b1;
    #0;
    check(req_valid_o === 1'b1, "request presented");
    check(req_o.data == tag, "request data forwarded");
    tick(); // the accepting edge
    req_valid_i[arm] = 1'b0;
    tick();
    out_id = req_id_q;
  endtask

  logic rsp_taken, rsp_taken_now;

  // Present one response and HOLD it until the remapper takes it, which is what
  // a response network does: a response cannot be withdrawn, so the tile either
  // accepts it or stops accepting from the network altogether. `taken` is 0 if
  // it was still not accepted after 32 clocks.
  task automatic hold_rsp(input logic [31:0] id, input logic [DataWidth-1:0] tag,
                          output logic taken);
    resp_i.data  = tag;
    resp_i.id    = id[snitch_pkg::MetaIdWidth-1:0];
    resp_i.write = 1'b0;
    resp_i.error = 1'b0;
    resp_valid_i = 1'b1;
    taken        = 1'b0;
    rsp_taken_now = 1'b0;
    for (int unsigned i = 0; i < 32 && !taken; i++) begin
      #0;
      if (resp_ready_o) begin
        taken = 1'b1;
        if (i == 0)
          rsp_taken_now = 1'b1;
      end else begin
        tick();
      end
    end
  endtask

  logic [31:0] id0, id1, id2;

  initial begin : p_main
    req_i        = '0;
    resp_i       = '0;
    req_valid_i  = '0;
    resp_valid_i = 1'b0;
    req_ready_i  = 1'b1;
    resp_ready_i = '0;
    repeat (4) tick();
    rst_n = 1'b1;
    repeat (2) tick();

    // ---- 1. arm 0 requests, then refuses its own response ------------------
    send_req(0, 32'hA0, 2'd2, id0);
    hold_rsp(id0, 32'hA0, rsp_taken);
    check(rsp_taken, "network accept while arm 0 refuses");
    tick(); // the response lands in arm 0's entry
    check(resp_valid_o[0] === 1'b1, "arm 0 response held");
    check(resp_o[0].data == 32'hA0, "arm 0 held data");
    check(resp_o[0].id == 2'd2, "arm 0 held id is its own id");
    resp_valid_i = 1'b0;

    // ---- 2. THE GATE: arm 1 is ready, arm 0 is not -------------------------
    send_req(1, 32'hB1, 2'd3, id1);
    resp_ready_i[1] = 1'b1;
    hold_rsp(id1, 32'hB1, rsp_taken);
    check(rsp_taken, "arm 1 response taken while arm 0 stalls");
    check(rsp_taken_now === 1'b1, "arm 1 delivered WITHOUT waiting");
    check(resp_valid_o[1] === 1'b1, "arm 1 delivered while arm 0 stalls");
    check(resp_o[1].data == 32'hB1, "arm 1 data");
    check(resp_o[1].id == 2'd3, "arm 1 id is its own id");
    tick();                 // the accepting edge
    resp_valid_i = 1'b0;    // the network's response is now gone
    tick();
    check(resp_valid_o[1] === 1'b0, "arm 1 not delivered twice");

    // ---- 3. the documented limit -------------------------------------------
    // A DIFFERENT own id on purpose: if the remapper resolved the id when the
    // arm collects the buffered response instead of when it arrived, this newer
    // request would have overwritten the mapping and arm 0 would be handed 1.
    send_req(0, 32'hA2, 2'd1, id2);
    hold_rsp(id2, 32'hA2, rsp_taken);
    check(!rsp_taken, "second response for a full arm is refused");
    check(resp_ready_o === 1'b0, "and the port says so");

    // ---- 4. drain arm 0; the refused response lands as it frees ----------
    resp_i.data     = 32'hA2;
    resp_i.id       = id2[snitch_pkg::MetaIdWidth-1:0];
    resp_valid_i    = 1'b1; // still presented: it was never taken
    resp_ready_i[0] = 1'b1;
    #0;
    check(resp_valid_o[0] === 1'b1, "arm 0 response presented");
    check(resp_o[0].data == 32'hA0, "arm 0 gets its own response");
    check(resp_o[0].id == 2'd2, "its OWN id, not the newer request's");
    tick();                // the drain, and the refused response is taken
    resp_valid_i = 1'b0;   // the network's response is now gone
    #0;
    check(resp_valid_o[0] === 1'b1, "the refused response is now held");
    check(resp_o[0].data == 32'hA2, "the refused response landed");
    check(resp_ready_o === 1'b1, "port ready for the next response");
    tick();                // drain it too
    resp_ready_i[0] = 1'b0;
    tick();
    check(resp_valid_o[0] === 1'b0, "arm 0 entry empty again");

    $display("[tb_tcdm_id_remapper] %0d errors", errors);
    $display("[EOC] Simulation ended at %t (retval = %0d).", $time,
             errors == 0 ? 0 : 1);
    $finish;
  end

  // The failure this gate exists for shows up as a hang, so bound the run.
  initial begin : p_watchdog
    repeat (3000) @(posedge clk);
    $display("[tb_tcdm_id_remapper] FAIL: timeout");
    $display("[EOC] Simulation ended at %t (retval = 1).", $time);
    $finish;
  end

endmodule : tb_tcdm_id_remapper
