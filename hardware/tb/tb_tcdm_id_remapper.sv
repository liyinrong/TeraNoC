// Copyright 2026 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Exercise the LSU contract: a sink may wait for valid before asserting ready.
// Check identity and exactly-once delivery while the shared response id changes.
// This does not assume that an indefinitely stalled sink must be bypassed.
`ifndef REMAPPER_DUT
`define REMAPPER_DUT tcdm_id_remapper
`endif

module tb_tcdm_id_remapper;
  timeunit 1ns;
  timeprecision 1ps;

  import snitch_pkg::*;
  logic clk = 0;
  logic rst_n = 0;
  always #1ns clk = ~clk;

  dreq_t [1:0] req_i;
  dresp_t [1:0] resp_o;
  logic [1:0] req_valid_i, req_ready_o, resp_valid_o, resp_ready_i;
  logic [1:0] allow_response;
  dreq_t req_o;
  dresp_t resp_i;
  logic req_valid_o, req_ready_i, resp_valid_i, resp_ready_o;

  // Both the scalar RVV LSU and FP LSU may assert ready in response to valid.
  assign resp_ready_i = resp_valid_o & allow_response;

  `REMAPPER_DUT #(.NumIn(2)) i_dut (
    .clk_i(clk), .rst_ni(rst_n),
    .req_i, .req_valid_i, .req_ready_o,
    .resp_o, .resp_valid_o, .resp_ready_i,
    .req_o, .req_valid_o, .req_ready_i,
    .resp_i, .resp_valid_i, .resp_ready_o
  );

  bit pending[2][2**MetaIdWidth];
  logic [31:0] expected[2][2**MetaIdWidth];
  int unsigned delivered = 0;
  int unsigned errors = 0;
  logic [MetaIdWidth-1:0] id0, id1, id2;

  task automatic check(input logic ok, input string message);
    if (ok !== 1'b1) begin
      errors++;
      $display("[id-remap] FAIL: %s", message);
    end
  endtask

  always @(posedge clk) begin
    if (rst_n) begin
      for (int arm = 0; arm < 2; arm++) begin
        if (resp_valid_o[arm] && resp_ready_i[arm]) begin
          check(!$isunknown(resp_o[arm]), "response contains X");
          check(pending[arm][resp_o[arm].id], "unexpected or duplicate response");
          check(resp_o[arm].data == expected[arm][resp_o[arm].id], "response data/id");
          pending[arm][resp_o[arm].id] = 0;
          delivered++;
        end
      end
    end
  end

  task automatic issue(input int arm, input logic [MetaIdWidth-1:0] own_id,
                       input logic [31:0] data, output logic [MetaIdWidth-1:0] net_id);
    @(negedge clk);
    pending[arm][own_id] = 1;
    expected[arm][own_id] = data;
    req_i[arm] = '0;
    req_i[arm].id = own_id;
    req_i[arm].data = data;
    req_valid_i[arm] = 1;
    do @(posedge clk); while (!req_ready_o[arm]);
    net_id = req_o.id;
    @(negedge clk);
    req_valid_i[arm] = 0;
  endtask

  task automatic respond(input logic [MetaIdWidth-1:0] net_id,
                         input logic [31:0] data,
                         input logic [MetaIdWidth-1:0] idle_id);
    @(negedge clk);
    resp_i = '0;
    resp_i.id = net_id;
    resp_i.data = data;
    resp_valid_i = 1;
    do @(posedge clk); while (!resp_ready_o);
    @(negedge clk);
    resp_valid_i = 0;
    // Idle bus contents are not required to preserve the previous selection.
    resp_i.id = idle_id;
    repeat (4) @(negedge clk);
  endtask

  initial begin
    req_i = '0;
    req_valid_i = '0;
    req_ready_i = 1;
    resp_i = '0;
    resp_valid_i = 0;
    allow_response = '1;
    repeat (4) @(negedge clk);
    rst_n = 1;

    issue(0, 2, 32'hA0, id0);
    issue(1, 3, 32'hB1, id1);
    // Out-of-order return across arms; the idle id selects the other arm.
    respond(id1, 32'hB1, id0);
    respond(id0, 32'hA0, id1);
    check(delivered == 2, "both valid-dependent sinks retire exactly once");

    // Recycle the network id with a different requester-owned id.
    issue(1, 1, 32'hC2, id2);
    respond(id2, 32'hC2, id0);
    check(delivered == 3, "recycled id retains requester identity");

    // A finite sink stall must preserve the response until it is accepted.
    issue(0, 4, 32'hD3, id0);
    allow_response[0] = 0;
    fork
      respond(id0, 32'hD3, id1);
      begin
        repeat (5) @(negedge clk);
        allow_response[0] = 1;
      end
    join
    check(delivered == 4, "stalled response retires exactly once");
    foreach (pending[arm, id]) check(!pending[arm][id], "all requests completed");
    $display("[id-remap] errors=%0d delivered=%0d", errors, delivered);
    $display("[EOC] Simulation ended (retval = %0d).", errors == 0 ? 0 : 1);
    $finish;
  end

  initial begin
    repeat (500) @(posedge clk);
    $fatal(1, "[id-remap] response progress timeout");
  end
endmodule
