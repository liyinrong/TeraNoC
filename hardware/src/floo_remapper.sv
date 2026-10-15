// Copyright 2024 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Author: Yinrong Li <yinrli@student.ethz.ch>

module floo_remapper #(
  /// Number of inputs (`> 0`).
  parameter int unsigned NumInp      = 32'd0,
  /// Number of outputs, equal to the inputs (`> 0`).
  parameter int unsigned NumOut      = 32'd0,
  /// Data width of the stream. Can be overwritten by defining the type parameter `payload_t`.
  parameter int unsigned DataWidth   = 32'd1,
  /// Payload type of the data ports, only usage of parameter `DataWidth`.
  parameter type         payload_t   = logic [DataWidth-1:0],
  /// Max number of routers in a remapping group.
  parameter int unsigned GroupSize   = 32'd2,
  /// Interleaved group.
  parameter bit          Interleaved = 1'b0
) (
  input  logic              clk_i,
  input  logic              rst_ni,

  input  payload_t [NumInp-1:0] inp_data_i,
  input  logic     [NumInp-1:0] inp_valid_i,
  output logic     [NumInp-1:0] inp_ready_o,

  output payload_t [NumOut-1:0] oup_data_o,
  output logic     [NumOut-1:0] oup_valid_o,
  input  logic     [NumOut-1:0] oup_ready_i
);

// pragma translate_off
`ifndef VERILATOR
  initial begin : p_assert
    assert(NumInp == NumOut)
      else $fatal(1, "Number of inputs and outputs must be equal.");
    assert ((NumInp % GroupSize) == 0)
      else $fatal(1, "Number of inputs must be divisible by GroupSize.");
    assert ((GroupSize & (GroupSize - 1)) == 0)
      else $fatal(1, "Group size must be a power of 2.");
  end
`endif
// pragma translate_on

localparam int NumGroup = NumInp / GroupSize;
localparam int SelWidth   = $clog2(GroupSize);
localparam type sel_oup_t = logic [SelWidth-1:0];

generate
  for (genvar g = 0; g < NumGroup; g++) begin : gen_remapper
    localparam int start_idx  = g * GroupSize;

    // sel_q[i]: the output of input i. inv_q[o]: the input of output o. sel_q
    // rotates by one input per cycle, so every inv_q entry steps down by one.
    sel_oup_t [GroupSize-1:0] sel_q, sel_d, inv_q, inv_d;

    always_ff @(posedge clk_i or negedge rst_ni) begin : p_sel_regs
      if(!rst_ni) begin
        for(int i = 0; i < (GroupSize / 2); i++) begin : gen_sel_even
          sel_q[i] <= (i * 2);
          inv_q[i * 2] <= i;
        end
        for(int i = (GroupSize / 2); i < GroupSize; i++) begin : gen_sel_odd
          sel_q[i] <= (i * 2 - GroupSize + 1);
          inv_q[i * 2 - GroupSize + 1] <= i;
        end
      end else begin
        sel_q <= sel_d;
        inv_q <= inv_d;
      end
    end

    assign sel_d = {sel_q[0], sel_q[GroupSize-1:1]};

    // Group port i is flat port g + i * NumGroup when interleaved, else start_idx + i.
    payload_t [GroupSize-1:0] grp_inp_data, grp_oup_data;
    logic     [GroupSize-1:0] grp_inp_valid, grp_inp_ready, grp_oup_valid, grp_oup_ready;
    for (genvar i = 0; i < GroupSize; i++) begin : gen_port
      localparam int unsigned Idx = Interleaved ? g + i * NumGroup : start_idx + i;
      assign grp_inp_data[i]  = inp_data_i[Idx];
      assign grp_inp_valid[i] = inp_valid_i[Idx];
      assign inp_ready_o[Idx] = grp_inp_ready[i];
      assign oup_data_o[Idx]  = grp_oup_data[i];
      assign oup_valid_o[Idx] = grp_oup_valid[i];
      assign grp_oup_ready[i] = oup_ready_i[Idx];
    end

    // sel_q is a permutation, so every output has exactly one source and nothing
    // needs arbitrating: a plain mux per output, selected straight from flops.
    for (genvar o = 0; o < GroupSize; o++) begin : gen_oup
      assign inv_d[o]         = inv_q[o] - sel_oup_t'(1);
      assign grp_oup_data[o]  = grp_inp_data[inv_q[o]];
      assign grp_oup_valid[o] = grp_inp_valid[inv_q[o]];
      // pragma translate_off
      `ifndef VERILATOR
      assert property (@(posedge clk_i) disable iff (!rst_ni) sel_q[inv_q[o]] == sel_oup_t'(o))
        else $fatal(1, "floo_remapper: inv_q is not the inverse of sel_q");
      `endif
      // pragma translate_on
    end
    for (genvar i = 0; i < GroupSize; i++) begin : gen_inp
      assign grp_inp_ready[i] = grp_oup_ready[sel_q[i]];
    end

  end
endgenerate

endmodule
