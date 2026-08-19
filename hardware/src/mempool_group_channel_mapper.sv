// Copyright 2026 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Concentrates generic tile streams onto two contiguous classes of physical
// group channels. Each logical port has a fixed class-local channel; the
// stream crossbar arbitrates fairly when multiple ports share that channel.
module mempool_group_channel_mapper
  import cf_math_pkg::idx_width;
#(
  parameter int unsigned NumInp       = 1,
  parameter int unsigned NumClass0    = 1,
  parameter int unsigned NumClass1    = 0,
  parameter type         payload_t    = logic,
  localparam int unsigned NumOut      = NumClass0 + NumClass1,
  localparam int unsigned SelectWidth = idx_width(NumOut)
) (
  input  logic                                      clk_i,
  input  logic                                      rst_ni,

  input  payload_t                 [NumInp-1:0]     inp_data_i,
  input  logic                     [NumInp-1:0]     inp_valid_i,
  output logic                     [NumInp-1:0]     inp_ready_o,
  input  logic                     [NumInp-1:0]     inp_class1_i,

  output payload_t                 [NumOut-1:0]     oup_data_o,
  output logic                     [NumOut-1:0]     oup_valid_o,
  input  logic                     [NumOut-1:0]     oup_ready_i
);

  logic [NumInp-1:0][SelectWidth-1:0] channel_sel;

  for (genvar i = 0; i < NumInp; i++) begin : gen_channel_sel
    if (NumClass0 == 0) begin : gen_class1_only
      assign channel_sel[i] = SelectWidth'(i % NumClass1);
    end else if (NumClass1 == 0) begin : gen_class0_only
      assign channel_sel[i] = SelectWidth'(i % NumClass0);
    end else begin : gen_two_classes
      assign channel_sel[i] = inp_class1_i[i] ?
          SelectWidth'(NumClass0 + (i % NumClass1)) :
          SelectWidth'(i % NumClass0);
    end
  end

  stream_xbar #(
    .NumInp   (NumInp   ),
    .NumOut   (NumOut   ),
    .payload_t(payload_t),
    .AxiVldRdy(1'b1     ),
    .LockIn   (1'b1     )
  ) i_stream_xbar (
    .clk_i  (clk_i       ),
    .rst_ni (rst_ni      ),
    .flush_i(1'b0        ),
    .rr_i   ('0          ),
    .data_i (inp_data_i  ),
    .sel_i  (channel_sel ),
    .valid_i(inp_valid_i ),
    .ready_o(inp_ready_o ),
    .data_o (oup_data_o  ),
    .idx_o  (/* unused */),
    .valid_o(oup_valid_o ),
    .ready_i(oup_ready_i )
  );

  // pragma translate_off
  initial begin : p_assert
    assert (NumInp > 0)
      else $fatal(1, "[mempool_group_channel_mapper] At least one input is required.");
    assert (NumOut > 0)
      else $fatal(1, "[mempool_group_channel_mapper] At least one output is required.");
    // A class WIDER than the input count is legal and deliberate. The select
    // below is `i % NumClass`, so with NumClass > NumInp the upper channels are
    // never picked here -- they become reachable only once a downstream
    // floo_remapper rotates the binding (noc_wide_router_remapping). At most
    // NumInp channels can still be busy in any one cycle; the extra channels buy
    // independent meshes, not injection bandwidth. The caller owns that pairing:
    // hardware/Makefile rejects an over-wide class whose remapper is off, since
    // without rotation those channels are provably dead silicon.
  end
  // pragma translate_on

endmodule
