// Copyright 2021 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
//
// Description: Boot ROM with a configurable data width

module bootrom #(
  parameter int unsigned DataWidth = 512,
  parameter int unsigned AddrWidth = 32
) (
  input  logic                 clk_i,
  input  logic                 req_i,
  input  logic [AddrWidth-1:0] addr_i,
  output logic [DataWidth-1:0] rdata_o
);

  import bootrom_image_pkg::*;

  localparam int unsigned WordsPerBeat =
      DataWidth >= BootromWordWidth ? DataWidth / BootromWordWidth : 1;
  localparam int unsigned RomSize =
      (BootromWordCount + WordsPerBeat - 1) / WordsPerBeat;
  localparam int unsigned AddrBits = RomSize > 1 ? $clog2(RomSize) : 1;
  localparam int unsigned DataOffset = $clog2(DataWidth / 8);

  logic [AddrBits-1:0] addr_q;

  // pragma translate_off
  initial begin : p_assert
    assert (DataWidth >= BootromWordWidth)
      else $fatal(1, "DataWidth must be at least %0d bits.", BootromWordWidth);
    assert ((DataWidth % BootromWordWidth) == 0)
      else $fatal(1, "DataWidth must be a multiple of %0d bits.", BootromWordWidth);
    assert ((DataWidth & (DataWidth - 1)) == 0)
      else $fatal(1, "DataWidth must be a power of two.");
  end
  // pragma translate_on

  always_ff @(posedge clk_i) begin
    if (req_i) begin
      addr_q <= addr_i[AddrBits-1+DataOffset:DataOffset];
    end
  end

  // Pack fixed 32-bit boot image words into the configured AXI beat width.
  // The bounds check zero-fills a partial final beat and suppresses address Xes.
  always_comb begin
    rdata_o = '0;
    for (int unsigned i = 0; i < WordsPerBeat; i++) begin
      if ((addr_q * WordsPerBeat + i) < BootromWordCount) begin
        rdata_o[i * BootromWordWidth +: BootromWordWidth] =
            BootromWords[addr_q * WordsPerBeat + i];
      end
    end
  end

endmodule
