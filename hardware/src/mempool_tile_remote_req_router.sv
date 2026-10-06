// Copyright 2021 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

`include "mempool/mempool.svh"

// Generates the per-core remote-request routing metadata from the (already
// scrambled) target address: the remote TCDM target address, the target group
// id, and the read/write demux select that picks the remote request port.
module mempool_tile_remote_req_router
  import mempool_pkg::*;
  import cf_math_pkg::idx_width;
#(
  parameter int unsigned NumCoresPerTile              = 4,
  parameter int unsigned NumNarrowRemoteReqPortsPerTile     = 4,
  parameter int unsigned NumNarrowRoReqPortsPerTile   = 2,
  parameter int unsigned NumNarrowWoReqPortsPerTile   = 2,
  parameter int unsigned NumNarrowRwReqPortsPerTile = 0,
  parameter int unsigned NumBanksPerTile              = 16,
  parameter int unsigned NumTilesPerGroup             = 4,
  parameter int unsigned NumGroups                    = 4,
  parameter int unsigned ByteOffset                   = 2,
  // Wide TCDM plane client ports and the tile's wide remote links. As on the
  // narrow side, the same-group links come first, then the inter-group RO and
  // WO request ports.
  parameter int unsigned NumWideLocalPorts             = 1,
  parameter int unsigned NumWideIntraGroupPortsPerTile = 1,
  parameter int unsigned NumWideRoReqPortsPerTile      = 1,
  parameter int unsigned NumWideWoReqPortsPerTile      = 1,
  // Bytes in one wide beat: the bits below it never select a port.
  parameter int unsigned WideBeatBytes                 = 64,
  // Payload-bearing request ports: the RW and WO classes share one port range.
  localparam int unsigned NumNarrowPayloadReqPortsPerTile =
      NumNarrowRwReqPortsPerTile + NumNarrowWoReqPortsPerTile,
  localparam int unsigned NumWideRemoteReqPortsPerTile =
      NumWideIntraGroupPortsPerTile + NumWideRoReqPortsPerTile + NumWideWoReqPortsPerTile
) (
  input  group_id_t                                                           group_id_i,
  input  logic                [NumCoresPerTile-1:0]                           remote_req_interco_wen_i,
  input  logic                [NumCoresPerTile-1:0]                           remote_req_interco_amoen_i,
  input  addr_t               [NumCoresPerTile-1:0]                           remote_req_interco_addr_int_i,
  output tcdm_addr_t          [NumCoresPerTile-1:0]                           remote_req_tgt_addr_o,
  output group_id_t           [NumCoresPerTile-1:0]                           remote_req_tgt_group_id_o,
  output logic [NumCoresPerTile-1:0][idx_width(NumNarrowRemoteReqPortsPerTile)-1:0] remote_req_tgt_sel_o
  `ifdef USE_WIDE_TCDM_PLANE
  ,
  input  logic                [NumWideLocalPorts-1:0]                         remote_wide_req_interco_wen_i,
  input  addr_t               [NumWideLocalPorts-1:0]                         remote_wide_req_interco_addr_int_i,
  output tcdm_addr_t          [NumWideLocalPorts-1:0]                         remote_wide_req_tgt_addr_o,
  output group_id_t           [NumWideLocalPorts-1:0]                         remote_wide_req_tgt_group_id_o,
  output logic [NumWideLocalPorts-1:0][idx_width(NumWideRemoteReqPortsPerTile)-1:0] remote_wide_req_tgt_sel_o
  `endif
);

  /*******************
   *   Core De/mux   *
   *******************/
  group_id_t           [NumCoresPerTile-1:0]                    tgt_group_id;
  logic                [NumCoresPerTile-1:0]                    group_id_is_local;

  for (genvar c = 0; c < NumCoresPerTile; c++) begin: gen_core_mux
    assign tgt_group_id[c] = remote_req_interco_addr_int_i[c][ByteOffset + $clog2(NumBanksPerTile) + $clog2(NumTilesPerGroup) +: $clog2(NumGroups)];
    assign group_id_is_local[c] = tgt_group_id[c] == group_id_i;
    assign remote_req_tgt_group_id_o[c] = tgt_group_id[c];

    // Remote target address: switch tile and bank indexes for correct upper-level
    // routing, and remove the group index.
    if (NumTilesPerGroup == 1) begin : gen_remote_req_tgt_addr
      assign remote_req_tgt_addr_o[c] =
      tcdm_addr_t'({remote_req_interco_addr_int_i[c][ByteOffset + idx_width(NumBanksPerTile) + $clog2(NumGroups) +: TCDMAddrMemWidth], // Bank address
         remote_req_interco_addr_int_i[c][ByteOffset +: idx_width(NumBanksPerTile)]}); // Tile
    end else begin : gen_remote_req_tgt_addr
      assign remote_req_tgt_addr_o[c] =
      tcdm_addr_t'({remote_req_interco_addr_int_i[c][ByteOffset + idx_width(NumBanksPerTile) + $clog2(NumTilesPerGroup) + $clog2(NumGroups) +: TCDMAddrMemWidth], // Bank address
         remote_req_interco_addr_int_i[c][ByteOffset +: idx_width(NumBanksPerTile)],                                                                              // Bank
         remote_req_interco_addr_int_i[c][ByteOffset + idx_width(NumBanksPerTile) +: $clog2(NumTilesPerGroup)]}); // Tile
    end

    // Map the requests from cores to the
    // channels with different usage:
    //                                 port id
    // ------     ------   -> local    [ 0 (lsb for local port)
    // |    | ->  |    |   -> r     Low  1
    // |Tile| ->  |xbar|   -> r     ||   2
    // |    | ->  |    |   -> rw    ||   3
    // |    | ->  |    |   -> w     \/   4
    // ------     ------   -> w    High   5 ]

    if((NumNarrowRoReqPortsPerTile > 0) || (NumNarrowWoReqPortsPerTile > 0)) begin
      assign remote_req_tgt_sel_o[c] = group_id_is_local[c] ? 0 :
                                      ~(remote_req_interco_wen_i[c] | remote_req_interco_amoen_i[c]) ?
                                       (1 + (c % (NumNarrowRoReqPortsPerTile + NumNarrowRwReqPortsPerTile))) :
                                       (1 + NumNarrowRoReqPortsPerTile + (c % (NumNarrowPayloadReqPortsPerTile)));
    end else begin
      assign remote_req_tgt_sel_o[c] = group_id_is_local[c] ? 0 : (1 + (c % (NumNarrowRemoteReqPortsPerTile - 1)));
    end
  end

  `ifdef USE_WIDE_TCDM_PLANE
  /*******************
   *   Wide De/mux   *
   *******************/
  // Same metadata as the narrow path above, for the wide TCDM plane's client
  // ports: target address, target group, and the remote-port select. Same-group
  // traffic spreads over the intra-group links by target tile. Inter-group writes
  // take a WO port, reads an RO port picked by a hash: the parity of the beat,
  // tile and three low row bits of the address XOR the client's slot. One engine
  // then alternates RO ports along its stream, and clients hitting the same
  // address take different ports, while a client keeps one port per address.
  localparam int unsigned WideTileBit = ByteOffset + $clog2(NumBanksPerTile);
  localparam int unsigned WideRowBit  =
      WideTileBit + $clog2(NumTilesPerGroup) + $clog2(NumGroups);
  localparam addr_t WideHashMask = addr_t'(
      (64'(1) << $clog2(WideBeatBytes)) |
      (((64'(1) << $clog2(NumTilesPerGroup)) - 1) << WideTileBit) |
      (64'(7) << WideRowBit));

  // Fold the masked address bits into $clog2(nb_ports) lanes, round robin.
  function automatic logic [31:0] wide_port_hash(input addr_t addr,
                                                 input int unsigned nb_ports);
    automatic logic [31:0] hash = '0;
    automatic int unsigned lanes = $clog2(nb_ports);
    automatic int unsigned j = 0;
    for (int unsigned b = 0; b < $bits(addr_t); b++) begin
      if (WideHashMask[b]) begin
        if (lanes > 0) hash[j % lanes] = hash[j % lanes] ^ addr[b];
        j++;
      end
    end
    return hash;
  endfunction

  group_id_t [NumWideLocalPorts-1:0]                                 wide_tgt_group_id;
  logic      [NumWideLocalPorts-1:0]                                 wide_group_id_is_local;
  logic      [NumWideLocalPorts-1:0][idx_width(NumTilesPerGroup)-1:0] wide_tgt_tile;

  for (genvar w = 0; w < NumWideLocalPorts; w++) begin: gen_wide_mux
    if (NumGroups == 1) begin : gen_wide_tgt_group
      assign wide_tgt_group_id[w] = '0;
    end else begin : gen_wide_tgt_group
      assign wide_tgt_group_id[w] = remote_wide_req_interco_addr_int_i[w][ByteOffset + $clog2(NumBanksPerTile) + $clog2(NumTilesPerGroup) +: $clog2(NumGroups)];
    end
    assign wide_group_id_is_local[w]         = wide_tgt_group_id[w] == group_id_i;
    assign remote_wide_req_tgt_group_id_o[w] = wide_tgt_group_id[w];

    if (NumTilesPerGroup == 1) begin : gen_wide_tgt_tile
      assign wide_tgt_tile[w] = '0;
    end else begin : gen_wide_tgt_tile
      assign wide_tgt_tile[w] = remote_wide_req_interco_addr_int_i[w][ByteOffset + $clog2(NumBanksPerTile) +: idx_width(NumTilesPerGroup)];
    end

    // Remote target address: same {bank address, bank, tile} layout the narrow
    // path builds, with the group index removed.
    if (NumTilesPerGroup == 1) begin : gen_remote_wide_req_tgt_addr
      assign remote_wide_req_tgt_addr_o[w] =
      tcdm_addr_t'({remote_wide_req_interco_addr_int_i[w][ByteOffset + idx_width(NumBanksPerTile) + $clog2(NumGroups) +: TCDMAddrMemWidth], // Bank address
         remote_wide_req_interco_addr_int_i[w][ByteOffset +: idx_width(NumBanksPerTile)]}); // Bank
    end else begin : gen_remote_wide_req_tgt_addr
      assign remote_wide_req_tgt_addr_o[w] =
      tcdm_addr_t'({remote_wide_req_interco_addr_int_i[w][ByteOffset + idx_width(NumBanksPerTile) + $clog2(NumTilesPerGroup) + $clog2(NumGroups) +: TCDMAddrMemWidth], // Bank address
         remote_wide_req_interco_addr_int_i[w][ByteOffset +: idx_width(NumBanksPerTile)],                                                                             // Bank
         remote_wide_req_interco_addr_int_i[w][ByteOffset + idx_width(NumBanksPerTile) +: $clog2(NumTilesPerGroup)]}); // Tile
    end

    assign remote_wide_req_tgt_sel_o[w] =
        idx_width(NumWideRemoteReqPortsPerTile)'(
            wide_group_id_is_local[w] ?
                (wide_tgt_tile[w] % NumWideIntraGroupPortsPerTile) :
            remote_wide_req_interco_wen_i[w] ?
                (NumWideIntraGroupPortsPerTile + NumWideRoReqPortsPerTile +
                 ((wide_port_hash(remote_wide_req_interco_addr_int_i[w], NumWideWoReqPortsPerTile)
                   ^ w) % NumWideWoReqPortsPerTile)) :
                (NumWideIntraGroupPortsPerTile +
                 ((wide_port_hash(remote_wide_req_interco_addr_int_i[w], NumWideRoReqPortsPerTile)
                   ^ w) % NumWideRoReqPortsPerTile)));
  end
  `endif

endmodule
