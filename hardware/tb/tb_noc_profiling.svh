// Copyright 2025 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

`ifndef TB_NOC_PROFILING_SVH_
`define TB_NOC_PROFILING_SVH_

`ifndef TARGET_SYNTHESIS
`ifndef TARGET_VERILATOR
  logic [63:0] cycle_q;
  always_ff @(posedge clk or negedge rst_n) begin
    if(~rst_n) begin
      cycle_q   <= '0;
    end else begin
      cycle_q   <= cycle_q + 64'd1;
    end
  end

`ifdef NOC_PROFILING
  // Hierarchical path to group g's FlooNoC wrapper (i_group).
  `define NOC_GRP(grp) dut.i_mempool_cluster.gen_groups_x[(grp) / NumY].gen_groups_y[(grp) % NumY].gen_rtl_group.i_group

  // ONE FULL per-router / per-tile NoC trace; the export-time --noc-slices flag
  // chooses the granularity. Each line is one of:
  //   S <portidx> <io> <start> <end> <state>   run-length: idle(0)/stall(1)/read(2)/write(3)
  //   P ... one accepted REQUEST flit (addr + XY routing), emitted every handshake cycle
  //   portidx 0..3 = N/E/S/W mesh ports, 4 = local; io 0=input, 1=output.
  // To stay under VCS's open-file limit, files are MERGED and lines PREFIXED with a demux idx:
  //   routers -> router_g<g>_{req,resp}.log  (one per group; <idx> = in-group router id
  //              <rid>, a flat slot t*ports+p -- routers are remapped so NOT tile/port)
  //   cores   -> core_g<g>_t<t>.log          (one per tile; <idx> = FLAT local-port
  //              index ports_per_core*c + p: the Snitch scalar data port at p=0
  //              and every Spatz vector port at p>=1)
  //   RedMulE -> rm_g<g>_t<t>.log            (one per tile; <idx> = master port,
  //              always 0 on the wide plane, which has one client per tile)
  //   tiles   -> tile_g<g>_t<t>.log          (already one per tile; NO prefix)
  // core_ and rm_ open with `D <data_width_bits>`: the width of ONE record, so a
  // consumer can normalise request counts to bytes across the two planes.
  // This directory holds the whole LIFE OF EACH MEMORY REQUEST and nothing else;
  // requester-internal state belongs in tb_<engine>_profiling.svh.
  string  app, log_path;
  integer retval;
  // MERGED file handles: routers -> ONE req + ONE resp file per group (line tagged with
  // router id); core ports / RedMulE master ports -> ONE file per tile (line tagged with
  // core-port / port idx). core_ covers the Snitch scalar data port AND every
  // Spatz vector port, on the flat local-port axis.
  int f_rreq  [NumGroups];
  int f_rresp [NumGroups];
  int f_tile  [NumGroups][NumTilesPerGroup];
  int f_core    [NumGroups][NumTilesPerGroup];
  // RMPorts>=1 keeps the state-register array legal when RedMulE is disabled.
  localparam int unsigned RMPorts = (RMMasterPorts > 0) ? RMMasterPorts : 1;
  int f_rm    [NumGroups][NumTilesPerGroup];
`ifdef USE_WIDE_TCDM_PLANE
  // WP2 wide plane. The engine port keeps the rm_ family: same grammar, same
  // fields, one client instead of several. What differs is the width of one
  // record -- WideTcdmDataWidth bits instead of 32 -- and that is carried by
  // the file's own `D <data_width_bits>` header rather than by the file name,
  // so a consumer normalises to bytes from the data instead of from a naming
  // convention.
  //   router_wide_g<g>_{req,resp}  the wide net's own routers (one file per group)
  // The wide tile boundary rides the EXISTING tile_g<g>_t<t>.log under its own
  // record letters (SW/PW/QW), so a tile's two planes stay in one file.
  int f_wrreq  [NumGroups];
  int f_wrresp [NumGroups];
`endif

  // Global destination TILE of a request, decoded from the tile-side SCRAMBLED
  // address (i.e. after the seq/interleave swap and the tile-id remap), so an
  // offline pass can split traffic into local-tile / intra-group / inter-group
  // without re-implementing mempool_addr_scrambler. Mirrors the tile's own
  // local/remote address map (mempool_tile.sv:899) and its bank-address slice
  // (mempool_tile.sv:910); destination group = dst / NumTilesPerGroup
  // (mempool_tile.sv:91). Returns -1 for a non-TCDM (SoC/L2) access.
  localparam int unsigned MemTileLsb  = ByteOffset + idx_width(NumBanksPerTile);
  localparam int unsigned MemTileBits = idx_width(NumTiles);
  // The testbench instantiates the DUT with TCDMBaseAddr = 32'h0
  // (mempool_tb.sv:125); same test as mempool_addr_scrambler's not_io_address.
  localparam addr_t       MemTcdmBase = 32'h0;

  function automatic int mem_dst_tile(input addr_t scrambled);
    mem_dst_tile = ((scrambled & mempool_pkg::TCDMMask) == MemTcdmBase)
                 ? int'(scrambled[MemTileLsb +: MemTileBits]) : -1;
  endfunction

  initial begin
    void'($value$plusargs("APP=%s", app));
    log_path = "noc_profiling";
    retval   = $system({"mkdir -p ", log_path});
    // Tiles keep their id (a tile is a fixed compute node, never remapped). Routers
    // are identified by a FLAT router id within the group's network -- because the
    // req/resp remappers shuffle which logical traffic lands on which physical
    // router, the [tile][port] index is NOT a tile/port assignment, just a physical
    // slot, so we expose it as a plain router id (r = t*NumPortsPerTile + p).
    for (int g = 0; g < NumGroups; g++) begin
      f_rreq[g]  = $fopen($sformatf("%s/router_g%0d_req.log",  log_path, g), "w");
      f_rresp[g] = $fopen($sformatf("%s/router_g%0d_resp.log", log_path, g), "w");
`ifdef USE_WIDE_TCDM_PLANE
      f_wrreq[g]  = $fopen($sformatf("%s/router_wide_g%0d_req.log",  log_path, g), "w");
      f_wrresp[g] = $fopen($sformatf("%s/router_wide_g%0d_resp.log", log_path, g), "w");
`endif
      for (int t = 0; t < NumTilesPerGroup; t++) begin
        f_tile[g][t] = $fopen($sformatf("%s/tile_g%0d_t%0d.log", log_path, g, t), "w");
        f_core[g][t]   = $fopen($sformatf("%s/core_g%0d_t%0d.log",   log_path, g, t), "w");
        // `D <data_width_bits>`: one record of this file moves that many bits.
        // Core ports are always the 32-bit narrow plane; the RedMulE port is
        // 32 or WideTcdmDataWidth depending on which plane carries it, which is
        // why the width is data and not part of the file name.
        $fwrite(f_core[g][t], "D 32\n");
        if (t < NumRMTilesPerGroup) begin
          f_rm[g][t] = $fopen($sformatf("%s/rm_g%0d_t%0d.log", log_path, g, t), "w");
`ifndef USE_WIDE_TCDM_PLANE
          $fwrite(f_rm[g][t], "D 32\n");
`else
          $fwrite(f_rm[g][t], "D %0d\n", mempool_pkg::WideTcdmDataWidth);
`endif
        end
      end
    end
  end

  // Module-scope state registers so the end-of-sim `final` below can flush them.
  // router: [group][tile][port][portidx 0..4][io 0..1]
  logic [1:0]  rsq_st    [NumGroups][NumTilesPerGroup][NumNarrowPayloadReqPortsPerTile][5][2];
  logic [63:0] rsq_start [NumGroups][NumTilesPerGroup][NumNarrowPayloadReqPortsPerTile][5][2];
  logic [1:0]  rsp_st    [NumGroups][NumTilesPerGroup][NumNarrowRemoteRespPortsPerTile-1][5][2];
  logic [63:0] rsp_start [NumGroups][NumTilesPerGroup][NumNarrowRemoteRespPortsPerTile-1][5][2];
  // tile: [group][tile][port]
  logic [1:0]  ts_mreq_st [NumGroups][NumTilesPerGroup][NumNarrowRemoteReqPortsPerTile];  logic [63:0] ts_mreq_s [NumGroups][NumTilesPerGroup][NumNarrowRemoteReqPortsPerTile];
  logic [1:0]  ts_sreq_st [NumGroups][NumTilesPerGroup][NumNarrowRemoteReqPortsPerTile];  logic [63:0] ts_sreq_s [NumGroups][NumTilesPerGroup][NumNarrowRemoteReqPortsPerTile];
  logic [1:0]  ts_mrsp_st [NumGroups][NumTilesPerGroup][NumNarrowRemoteRespPortsPerTile]; logic [63:0] ts_mrsp_s [NumGroups][NumTilesPerGroup][NumNarrowRemoteRespPortsPerTile];
  logic [1:0]  ts_srsp_st [NumGroups][NumTilesPerGroup][NumNarrowRemoteRespPortsPerTile]; logic [63:0] ts_srsp_s [NumGroups][NumTilesPerGroup][NumNarrowRemoteRespPortsPerTile];
  // PE (core data port): req = data_q* (out, read/write by qwrite), resp = data_p* (in).
  // One entry per CORE PORT, not per core: the cores' TCDM ports are flattened
  // as port p of core c = core_port_base(c) + p (mempool_pkg.sv; a Snitch-only
  // core has one port). Scalar builds have one port per core, so the
  // index stays the core id; Spatz builds add the NumMemPortsPerSpatz vector
  // ports after each core's scalar port.
  localparam int unsigned NumCorePorts = NumCorePortsPerTile;
  logic [1:0]  core_req_st [NumGroups][NumTilesPerGroup][NumCorePorts]; logic [63:0] core_req_s [NumGroups][NumTilesPerGroup][NumCorePorts];
  logic [1:0]  core_rsp_st [NumGroups][NumTilesPerGroup][NumCorePorts]; logic [63:0] core_rsp_s [NumGroups][NumTilesPerGroup][NumCorePorts];
  // RedMulE master-port state (RedMulE tiles only); same S/P format as PE cores.
  logic [1:0]  rm_req_st [NumGroups][NumTilesPerGroup][RMPorts]; logic [63:0] rm_req_s [NumGroups][NumTilesPerGroup][RMPorts];
  logic [1:0]  rm_rsp_st [NumGroups][NumTilesPerGroup][RMPorts]; logic [63:0] rm_rsp_s [NumGroups][NumTilesPerGroup][RMPorts];

`ifdef USE_WIDE_TCDM_PLANE
  // ---- WP2 wide plane state ----
  // The wide engine port is ONE port per RedMulE tile (mempool_tile.sv's
  // WidePortIdx), so it needs no port dimension. On the shared local-port axis
  // (see the PE note above) it sits where the narrow RedMulE ports would start,
  // i.e. at index NumCorePorts -- the tap emits that index as the `core` field
  // of every wide packet so a wide NoC/bank record keys the same way a narrow
  // RedMulE one does, and _req_eng renders it as RM0.
  localparam int unsigned WideEngCoreId = NumCorePorts;
  logic [1:0]  wrm_req_st [NumGroups][NumTilesPerGroup]; logic [63:0] wrm_req_s [NumGroups][NumTilesPerGroup];
  logic [1:0]  wrm_rsp_st [NumGroups][NumTilesPerGroup]; logic [63:0] wrm_rsp_s [NumGroups][NumTilesPerGroup];
  // wide tile boundary: [group][tile][port], same four sides as the narrow one
  logic [1:0]  wts_mreq_st [NumGroups][NumTilesPerGroup][NumWideRemoteReqPortsPerTile];  logic [63:0] wts_mreq_s [NumGroups][NumTilesPerGroup][NumWideRemoteReqPortsPerTile];
  logic [1:0]  wts_sreq_st [NumGroups][NumTilesPerGroup][NumWideRemoteReqPortsPerTile];  logic [63:0] wts_sreq_s [NumGroups][NumTilesPerGroup][NumWideRemoteReqPortsPerTile];
  logic [1:0]  wts_mrsp_st [NumGroups][NumTilesPerGroup][NumWideRemoteRespPortsPerTile]; logic [63:0] wts_mrsp_s [NumGroups][NumTilesPerGroup][NumWideRemoteRespPortsPerTile];
  logic [1:0]  wts_srsp_st [NumGroups][NumTilesPerGroup][NumWideRemoteRespPortsPerTile]; logic [63:0] wts_srsp_s [NumGroups][NumTilesPerGroup][NumWideRemoteRespPortsPerTile];
  // wide routers: the two request classes share one flat lane axis per group --
  // the RO port routers (tile * ports + port) first, then the merged WO
  // channels -- likewise the RW response port routers and the WO response
  // channels. The class a flat index belongs to is emitted as the `cls` column,
  // so one file per group per net still holds every wide router.
  localparam int unsigned NumWROReqLanes  = NumTilesPerGroup * NumWideRoReqPortsPerTile;
  localparam int unsigned NumWRWRespLanes = NumTilesPerGroup * NumWideRwRespPortsPerTile;
  localparam int unsigned NumWReqCh  = NumWROReqLanes + NumWideWoReqChannelsPerGroup;
  localparam int unsigned NumWRespCh = NumWRWRespLanes + NumWideWoRespChannelsPerGroup;
  logic [1:0]  wrq_st [NumGroups][NumWReqCh][5][2];  logic [63:0] wrq_s [NumGroups][NumWReqCh][5][2];
  logic [1:0]  wrs_st [NumGroups][NumWRespCh][5][2]; logic [63:0] wrs_s [NumGroups][NumWRespCh][5][2];
`endif

  // ------------------------------------------------------------
  // Router port capture (4 mesh dirs + local, input + output) -> per-router file.
  // S lines RLE idle/stall/read/write; P lines log every accepted request flit
  // (req routers only). read/write split via payload.wen.
  // ------------------------------------------------------------
  generate
    for (genvar g = 0; g < NumGroups; g++) begin : gen_rstate_g
      always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
          rsq_st[g] <= '{default: '0}; rsq_start[g] <= '{default: '0};
          rsp_st[g] <= '{default: '0}; rsp_start[g] <= '{default: '0};
        end else begin
          for (int t = 0; t < NumTilesPerGroup; t++) begin
            // ---- narrow payload-bearing RW/WO request routers ----
            for (int p = 0; p < NumNarrowPayloadReqPortsPerTile; p++) begin
              automatic int rid = t*NumNarrowPayloadReqPortsPerTile + p;  // merged-file router tag
              for (int d = 0; d < 4; d++) begin
                automatic logic [1:0] si = `NOC_GRP(g).floo_tcdm_narrow_rw_req_valid_in_trans[t][p][d]
                  ? (`NOC_GRP(g).floo_tcdm_narrow_rw_req_ready_out_trans[t][p][d]
                     ? (`NOC_GRP(g).floo_tcdm_narrow_rw_req_in_trans[t][p][d].payload.wen ? 2'd3 : 2'd2) : 2'd1) : 2'd0;
                automatic logic [1:0] so = `NOC_GRP(g).floo_tcdm_narrow_rw_req_valid_out_trans[t][p][d]
                  ? (`NOC_GRP(g).floo_tcdm_narrow_rw_req_ready_in_trans[t][p][d]
                     ? (`NOC_GRP(g).floo_tcdm_narrow_rw_req_out_trans[t][p][d].payload.wen ? 2'd3 : 2'd2) : 2'd1) : 2'd0;
                if (si != rsq_st[g][t][p][d][0]) begin
                  $fwrite(f_rreq[g], "%0d S %0d 0 %0d %0d %0d\n", rid, d, rsq_start[g][t][p][d][0], cycle_q, rsq_st[g][t][p][d][0]);
                  rsq_st[g][t][p][d][0] <= si; rsq_start[g][t][p][d][0] <= cycle_q;
                end
                if (so != rsq_st[g][t][p][d][1]) begin
                  $fwrite(f_rreq[g], "%0d S %0d 1 %0d %0d %0d\n", rid, d, rsq_start[g][t][p][d][1], cycle_q, rsq_st[g][t][p][d][1]);
                  rsq_st[g][t][p][d][1] <= so; rsq_start[g][t][p][d][1] <= cycle_q;
                end
                // P: portidx io cycle wen tgt_addr dst_x dst_y src_x src_y src_tile core meta_id
                if (si >= 2)  // input handshake -> one request flit accepted
                  $fwrite(f_rreq[g], "%0d P %0d 0 %0d %0d %0h %0d %0d %0d %0d %0d %0d %0d\n", rid, d, cycle_q,
                    `NOC_GRP(g).floo_tcdm_narrow_rw_req_in_trans[t][p][d].payload.wen,
                    `NOC_GRP(g).floo_tcdm_narrow_rw_req_in_trans[t][p][d].hdr.tgt_addr,
                    `NOC_GRP(g).floo_tcdm_narrow_rw_req_in_trans[t][p][d].hdr.dst_id.x,
                    `NOC_GRP(g).floo_tcdm_narrow_rw_req_in_trans[t][p][d].hdr.dst_id.y,
                    `NOC_GRP(g).floo_tcdm_narrow_rw_req_in_trans[t][p][d].hdr.src_id.x,
                    `NOC_GRP(g).floo_tcdm_narrow_rw_req_in_trans[t][p][d].hdr.src_id.y,
                    `NOC_GRP(g).floo_tcdm_narrow_rw_req_in_trans[t][p][d].hdr.src_tile_id,
                    `NOC_GRP(g).floo_tcdm_narrow_rw_req_in_trans[t][p][d].hdr.core_id,
                    `NOC_GRP(g).floo_tcdm_narrow_rw_req_in_trans[t][p][d].hdr.meta_id);
                if (so >= 2)  // output handshake
                  $fwrite(f_rreq[g], "%0d P %0d 1 %0d %0d %0h %0d %0d %0d %0d %0d %0d %0d\n", rid, d, cycle_q,
                    `NOC_GRP(g).floo_tcdm_narrow_rw_req_out_trans[t][p][d].payload.wen,
                    `NOC_GRP(g).floo_tcdm_narrow_rw_req_out_trans[t][p][d].hdr.tgt_addr,
                    `NOC_GRP(g).floo_tcdm_narrow_rw_req_out_trans[t][p][d].hdr.dst_id.x,
                    `NOC_GRP(g).floo_tcdm_narrow_rw_req_out_trans[t][p][d].hdr.dst_id.y,
                    `NOC_GRP(g).floo_tcdm_narrow_rw_req_out_trans[t][p][d].hdr.src_id.x,
                    `NOC_GRP(g).floo_tcdm_narrow_rw_req_out_trans[t][p][d].hdr.src_id.y,
                    `NOC_GRP(g).floo_tcdm_narrow_rw_req_out_trans[t][p][d].hdr.src_tile_id,
                    `NOC_GRP(g).floo_tcdm_narrow_rw_req_out_trans[t][p][d].hdr.core_id,
                    `NOC_GRP(g).floo_tcdm_narrow_rw_req_out_trans[t][p][d].hdr.meta_id);
              end
              begin : req_local
                automatic logic [1:0] sli = `NOC_GRP(g).floo_tcdm_narrow_rw_req_to_router_vc_valid[t][p]
                  ? (`NOC_GRP(g).floo_tcdm_narrow_rw_req_to_router_vc_ready[t][p]
                     ? (`NOC_GRP(g).floo_tcdm_narrow_rw_req_to_router[t][p].payload.wen ? 2'd3 : 2'd2) : 2'd1) : 2'd0;
                automatic logic [1:0] slo = `NOC_GRP(g).floo_tcdm_narrow_rw_req_from_router_vc_valid[t][p]
                  ? (`NOC_GRP(g).floo_tcdm_narrow_rw_req_from_router_vc_ready[t][p]
                     ? (`NOC_GRP(g).floo_tcdm_narrow_rw_req_from_router[t][p].payload.wen ? 2'd3 : 2'd2) : 2'd1) : 2'd0;
                if (sli != rsq_st[g][t][p][4][0]) begin
                  $fwrite(f_rreq[g], "%0d S 4 0 %0d %0d %0d\n", rid, rsq_start[g][t][p][4][0], cycle_q, rsq_st[g][t][p][4][0]);
                  rsq_st[g][t][p][4][0] <= sli; rsq_start[g][t][p][4][0] <= cycle_q;
                end
                if (slo != rsq_st[g][t][p][4][1]) begin
                  $fwrite(f_rreq[g], "%0d S 4 1 %0d %0d %0d\n", rid, rsq_start[g][t][p][4][1], cycle_q, rsq_st[g][t][p][4][1]);
                  rsq_st[g][t][p][4][1] <= slo; rsq_start[g][t][p][4][1] <= cycle_q;
                end
                if (sli >= 2)  // local input (injection from tile) handshake
                  $fwrite(f_rreq[g], "%0d P 4 0 %0d %0d %0h %0d %0d %0d %0d %0d %0d %0d\n", rid, cycle_q,
                    `NOC_GRP(g).floo_tcdm_narrow_rw_req_to_router[t][p].payload.wen,
                    `NOC_GRP(g).floo_tcdm_narrow_rw_req_to_router[t][p].hdr.tgt_addr,
                    `NOC_GRP(g).floo_tcdm_narrow_rw_req_to_router[t][p].hdr.dst_id.x,
                    `NOC_GRP(g).floo_tcdm_narrow_rw_req_to_router[t][p].hdr.dst_id.y,
                    `NOC_GRP(g).floo_tcdm_narrow_rw_req_to_router[t][p].hdr.src_id.x,
                    `NOC_GRP(g).floo_tcdm_narrow_rw_req_to_router[t][p].hdr.src_id.y,
                    `NOC_GRP(g).floo_tcdm_narrow_rw_req_to_router[t][p].hdr.src_tile_id,
                    `NOC_GRP(g).floo_tcdm_narrow_rw_req_to_router[t][p].hdr.core_id,
                    `NOC_GRP(g).floo_tcdm_narrow_rw_req_to_router[t][p].hdr.meta_id);
                if (slo >= 2)  // local output (ejection to tile) handshake
                  $fwrite(f_rreq[g], "%0d P 4 1 %0d %0d %0h %0d %0d %0d %0d %0d %0d %0d\n", rid, cycle_q,
                    `NOC_GRP(g).floo_tcdm_narrow_rw_req_from_router[t][p].payload.wen,
                    `NOC_GRP(g).floo_tcdm_narrow_rw_req_from_router[t][p].hdr.tgt_addr,
                    `NOC_GRP(g).floo_tcdm_narrow_rw_req_from_router[t][p].hdr.dst_id.x,
                    `NOC_GRP(g).floo_tcdm_narrow_rw_req_from_router[t][p].hdr.dst_id.y,
                    `NOC_GRP(g).floo_tcdm_narrow_rw_req_from_router[t][p].hdr.src_id.x,
                    `NOC_GRP(g).floo_tcdm_narrow_rw_req_from_router[t][p].hdr.src_id.y,
                    `NOC_GRP(g).floo_tcdm_narrow_rw_req_from_router[t][p].hdr.src_tile_id,
                    `NOC_GRP(g).floo_tcdm_narrow_rw_req_from_router[t][p].hdr.core_id,
                    `NOC_GRP(g).floo_tcdm_narrow_rw_req_from_router[t][p].hdr.meta_id);
              end
            end
            // ---- resp routers (router slot p+1; no read/write split). P lines log every
            //      accepted RESPONSE flit; a resp flit has no tgt_addr. hdr carries dst_id
            //      (requester it returns to), src_id (responder), tile_id/core_id (requester).
            //   P <portidx> <io> <cycle> <dst_x> <dst_y> <src_x> <src_y> <req_tile> <req_core> <meta_id> ----
            for (int p = 0; p < NumNarrowRemoteRespPortsPerTile-1; p++) begin
              automatic int rid = t*(NumNarrowRemoteRespPortsPerTile-1) + p;  // merged-file router tag
              for (int d = 0; d < 4; d++) begin
                automatic logic [1:0] si = `NOC_GRP(g).floo_tcdm_narrow_rw_resp_valid_in_trans[t][p+1][d]
                  ? (`NOC_GRP(g).floo_tcdm_narrow_rw_resp_ready_out_trans[t][p+1][d] ? 2'd2 : 2'd1) : 2'd0;
                automatic logic [1:0] so = `NOC_GRP(g).floo_tcdm_narrow_rw_resp_valid_out_trans[t][p+1][d]
                  ? (`NOC_GRP(g).floo_tcdm_narrow_rw_resp_ready_in_trans[t][p+1][d] ? 2'd2 : 2'd1) : 2'd0;
                if (si != rsp_st[g][t][p][d][0]) begin
                  $fwrite(f_rresp[g], "%0d S %0d 0 %0d %0d %0d\n", rid, d, rsp_start[g][t][p][d][0], cycle_q, rsp_st[g][t][p][d][0]);
                  rsp_st[g][t][p][d][0] <= si; rsp_start[g][t][p][d][0] <= cycle_q;
                end
                if (so != rsp_st[g][t][p][d][1]) begin
                  $fwrite(f_rresp[g], "%0d S %0d 1 %0d %0d %0d\n", rid, d, rsp_start[g][t][p][d][1], cycle_q, rsp_st[g][t][p][d][1]);
                  rsp_st[g][t][p][d][1] <= so; rsp_start[g][t][p][d][1] <= cycle_q;
                end
                if (si >= 2)  // mesh input handshake -> one response flit accepted
                  $fwrite(f_rresp[g], "%0d P %0d 0 %0d %0d %0d %0d %0d %0d %0d %0d\n", rid, d, cycle_q,
                    `NOC_GRP(g).floo_tcdm_narrow_rw_resp_in_trans[t][p+1][d].hdr.dst_id.x,
                    `NOC_GRP(g).floo_tcdm_narrow_rw_resp_in_trans[t][p+1][d].hdr.dst_id.y,
                    `NOC_GRP(g).floo_tcdm_narrow_rw_resp_in_trans[t][p+1][d].hdr.src_id.x,
                    `NOC_GRP(g).floo_tcdm_narrow_rw_resp_in_trans[t][p+1][d].hdr.src_id.y,
                    `NOC_GRP(g).floo_tcdm_narrow_rw_resp_in_trans[t][p+1][d].hdr.tile_id,
                    `NOC_GRP(g).floo_tcdm_narrow_rw_resp_in_trans[t][p+1][d].hdr.core_id,
                    `NOC_GRP(g).floo_tcdm_narrow_rw_resp_in_trans[t][p+1][d].hdr.meta_id);
                if (so >= 2)  // mesh output handshake
                  $fwrite(f_rresp[g], "%0d P %0d 1 %0d %0d %0d %0d %0d %0d %0d %0d\n", rid, d, cycle_q,
                    `NOC_GRP(g).floo_tcdm_narrow_rw_resp_out_trans[t][p+1][d].hdr.dst_id.x,
                    `NOC_GRP(g).floo_tcdm_narrow_rw_resp_out_trans[t][p+1][d].hdr.dst_id.y,
                    `NOC_GRP(g).floo_tcdm_narrow_rw_resp_out_trans[t][p+1][d].hdr.src_id.x,
                    `NOC_GRP(g).floo_tcdm_narrow_rw_resp_out_trans[t][p+1][d].hdr.src_id.y,
                    `NOC_GRP(g).floo_tcdm_narrow_rw_resp_out_trans[t][p+1][d].hdr.tile_id,
                    `NOC_GRP(g).floo_tcdm_narrow_rw_resp_out_trans[t][p+1][d].hdr.core_id,
                    `NOC_GRP(g).floo_tcdm_narrow_rw_resp_out_trans[t][p+1][d].hdr.meta_id);
              end
              begin : rsp_local
                automatic logic [1:0] sli = `NOC_GRP(g).floo_tcdm_narrow_rw_resp_to_router_vc_valid[t][p+1]
                  ? (`NOC_GRP(g).floo_tcdm_narrow_rw_resp_to_router_vc_ready[t][p+1] ? 2'd2 : 2'd1) : 2'd0;
                automatic logic [1:0] slo = `NOC_GRP(g).floo_tcdm_narrow_rw_resp_from_router_vc_valid[t][p+1]
                  ? (`NOC_GRP(g).floo_tcdm_narrow_rw_resp_from_router_vc_ready[t][p+1] ? 2'd2 : 2'd1) : 2'd0;
                if (sli != rsp_st[g][t][p][4][0]) begin
                  $fwrite(f_rresp[g], "%0d S 4 0 %0d %0d %0d\n", rid, rsp_start[g][t][p][4][0], cycle_q, rsp_st[g][t][p][4][0]);
                  rsp_st[g][t][p][4][0] <= sli; rsp_start[g][t][p][4][0] <= cycle_q;
                end
                if (slo != rsp_st[g][t][p][4][1]) begin
                  $fwrite(f_rresp[g], "%0d S 4 1 %0d %0d %0d\n", rid, rsp_start[g][t][p][4][1], cycle_q, rsp_st[g][t][p][4][1]);
                  rsp_st[g][t][p][4][1] <= slo; rsp_start[g][t][p][4][1] <= cycle_q;
                end
                if (sli >= 2)  // local input (injection from tile = slave_resp) handshake
                  $fwrite(f_rresp[g], "%0d P 4 0 %0d %0d %0d %0d %0d %0d %0d %0d\n", rid, cycle_q,
                    `NOC_GRP(g).floo_tcdm_narrow_rw_resp_to_router[t][p+1].hdr.dst_id.x,
                    `NOC_GRP(g).floo_tcdm_narrow_rw_resp_to_router[t][p+1].hdr.dst_id.y,
                    `NOC_GRP(g).floo_tcdm_narrow_rw_resp_to_router[t][p+1].hdr.src_id.x,
                    `NOC_GRP(g).floo_tcdm_narrow_rw_resp_to_router[t][p+1].hdr.src_id.y,
                    `NOC_GRP(g).floo_tcdm_narrow_rw_resp_to_router[t][p+1].hdr.tile_id,
                    `NOC_GRP(g).floo_tcdm_narrow_rw_resp_to_router[t][p+1].hdr.core_id,
                    `NOC_GRP(g).floo_tcdm_narrow_rw_resp_to_router[t][p+1].hdr.meta_id);
                if (slo >= 2)  // local output (ejection to tile = master_resp) handshake
                  $fwrite(f_rresp[g], "%0d P 4 1 %0d %0d %0d %0d %0d %0d %0d %0d\n", rid, cycle_q,
                    `NOC_GRP(g).floo_tcdm_narrow_rw_resp_from_router[t][p+1].hdr.dst_id.x,
                    `NOC_GRP(g).floo_tcdm_narrow_rw_resp_from_router[t][p+1].hdr.dst_id.y,
                    `NOC_GRP(g).floo_tcdm_narrow_rw_resp_from_router[t][p+1].hdr.src_id.x,
                    `NOC_GRP(g).floo_tcdm_narrow_rw_resp_from_router[t][p+1].hdr.src_id.y,
                    `NOC_GRP(g).floo_tcdm_narrow_rw_resp_from_router[t][p+1].hdr.tile_id,
                    `NOC_GRP(g).floo_tcdm_narrow_rw_resp_from_router[t][p+1].hdr.core_id,
                    `NOC_GRP(g).floo_tcdm_narrow_rw_resp_from_router[t][p+1].hdr.meta_id);
              end
            end
          end
        end
      end
    end
  endgenerate

  // ------------------------------------------------------------
  // Tile port capture -> per-tile file. req=master_req(out)/slave_req(in) split
  // read/write by .wen + P packet lines; resp=master_resp(in)/slave_resp(out)
  // single handshake state (no address). genvar t: gen_tiles[t].i_tile constant.
  // ------------------------------------------------------------
  generate
    for (genvar g = 0; g < NumGroups; g++) begin : gen_tstate_g
      for (genvar t = 0; t < NumTilesPerGroup; t++) begin : gen_tstate_t
        always_ff @(posedge clk or negedge rst_n) begin
          if (!rst_n) begin
            ts_mreq_st[g][t] <= '{default: '0}; ts_mreq_s[g][t] <= '{default: '0};
            ts_sreq_st[g][t] <= '{default: '0}; ts_sreq_s[g][t] <= '{default: '0};
            ts_mrsp_st[g][t] <= '{default: '0}; ts_mrsp_s[g][t] <= '{default: '0};
            ts_srsp_st[g][t] <= '{default: '0}; ts_srsp_s[g][t] <= '{default: '0};
          end else begin
            for (int p = 0; p < NumNarrowRemoteReqPortsPerTile; p++) begin
              automatic logic [1:0] smo = `NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.tcdm_master_req_valid_o[p]
                ? (`NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.tcdm_master_req_ready_i[p]
                   ? (`NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.tcdm_master_req_o[p].wen ? 2'd3 : 2'd2) : 2'd1) : 2'd0;
              automatic logic [1:0] ssi = `NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.tcdm_slave_req_valid_i[p]
                ? (`NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.tcdm_slave_req_ready_o[p]
                   ? (`NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.tcdm_slave_req_i[p].wen ? 2'd3 : 2'd2) : 2'd1) : 2'd0;
              if (smo != ts_mreq_st[g][t][p]) begin
                $fwrite(f_tile[g][t], "S 0 1 %0d %0d %0d %0d\n", p, ts_mreq_s[g][t][p], cycle_q, ts_mreq_st[g][t][p]);
                ts_mreq_st[g][t][p] <= smo; ts_mreq_s[g][t][p] <= cycle_q;
              end
              if (ssi != ts_sreq_st[g][t][p]) begin
                $fwrite(f_tile[g][t], "S 0 0 %0d %0d %0d %0d\n", p, ts_sreq_s[g][t][p], cycle_q, ts_sreq_st[g][t][p]);
                ts_sreq_st[g][t][p] <= ssi; ts_sreq_s[g][t][p] <= cycle_q;
              end
              // P: io port cycle wen tgt_addr src_group dst_group req_tile req_core meta_id.
              //   src/dst groups are LINEAR ids (exporter renders as mesh (x,y));
              //   requester group == src group, so not emitted separately.
              if (smo >= 2)  // master_req out (this tile issues a request)
                $fwrite(f_tile[g][t], "P 1 %0d %0d %0d %0h %0d %0d %0d %0d %0d\n", p, cycle_q,
                  `NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.tcdm_master_req_o[p].wen,
                  `NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.tcdm_master_req_o[p].tgt_addr,
                  g,                                                                                  // src group
                  `NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.tcdm_master_req_o[p].tgt_group_id, // dst group
                  t,                                                                                  // requester tile
                  `NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.tcdm_master_req_o[p].wdata.core_id, // requester core
                  `NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.tcdm_master_req_o[p].wdata.meta_id);// meta_id
              if (ssi >= 2)  // slave_req in (a request arrives at this tile's SPM)
                $fwrite(f_tile[g][t], "P 0 %0d %0d %0d %0h %0d %0d %0d %0d %0d\n", p, cycle_q,
                  `NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.tcdm_slave_req_i[p].wen,
                  `NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.tcdm_slave_req_i[p].tgt_addr,
                  `NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.tcdm_slave_req_i[p].src_group_id,   // src group
                  g,                                                                                  // dst group
                  `NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.tcdm_slave_req_i[p].ini_addr,       // requester tile
                  `NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.tcdm_slave_req_i[p].wdata.core_id,  // requester core
                  `NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.tcdm_slave_req_i[p].wdata.meta_id); // meta_id
            end
            for (int p = 0; p < NumNarrowRemoteRespPortsPerTile; p++) begin
              automatic logic [1:0] smi = `NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.tcdm_master_resp_valid_i[p]
                ? (`NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.tcdm_master_resp_ready_o[p] ? 2'd2 : 2'd1) : 2'd0;
              automatic logic [1:0] sso = `NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.tcdm_slave_resp_valid_o[p]
                ? (`NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.tcdm_slave_resp_ready_i[p] ? 2'd2 : 2'd1) : 2'd0;
              // Tile RESP packets use Q lines (kept apart from P req lines):
              //   Q <io> <port> <cycle> <srcvalid> <src_grp> <dst_grp> <req_tile> <req_core> <meta_id>
              //   master_resp(in,io0): responder not on tcdm boundary -> srcvalid=0, dst=this group.
              //   slave_resp(out,io1): src=this group, dst=src_group_id (requester it returns to).
              if (smi != ts_mrsp_st[g][t][p]) begin
                $fwrite(f_tile[g][t], "S 1 0 %0d %0d %0d %0d\n", p, ts_mrsp_s[g][t][p], cycle_q, ts_mrsp_st[g][t][p]);
                ts_mrsp_st[g][t][p] <= smi; ts_mrsp_s[g][t][p] <= cycle_q;
              end
              if (smi >= 2)  // master_resp in: response arrives for this tile's core
                $fwrite(f_tile[g][t], "Q 0 %0d %0d 0 %0d %0d %0d %0d %0d\n", p, cycle_q,
                  g, g, t,
                  `NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.tcdm_master_resp_i[p].rdata.core_id,
                  `NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.tcdm_master_resp_i[p].rdata.meta_id);
              if (sso != ts_srsp_st[g][t][p]) begin
                $fwrite(f_tile[g][t], "S 1 1 %0d %0d %0d %0d\n", p, ts_srsp_s[g][t][p], cycle_q, ts_srsp_st[g][t][p]);
                ts_srsp_st[g][t][p] <= sso; ts_srsp_s[g][t][p] <= cycle_q;
              end
              if (sso >= 2)  // slave_resp out: this tile's SPM responds to a requester
                $fwrite(f_tile[g][t], "Q 1 %0d %0d 1 %0d %0d %0d %0d %0d\n", p, cycle_q,
                  g,
                  `NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.tcdm_slave_resp_o[p].src_group_id,
                  `NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.tcdm_slave_resp_o[p].ini_addr,
                  `NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.tcdm_slave_resp_o[p].rdata.core_id,
                  `NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.tcdm_slave_resp_o[p].rdata.meta_id);
            end
          end
        end
      end
    end
  endgenerate

`ifdef USE_WIDE_TCDM_PLANE
  // ------------------------------------------------------------
  // WIDE tile port capture -> the SAME per-tile file as the narrow boundary,
  // under its own record letters so an old parser skips them and a new one can
  // show both planes of a tile side by side:
  //   SW <net> <io> <port> <start> <end> <state>
  //   PW <io> <port> <cyc> <wen> <tgt_addr> <src_grp> <dst_grp> <req_tile> <req_core> <meta_id>
  //   QW <io> <port> <cyc> <srcvalid> <src_grp> <dst_grp> <req_tile> <req_core> <meta_id>
  // Field-for-field identical to the narrow S/P/Q lines. req_core is the wide
  // engine's slot on the shared local-port axis (WideEngCoreId): the plane
  // carries core_id=0 on the wire because a tile has exactly ONE wide client,
  // so the wire value would collide with core 0's narrow requests.
  // ------------------------------------------------------------
  generate
    for (genvar g = 0; g < NumGroups; g++) begin : gen_wtstate_g
      for (genvar t = 0; t < NumTilesPerGroup; t++) begin : gen_wtstate_t
        always_ff @(posedge clk or negedge rst_n) begin
          if (!rst_n) begin
            wts_mreq_st[g][t] <= '{default: '0}; wts_mreq_s[g][t] <= '{default: '0};
            wts_sreq_st[g][t] <= '{default: '0}; wts_sreq_s[g][t] <= '{default: '0};
            wts_mrsp_st[g][t] <= '{default: '0}; wts_mrsp_s[g][t] <= '{default: '0};
            wts_srsp_st[g][t] <= '{default: '0}; wts_srsp_s[g][t] <= '{default: '0};
          end else begin
            for (int p = 0; p < NumWideRemoteReqPortsPerTile; p++) begin
              automatic logic [1:0] smo = `NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.tcdm_wide_master_req_valid_o[p]
                ? (`NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.tcdm_wide_master_req_ready_i[p]
                   ? (`NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.tcdm_wide_master_req_o[p].wen ? 2'd3 : 2'd2) : 2'd1) : 2'd0;
              automatic logic [1:0] ssi = `NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.tcdm_wide_slave_req_valid_i[p]
                ? (`NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.tcdm_wide_slave_req_ready_o[p]
                   ? (`NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.tcdm_wide_slave_req_i[p].wen ? 2'd3 : 2'd2) : 2'd1) : 2'd0;
              if (smo != wts_mreq_st[g][t][p]) begin
                $fwrite(f_tile[g][t], "SW 0 1 %0d %0d %0d %0d\n", p, wts_mreq_s[g][t][p], cycle_q, wts_mreq_st[g][t][p]);
                wts_mreq_st[g][t][p] <= smo; wts_mreq_s[g][t][p] <= cycle_q;
              end
              if (ssi != wts_sreq_st[g][t][p]) begin
                $fwrite(f_tile[g][t], "SW 0 0 %0d %0d %0d %0d\n", p, wts_sreq_s[g][t][p], cycle_q, wts_sreq_st[g][t][p]);
                wts_sreq_st[g][t][p] <= ssi; wts_sreq_s[g][t][p] <= cycle_q;
              end
              if (smo >= 2)  // master_req out (this tile's engine issues a wide request)
                $fwrite(f_tile[g][t], "PW 1 %0d %0d %0d %0h %0d %0d %0d %0d %0d\n", p, cycle_q,
                  `NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.tcdm_wide_master_req_o[p].wen,
                  `NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.tcdm_wide_master_req_o[p].tgt_addr,
                  g,                                                                                       // src group
                  `NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.tcdm_wide_master_req_o[p].tgt_group_id,   // dst group
                  t,                                                                                       // requester tile
                  WideEngCoreId,                                                                           // requester port
                  `NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.tcdm_wide_master_req_o[p].wdata.meta_id);
              if (ssi >= 2)  // slave_req in (a wide request arrives at this tile's SPM)
                $fwrite(f_tile[g][t], "PW 0 %0d %0d %0d %0h %0d %0d %0d %0d %0d\n", p, cycle_q,
                  `NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.tcdm_wide_slave_req_i[p].wen,
                  `NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.tcdm_wide_slave_req_i[p].tgt_addr,
                  `NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.tcdm_wide_slave_req_i[p].src_group_id,    // src group
                  g,                                                                                       // dst group
                  `NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.tcdm_wide_slave_req_i[p].ini_addr,        // requester tile
                  WideEngCoreId,                                                                           // requester port
                  `NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.tcdm_wide_slave_req_i[p].wdata.meta_id);
            end
            for (int p = 0; p < NumWideRemoteRespPortsPerTile; p++) begin
              automatic logic [1:0] smi = `NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.tcdm_wide_master_resp_valid_i[p]
                ? (`NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.tcdm_wide_master_resp_ready_o[p] ? 2'd2 : 2'd1) : 2'd0;
              automatic logic [1:0] sso = `NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.tcdm_wide_slave_resp_valid_o[p]
                ? (`NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.tcdm_wide_slave_resp_ready_i[p] ? 2'd2 : 2'd1) : 2'd0;
              if (smi != wts_mrsp_st[g][t][p]) begin
                $fwrite(f_tile[g][t], "SW 1 0 %0d %0d %0d %0d\n", p, wts_mrsp_s[g][t][p], cycle_q, wts_mrsp_st[g][t][p]);
                wts_mrsp_st[g][t][p] <= smi; wts_mrsp_s[g][t][p] <= cycle_q;
              end
              if (smi >= 2)  // master_resp in: a wide response returns to this tile's engine
                $fwrite(f_tile[g][t], "QW 0 %0d %0d 0 %0d %0d %0d %0d %0d\n", p, cycle_q,
                  g, g, t, WideEngCoreId,
                  `NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.tcdm_wide_master_resp_i[p].rdata.meta_id);
              if (sso != wts_srsp_st[g][t][p]) begin
                $fwrite(f_tile[g][t], "SW 1 1 %0d %0d %0d %0d\n", p, wts_srsp_s[g][t][p], cycle_q, wts_srsp_st[g][t][p]);
                wts_srsp_st[g][t][p] <= sso; wts_srsp_s[g][t][p] <= cycle_q;
              end
              if (sso >= 2)  // slave_resp out: this tile's SPM answers a wide requester
                $fwrite(f_tile[g][t], "QW 1 %0d %0d 1 %0d %0d %0d %0d %0d\n", p, cycle_q,
                  g,
                  `NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.tcdm_wide_slave_resp_o[p].src_group_id,
                  `NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.tcdm_wide_slave_resp_o[p].ini_addr,
                  WideEngCoreId,
                  `NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.tcdm_wide_slave_resp_o[p].rdata.meta_id);
            end
          end
        end
      end
    end
  endgenerate
`endif

  // ------------------------------------------------------------
  // PE (core) data-port capture -> per-tile file (one line per CORE PORT). Same
  // format as a tile port; req = snitch_data_q* (out, read/write by qwrite, addr=qaddr),
  // resp = snitch_data_p* (in). No bw/util derived for PE ports.
  //   S <rr> <io> <port> <start> <end> <state>
  //   req  P 1 0 <cyc> <wen> <addr(hex)> <meta_id> <dst_tile>
  //   resp P 0 0 <cyc> <meta_id>
  // dst_tile = global destination tile (mem_dst_tile of the SCRAMBLED address),
  // -1 for a SoC/L2 access; pairing req->resp by meta_id gives the port's
  // round-trip latency, split by address class (scripts/interco_perf.py).
  // The leading demux index is the FLAT core-port index (core c port p =
  // core_port_base(c) + p), so Spatz builds log the scalar port AND all
  // NumMemPortsPerSpatz vector ports of every core, not just the first four
  // ports of core 0.
  // ------------------------------------------------------------
  generate
    for (genvar g = 0; g < NumGroups; g++) begin : gen_core_g
      for (genvar t = 0; t < NumTilesPerGroup; t++) begin : gen_core_t
        always_ff @(posedge clk or negedge rst_n) begin
          if (!rst_n) begin
            core_req_st[g][t] <= '{default: '0}; core_req_s[g][t] <= '{default: '0};
            core_rsp_st[g][t] <= '{default: '0}; core_rsp_s[g][t] <= '{default: '0};
          end else begin
            for (int c = 0; c < NumCorePorts; c++) begin
              automatic logic [1:0] rq = `NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.snitch_data_qvalid[c]
                ? (`NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.snitch_data_qready[c]
                   ? (`NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.snitch_data_qwrite[c] ? 2'd3 : 2'd2) : 2'd1) : 2'd0;
              automatic logic [1:0] rp = `NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.snitch_data_pvalid[c]
                ? (`NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.snitch_data_pready[c] ? 2'd2 : 2'd1) : 2'd0;
              if (rq != core_req_st[g][t][c]) begin
                $fwrite(f_core[g][t], "%0d S 0 1 0 %0d %0d %0d\n", c, core_req_s[g][t][c], cycle_q, core_req_st[g][t][c]);
                core_req_st[g][t][c] <= rq; core_req_s[g][t][c] <= cycle_q;
              end
              if (rp != core_rsp_st[g][t][c]) begin
                $fwrite(f_core[g][t], "%0d S 1 0 0 %0d %0d %0d\n", c, core_rsp_s[g][t][c], cycle_q, core_rsp_st[g][t][c]);
                core_rsp_st[g][t][c] <= rp; core_rsp_s[g][t][c] <= cycle_q;
              end
              if (rq >= 2)  // req handshake -> one core load/store accepted (qid = meta_id)
                // mask qid to the low snitch_pkg::MetaIdWidth bits (upper bits are X) so the
                // printed id matches the NoC's zero-extended meta_id for this request.
                // dst = destination tile (from the scrambled address), -1 = SoC/L2.
                $fwrite(f_core[g][t], "%0d P 1 0 %0d %0d %0h %0d %0d\n", c, cycle_q,
                  `NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.snitch_data_qwrite[c],
                  `NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.snitch_data_qaddr[c],
                  `NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.snitch_data_qid[c][snitch_pkg::MetaIdWidth-1:0],
                  mem_dst_tile(`NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.snitch_data_qaddr_scrambled[c]));
              if (rp >= 2)  // resp handshake -> a response returns to this core (pid = meta_id)
                $fwrite(f_core[g][t], "%0d P 0 0 %0d %0d\n", c, cycle_q,
                  `NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.snitch_data_pid[c]);
            end
          end
        end
      end
    end
  endgenerate

  // ------------------------------------------------------------
  // RedMulE master-port capture -> per-tile file (RedMulE tiles only). Tapped at the
  // per-port tcdm_shim INPUT (gen_redmule.redmule_tcdm_req/resp[p]) -- BEFORE the shim's
  // local/remote split -- so it sees every access (local AND remote NoC ones). Remote
  // requests carry core_id = NumCoresPerTile+p, so a NoC packet maps back to rm port p.
  // Same S/P format as a PE core port:
  //   S <rr> <io> 0 <start> <end> <state> ;
  //   req  P 1 0 <cyc> <wen> <addr> <meta_id> <dst_tile> ; resp P 0 0 <cyc> <meta_id>.
  // ------------------------------------------------------------
  // NOTE: this tap reads gen_redmule.redmule_tcdm_req/resp[p], the per-port
  // NARROW RedMulE signals. The WP2 wide plane replaces that per-port unpack with a
  // single wide HCI access, so those signals do not exist and the XMRefs cannot
  // resolve. Skip the tap when the wide plane is on (a wide-plane RedMulE tap would
  // have to read the wide master port / engine port instead).
`ifndef USE_WIDE_TCDM_PLANE
  generate
    if (NumRMTiles > 0) begin : gen_rm
      for (genvar g = 0; g < NumGroups; g++) begin : gen_rm_g
        for (genvar t = 0; t < NumRMTilesPerGroup; t++) begin : gen_rm_t
          always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
              rm_req_st[g][t] <= '{default: '0}; rm_req_s[g][t] <= '{default: '0};
              rm_rsp_st[g][t] <= '{default: '0}; rm_rsp_s[g][t] <= '{default: '0};
            end else begin
              for (int p = 0; p < RMMasterPorts; p++) begin
                automatic logic [1:0] rq = `NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.gen_redmule.redmule_tcdm_req_valid[p]
                  ? (`NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.gen_redmule.redmule_tcdm_req_ready[p]
                     ? (`NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.gen_redmule.redmule_tcdm_req[p].write ? 2'd3 : 2'd2) : 2'd1) : 2'd0;
                automatic logic [1:0] rp = `NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.gen_redmule.redmule_tcdm_resp_valid[p]
                  ? (`NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.gen_redmule.redmule_tcdm_resp_ready[p] ? 2'd2 : 2'd1) : 2'd0;
                if (rq != rm_req_st[g][t][p]) begin
                  $fwrite(f_rm[g][t], "%0d S 0 1 0 %0d %0d %0d\n", p, rm_req_s[g][t][p], cycle_q, rm_req_st[g][t][p]);
                  rm_req_st[g][t][p] <= rq; rm_req_s[g][t][p] <= cycle_q;
                end
                if (rp != rm_rsp_st[g][t][p]) begin
                  $fwrite(f_rm[g][t], "%0d S 1 0 0 %0d %0d %0d\n", p, rm_rsp_s[g][t][p], cycle_q, rm_rsp_st[g][t][p]);
                  rm_rsp_st[g][t][p] <= rp; rm_rsp_s[g][t][p] <= cycle_q;
                end
                if (rq >= 2)  // req handshake -> one RedMulE load/store accepted (id = meta_id)
                  // dst = destination tile (from the scrambled address), -1 = SoC/L2.
                  $fwrite(f_rm[g][t], "%0d P 1 0 %0d %0d %0h %0d %0d\n", p, cycle_q,
                    `NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.gen_redmule.redmule_tcdm_req[p].write,
                    `NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.gen_redmule.redmule_tcdm_req[p].addr,
                    `NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.gen_redmule.redmule_tcdm_req[p].id,
                    mem_dst_tile(`NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.gen_redmule.redmule_addr_scrambled[p]));
                if (rp >= 2)  // resp handshake -> a response returns to this RedMulE port (id = meta_id)
                  $fwrite(f_rm[g][t], "%0d P 0 0 %0d %0d\n", p, cycle_q,
                    `NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.gen_redmule.redmule_tcdm_resp[p].id);
              end
            end
          end
        end
      end
    end
  endgenerate
`endif

`ifdef USE_WIDE_TCDM_PLANE
  // ------------------------------------------------------------
  // WIDE RedMulE engine-port capture -> rmw_g<g>_t<t>.log (one wide port per
  // RedMulE tile). Tapped at the wide tcdm_shim INPUT -- the single HCI
  // translation point in mempool_tile.sv -- so it sees every access, own-tile
  // and remote alike, exactly as the narrow rm_ tap does one plane over.
  // Same S/P grammar as core_/rm_, demux index 0 (a tile has ONE wide client):
  //   S <rr> <io> 0 <start> <end> <state>
  //   req  P 1 0 <cyc> <wen> <addr> <meta_id> <dst_tile>
  //   resp P 0 0 <cyc> <meta_id>
  // `addr` is the RAW HCI address (pre-scrambler), matching core_/rm_; the wide
  // shim consumes the SCRAMBLED one (i_rm_wide_scrambler drives
  // tcdm_wide_rm_req.addr directly), which is what mem_dst_tile decodes.
  // ------------------------------------------------------------
  `define RMW_ENG(gg,tt) `NOC_GRP(gg).i_mempool_group.gen_tiles[tt].i_tile.gen_redmule
  generate
    if (NumRMTiles > 0) begin : gen_rmw
      for (genvar g = 0; g < NumGroups; g++) begin : gen_rmw_g
        for (genvar t = 0; t < NumRMTilesPerGroup; t++) begin : gen_rmw_t
          always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
              wrm_req_st[g][t] <= '0; wrm_req_s[g][t] <= '0;
              wrm_rsp_st[g][t] <= '0; wrm_rsp_s[g][t] <= '0;
            end else begin
              automatic logic [1:0] rq = `RMW_ENG(g,t).tcdm_wide_rm_req_valid
                ? (`RMW_ENG(g,t).tcdm_wide_rm_req_ready
                   ? (`RMW_ENG(g,t).tcdm_wide_rm_req.write ? 2'd3 : 2'd2) : 2'd1) : 2'd0;
              automatic logic [1:0] rp = `RMW_ENG(g,t).tcdm_wide_rm_resp_valid
                ? (`RMW_ENG(g,t).tcdm_wide_rm_resp_ready ? 2'd2 : 2'd1) : 2'd0;
              if (rq != wrm_req_st[g][t]) begin
                $fwrite(f_rm[g][t], "0 S 0 1 0 %0d %0d %0d\n", wrm_req_s[g][t], cycle_q, wrm_req_st[g][t]);
                wrm_req_st[g][t] <= rq; wrm_req_s[g][t] <= cycle_q;
              end
              if (rp != wrm_rsp_st[g][t]) begin
                $fwrite(f_rm[g][t], "0 S 1 0 0 %0d %0d %0d\n", wrm_rsp_s[g][t], cycle_q, wrm_rsp_st[g][t]);
                wrm_rsp_st[g][t] <= rp; wrm_rsp_s[g][t] <= cycle_q;
              end
              if (rq >= 2)  // req handshake -> one wide engine access accepted
                $fwrite(f_rm[g][t], "0 P 1 0 %0d %0d %0h %0d %0d\n", cycle_q,
                  `RMW_ENG(g,t).tcdm_wide_rm_req.write,
                  `RMW_ENG(g,t).tcdm.add,
                  `RMW_ENG(g,t).tcdm_wide_rm_req.id,
                  mem_dst_tile(`RMW_ENG(g,t).tcdm_wide_rm_req.addr));
              if (rp >= 2)  // resp handshake -> a wide response returns to the engine
                $fwrite(f_rm[g][t], "0 P 0 0 %0d %0d\n", cycle_q,
                  `RMW_ENG(g,t).tcdm_wide_rm_resp.id);
            end
          end
        end
      end
    end
  endgenerate

  // ------------------------------------------------------------
  // WIDE router capture -> router_g<g>_wide_{req,resp}.log. The wide plane runs
  // its own routers, one per tile port (RO request, RW response) or merged
  // channel (WO), so the demux index is a flat lane id over the classes of that
  // net (RO then WO on the request net, RW then WO on the response net) and
  // `cls` names which class it is:
  //   <c> S <cls> <dir> <io> <start> <end> <state>
  //   req  <c> P <cls> <dir> <io> <cyc> <wen> <tgt_addr> <dstx> <dsty> <srcx> <srcy> <src_tile> <core> <meta_id>
  //   resp <c> P <cls> <dir> <io> <cyc> <dstx> <dsty> <srcx> <srcy> <req_tile> <req_core> <meta_id>
  // dir 0..3 = N/E/S/W, 4 = local; io 0=in 1=out -- as the narrow routers.
  // `core` is WideEngCoreId, not the wire's core_id (see the tile note above).
  // The three macros exist because the classes differ ONLY in the flit's write
  // flag: an RO request flit is meta-only, so it is a read by construction.
  // ------------------------------------------------------------
  `define WIDE_REQ_ROUTER_BODY(NM, CLS, BASE, NCH, IDX, WEN_I, WEN_O, WEN_T, WEN_F)   \
    for (int k = 0; k < NCH; k++) begin                                       \
      automatic int c = BASE + k;                                                \
      for (int d = 0; d < 4; d++) begin                                          \
        automatic logic [1:0] si = `NOC_GRP(g).NM``_valid_in_trans IDX [d]         \
          ? (`NOC_GRP(g).NM``_ready_out_trans IDX [d] ? (WEN_I ? 2'd3 : 2'd2) : 2'd1) : 2'd0; \
        automatic logic [1:0] so = `NOC_GRP(g).NM``_valid_out_trans IDX [d]        \
          ? (`NOC_GRP(g).NM``_ready_in_trans IDX [d] ? (WEN_O ? 2'd3 : 2'd2) : 2'd1) : 2'd0;  \
        if (si != wrq_st[g][c][d][0]) begin                                      \
          $fwrite(f_wrreq[g], "%0d S %0d %0d 0 %0d %0d %0d\n", c, CLS, d, wrq_s[g][c][d][0], cycle_q, wrq_st[g][c][d][0]); \
          wrq_st[g][c][d][0] <= si; wrq_s[g][c][d][0] <= cycle_q;                \
        end                                                                      \
        if (so != wrq_st[g][c][d][1]) begin                                      \
          $fwrite(f_wrreq[g], "%0d S %0d %0d 1 %0d %0d %0d\n", c, CLS, d, wrq_s[g][c][d][1], cycle_q, wrq_st[g][c][d][1]); \
          wrq_st[g][c][d][1] <= so; wrq_s[g][c][d][1] <= cycle_q;                \
        end                                                                      \
        if (si >= 2)                                                             \
          $fwrite(f_wrreq[g], "%0d P %0d %0d 0 %0d %0d %0h %0d %0d %0d %0d %0d %0d %0d\n", c, CLS, d, cycle_q, \
            WEN_I,                                                               \
            `NOC_GRP(g).NM``_in_trans IDX [d].hdr.tgt_addr,                        \
            `NOC_GRP(g).NM``_in_trans IDX [d].hdr.dst_id.x,                        \
            `NOC_GRP(g).NM``_in_trans IDX [d].hdr.dst_id.y,                        \
            `NOC_GRP(g).NM``_in_trans IDX [d].hdr.src_id.x,                        \
            `NOC_GRP(g).NM``_in_trans IDX [d].hdr.src_id.y,                        \
            `NOC_GRP(g).NM``_in_trans IDX [d].hdr.src_tile_id,                     \
            WideEngCoreId,                                                       \
            `NOC_GRP(g).NM``_in_trans IDX [d].hdr.meta_id);                        \
        if (so >= 2)                                                             \
          $fwrite(f_wrreq[g], "%0d P %0d %0d 1 %0d %0d %0h %0d %0d %0d %0d %0d %0d %0d\n", c, CLS, d, cycle_q, \
            WEN_O,                                                               \
            `NOC_GRP(g).NM``_out_trans IDX [d].hdr.tgt_addr,                       \
            `NOC_GRP(g).NM``_out_trans IDX [d].hdr.dst_id.x,                       \
            `NOC_GRP(g).NM``_out_trans IDX [d].hdr.dst_id.y,                       \
            `NOC_GRP(g).NM``_out_trans IDX [d].hdr.src_id.x,                       \
            `NOC_GRP(g).NM``_out_trans IDX [d].hdr.src_id.y,                       \
            `NOC_GRP(g).NM``_out_trans IDX [d].hdr.src_tile_id,                    \
            WideEngCoreId,                                                       \
            `NOC_GRP(g).NM``_out_trans IDX [d].hdr.meta_id);                       \
      end                                                                        \
      begin                                                                      \
        automatic logic [1:0] sli = `NOC_GRP(g).NM``_to_router_valid IDX            \
          ? (`NOC_GRP(g).NM``_to_router_ready IDX  ? (WEN_T ? 2'd3 : 2'd2) : 2'd1) : 2'd0;    \
        automatic logic [1:0] slo = `NOC_GRP(g).NM``_from_router_valid IDX          \
          ? (`NOC_GRP(g).NM``_from_router_ready IDX  ? (WEN_F ? 2'd3 : 2'd2) : 2'd1) : 2'd0;  \
        if (sli != wrq_st[g][c][4][0]) begin                                     \
          $fwrite(f_wrreq[g], "%0d S %0d 4 0 %0d %0d %0d\n", c, CLS, wrq_s[g][c][4][0], cycle_q, wrq_st[g][c][4][0]); \
          wrq_st[g][c][4][0] <= sli; wrq_s[g][c][4][0] <= cycle_q;               \
        end                                                                      \
        if (slo != wrq_st[g][c][4][1]) begin                                     \
          $fwrite(f_wrreq[g], "%0d S %0d 4 1 %0d %0d %0d\n", c, CLS, wrq_s[g][c][4][1], cycle_q, wrq_st[g][c][4][1]); \
          wrq_st[g][c][4][1] <= slo; wrq_s[g][c][4][1] <= cycle_q;               \
        end                                                                      \
        if (sli >= 2)                                                            \
          $fwrite(f_wrreq[g], "%0d P %0d 4 0 %0d %0d %0h %0d %0d %0d %0d %0d %0d %0d\n", c, CLS, cycle_q, \
            WEN_T,                                                               \
            `NOC_GRP(g).NM``_to_router IDX .hdr.tgt_addr,                          \
            `NOC_GRP(g).NM``_to_router IDX .hdr.dst_id.x,                          \
            `NOC_GRP(g).NM``_to_router IDX .hdr.dst_id.y,                          \
            `NOC_GRP(g).NM``_to_router IDX .hdr.src_id.x,                          \
            `NOC_GRP(g).NM``_to_router IDX .hdr.src_id.y,                          \
            `NOC_GRP(g).NM``_to_router IDX .hdr.src_tile_id,                       \
            WideEngCoreId,                                                       \
            `NOC_GRP(g).NM``_to_router IDX .hdr.meta_id);                          \
        if (slo >= 2)                                                            \
          $fwrite(f_wrreq[g], "%0d P %0d 4 1 %0d %0d %0h %0d %0d %0d %0d %0d %0d %0d\n", c, CLS, cycle_q, \
            WEN_F,                                                               \
            `NOC_GRP(g).NM``_from_router IDX .hdr.tgt_addr,                        \
            `NOC_GRP(g).NM``_from_router IDX .hdr.dst_id.x,                        \
            `NOC_GRP(g).NM``_from_router IDX .hdr.dst_id.y,                        \
            `NOC_GRP(g).NM``_from_router IDX .hdr.src_id.x,                        \
            `NOC_GRP(g).NM``_from_router IDX .hdr.src_id.y,                        \
            `NOC_GRP(g).NM``_from_router IDX .hdr.src_tile_id,                     \
            WideEngCoreId,                                                       \
            `NOC_GRP(g).NM``_from_router IDX .hdr.meta_id);                        \
      end                                                                        \
    end

  // A response flit has no address and no write flag: one macro serves the RD
  // (data-bearing) and WR (meta-only ACK) classes alike.
  `define WIDE_RESP_ROUTER_BODY(NM, CLS, BASE, NCH, IDX)                              \
    for (int k = 0; k < NCH; k++) begin                                       \
      automatic int c = BASE + k;                                                \
      for (int d = 0; d < 4; d++) begin                                          \
        automatic logic [1:0] si = `NOC_GRP(g).NM``_valid_in_trans IDX [d]         \
          ? (`NOC_GRP(g).NM``_ready_out_trans IDX [d] ? 2'd2 : 2'd1) : 2'd0;       \
        automatic logic [1:0] so = `NOC_GRP(g).NM``_valid_out_trans IDX [d]        \
          ? (`NOC_GRP(g).NM``_ready_in_trans IDX [d] ? 2'd2 : 2'd1) : 2'd0;        \
        if (si != wrs_st[g][c][d][0]) begin                                      \
          $fwrite(f_wrresp[g], "%0d S %0d %0d 0 %0d %0d %0d\n", c, CLS, d, wrs_s[g][c][d][0], cycle_q, wrs_st[g][c][d][0]); \
          wrs_st[g][c][d][0] <= si; wrs_s[g][c][d][0] <= cycle_q;                \
        end                                                                      \
        if (so != wrs_st[g][c][d][1]) begin                                      \
          $fwrite(f_wrresp[g], "%0d S %0d %0d 1 %0d %0d %0d\n", c, CLS, d, wrs_s[g][c][d][1], cycle_q, wrs_st[g][c][d][1]); \
          wrs_st[g][c][d][1] <= so; wrs_s[g][c][d][1] <= cycle_q;                \
        end                                                                      \
        if (si >= 2)                                                             \
          $fwrite(f_wrresp[g], "%0d P %0d %0d 0 %0d %0d %0d %0d %0d %0d %0d %0d\n", c, CLS, d, cycle_q, \
            `NOC_GRP(g).NM``_in_trans IDX [d].hdr.dst_id.x,                        \
            `NOC_GRP(g).NM``_in_trans IDX [d].hdr.dst_id.y,                        \
            `NOC_GRP(g).NM``_in_trans IDX [d].hdr.src_id.x,                        \
            `NOC_GRP(g).NM``_in_trans IDX [d].hdr.src_id.y,                        \
            `NOC_GRP(g).NM``_in_trans IDX [d].hdr.tile_id,                         \
            WideEngCoreId,                                                       \
            `NOC_GRP(g).NM``_in_trans IDX [d].hdr.meta_id);                        \
        if (so >= 2)                                                             \
          $fwrite(f_wrresp[g], "%0d P %0d %0d 1 %0d %0d %0d %0d %0d %0d %0d %0d\n", c, CLS, d, cycle_q, \
            `NOC_GRP(g).NM``_out_trans IDX [d].hdr.dst_id.x,                       \
            `NOC_GRP(g).NM``_out_trans IDX [d].hdr.dst_id.y,                       \
            `NOC_GRP(g).NM``_out_trans IDX [d].hdr.src_id.x,                       \
            `NOC_GRP(g).NM``_out_trans IDX [d].hdr.src_id.y,                       \
            `NOC_GRP(g).NM``_out_trans IDX [d].hdr.tile_id,                        \
            WideEngCoreId,                                                       \
            `NOC_GRP(g).NM``_out_trans IDX [d].hdr.meta_id);                       \
      end                                                                        \
      begin                                                                      \
        automatic logic [1:0] sli = `NOC_GRP(g).NM``_to_router_valid IDX           \
          ? (`NOC_GRP(g).NM``_to_router_ready IDX  ? 2'd2 : 2'd1) : 2'd0;          \
        automatic logic [1:0] slo = `NOC_GRP(g).NM``_from_router_valid IDX         \
          ? (`NOC_GRP(g).NM``_from_router_ready IDX  ? 2'd2 : 2'd1) : 2'd0;        \
        if (sli != wrs_st[g][c][4][0]) begin                                     \
          $fwrite(f_wrresp[g], "%0d S %0d 4 0 %0d %0d %0d\n", c, CLS, wrs_s[g][c][4][0], cycle_q, wrs_st[g][c][4][0]); \
          wrs_st[g][c][4][0] <= sli; wrs_s[g][c][4][0] <= cycle_q;               \
        end                                                                      \
        if (slo != wrs_st[g][c][4][1]) begin                                     \
          $fwrite(f_wrresp[g], "%0d S %0d 4 1 %0d %0d %0d\n", c, CLS, wrs_s[g][c][4][1], cycle_q, wrs_st[g][c][4][1]); \
          wrs_st[g][c][4][1] <= slo; wrs_s[g][c][4][1] <= cycle_q;               \
        end                                                                      \
        if (sli >= 2)                                                            \
          $fwrite(f_wrresp[g], "%0d P %0d 4 0 %0d %0d %0d %0d %0d %0d %0d %0d\n", c, CLS, cycle_q, \
            `NOC_GRP(g).NM``_to_router IDX .hdr.dst_id.x,                          \
            `NOC_GRP(g).NM``_to_router IDX .hdr.dst_id.y,                          \
            `NOC_GRP(g).NM``_to_router IDX .hdr.src_id.x,                          \
            `NOC_GRP(g).NM``_to_router IDX .hdr.src_id.y,                          \
            `NOC_GRP(g).NM``_to_router IDX .hdr.tile_id,                           \
            WideEngCoreId,                                                       \
            `NOC_GRP(g).NM``_to_router IDX .hdr.meta_id);                          \
        if (slo >= 2)                                                            \
          $fwrite(f_wrresp[g], "%0d P %0d 4 1 %0d %0d %0d %0d %0d %0d %0d %0d\n", c, CLS, cycle_q, \
            `NOC_GRP(g).NM``_from_router IDX .hdr.dst_id.x,                        \
            `NOC_GRP(g).NM``_from_router IDX .hdr.dst_id.y,                        \
            `NOC_GRP(g).NM``_from_router IDX .hdr.src_id.x,                        \
            `NOC_GRP(g).NM``_from_router IDX .hdr.src_id.y,                        \
            `NOC_GRP(g).NM``_from_router IDX .hdr.tile_id,                         \
            WideEngCoreId,                                                       \
            `NOC_GRP(g).NM``_from_router IDX .hdr.meta_id);                        \
      end                                                                        \
    end

  generate
    for (genvar g = 0; g < NumGroups; g++) begin : gen_wrstate_g
      always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
          wrq_st[g] <= '{default: '0}; wrq_s[g] <= '{default: '0};
          wrs_st[g] <= '{default: '0}; wrs_s[g] <= '{default: '0};
        end else begin
          // RO request flits are meta-only: a read by construction.
          `WIDE_REQ_ROUTER_BODY(floo_tcdm_wide_ro_req, 0, 0, NumWROReqLanes,
                                [k / NumWideRoReqPortsPerTile][k % NumWideRoReqPortsPerTile],
                                1'b0, 1'b0, 1'b0, 1'b0)
          `WIDE_REQ_ROUTER_BODY(floo_tcdm_wide_wo_req, 1, NumWROReqLanes,
                                NumWideWoReqChannelsPerGroup, [k],
                                `NOC_GRP(g).floo_tcdm_wide_wo_req_in_trans[k][d].payload.wen,
                                `NOC_GRP(g).floo_tcdm_wide_wo_req_out_trans[k][d].payload.wen,
                                `NOC_GRP(g).floo_tcdm_wide_wo_req_to_router[k].payload.wen,
                                `NOC_GRP(g).floo_tcdm_wide_wo_req_from_router[k].payload.wen)
          `WIDE_RESP_ROUTER_BODY(floo_tcdm_wide_rw_resp, 0, 0, NumWRWRespLanes,
                                 [k / NumWideRwRespPortsPerTile][k % NumWideRwRespPortsPerTile])
          `WIDE_RESP_ROUTER_BODY(floo_tcdm_wide_wo_resp, 1, NumWRWRespLanes,
                                 NumWideWoRespChannelsPerGroup, [k])
        end
      end
    end
  endgenerate
`endif

  // ------------------------------------------------------------
  // End-of-sim flush: write every port's still-open S run, then close every file.
  // P lines are written as they happen, so they need no flush.
  // ------------------------------------------------------------
  final begin
    for (int g = 0; g < NumGroups; g++) begin
      for (int t = 0; t < NumTilesPerGroup; t++) begin
        for (int p = 0; p < NumNarrowPayloadReqPortsPerTile; p++) begin
          automatic int rid = t*NumNarrowPayloadReqPortsPerTile + p;
          for (int d = 0; d < 4; d++) begin
            $fwrite(f_rreq[g], "%0d S %0d 0 %0d %0d %0d\n", rid, d, rsq_start[g][t][p][d][0], cycle_q, rsq_st[g][t][p][d][0]);
            $fwrite(f_rreq[g], "%0d S %0d 1 %0d %0d %0d\n", rid, d, rsq_start[g][t][p][d][1], cycle_q, rsq_st[g][t][p][d][1]);
          end
          $fwrite(f_rreq[g], "%0d S 4 0 %0d %0d %0d\n", rid, rsq_start[g][t][p][4][0], cycle_q, rsq_st[g][t][p][4][0]);
          $fwrite(f_rreq[g], "%0d S 4 1 %0d %0d %0d\n", rid, rsq_start[g][t][p][4][1], cycle_q, rsq_st[g][t][p][4][1]);
        end
        for (int p = 0; p < NumNarrowRemoteRespPortsPerTile-1; p++) begin
          automatic int rid = t*(NumNarrowRemoteRespPortsPerTile-1) + p;
          for (int d = 0; d < 4; d++) begin
            $fwrite(f_rresp[g], "%0d S %0d 0 %0d %0d %0d\n", rid, d, rsp_start[g][t][p][d][0], cycle_q, rsp_st[g][t][p][d][0]);
            $fwrite(f_rresp[g], "%0d S %0d 1 %0d %0d %0d\n", rid, d, rsp_start[g][t][p][d][1], cycle_q, rsp_st[g][t][p][d][1]);
          end
          $fwrite(f_rresp[g], "%0d S 4 0 %0d %0d %0d\n", rid, rsp_start[g][t][p][4][0], cycle_q, rsp_st[g][t][p][4][0]);
          $fwrite(f_rresp[g], "%0d S 4 1 %0d %0d %0d\n", rid, rsp_start[g][t][p][4][1], cycle_q, rsp_st[g][t][p][4][1]);
        end
        for (int p = 0; p < NumNarrowRemoteReqPortsPerTile; p++) begin
          $fwrite(f_tile[g][t], "S 0 1 %0d %0d %0d %0d\n", p, ts_mreq_s[g][t][p], cycle_q, ts_mreq_st[g][t][p]);
          $fwrite(f_tile[g][t], "S 0 0 %0d %0d %0d %0d\n", p, ts_sreq_s[g][t][p], cycle_q, ts_sreq_st[g][t][p]);
        end
        for (int p = 0; p < NumNarrowRemoteRespPortsPerTile; p++) begin
          $fwrite(f_tile[g][t], "S 1 0 %0d %0d %0d %0d\n", p, ts_mrsp_s[g][t][p], cycle_q, ts_mrsp_st[g][t][p]);
          $fwrite(f_tile[g][t], "S 1 1 %0d %0d %0d %0d\n", p, ts_srsp_s[g][t][p], cycle_q, ts_srsp_st[g][t][p]);
        end
`ifdef USE_WIDE_TCDM_PLANE
        for (int p = 0; p < NumWideRemoteReqPortsPerTile; p++) begin
          $fwrite(f_tile[g][t], "SW 0 1 %0d %0d %0d %0d\n", p, wts_mreq_s[g][t][p], cycle_q, wts_mreq_st[g][t][p]);
          $fwrite(f_tile[g][t], "SW 0 0 %0d %0d %0d %0d\n", p, wts_sreq_s[g][t][p], cycle_q, wts_sreq_st[g][t][p]);
        end
        for (int p = 0; p < NumWideRemoteRespPortsPerTile; p++) begin
          $fwrite(f_tile[g][t], "SW 1 0 %0d %0d %0d %0d\n", p, wts_mrsp_s[g][t][p], cycle_q, wts_mrsp_st[g][t][p]);
          $fwrite(f_tile[g][t], "SW 1 1 %0d %0d %0d %0d\n", p, wts_srsp_s[g][t][p], cycle_q, wts_srsp_st[g][t][p]);
        end
`endif
        $fclose(f_tile[g][t]);
        for (int c = 0; c < NumCorePorts; c++) begin
          $fwrite(f_core[g][t], "%0d S 0 1 0 %0d %0d %0d\n", c, core_req_s[g][t][c], cycle_q, core_req_st[g][t][c]);
          $fwrite(f_core[g][t], "%0d S 1 0 0 %0d %0d %0d\n", c, core_rsp_s[g][t][c], cycle_q, core_rsp_st[g][t][c]);
        end
        $fclose(f_core[g][t]);
        if (t < NumRMTilesPerGroup) begin
`ifndef USE_WIDE_TCDM_PLANE
          for (int p = 0; p < RMMasterPorts; p++) begin
            $fwrite(f_rm[g][t], "%0d S 0 1 0 %0d %0d %0d\n", p, rm_req_s[g][t][p], cycle_q, rm_req_st[g][t][p]);
            $fwrite(f_rm[g][t], "%0d S 1 0 0 %0d %0d %0d\n", p, rm_rsp_s[g][t][p], cycle_q, rm_rsp_st[g][t][p]);
          end
          $fclose(f_rm[g][t]);
`else
          $fwrite(f_rm[g][t], "0 S 0 1 0 %0d %0d %0d\n", wrm_req_s[g][t], cycle_q, wrm_req_st[g][t]);
          $fwrite(f_rm[g][t], "0 S 1 0 0 %0d %0d %0d\n", wrm_rsp_s[g][t], cycle_q, wrm_rsp_st[g][t]);
          $fclose(f_rm[g][t]);
`endif
        end
      end
`ifdef USE_WIDE_TCDM_PLANE
      for (int c = 0; c < NumWReqCh; c++) begin
        automatic int cls = (c < NumWROReqLanes) ? 0 : 1;
        for (int d = 0; d < 5; d++) begin
          $fwrite(f_wrreq[g], "%0d S %0d %0d 0 %0d %0d %0d\n", c, cls, d, wrq_s[g][c][d][0], cycle_q, wrq_st[g][c][d][0]);
          $fwrite(f_wrreq[g], "%0d S %0d %0d 1 %0d %0d %0d\n", c, cls, d, wrq_s[g][c][d][1], cycle_q, wrq_st[g][c][d][1]);
        end
      end
      for (int c = 0; c < NumWRespCh; c++) begin
        automatic int cls = (c < NumWRWRespLanes) ? 0 : 1;
        for (int d = 0; d < 5; d++) begin
          $fwrite(f_wrresp[g], "%0d S %0d %0d 0 %0d %0d %0d\n", c, cls, d, wrs_s[g][c][d][0], cycle_q, wrs_st[g][c][d][0]);
          $fwrite(f_wrresp[g], "%0d S %0d %0d 1 %0d %0d %0d\n", c, cls, d, wrs_s[g][c][d][1], cycle_q, wrs_st[g][c][d][1]);
        end
      end
      $fclose(f_wrreq[g]);
      $fclose(f_wrresp[g]);
`endif
      $fclose(f_rreq[g]);
      $fclose(f_rresp[g]);
    end
  end

`endif // NOC_PROFILING

`endif // TARGET_VERILATOR
`endif // TARGET_SYNTHESIS

`endif // TB_NOC_PROFILING_SVH_
