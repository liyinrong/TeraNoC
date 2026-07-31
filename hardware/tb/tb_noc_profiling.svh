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
  //   cores   -> pe_g<g>_t<t>.log            (one per tile; <idx> = core)
  //   RedMulE -> rm_g<g>_t<t>.log            (one per tile; <idx> = master port)
  //   tiles   -> tile_g<g>_t<t>.log          (already one per tile; NO prefix)
  string  app, log_path;
  integer retval;
  // MERGED file handles: routers -> ONE req + ONE resp file per group (line tagged with
  // router id); Snitch cores / RedMulE master ports -> ONE file per tile (line tagged with
  // core / port idx). PE = the Snitch core's data memory port.
  int f_rreq  [NumGroups];
  int f_rresp [NumGroups];
  int f_tile  [NumGroups][NumTilesPerGroup];
  int f_pe    [NumGroups][NumTilesPerGroup];
  // RMPorts>=1 keeps the state-register array legal when RedMulE is disabled.
  localparam int unsigned RMPorts = (RMMasterPorts > 0) ? RMMasterPorts : 1;
  int f_rm    [NumGroups][NumTilesPerGroup];
  // Spatz VLSU per-request memory trace (Spatz builds only), one file per tile.
  int f_spmem [NumGroups][NumTilesPerGroup];

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
      for (int t = 0; t < NumTilesPerGroup; t++) begin
        f_tile[g][t] = $fopen($sformatf("%s/tile_g%0d_t%0d.log", log_path, g, t), "w");
        f_pe[g][t]   = $fopen($sformatf("%s/pe_g%0d_t%0d.log",   log_path, g, t), "w");
        if (t < NumRMTilesPerGroup)
          f_rm[g][t] = $fopen($sformatf("%s/rm_g%0d_t%0d.log",   log_path, g, t), "w");
        if (mempool_pkg::RVV) begin
          f_spmem[g][t] = $fopen($sformatf("%s/spmem_g%0d_t%0d.log", log_path, g, t), "w");
          $fwrite(f_spmem[g][t], "D %0d %0d %0d %0d %0d\n",
                  mempool_pkg::NumMemPortsPerSpatz, NumCoresPerTile,
                  NumTilesPerGroup, NumTiles,
                  $clog2(snitch_pkg::NumIntOutstandingLoads));
        end
      end
    end
  end

  // Module-scope state registers so the end-of-sim `final` below can flush them.
  // router: [group][tile][port][portidx 0..4][io 0..1]
  logic [1:0]  rsq_st    [NumGroups][NumTilesPerGroup][NumNarrowRwReqPortsPerTile][5][2];
  logic [63:0] rsq_start [NumGroups][NumTilesPerGroup][NumNarrowRwReqPortsPerTile][5][2];
  logic [1:0]  rsp_st    [NumGroups][NumTilesPerGroup][NumNarrowRemoteRespPortsPerTile-1][5][2];
  logic [63:0] rsp_start [NumGroups][NumTilesPerGroup][NumNarrowRemoteRespPortsPerTile-1][5][2];
  // tile: [group][tile][port]
  logic [1:0]  ts_mreq_st [NumGroups][NumTilesPerGroup][NumNarrowRemoteReqPortsPerTile];  logic [63:0] ts_mreq_s [NumGroups][NumTilesPerGroup][NumNarrowRemoteReqPortsPerTile];
  logic [1:0]  ts_sreq_st [NumGroups][NumTilesPerGroup][NumNarrowRemoteReqPortsPerTile];  logic [63:0] ts_sreq_s [NumGroups][NumTilesPerGroup][NumNarrowRemoteReqPortsPerTile];
  logic [1:0]  ts_mrsp_st [NumGroups][NumTilesPerGroup][NumNarrowRemoteRespPortsPerTile]; logic [63:0] ts_mrsp_s [NumGroups][NumTilesPerGroup][NumNarrowRemoteRespPortsPerTile];
  logic [1:0]  ts_srsp_st [NumGroups][NumTilesPerGroup][NumNarrowRemoteRespPortsPerTile]; logic [63:0] ts_srsp_s [NumGroups][NumTilesPerGroup][NumNarrowRemoteRespPortsPerTile];
  // PE (core data port): req = data_q* (out, read/write by qwrite), resp = data_p* (in).
  // One entry per CORE PORT, not per core: a core drives NumDataPortsPerCore
  // TCDM ports flattened as port p of core c = NumDataPortsPerCore*c + p
  // (mempool_tile.sv:122). Scalar builds have NumDataPortsPerCore == 1, so the
  // index stays the core id; Spatz builds add the NumMemPortsPerSpatz vector
  // ports after each core's scalar port.
  localparam int unsigned NumPeCorePorts = NumCoresPerTile * NumDataPortsPerCore;
  logic [1:0]  pe_req_st [NumGroups][NumTilesPerGroup][NumPeCorePorts]; logic [63:0] pe_req_s [NumGroups][NumTilesPerGroup][NumPeCorePorts];
  logic [1:0]  pe_rsp_st [NumGroups][NumTilesPerGroup][NumPeCorePorts]; logic [63:0] pe_rsp_s [NumGroups][NumTilesPerGroup][NumPeCorePorts];
  // RedMulE master-port state (RedMulE tiles only); same S/P format as PE cores.
  logic [1:0]  rm_req_st [NumGroups][NumTilesPerGroup][RMPorts]; logic [63:0] rm_req_s [NumGroups][NumTilesPerGroup][RMPorts];
  logic [1:0]  rm_rsp_st [NumGroups][NumTilesPerGroup][RMPorts]; logic [63:0] rm_rsp_s [NumGroups][NumTilesPerGroup][RMPorts];

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
            // ---- narrow payload-bearing RW request routers ----
            for (int p = 0; p < NumNarrowRwReqPortsPerTile; p++) begin
              automatic int rid = t*NumNarrowRwReqPortsPerTile + p;  // merged-file router tag
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
            // ---- RW/WO response routers (router slot p+1). P lines log every
            //      accepted RESPONSE flit; a resp flit has no tgt_addr. hdr carries dst_id
            //      (requester it returns to), src_id (responder), tile_id/core_id (requester).
            //   P <portidx> <io> <cycle> <dst_x> <dst_y> <src_x> <src_y> <req_tile> <req_core> <meta_id> ----
            for (int p = 0; p < NumNarrowRemoteRespPortsPerTile-1; p++) begin
              automatic int rid = t*(NumNarrowRemoteRespPortsPerTile-1) + p;  // merged-file router tag
              for (int d = 0; d < 4; d++) begin
                automatic logic [1:0] si = `NOC_GRP(g).floo_tcdm_narrow_resp_valid_in_trans[t][p+1][d]
                  ? (`NOC_GRP(g).floo_tcdm_narrow_resp_ready_out_trans[t][p+1][d] ? 2'd2 : 2'd1) : 2'd0;
                automatic logic [1:0] so = `NOC_GRP(g).floo_tcdm_narrow_resp_valid_out_trans[t][p+1][d]
                  ? (`NOC_GRP(g).floo_tcdm_narrow_resp_ready_in_trans[t][p+1][d] ? 2'd2 : 2'd1) : 2'd0;
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
                    `NOC_GRP(g).floo_tcdm_narrow_resp_in_trans[t][p+1][d].hdr.dst_id.x,
                    `NOC_GRP(g).floo_tcdm_narrow_resp_in_trans[t][p+1][d].hdr.dst_id.y,
                    `NOC_GRP(g).floo_tcdm_narrow_resp_in_trans[t][p+1][d].hdr.src_id.x,
                    `NOC_GRP(g).floo_tcdm_narrow_resp_in_trans[t][p+1][d].hdr.src_id.y,
                    `NOC_GRP(g).floo_tcdm_narrow_resp_in_trans[t][p+1][d].hdr.tile_id,
                    `NOC_GRP(g).floo_tcdm_narrow_resp_in_trans[t][p+1][d].hdr.core_id,
                    `NOC_GRP(g).floo_tcdm_narrow_resp_in_trans[t][p+1][d].hdr.meta_id);
                if (so >= 2)  // mesh output handshake
                  $fwrite(f_rresp[g], "%0d P %0d 1 %0d %0d %0d %0d %0d %0d %0d %0d\n", rid, d, cycle_q,
                    `NOC_GRP(g).floo_tcdm_narrow_resp_out_trans[t][p+1][d].hdr.dst_id.x,
                    `NOC_GRP(g).floo_tcdm_narrow_resp_out_trans[t][p+1][d].hdr.dst_id.y,
                    `NOC_GRP(g).floo_tcdm_narrow_resp_out_trans[t][p+1][d].hdr.src_id.x,
                    `NOC_GRP(g).floo_tcdm_narrow_resp_out_trans[t][p+1][d].hdr.src_id.y,
                    `NOC_GRP(g).floo_tcdm_narrow_resp_out_trans[t][p+1][d].hdr.tile_id,
                    `NOC_GRP(g).floo_tcdm_narrow_resp_out_trans[t][p+1][d].hdr.core_id,
                    `NOC_GRP(g).floo_tcdm_narrow_resp_out_trans[t][p+1][d].hdr.meta_id);
              end
              begin : rsp_local
                automatic logic [1:0] sli = `NOC_GRP(g).floo_tcdm_narrow_resp_to_router_valid[t][p+1]
                  ? (`NOC_GRP(g).floo_tcdm_narrow_resp_to_router_ready[t][p+1] ? 2'd2 : 2'd1) : 2'd0;
                automatic logic [1:0] slo = `NOC_GRP(g).floo_tcdm_narrow_resp_from_router_before_xbar_valid[t][p+1]
                  ? (`NOC_GRP(g).floo_tcdm_narrow_resp_from_router_before_xbar_ready[t][p+1] ? 2'd2 : 2'd1) : 2'd0;
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
                    `NOC_GRP(g).floo_tcdm_narrow_resp_to_router[t][p+1].hdr.dst_id.x,
                    `NOC_GRP(g).floo_tcdm_narrow_resp_to_router[t][p+1].hdr.dst_id.y,
                    `NOC_GRP(g).floo_tcdm_narrow_resp_to_router[t][p+1].hdr.src_id.x,
                    `NOC_GRP(g).floo_tcdm_narrow_resp_to_router[t][p+1].hdr.src_id.y,
                    `NOC_GRP(g).floo_tcdm_narrow_resp_to_router[t][p+1].hdr.tile_id,
                    `NOC_GRP(g).floo_tcdm_narrow_resp_to_router[t][p+1].hdr.core_id,
                    `NOC_GRP(g).floo_tcdm_narrow_resp_to_router[t][p+1].hdr.meta_id);
                if (slo >= 2)  // local output (ejection to tile = master_resp) handshake
                  $fwrite(f_rresp[g], "%0d P 4 1 %0d %0d %0d %0d %0d %0d %0d %0d\n", rid, cycle_q,
                    `NOC_GRP(g).floo_tcdm_narrow_resp_from_router[t][p+1].hdr.dst_id.x,
                    `NOC_GRP(g).floo_tcdm_narrow_resp_from_router[t][p+1].hdr.dst_id.y,
                    `NOC_GRP(g).floo_tcdm_narrow_resp_from_router[t][p+1].hdr.src_id.x,
                    `NOC_GRP(g).floo_tcdm_narrow_resp_from_router[t][p+1].hdr.src_id.y,
                    `NOC_GRP(g).floo_tcdm_narrow_resp_from_router[t][p+1].hdr.tile_id,
                    `NOC_GRP(g).floo_tcdm_narrow_resp_from_router[t][p+1].hdr.core_id,
                    `NOC_GRP(g).floo_tcdm_narrow_resp_from_router[t][p+1].hdr.meta_id);
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
  // NumDataPortsPerCore*c + p), so Spatz builds log the scalar port AND all
  // NumMemPortsPerSpatz vector ports of every core, not just the first four
  // ports of core 0.
  // ------------------------------------------------------------
  generate
    for (genvar g = 0; g < NumGroups; g++) begin : gen_pe_g
      for (genvar t = 0; t < NumTilesPerGroup; t++) begin : gen_pe_t
        always_ff @(posedge clk or negedge rst_n) begin
          if (!rst_n) begin
            pe_req_st[g][t] <= '{default: '0}; pe_req_s[g][t] <= '{default: '0};
            pe_rsp_st[g][t] <= '{default: '0}; pe_rsp_s[g][t] <= '{default: '0};
          end else begin
            for (int c = 0; c < NumPeCorePorts; c++) begin
              automatic logic [1:0] rq = `NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.snitch_data_qvalid[c]
                ? (`NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.snitch_data_qready[c]
                   ? (`NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.snitch_data_qwrite[c] ? 2'd3 : 2'd2) : 2'd1) : 2'd0;
              automatic logic [1:0] rp = `NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.snitch_data_pvalid[c]
                ? (`NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.snitch_data_pready[c] ? 2'd2 : 2'd1) : 2'd0;
              if (rq != pe_req_st[g][t][c]) begin
                $fwrite(f_pe[g][t], "%0d S 0 1 0 %0d %0d %0d\n", c, pe_req_s[g][t][c], cycle_q, pe_req_st[g][t][c]);
                pe_req_st[g][t][c] <= rq; pe_req_s[g][t][c] <= cycle_q;
              end
              if (rp != pe_rsp_st[g][t][c]) begin
                $fwrite(f_pe[g][t], "%0d S 1 0 0 %0d %0d %0d\n", c, pe_rsp_s[g][t][c], cycle_q, pe_rsp_st[g][t][c]);
                pe_rsp_st[g][t][c] <= rp; pe_rsp_s[g][t][c] <= cycle_q;
              end
              if (rq >= 2)  // req handshake -> one core load/store accepted (qid = meta_id)
                // mask qid to the low snitch_pkg::MetaIdWidth bits (upper bits are X) so the
                // printed id matches the NoC's zero-extended meta_id for this request.
                // dst = destination tile (from the scrambled address), -1 = SoC/L2.
                $fwrite(f_pe[g][t], "%0d P 1 0 %0d %0d %0h %0d %0d\n", c, cycle_q,
                  `NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.snitch_data_qwrite[c],
                  `NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.snitch_data_qaddr[c],
                  `NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.snitch_data_qid[c][snitch_pkg::MetaIdWidth-1:0],
                  mem_dst_tile(`NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.snitch_data_qaddr_scrambled[c]));
              if (rp >= 2)  // resp handshake -> a response returns to this core (pid = meta_id)
                $fwrite(f_pe[g][t], "%0d P 0 0 %0d %0d\n", c, cycle_q,
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

  // ------------------------------------------------------------
  // Spatz VLSU per-REQUEST memory trace -> per-tile file (Spatz builds only).
  // The pe_ capture above already shows these ports as state runs + packets; this
  // adds the one thing the state runs cannot give: the ROUND TRIP of each word
  // request, i.e. the two instants that bracket it --
  //   Q = the request-ACCEPTED handshake (valid & ready; bare valid is already
  //       covered by the Spatz engine trace's `memact`)
  //   P = the response arriving back at the Spatz boundary (no ready there)
  // Pairing key = the per-port ROB id the VLSU already puts on the request
  // (spatz_vlsu.sv: mem_req_id[port] = rob_id[port]) and gets back on the
  // response, so no new RTL signal is needed. Stores are logged too: their ROB
  // entry frees at send rather than at response, so `wr` keeps a store-ack
  // distinguishable from a load reply that reuses the same id. Only the low
  // IdBits of `id` round-trip (mempool_cc.sv packs {write, id[IdBits-1:0]} into
  // meta_id), so a matcher must mask to IdBits -- hence IdBits in the D header.
  //
  // One file per TILE (merged like pe_/rm_, line PREFIXED with the core index):
  //   spmem_g<g>_t<t>.log
  //     D <NrMemPorts> <NumCoresPerTile> <NumTilesPerGroup> <NumTiles> <IdBits>
  //     <c> Q <cyc> <port> <id> <addr_hex> <wr> <dst_tile>
  //     <c> P <cyc> <port> <id> <wr>
  //     E <last_cyc>
  // `addr` is the RAW core address (pre-scrambler) so it can be matched back to
  // the kernel's data structures. `dst_tile` is the GLOBAL destination tile id
  // taken from the tile's SCRAMBLED address of the SAME request -- i.e. after the
  // seq/interleave swap and the tile-id remap -- or -1 for a non-TCDM (SoC/L2)
  // access; the Spatz port feeds the tile's core data port combinationally
  // (mempool_cc.sv:451), so both are valid in the handshake cycle. Destination
  // group = dst_tile / NumTilesPerGroup (mempool_tile.sv:91), so an offline pass
  // classifies every request as local-tile / intra-group / inter-group / SoC
  // without re-implementing the address scrambler.
  // ------------------------------------------------------------
  // Core c's Spatz vector unit (mempool_cc.riscv_core.gen_spatz.i_spatz).
  `define SPMEM_SPATZ(gg,tt,cc) `NOC_GRP(gg).i_mempool_group.gen_tiles[tt].i_tile.gen_cores[cc].gen_mempool_cc.riscv_core.gen_spatz.i_spatz
  generate
    if (mempool_pkg::RVV) begin : gen_spmem
      for (genvar g = 0; g < NumGroups; g++) begin : gen_spmem_g
        for (genvar t = 0; t < NumTilesPerGroup; t++) begin : gen_spmem_t
          for (genvar c = 0; c < NumCoresPerTile; c++) begin : gen_spmem_c
            always_ff @(posedge clk or negedge rst_n) begin
              if (rst_n) begin
                for (int p = 0; p < mempool_pkg::NumMemPortsPerSpatz; p++) begin
                  if (`SPMEM_SPATZ(g,t,c).spatz_mem_req_valid_o[p] &&
                      `SPMEM_SPATZ(g,t,c).spatz_mem_req_ready_i[p])
                    $fwrite(f_spmem[g][t], "%0d Q %0d %0d %0d %0h %0d %0d\n",
                      c, cycle_q, p,
                      `SPMEM_SPATZ(g,t,c).spatz_mem_req_o[p].id,
                      `SPMEM_SPATZ(g,t,c).spatz_mem_req_o[p].addr,
                      `SPMEM_SPATZ(g,t,c).spatz_mem_req_o[p].write,
                      mem_dst_tile(`NOC_GRP(g).i_mempool_group.gen_tiles[t].i_tile.snitch_data_qaddr_scrambled[
                                     mempool_pkg::NumDataPortsPerCore*c + 1 + p]));
                  if (`SPMEM_SPATZ(g,t,c).spatz_mem_rsp_valid_i[p])
                    $fwrite(f_spmem[g][t], "%0d P %0d %0d %0d %0d\n",
                      c, cycle_q, p,
                      `SPMEM_SPATZ(g,t,c).spatz_mem_rsp_i[p].id,
                      `SPMEM_SPATZ(g,t,c).spatz_mem_rsp_i[p].write);
                end
              end
            end
          end
        end
      end
    end
  endgenerate

  // ------------------------------------------------------------
  // End-of-sim flush: write every port's still-open S run, then close every file.
  // P lines are written as they happen, so they need no flush.
  // ------------------------------------------------------------
  final begin
    for (int g = 0; g < NumGroups; g++) begin
      for (int t = 0; t < NumTilesPerGroup; t++) begin
        for (int p = 0; p < NumNarrowRwReqPortsPerTile; p++) begin
          automatic int rid = t*NumNarrowRwReqPortsPerTile + p;
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
        $fclose(f_tile[g][t]);
        for (int c = 0; c < NumPeCorePorts; c++) begin
          $fwrite(f_pe[g][t], "%0d S 0 1 0 %0d %0d %0d\n", c, pe_req_s[g][t][c], cycle_q, pe_req_st[g][t][c]);
          $fwrite(f_pe[g][t], "%0d S 1 0 0 %0d %0d %0d\n", c, pe_rsp_s[g][t][c], cycle_q, pe_rsp_st[g][t][c]);
        end
        $fclose(f_pe[g][t]);
        if (mempool_pkg::RVV) begin
          $fwrite(f_spmem[g][t], "E %0d\n", cycle_q);
          $fclose(f_spmem[g][t]);
        end
        if (t < NumRMTilesPerGroup) begin
          for (int p = 0; p < RMMasterPorts; p++) begin
            $fwrite(f_rm[g][t], "%0d S 0 1 0 %0d %0d %0d\n", p, rm_req_s[g][t][p], cycle_q, rm_req_st[g][t][p]);
            $fwrite(f_rm[g][t], "%0d S 1 0 0 %0d %0d %0d\n", p, rm_rsp_s[g][t][p], cycle_q, rm_rsp_st[g][t][p]);
          end
          $fclose(f_rm[g][t]);
        end
      end
      $fclose(f_rreq[g]);
      $fclose(f_rresp[g]);
    end
  end

`endif // NOC_PROFILING

`endif // TARGET_VERILATOR
`endif // TARGET_SYNTHESIS

`endif // TB_NOC_PROFILING_SVH_
