// Copyright 2025 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

`ifndef TB_SPATZ_PROFILING_SVH_
`define TB_SPATZ_PROFILING_SVH_

`ifndef TARGET_SYNTHESIS
`ifndef TARGET_VERILATOR

  // ===========================================================================
  // Spatz VECTOR ENGINE trace -- ALWAYS ON whenever the cores are Spatz cores
  // (mempool_pkg::RVV). Engine INTERNALS only (controller scoreboard occupancy,
  // VFU / VLSU / VSLDU busy + sub-state, issue/retire handshakes, active memory
  // ports); the Spatz memory-port traffic itself stays under NOC_PROFILING.
  //
  // Spatz is per-CORE (not per-tile like RedMulE), so one log file per Spatz
  // core (reuses cycle_q from tb_noc_profiling.svh):
  //   D <VLEN> <N_FU> <NrMemPorts> <VRFWordBytes> <ELENB> <RobDepth>
  //                                                                 (header, once)
  //   C <cyc> <occ> <stall> <issue> <vfu_busy> <vfu_st> <vlsu_busy> <vlsu_st>
  //       <vsldu_busy> <vfu_rsp> <vlsu_rsp> <vsldu_rsp> <memact>
  //       <istall> <vstall> <lin> <lout> <vlsu_out> <rob_full> <rob_empty>
  //   L <cyc> <vsew> <nr_elem_word> <elems> <lane0> .. <lane(N_FU-1)>
  //                                                            (VFU lane state)
  //   B <cyc> <pend0> .. <pend(NrMemPorts-1)>                  (VLSU ROB level)
  //   I <cyc> <id> <unit> <vl> <vsew> <vlmul>              (vector insn issued)
  //   R <cyc> <unit> <id>                                  (vector insn retired)
  //   E <last_cyc>                                                  (footer, once)
  //
  // L and B are CHANGE-DETECTED state runs (a line only when the value moves,
  // like the NoC port traces), not one line per cycle: per-cycle detail is
  // preserved because a run spans until the next line, at a fraction of the
  // volume. They are separate record letters rather than more C columns on
  // purpose -- widening C has silently broken its parsers twice.
  // A C line is written every cycle the engine is active; a cycle with no C line
  // means IDLE. The exporter derives the FSM slices, util counters and the
  // occupancy/CPI breakdown from these raw samples, so the policy can change
  // without re-simulating.
  //
  //   occ        = $countones(controller.running_insn_q) -- in-flight vinsns
  //   stall      = controller.stall (issue stalled: target ex-unit not ready)
  //   issue      = controller issue handshake (issue_valid & issue_ready)
  //   vfu_busy   = i_vfu.busy_q ;  vfu_st  = i_vfu.state_q  (0=IPU,1=FPU)
  //   vlsu_busy  = i_vlsu.busy_q;  vlsu_st = i_vlsu.state_q (0=load,1=store)
  //   vsldu_busy = |i_vsldu.running_q
  //   *_rsp      = per-unit retire-valid (vfu/vlsu/vsldu_rsp_valid)
  //   memact     = $countones(spatz_mem_req_valid_o) -- TCDM ports requesting
  //   vlsu_out   = sum of i_vlsu.mem_pending_q -- requests ISSUED and not yet
  //                answered, summed over the NrMemPorts reorder buffers. This
  //                is the engine's own view of its outstanding memory work; the
  //                per-request round trip itself is in noc_profiling/core_.
  //   rob_full   = i_vlsu.rob_full  (bitmap, one bit per mem port) -- the port
  //                cannot accept another request: NrOutstandingLoads reached.
  //   rob_empty  = i_vlsu.rob_empty (bitmap) -- the port has fully drained.
  //                &rob_empty is what the VLSU FSM waits on to change state.
  //
  // WHY an instruction did not issue -- `stall` is one bit, so it cannot tell a
  // busy target unit from a full scoreboard. `istall` is the bitmap behind it
  // (spatz_controller.sv:376 and its req_buffer_pop condition):
  //   b0 vfu_stall   b1 vlsu_stall   b2 vsldu_stall  -- target unit not ready
  //   b3 running_insn_full                           -- no free scoreboard id
  //   b4 req_buffer_valid                            -- an insn IS waiting
  // b4 separates "blocked" from "nothing to run": with b4 clear the engine has
  // no work (Snitch has not offloaded), which is not a stall.
  //
  // WHY the VFU produced nothing -- `vstall`, valid only while the VFU holds a
  // request (i_vfu.spatz_req_valid), else 0:
  //   b0/b1/b2 !op{1,2,3}_is_ready  -- waiting on a VRF read (operand hazard)
  //   b3 FU inputs not ready        -- ~&(in_ready | ~valid_operations)
  //   b4 i_vfu.stall                -- IPU<->FPU datapath switch drain
  //   b5 scalar writeback wait      -- is_scalar & ~vfu_rsp_ready_i
  //
  // HOW MUCH of the datapath worked -- busy is one bit and says nothing about
  // how many lanes ran (a vl=4 tail looks like a full-width operation):
  //   lin  = $countones(valid_operations) on the cycle a word is fed to the FUs
  //          (i_vfu.word_issued), else 0. BYTES of real work fed this cycle, so
  //          lin / VRFWordBytes is the true VFU lane utilization.
  //   lout = $countones(i_vfu.result_valid) -- bytes whose result is asserted.
  //          Held across writeback back-pressure, so it is occupancy-WITH-
  //          back-pressure (like memact on the memory side), not a work count.
  //   VRFWordBytes = N_FU*ELENB, the per-cycle peak (in the D header).
  //
  // PER-LANE work and the SEW-relative peak -- the `L` record.
  //
  // NOTE what is NOT usable here: valid_operations is '1 for every ordinary
  // vector op (spatz_vfu.sv:140-142) -- it only narrows for scalar and
  // reduction ops -- and vreg_wbe likewise. So neither carries the vl tail,
  // and a per-lane popcount of them would be a constant ELENB. `lin` above
  // therefore measures word_issued, not lane fill, whatever its name suggests.
  // The two signals that DO carry it:
  //   elems = min(spatz_req.vl - vl_q, nr_elem_word) on an issued cycle, else
  //          0 -- the ELEMENTS actually fed this cycle. Spatz advances vl_q by
  //          a whole nr_elem_word per word (spatz_vfu.sv:228), so only the LAST
  //          word of a vector is partial; this is what makes that visible.
  //   nr_elem_word = spatz_vfu.sv:103 -- the elements one VRF word holds at the
  //          CURRENT vsew (N_FU << (MAXEW-vsew), halved when narrowing): 16 fp8
  //          / 8 fp16 / 4 fp32 on N_FU=4, ELEN=32. elems/nr_elem_word is
  //          therefore throughput as a fraction of the peak AT THIS ELEMENT
  //          WIDTH, so a kernel that fills its lanes with fp8 reads 100% rather
  //          than being penalised for not filling the byte width.
  //   lane l = $countones(result_valid[l*ELENB +: ELENB]) -- the OUTPUT side,
  //          per lane. This is the only genuinely per-lane signal in the VFU:
  //          the input side is lock-step by construction (all lanes take a
  //          slice of the same VRF word), so lanes differ only through FU
  //          latency, IPU/FPU switches and drain, which is exactly what the
  //          per-lane lines show.
  //   All are 0 on a cycle with no word issued, so "not feeding" is its own
  //   state rather than a stale hold.
  //
  // VLSU ROB LEVEL -- the `B` record: mem_pending_q[port], the per-port count
  // of requests issued and not yet answered (0..RobDepth). C carries only the
  // sum (vlsu_out) and the full/empty bitmaps; the level per port is what shows
  // WHICH port saturated and how deep it ran.
  //
  // WHAT was issued -- one `I` line per vector instruction accepted by the
  // controller (spatz_req_valid_o), so per-instruction metrics can be
  // normalised by length instead of averaging over mixed shapes:
  //   unit = spatz_req.ex_unit (0=CON 1=LSU 2=SLD 3=VFU)
  //   vl   = elements as issued (widening ops carry the doubled value)
  //   vsew = 0..3 -> 8/16/32/64-bit elements; vlmul 0..3 = 1,2,4,8 and
  //          5..7 = 1/8,1/4,1/2 (rvv_pkg vew_e / vlmul_e)
  // `R` closes it -- retire of scoreboard id <id> by <unit> (0=VFU 1=VLSU
  // 2=VSLDU) -- so pairing I->R per (unit,id) gives per-instruction latency.
  // ===========================================================================

  // Path to core c's Spatz vector unit. Mirrors RM_TOP but Spatz lives per core
  // inside the unified core complex (mempool_cc.riscv_core.gen_spatz.i_spatz).
  `define SPATZ_TOP(gg,tt,cc) dut.i_mempool_cluster.gen_groups_x[(gg)/NumY].gen_groups_y[(gg)%NumY].gen_rtl_group.i_group.i_mempool_group.gen_tiles[tt].i_tile.gen_cores[cc].gen_mempool_cc.riscv_core.gen_spatz.i_spatz

  if (mempool_pkg::RVV) begin : gen_spatz_eng_prof
    int f_speng [NumGroups][NumTilesPerGroup][NumCoresPerTile];

    initial begin
      string  sp_log_path;
      integer sp_retval;
      sp_log_path = "spatz_profiling";
      sp_retval   = $system({"mkdir -p ", sp_log_path});
      for (int g = 0; g < NumGroups; g++)
        for (int t = 0; t < NumTilesPerGroup; t++)
          for (int c = 0; c < NumCoresPerTile; c++) begin
            f_speng[g][t][c] = $fopen($sformatf("%s/spatz_g%0d_t%0d_c%0d.log",
                                                sp_log_path, g, t, c), "w");
            // ELENB sizes one lane; RobDepth (NrOutstandingLoads, read off the
            // VLSU instance) scales the per-port ROB level.
            $fwrite(f_speng[g][t][c], "D %0d %0d %0d %0d %0d %0d\n",
                    spatz_pkg::VLEN, spatz_pkg::N_FU,
                    mempool_pkg::NumMemPortsPerSpatz,
                    spatz_pkg::N_FU * spatz_pkg::ELENB,
                    spatz_pkg::ELENB,
                    `SPATZ_TOP(0,0,0).i_vlsu.NrOutstandingLoads);
          end
    end

    for (genvar g = 0; g < NumGroups; g++) begin : gen_g
      for (genvar t = 0; t < NumTilesPerGroup; t++) begin : gen_t
        for (genvar c = 0; c < NumCoresPerTile; c++) begin : gen_c
          // Change-detection state for the L/B records: the last values
          // written, so a line is emitted only when one of them moves. Cleared
          // in the reset branch below -- NOT by a declaration initializer,
          // which VCS rejects as a second procedural driver on a variable an
          // always_ff also writes (ICPD_INIT); Questa accepts it silently.
          // lb_init makes cycle 0 anchor the run rather than depending on the
          // first compare.
          logic [spatz_pkg::N_FU-1:0][7:0]                  prev_lane;
          logic [2:0]                                       prev_sew;
          int unsigned                                      prev_nelem;
          int unsigned                                      prev_elems;
          logic [mempool_pkg::NumMemPortsPerSpatz-1:0][7:0] prev_pend;
          logic                                             lb_init;

          always_ff @(posedge clk or negedge rst_n) begin
            if (rst_n) begin
              automatic int unsigned occ        = $countones(`SPATZ_TOP(g,t,c).i_controller.running_insn_q);
              automatic logic        stall      = `SPATZ_TOP(g,t,c).i_controller.stall;
              automatic logic        issue      = `SPATZ_TOP(g,t,c).issue_valid & `SPATZ_TOP(g,t,c).issue_ready;
              automatic logic        vfu_busy   = `SPATZ_TOP(g,t,c).i_vfu.busy_q;
              automatic int unsigned vfu_st     = `SPATZ_TOP(g,t,c).i_vfu.state_q;
              automatic logic        vlsu_busy  = `SPATZ_TOP(g,t,c).i_vlsu.busy_q;
              automatic int unsigned vlsu_st    = `SPATZ_TOP(g,t,c).i_vlsu.state_q;
              automatic logic        vsldu_busy = |`SPATZ_TOP(g,t,c).i_vsldu.running_q;
              automatic logic        vfu_rsp    = `SPATZ_TOP(g,t,c).vfu_rsp_valid;
              automatic logic        vlsu_rsp   = `SPATZ_TOP(g,t,c).vlsu_rsp_valid;
              automatic logic        vsldu_rsp  = `SPATZ_TOP(g,t,c).vsldu_rsp_valid;
              automatic int unsigned memact     = $countones(`SPATZ_TOP(g,t,c).spatz_mem_req_valid_o);
              // Issue-stall causes (see the header): which target unit was not
              // ready, whether the scoreboard was full, and whether anything
              // was waiting at all.
              automatic logic [4:0]  istall     = {`SPATZ_TOP(g,t,c).i_controller.req_buffer_valid,
                                                   `SPATZ_TOP(g,t,c).i_controller.running_insn_full,
                                                   `SPATZ_TOP(g,t,c).i_controller.vsldu_stall,
                                                   `SPATZ_TOP(g,t,c).i_controller.vlsu_stall,
                                                   `SPATZ_TOP(g,t,c).i_controller.vfu_stall};
              // VFU-side causes; op*_is_ready already include the request-valid
              // term, so gate the whole bitmap on it to avoid phantom stalls.
              automatic logic        vfu_req    = `SPATZ_TOP(g,t,c).i_vfu.spatz_req_valid;
              automatic logic [5:0]  vstall     = !vfu_req ? 6'd0 : {
                  `SPATZ_TOP(g,t,c).i_vfu.spatz_req.op_arith.is_scalar &
                    ~`SPATZ_TOP(g,t,c).i_vfu.vfu_rsp_ready_i,
                  `SPATZ_TOP(g,t,c).i_vfu.stall,
                  ~&(`SPATZ_TOP(g,t,c).i_vfu.in_ready |
                     ~`SPATZ_TOP(g,t,c).i_vfu.valid_operations),
                  ~`SPATZ_TOP(g,t,c).i_vfu.op3_is_ready,
                  ~`SPATZ_TOP(g,t,c).i_vfu.op2_is_ready,
                  ~`SPATZ_TOP(g,t,c).i_vfu.op1_is_ready};
              // Lane occupancy: bytes actually fed to the FUs this cycle, and
              // bytes whose result is asserted (see the header for why they
              // differ).
              automatic int unsigned lin        = `SPATZ_TOP(g,t,c).i_vfu.word_issued
                                                ? $countones(`SPATZ_TOP(g,t,c).i_vfu.valid_operations) : 0;
              automatic int unsigned lout       = $countones(`SPATZ_TOP(g,t,c).i_vfu.result_valid);
              // VLSU reorder-buffer state. The per-request memory trace lives
              // in core_ (noc_profiling) -- what belongs HERE is the engine's
              // own view of it: how many requests it has outstanding and
              // whether a port is saturated or drained.
              automatic int unsigned vlsu_out   = 0;
              automatic logic [mempool_pkg::NumMemPortsPerSpatz-1:0] rob_full =
                  `SPATZ_TOP(g,t,c).i_vlsu.rob_full;
              automatic logic [mempool_pkg::NumMemPortsPerSpatz-1:0] rob_empty =
                  `SPATZ_TOP(g,t,c).i_vlsu.rob_empty;
              // Per-port ROB level (the `B` record) and per-lane VFU bytes with
              // their SEW peak (the `L` record); both 0 when nothing is
              // pending / no word is fed, so idle is its own state.
              automatic logic [mempool_pkg::NumMemPortsPerSpatz-1:0][7:0] pend = '0;
              automatic logic [spatz_pkg::N_FU-1:0][7:0] lane  = '0;
              automatic logic [2:0]                      sew   = '0;
              automatic int unsigned                     nelem = 0;
              automatic int unsigned                     elems = 0;
              for (int mp = 0; mp < mempool_pkg::NumMemPortsPerSpatz; mp++) begin
                pend[mp]  = `SPATZ_TOP(g,t,c).i_vlsu.mem_pending_q[mp];
                vlsu_out += `SPATZ_TOP(g,t,c).i_vlsu.mem_pending_q[mp];
              end
              if (`SPATZ_TOP(g,t,c).i_vfu.word_issued) begin
                for (int l = 0; l < spatz_pkg::N_FU; l++)
                  lane[l] = $countones(`SPATZ_TOP(g,t,c).i_vfu.result_valid[
                                       l*spatz_pkg::ELENB +: spatz_pkg::ELENB]);
                sew   = `SPATZ_TOP(g,t,c).i_vfu.spatz_req.vtype.vsew;
                nelem = `SPATZ_TOP(g,t,c).i_vfu.nr_elem_word;
                // Only the last word of a vector is partial (vl_q advances by a
                // whole nr_elem_word), so this clamp is the tail.
                elems = (`SPATZ_TOP(g,t,c).i_vfu.spatz_req.vl >
                         `SPATZ_TOP(g,t,c).i_vfu.vl_q)
                      ? ((`SPATZ_TOP(g,t,c).i_vfu.spatz_req.vl -
                          `SPATZ_TOP(g,t,c).i_vfu.vl_q) < nelem
                         ? (`SPATZ_TOP(g,t,c).i_vfu.spatz_req.vl -
                            `SPATZ_TOP(g,t,c).i_vfu.vl_q)
                         : nelem)
                      : 0;
              end
              if (occ != 0 || vfu_busy || vlsu_busy || vsldu_busy || issue ||
                  memact != 0 || istall[3:0] != 4'd0 || vlsu_out != 0)
                $fwrite(f_speng[g][t][c],
                        "C %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d\n",
                        cycle_q, occ, stall, issue, vfu_busy, vfu_st,
                        vlsu_busy, vlsu_st, vsldu_busy,
                        vfu_rsp, vlsu_rsp, vsldu_rsp, memact,
                        istall, vstall, lin, lout,
                        vlsu_out, rob_full, rob_empty);
              // Change-detected state runs: one line only when the value moves.
              if (!lb_init || lane != prev_lane || sew != prev_sew ||
                  nelem != prev_nelem || elems != prev_elems) begin
                $fwrite(f_speng[g][t][c], "L %0d %0d %0d %0d",
                        cycle_q, sew, nelem, elems);
                for (int l = 0; l < spatz_pkg::N_FU; l++)
                  $fwrite(f_speng[g][t][c], " %0d", lane[l]);
                $fwrite(f_speng[g][t][c], "\n");
                prev_lane  <= lane;
                prev_sew   <= sew;
                prev_nelem <= nelem;
                prev_elems <= elems;
              end
              if (!lb_init || pend != prev_pend) begin
                $fwrite(f_speng[g][t][c], "B %0d", cycle_q);
                for (int mp = 0; mp < mempool_pkg::NumMemPortsPerSpatz; mp++)
                  $fwrite(f_speng[g][t][c], " %0d", pend[mp]);
                $fwrite(f_speng[g][t][c], "\n");
                prev_pend <= pend;
              end
              lb_init <= 1'b1;
              // One line per vector instruction issued / retired, so a metric
              // can be per instruction (and length-aware) instead of a mean
              // over whatever mix the kernel happened to run.
              if (`SPATZ_TOP(g,t,c).i_controller.spatz_req_valid_o)
                $fwrite(f_speng[g][t][c], "I %0d %0d %0d %0d %0d %0d\n", cycle_q,
                        `SPATZ_TOP(g,t,c).i_controller.spatz_req_o.id,
                        `SPATZ_TOP(g,t,c).i_controller.spatz_req_o.ex_unit,
                        `SPATZ_TOP(g,t,c).i_controller.spatz_req_o.vl,
                        `SPATZ_TOP(g,t,c).i_controller.spatz_req_o.vtype.vsew,
                        `SPATZ_TOP(g,t,c).i_controller.spatz_req_o.vtype.vlmul);
              if (vfu_rsp)
                $fwrite(f_speng[g][t][c], "R %0d 0 %0d\n", cycle_q,
                        `SPATZ_TOP(g,t,c).vfu_rsp.id);
              if (vlsu_rsp)
                $fwrite(f_speng[g][t][c], "R %0d 1 %0d\n", cycle_q,
                        `SPATZ_TOP(g,t,c).vlsu_rsp.id);
              if (vsldu_rsp)
                $fwrite(f_speng[g][t][c], "R %0d 2 %0d\n", cycle_q,
                        `SPATZ_TOP(g,t,c).vsldu_rsp.id);
            end else begin
              // Sole reset of the L/B change-detection state (see above).
              prev_lane  <= '0;
              prev_sew   <= '0;
              prev_nelem <= '0;
              prev_elems <= '0;
              prev_pend  <= '0;
              lb_init    <= 1'b0;
            end
          end
        end
      end
    end

    final begin
      // End-of-sim marker: last cycle, so the exporter can render IDLE across
      // every gap between active bursts and out to the end of the run.
      for (int g = 0; g < NumGroups; g++)
        for (int t = 0; t < NumTilesPerGroup; t++)
          for (int c = 0; c < NumCoresPerTile; c++) begin
            $fwrite(f_speng[g][t][c], "E %0d\n", cycle_q);
            $fclose(f_speng[g][t][c]);
          end
    end
  end

`endif // TARGET_VERILATOR
`endif // TARGET_SYNTHESIS
`endif // TB_SPATZ_PROFILING_SVH_
