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
  //   D <VLEN> <N_FU> <NrMemPorts> <VRFWordBytes>                   (header, once)
  //   C <cyc> <occ> <stall> <issue> <vfu_busy> <vfu_st> <vlsu_busy> <vlsu_st>
  //       <vsldu_busy> <vfu_rsp> <vlsu_rsp> <vsldu_rsp> <memact>
  //       <istall> <vstall> <lin> <lout>
  //   I <cyc> <id> <unit> <vl> <vsew> <vlmul>              (vector insn issued)
  //   R <cyc> <unit> <id>                                  (vector insn retired)
  //   E <last_cyc>                                                  (footer, once)
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
            $fwrite(f_speng[g][t][c], "D %0d %0d %0d %0d\n",
                    spatz_pkg::VLEN, spatz_pkg::N_FU,
                    mempool_pkg::NumMemPortsPerSpatz,
                    spatz_pkg::N_FU * spatz_pkg::ELENB);
          end
    end

    for (genvar g = 0; g < NumGroups; g++) begin : gen_g
      for (genvar t = 0; t < NumTilesPerGroup; t++) begin : gen_t
        for (genvar c = 0; c < NumCoresPerTile; c++) begin : gen_c
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
              if (occ != 0 || vfu_busy || vlsu_busy || vsldu_busy || issue ||
                  memact != 0 || istall[3:0] != 4'd0)
                $fwrite(f_speng[g][t][c],
                        "C %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d\n",
                        cycle_q, occ, stall, issue, vfu_busy, vfu_st,
                        vlsu_busy, vlsu_st, vsldu_busy,
                        vfu_rsp, vlsu_rsp, vsldu_rsp, memact,
                        istall, vstall, lin, lout);
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
