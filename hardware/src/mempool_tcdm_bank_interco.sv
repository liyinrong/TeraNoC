// Copyright 2025 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

`include "mempool/mempool.svh"

// Tile-internal TCDM bank-side interconnect: the narrow req/resp crossbars, one
// superbank handler per wide-word ingress, and the per-bank wide-over-narrow
// priority. Companion modules: mempool_tcdm_wide_superbank.sv (the per-ingress
// fork/join wide datapath) and mempool_bank_id_remapper.sv.
//
// Two kinds of wide-word ingress, each on its own ports and its own superbank
// handler: slv_dma_* is the DMA (always present) and slv_wide_* is the WP2 wide
// TCDM plane (the RedMulE tile paths, present only with USE_WIDE_TCDM_PLANE).
// `wide` therefore names the plane, never the DMA, even though both move a
// multi-bank word; WIDE_W need not equal the DMA width. The two share the banks
// through the priority below (wide > DMA > narrow), granted SUPERBANK-ATOMICALLY
// so ingress paths cannot interleave partial forks into the per-bank response
// queues (see gen_bank_mux).
//
// The superbank index (the upper bank-id bits, i.e. which superbank a wide
// access lands on) passes through unchanged so a wide chunk lands in the same
// superbank a narrow access of its base address would pick; each class's fork
// then rotates within that superbank to stay coherent with the narrow path.
module mempool_tcdm_bank_interco
  import mempool_pkg::*;
  import cf_math_pkg::idx_width;
#(
  parameter int unsigned NumNarrowReq      = 8,
  parameter int unsigned NumNarrowResp     = 8,
  parameter int unsigned NumBanksPerTile   = 16,
  parameter int unsigned NumSuperbanks     = 1,
  parameter int unsigned NarrowDataWidth   = 32,
  parameter int unsigned ByteOffset        = 2,
  parameter bit          SpmBankIdRemap    = 0,
  parameter type narrow_req_t  = tcdm_slave_req_t,
  parameter type narrow_resp_t = tcdm_slave_resp_t,
  parameter type group_id_t    = mempool_pkg::group_id_t,
  // Must be wide enough to address NumNarrowResp ports (logic [idx_width(NumNarrowResp)-1:0]).
  parameter type resp_idx_t    = logic [3:0],
  // DMA ingress. The `wide` qualifier stays on these because mempool_pkg already
  // owns DmaDataWidth and dma_req_t (the DMA frontend descriptor, unrelated).
  parameter int unsigned DmaWideDataWidth = 512,
  parameter type dma_wide_req_t  = tcdm_dma_req_t,
  parameter type dma_wide_resp_t = tcdm_dma_resp_t,
  // WP2 wide TCDM plane ingress. Defaults keep it inert when unused; a WP2 build
  // overrides these with WIDE_W and the wide req/resp types.
  parameter int unsigned NumWidePorts  = 1,
  parameter int unsigned WideDataWidth = 512,
  parameter type wide_req_t  = logic,
  parameter type wide_resp_t = logic,
  // derived
  localparam int unsigned NumBanksPerSB    = NumBanksPerTile / NumSuperbanks,
  localparam int unsigned BankOffsetBits   = $clog2(NumBanksPerTile),
  localparam int unsigned BankIdLoBits     = $clog2(NumBanksPerSB),
  localparam int unsigned NarrowBeWidth    = NarrowDataWidth / 8,
  // Bank-id remap offset width (== bank_id_lo width). NumBanksPerSB >= 2 is
  // required (asserted below); a single-bank superbank would zero-width the slices.
  localparam int unsigned BankRemapShiftAmt = BankIdLoBits,
  // Outstanding wide requests per superbank, per ingress.
  localparam int unsigned RotFifoDepth      = 2,
  // Per-bank class tags in flight. Size for every wide ingress to reach its
  // per-superbank outstanding-request limit without this queue becoming a
  // lower shared limit.
  localparam int unsigned ClassFifoDepth =
      (NumWidePorts + 1) * RotFifoDepth,
  // Per-ingress superbank geometry. The DMA and plane widths may differ.
  localparam int unsigned NumBanksPerDmaSB  = DmaWideDataWidth / NarrowDataWidth,
  localparam int unsigned NumDmaSB          = NumBanksPerTile / NumBanksPerDmaSB,
  localparam int unsigned NumBanksPerWideSB = WideDataWidth / NarrowDataWidth,
  localparam int unsigned NumWideSB         = NumBanksPerTile / NumBanksPerWideSB
) (
  input  logic                                          clk_i,
  input  logic                                          rst_ni,
  input  group_id_t                                     group_id_i,

  // ----- Narrow request inputs -----
  input  narrow_req_t  [NumNarrowReq-1:0]               slv_narrow_req_i,
  input  logic         [NumNarrowReq-1:0]               slv_narrow_req_valid_i,
  output logic         [NumNarrowReq-1:0]               slv_narrow_req_ready_o,

  // ----- Narrow response outputs -----
  output narrow_resp_t [NumNarrowResp-1:0]              slv_narrow_resp_o,
  output logic         [NumNarrowResp-1:0]              slv_narrow_resp_valid_o,
  input  logic         [NumNarrowResp-1:0]              slv_narrow_resp_ready_i,

  // ----- DMA request input (single, demuxed internally) -----
  input  dma_wide_req_t                                 slv_dma_req_i,
  input  logic                                          slv_dma_req_valid_i,
  output logic                                          slv_dma_req_ready_o,

  // ----- DMA response output (single, muxed internally) -----
  output dma_wide_resp_t                                slv_dma_resp_o,
  output logic                                          slv_dma_resp_valid_o,
  input  logic                                          slv_dma_resp_ready_i,

  // ----- Wide TCDM plane request/response ports (WP2 RedMulE) -----
  `ifdef USE_WIDE_TCDM_PLANE
  input  wide_req_t    [NumWidePorts-1:0]               slv_wide_req_i,
  input  logic         [NumWidePorts-1:0]               slv_wide_req_valid_i,
  output logic         [NumWidePorts-1:0]               slv_wide_req_ready_o,
  output wide_resp_t   [NumWidePorts-1:0]               slv_wide_resp_o,
  output logic         [NumWidePorts-1:0]               slv_wide_resp_valid_o,
  input  logic         [NumWidePorts-1:0]               slv_wide_resp_ready_i,
  `endif

  // ----- Bank-side request ports (per bank) -----
  output narrow_req_t  [NumBanksPerTile-1:0]            mst_req_o,
  output logic         [NumBanksPerTile-1:0]            mst_req_valid_o,
  input  logic         [NumBanksPerTile-1:0]            mst_req_ready_i,
  output logic         [NumBanksPerTile-1:0]            mst_req_wide_o,
  output resp_idx_t    [NumBanksPerTile-1:0]            mst_req_ini_addr_o,

  // ----- Bank-side response ports (per bank) -----
  input  narrow_resp_t [NumBanksPerTile-1:0]            mst_resp_i,
  input  logic         [NumBanksPerTile-1:0]            mst_resp_valid_i,
  output logic         [NumBanksPerTile-1:0]            mst_resp_ready_o,
  input  logic         [NumBanksPerTile-1:0]            mst_resp_wide_i,
  input  resp_idx_t    [NumBanksPerTile-1:0]            mst_resp_ini_addr_i
);

  // A single-bank superbank would zero-width the bank_id_lo slices.
  if (NumBanksPerSB < 2)
    $fatal(1, "mempool_tcdm_bank_interco: NumBanksPerSB (=%0d) must be >= 2.", NumBanksPerSB);
  // Every wide class must tile the banks exactly, or the atomic grant below would
  // index a partial superbank.
  if (NumDmaSB * NumBanksPerDmaSB != NumBanksPerTile)
    $fatal(1, "mempool_tcdm_bank_interco: DMA superbank (%0d banks) does not tile %0d banks.",
           NumBanksPerDmaSB, NumBanksPerTile);
  `ifdef USE_WIDE_TCDM_PLANE
  if (NumWidePorts < 1)
    $fatal(1, "mempool_tcdm_bank_interco: at least one wide TCDM plane port is required.");
  if (NumWideSB * NumBanksPerWideSB != NumBanksPerTile)
    $fatal(1, "mempool_tcdm_bank_interco: wide superbank (%0d banks) does not tile %0d banks.",
           NumBanksPerWideSB, NumBanksPerTile);
  `endif

  // ============================================================
  // Narrow-request: bank-id-lo remap + xbar
  // ============================================================
  // Within-superbank constraint: only the low BankIdLoBits of bank_id are rewritten.
  logic [NumNarrowReq-1:0][BankOffsetBits-1:0] narrow_sel;
  for (genvar i = 0; i < NumNarrowReq; i++) begin : gen_narrow_sel
    logic [BankIdLoBits-1:0] raw_lo, new_lo;
    assign raw_lo = slv_narrow_req_i[i].tgt_addr[0 +: BankIdLoBits];
    if (SpmBankIdRemap) begin : gen_remap_active
      // Offset comes from the row-id slice just above bank_id.
      logic [BankRemapShiftAmt-1:0] offset_lo;
      assign offset_lo = slv_narrow_req_i[i].tgt_addr[BankOffsetBits +: BankRemapShiftAmt];
      assign new_lo = raw_lo + offset_lo;
    end else begin : gen_remap_passthrough
      assign new_lo = raw_lo;
    end
    if (NumSuperbanks > 1) begin : gen_hi_passthrough
      // bank_id_hi passes through untouched
      assign narrow_sel[i] =
          {slv_narrow_req_i[i].tgt_addr[BankIdLoBits +: (BankOffsetBits - BankIdLoBits)],
           new_lo};
    end else begin : gen_no_hi
      assign narrow_sel[i] = new_lo;
    end
  end

  narrow_req_t [NumBanksPerTile-1:0] narrow_to_bank;
  logic        [NumBanksPerTile-1:0] narrow_to_bank_valid;
  logic        [NumBanksPerTile-1:0] narrow_to_bank_ready;
  resp_idx_t   [NumBanksPerTile-1:0] narrow_to_bank_idx;

  stream_xbar #(
    .NumInp   (NumNarrowReq    ),
    .NumOut   (NumBanksPerTile ),
    .payload_t(narrow_req_t    )
  ) i_narrow_req_xbar (
    .clk_i  (clk_i                  ),
    .rst_ni (rst_ni                 ),
    .flush_i(1'b0                   ),
    .rr_i   ('0                     ),
    .data_i (slv_narrow_req_i       ),
    .valid_i(slv_narrow_req_valid_i ),
    .ready_o(slv_narrow_req_ready_o ),
    .sel_i  (narrow_sel             ),
    .data_o (narrow_to_bank         ),
    .valid_o(narrow_to_bank_valid   ),
    .ready_i(narrow_to_bank_ready   ),
    .idx_o  (narrow_to_bank_idx     )
  );

  // ============================================================
  // Wide-word ingress: one DMA handler plus one handler per wide plane port.
  // The per-bank priority below is shared across all of them and narrow.
  // ============================================================
  narrow_req_t [NumBanksPerTile-1:0] dma_req_for_bank;
  logic        [NumBanksPerTile-1:0] dma_fork_valid, dma_fork_ready;
  logic        [NumBanksPerTile-1:0] dma_resp_valid, dma_resp_ready;
  narrow_req_t [NumWidePorts-1:0][NumBanksPerTile-1:0] wide_req_for_bank;
  logic [NumWidePorts-1:0][NumBanksPerTile-1:0] wide_fork_valid, wide_fork_ready;
  logic [NumWidePorts-1:0][NumBanksPerTile-1:0] wide_resp_valid, wide_resp_ready;

  // mempool_tcdm_wide_superbank is the generic per-class datapath, so ITS `wide`
  // prefix means "one wide word", not "the wide plane": the DMA drives it too.
  mempool_tcdm_wide_superbank #(
    .NumBanksPerTile(NumBanksPerTile),
    .NarrowDataWidth(NarrowDataWidth),
    .WideDataWidth  (DmaWideDataWidth),
    .SpmBankIdRemap (SpmBankIdRemap ),
    .RotFifoDepth   (RotFifoDepth   ),
    .wide_req_t     (dma_wide_req_t ),
    .wide_resp_t    (dma_wide_resp_t),
    .narrow_req_t   (narrow_req_t   ),
    .narrow_resp_t  (narrow_resp_t  ),
    .group_id_t     (group_id_t     ),
    .ini_addr_t     (tile_group_id_t)
  ) i_dma_superbank (
    .clk_i, .rst_ni,
    .group_id_i               (group_id_i           ),
    .slv_wide_req_i           (slv_dma_req_i        ),
    .slv_wide_req_valid_i     (slv_dma_req_valid_i  ),
    .slv_wide_req_ready_o     (slv_dma_req_ready_o  ),
    .slv_wide_resp_o          (slv_dma_resp_o       ),
    .slv_wide_resp_valid_o    (slv_dma_resp_valid_o ),
    .slv_wide_resp_ready_i    (slv_dma_resp_ready_i ),
    // The DMA response never leaves the tile: no source routing to carry, and the
    // DMA frontend does not need a read/write tag on its response.
    .slv_wide_ini_addr_i      ('0        ),
    .slv_wide_src_group_i     (group_id_i),
    .slv_wide_resp_ini_addr_o (/* unused */),
    .slv_wide_resp_src_group_o(/* unused */),
    .slv_wide_resp_is_write_o (/* unused */),
    .wide_req_for_bank_o(dma_req_for_bank), .wide_fork_valid_o(dma_fork_valid),
    .wide_fork_ready_i  (dma_fork_ready  ),
    .resp_i             (mst_resp_i      ), .resp_valid_i(dma_resp_valid),
    .resp_ready_o       (dma_resp_ready  )
  );

  // Per-bank class tag: routes each wide response back to the class that was
  // granted. Bank responses are in order, so a small FIFO per bank suffices.
  localparam int unsigned ClassTagWidth = idx_width(NumWidePorts + 1);
  typedef logic [ClassTagWidth-1:0] class_tag_t;
  logic       [NumBanksPerTile-1:0] ctag_push_valid, ctag_push_ready;
  logic       [NumBanksPerTile-1:0] ctag_pop_valid, ctag_pop_ready;
  class_tag_t [NumBanksPerTile-1:0] ctag_push_data, ctag_pop_data;

  `ifdef USE_WIDE_TCDM_PLANE
  // Each ingress path owns a superbank handler. Paths targeting different
  // superbanks can therefore progress in parallel; arbitration below occurs
  // only when their bank sets overlap.
  for (genvar p = 0; p < NumWidePorts; p++) begin : gen_wide_port
    wide_resp_t wide_submod_resp;
    tile_group_id_t wide_resp_ini_addr;
    group_id_t      wide_resp_src_group;
    logic           wide_resp_is_write;
    assign slv_wide_resp_o[p] = '{
      rdata:        wide_submod_resp.rdata,
      ini_addr:     wide_resp_ini_addr,
      src_group_id: wide_resp_src_group,
      is_write:     wide_resp_is_write
    };

    mempool_tcdm_wide_superbank #(
      .NumBanksPerTile(NumBanksPerTile ),
      .NarrowDataWidth(NarrowDataWidth ),
      .WideDataWidth  (WideDataWidth   ),
      .SpmBankIdRemap (SpmBankIdRemap  ),
      .RotFifoDepth   (RotFifoDepth    ),
      .wide_req_t     (wide_req_t      ),
      .wide_resp_t    (wide_resp_t     ),
      .narrow_req_t   (narrow_req_t    ),
      .narrow_resp_t  (narrow_resp_t   ),
      .group_id_t     (group_id_t      ),
      .ini_addr_t     (tile_group_id_t )
    ) i_wide_superbank (
      .clk_i,
      .rst_ni,
      .group_id_i               (group_id_i                    ),
      .slv_wide_req_i           (slv_wide_req_i[p]             ),
      .slv_wide_req_valid_i     (slv_wide_req_valid_i[p]       ),
      .slv_wide_req_ready_o     (slv_wide_req_ready_o[p]       ),
      .slv_wide_resp_o          (wide_submod_resp              ),
      .slv_wide_resp_valid_o    (slv_wide_resp_valid_o[p]      ),
      .slv_wide_resp_ready_i    (slv_wide_resp_ready_i[p]      ),
      .slv_wide_ini_addr_i      (slv_wide_req_i[p].ini_addr    ),
      .slv_wide_src_group_i     (slv_wide_req_i[p].src_group_id),
      .slv_wide_resp_ini_addr_o (wide_resp_ini_addr            ),
      .slv_wide_resp_src_group_o(wide_resp_src_group           ),
      .slv_wide_resp_is_write_o (wide_resp_is_write            ),
      .wide_req_for_bank_o      (wide_req_for_bank[p]          ),
      .wide_fork_valid_o        (wide_fork_valid[p]            ),
      .wide_fork_ready_i        (wide_fork_ready[p]            ),
      .resp_i                   (mst_resp_i                    ),
      .resp_valid_i             (wide_resp_valid[p]            ),
      .resp_ready_o             (wide_resp_ready[p]            )
    );
  end

  for (genvar b = 0; b < NumBanksPerTile; b++) begin : gen_ctag_fifo
    stream_fifo_optimal_wrap #(
      .Depth (ClassFifoDepth),
      .type_t(class_tag_t    )
    ) i_ctag_fifo (
      .clk_i (clk_i), .rst_ni(rst_ni), .flush_i(1'b0), .testmode_i(1'b0), .usage_o(),
      .data_i(ctag_push_data[b]), .valid_i(ctag_push_valid[b]), .ready_o(ctag_push_ready[b]),
      .data_o(ctag_pop_data[b]),  .valid_o(ctag_pop_valid[b]),  .ready_i(ctag_pop_ready[b])
    );
  end
  `else
  // No wide plane: tie it inert so the priority degrades to DMA > narrow, i.e.
  // exactly the single-wide-class behaviour. wide_resp_valid is NOT tied here --
  // gen_resp_split's always_comb already drives it (to 0, since ctag_pop_data is
  // tied low), and a continuous assign on top of that is a vopt-12003 error.
  assign wide_req_for_bank = '0;
  assign wide_fork_valid   = '0;
  assign wide_resp_ready   = '0;
  assign ctag_push_ready  = '1;   // never blocks a (DMA-only) wide grant
  // The DMA owns every wide response, so the tag is permanently available: this
  // feeds dma_resp_valid in gen_resp_split. Tying it low instead would stall the
  // DMA join forever and wedge the tile.
  assign ctag_pop_valid   = '1;
  assign ctag_pop_data    = '0;   // route every wide response to the DMA
  `endif

  // ============================================================
  // Wide-word arbitration: SUPERBANK-ATOMIC grant, wide plane > DMA > narrow.
  //
  // A wide class's fork is granted only when EVERY bank of its superbank can take
  // it in the SAME cycle. This is load-bearing, not an optimisation: with multiple
  // wide paths sharing banks, a partially granted fork lets the per-bank response
  // queues acquire DIFFERENT class orders (bank i holds [wide, dma] while bank j
  // holds [dma, wide]). Each class's stream_join then waits on an entry stuck
  // behind the other class's entry, and the two joins deadlock -- which is exactly
  // how rm_gemm wedged at ~cyc 5564. Granting atomically keeps every bank of a
  // superbank in lockstep, so the class order is identical at every bank and the
  // circular wait cannot form. (The original single-path design got this for free:
  // one path cannot deadlock against itself.)
  //
  // Priority is strict wide plane > DMA. Deadlock-free by the argument above; the
  // residual risk is DMA STARVATION if the plane streams without gaps. Acceptable
  // today (RedMulE and DMA phases are disjoint in the dense-LLM kernels) -- if a
  // measurement ever shows a starved DMA, add a round-robin toggle here rather
  // than reintroducing per-bank aging.
  // ============================================================
  // A bank can take a wide access iff it accepts AND its class-tag FIFO has room
  // for the response routing.
  logic [NumBanksPerTile-1:0] bank_ingress_ok;
  assign bank_ingress_ok = mst_req_ready_i & ctag_push_ready;

  logic [NumWideSB-1:0][NumWidePorts-1:0] wide_sb_req, wide_sb_gnt;
  logic [NumDmaSB-1:0] dma_sb_go;
  // Per bank: does a wide class want it (even if not yet feasible)? Used to hold
  // narrow off so a wide fork cannot be starved while it assembles.
  logic [NumBanksPerTile-1:0] ingress_wants, wide_grant_any, dma_grant;
  logic [NumWidePorts-1:0][NumBanksPerTile-1:0] wide_grant;
  narrow_req_t [NumBanksPerTile-1:0] wide_req_granted;
  class_tag_t [NumBanksPerTile-1:0] wide_grant_tag;

  for (genvar d = 0; d < NumWideSB; d++) begin : gen_wide_sb_go
    logic [NumBanksPerWideSB-1:0] ok;
    logic wide_sb_ready;
    for (genvar b = 0; b < NumBanksPerWideSB; b++) begin : gen_wide_sb_ok
      assign ok[b] = bank_ingress_ok[d*NumBanksPerWideSB + b];
    end
    assign wide_sb_ready = &ok;
    for (genvar p = 0; p < NumWidePorts; p++) begin : gen_wide_port_req
      logic [NumBanksPerWideSB-1:0] want;
      for (genvar b = 0; b < NumBanksPerWideSB; b++) begin : gen_wide_sb_want
        assign want[b] = wide_fork_valid[p][d*NumBanksPerWideSB + b];
      end
      assign wide_sb_req[d][p] = &want;
    end

    rr_arb_tree #(
      .NumIn    (NumWidePorts),
      .DataType (logic          ),
      .AxiVldRdy(1'b1           ),
      .LockIn   (1'b1           ),
      .FairArb  (1'b1           )
    ) i_wide_sb_arb (
      .clk_i  (clk_i        ),
      .rst_ni (rst_ni       ),
      .flush_i(1'b0         ),
      .rr_i   ('0           ),
      .req_i  (wide_sb_req[d]),
      .gnt_o  (wide_sb_gnt[d]),
      .data_i ('0           ),
      .req_o  (/* unused */ ),
      .gnt_i  (wide_sb_ready ),
      .data_o (/* unused */ ),
      .idx_o  (/* unused */ )
    );
  end

  for (genvar e = 0; e < NumDmaSB; e++) begin : gen_dma_sb_go
    logic [NumBanksPerDmaSB-1:0] want, ok, blocked;
    for (genvar b = 0; b < NumBanksPerDmaSB; b++) begin : gen_dma_sb_bit
      assign want[b]    = dma_fork_valid[e*NumBanksPerDmaSB + b];
      assign ok[b]      = bank_ingress_ok  [e*NumBanksPerDmaSB + b];
      // The plane outranks the DMA: block on WANT (not on grant), so the DMA can
      // never slip in between the banks of a wide fork that is still assembling.
      always_comb begin
        blocked[b] = 1'b0;
        for (int unsigned p = 0; p < NumWidePorts; p++)
          blocked[b] |= wide_fork_valid[p][e*NumBanksPerDmaSB + b];
      end
    end
    assign dma_sb_go[e] = (&want) & (&ok) & ~(|blocked);
  end

  for (genvar b = 0; b < NumBanksPerTile; b++) begin : gen_bank_mux
    assign dma_grant[b]  = dma_sb_go[b / NumBanksPerDmaSB];
    for (genvar p = 0; p < NumWidePorts; p++) begin : gen_wide_grant
      assign wide_grant[p][b] =
          wide_sb_req[b / NumBanksPerWideSB][p] &
          wide_sb_gnt[b / NumBanksPerWideSB][p];
    end

    always_comb begin
      ingress_wants[b]    = dma_fork_valid[b];
      wide_grant_any[b] = 1'b0;
      wide_req_granted[b] = '0;
      wide_grant_tag[b] = '0;
      for (int unsigned p = 0; p < NumWidePorts; p++) begin
        ingress_wants[b] |= wide_fork_valid[p][b];
        if (wide_grant[p][b]) begin
          wide_grant_any[b] = 1'b1;
          wide_req_granted[b] = wide_req_for_bank[p][b];
          wide_grant_tag[b] = class_tag_t'(p + 1);
        end
      end
    end

    always_comb begin
      mst_req_o[b]            = narrow_to_bank[b];
      mst_req_valid_o[b]      = 1'b0;
      mst_req_wide_o[b]       = 1'b0;
      mst_req_ini_addr_o[b]   = narrow_to_bank_idx[b];
      narrow_to_bank_ready[b] = 1'b0;
      if (wide_grant_any[b] | dma_grant[b]) begin
        mst_req_o[b]          = wide_grant_any[b] ? wide_req_granted[b] : dma_req_for_bank[b];
        mst_req_valid_o[b]    = 1'b1;
        mst_req_wide_o[b]     = 1'b1;
        mst_req_ini_addr_o[b] = '0;
      end else if (ingress_wants[b]) begin
        // A wide fork is assembling on this bank but is not yet feasible across
        // its whole superbank: hold the bank idle rather than letting narrow in,
        // which would break wide-over-narrow and could starve the fork.
        mst_req_valid_o[b]    = 1'b0;
      end else begin
        mst_req_valid_o[b]      = narrow_to_bank_valid[b];
        narrow_to_bank_ready[b] = mst_req_ready_i[b];
      end
    end

    // The wide arbiter and dma_sb_go include mst_req_ready_i for every bank
    // of the superbank, so the fork and bank handshake in the same cycle.
    for (genvar p = 0; p < NumWidePorts; p++) begin : gen_wide_ready
      assign wide_fork_ready[p][b] = wide_grant[p][b];
    end
    assign dma_fork_ready[b] = dma_grant[b];

    // Queue the granted class so its (in-order) response can be routed back.
    assign ctag_push_valid[b] = wide_grant_any[b] | dma_grant[b];
    assign ctag_push_data[b]  = wide_grant_any[b] ? wide_grant_tag[b] : '0;
  end

  // ============================================================
  // Per-bank response split: wide -> the granted class's join, narrow -> xbar.
  // ============================================================
  narrow_resp_t [NumBanksPerTile-1:0] narrow_resp_to_xbar;
  logic         [NumBanksPerTile-1:0] narrow_resp_to_xbar_valid;
  logic         [NumBanksPerTile-1:0] narrow_resp_to_xbar_ready;

  for (genvar b = 0; b < NumBanksPerTile; b++) begin : gen_resp_split
    assign narrow_resp_to_xbar[b] = mst_resp_i[b];
    for (genvar p = 0; p < NumWidePorts; p++) begin : gen_wide_resp_valid
      assign wide_resp_valid[p][b] = mst_resp_wide_i[b] && ctag_pop_valid[b] &&
          (ctag_pop_data[b] == class_tag_t'(p + 1)) && mst_resp_valid_i[b];
    end
    always_comb begin
      dma_resp_valid[b]            = 1'b0;
      narrow_resp_to_xbar_valid[b] = 1'b0;
      ctag_pop_ready[b]            = 1'b0;
      mst_resp_ready_o[b]          = 1'b0;
      if (mst_resp_wide_i[b]) begin
        if (ctag_pop_data[b] == '0) begin
          dma_resp_valid[b]   = ctag_pop_valid[b] & mst_resp_valid_i[b];
          mst_resp_ready_o[b] = dma_resp_ready[b];
        end else begin
          for (int unsigned p = 0; p < NumWidePorts; p++) begin
            if (ctag_pop_data[b] == class_tag_t'(p + 1)) begin
              mst_resp_ready_o[b]  = wide_resp_ready[p][b];
            end
          end
        end
        ctag_pop_ready[b] = mst_resp_valid_i[b] & mst_resp_ready_o[b];
      end else begin
        narrow_resp_to_xbar_valid[b] = mst_resp_valid_i[b];
        mst_resp_ready_o[b]          = narrow_resp_to_xbar_ready[b];
      end
    end
  end

  // ============================================================
  // Narrow response xbar (NumBanksPerTile -> NumNarrowResp).
  // sel = caller-supplied resp idx (already conditioned for any
  // NumNarrowRemoteReqPortsPerTile vs NumNarrowRemoteRespPortsPerTile asymmetry).
  // ============================================================
  stream_xbar #(
    .NumInp   (NumBanksPerTile ),
    .NumOut   (NumNarrowResp   ),
    .payload_t(narrow_resp_t   )
  ) i_narrow_resp_xbar (
    .clk_i  (clk_i                      ),
    .rst_ni (rst_ni                     ),
    .flush_i(1'b0                       ),
    .rr_i   ('0                         ),
    .data_i (narrow_resp_to_xbar        ),
    .valid_i(narrow_resp_to_xbar_valid  ),
    .ready_o(narrow_resp_to_xbar_ready  ),
    .sel_i  (mst_resp_ini_addr_i        ),
    .data_o (slv_narrow_resp_o          ),
    .valid_o(slv_narrow_resp_valid_o    ),
    .ready_i(slv_narrow_resp_ready_i    ),
    .idx_o  (/* unused */               )
  );

endmodule
