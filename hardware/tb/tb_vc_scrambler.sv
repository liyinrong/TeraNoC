// Copyright 2026 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Standalone replay of the WP3 virtual-cluster golden vectors against the
// RTL permuter. Instantiates ONLY mempool_addr_scrambler, so it runs in
// seconds and can be used long before any tile or group port changes.
//
// GATE item 1 of docs/plan/wp2-wp4-implementation-roadmap.md §T2.3: the
// bijection check is PRIMARY. Track A finding S4 showed that one of the two
// wrong readings of invariant #4 collides without ever escaping the
// rectangle's bounding box, so the confinement SVA of gate item 3 would pass
// a broken permuter. Only replaying the vectors catches it.
//
// The DUTs are instantiated with EnableTileIdRemap = 1 ON PURPOSE. That is
// the configuration in which a VC branch appended last would be swallowed by
// the tile-id branch (finding S9), so this testbench is the S9 regression.
//
// Vectors: scripts/vc/golden/vc_golden_{terapool,monsterpool8x8}_v1.txt,
// format documented in scripts/vc/README.md §"Vector file format". All
// columns are hex without a 0x prefix, and the offsets are window relative.

module vc_replay #(
  parameter string       GoldenFile        = "",
  parameter string       Label             = "",
  parameter int unsigned NumX              = 4,
  parameter int unsigned NumY              = 4,
  parameter int unsigned NumTilesPerGroup  = 16,
  parameter int unsigned NumBanksPerTile   = 16,
  parameter int unsigned BankSizeBytes     = 1024,
  parameter int unsigned SeqMemSizePerTile = 1024,
  parameter int unsigned AddrWidth         = 32
) (
  output int unsigned n_pass_o,
  output int unsigned n_fail_o,
  output int unsigned n_skip_o
);

  localparam int unsigned ByteOffset  = 2;
  localparam int unsigned NumGroups   = NumX * NumY;
  localparam int unsigned NumTiles    = NumTilesPerGroup * NumGroups;
  localparam logic [AddrWidth-1:0] TCDMSize =
      NumTiles * NumBanksPerTile * BankSizeBytes;
  localparam logic [AddrWidth-1:0] TCDMBase = '0;
  localparam int unsigned GroupIdWidth = $clog2(NumGroups);
  localparam int unsigned MeshLog2 = $clog2(NumX < NumY ? NumX : NumY);

  logic [AddrWidth-1:0]    address_i, address_o;
  logic [GroupIdWidth-1:0] group_id_i;

  mempool_addr_scrambler #(
    .AddrWidth               (AddrWidth         ),
    .ByteOffset              (ByteOffset        ),
    .NumTiles                (NumTiles          ),
    .NumTilesPerDma          (NumTilesPerGroup  ),
    .NumBanksPerTile         (NumBanksPerTile   ),
    .Bypass                  (1'b0              ),
    .SeqMemSizePerTile       (SeqMemSizePerTile ),
    .TCDMBaseAddr            (TCDMBase          ),
    .TCDMMask                (~(TCDMSize - 1)   ),
    .EnableSeqInterleaveSwap (1'b1              ),
    .EnableTileIdRemap       (1'b1              ), // S9 regression, see above
    .TCDMSize                (TCDMSize          ),
    .NumTilesPerGroup        (NumTilesPerGroup  ),
    .NumX                    (NumX              ),
    .NumY                    (NumY              ),
    .EnableVirtualCluster           (1'b1              )
  ) i_dut (
    .address_i  (address_i ),
    .group_id_i (group_id_i),
    .vc_hit_o   (/* unused */),
    .vc_level_o (/* unused */),
    .address_o  (address_o )
  );

  initial begin : p_replay
    automatic int fd;
    automatic string line;
    automatic string dir = "../../scripts/vc/golden/";
    automatic string path;
    automatic int unsigned k, gxi, gyi, cbase, crows, off, phys;
    automatic int unsigned n_pass = 0, n_fail = 0, n_skip = 0, n_line = 0;
    automatic logic [AddrWidth-1:0] base, exp;

    // Default is relative to hardware/<buildpath>/; +GOLDEN_DIR overrides.
    void'($value$plusargs("GOLDEN_DIR=%s", dir));
    path = {dir, GoldenFile};
    fd = $fopen(path, "r");
    if (fd == 0) $fatal(1, "[%s] cannot open %s", Label, path);

    while (!$feof(fd)) begin
      line = "";
      void'($fgets(line, fd));
      if (line.len() == 0) continue;
      if (line.substr(0, 0) == "#") continue;
      if ($sscanf(line, "%h %h %h %h %h %h %h",
                  k, gxi, gyi, cbase, crows, off, phys) != 7) continue;
      n_line++;

      // The TOP level has no window: its permuter is provably the identity,
      // so the global region already is that window (findings S7/S11). Those
      // vectors are covered by the DUT's own top-level identity assert; here
      // they are only audited for self-consistency and counted as skipped.
      if (k >= MeshLog2) begin
        if (phys != off)
          $fatal(1, "[%s] top-level vector is not the identity: k=%0d off=%h phys=%h",
                 Label, k, off, phys);
        n_skip++;
        continue;
      end

      // chunk_base is deleted by S12; the adopted profile must carry 0.
      if (cbase != 0)
        $fatal(1, "[%s] vector has chunk_base=%h; this file is not the 'adopted' profile",
               Label, cbase);

      base       = i_dut.vc_window_base(k);
      address_i  = base + off;
      group_id_i = GroupIdWidth'(gxi * NumY + gyi);
      exp        = TCDMBase + phys;
      #1ns;

      if (address_o !== exp) begin
        n_fail++;
        if (n_fail <= 10)
          $display("[%s] MISMATCH k=%0d g=(%0d,%0d) off=%h : got %h expected %h",
                   Label, k, gxi, gyi, off, address_o, exp);
      end else begin
        n_pass++;
      end
    end
    $fclose(fd);

    if (n_line == 0) $fatal(1, "[%s] no vectors parsed from %s", Label, path);
    $display("[%s] %0d vectors: %0d pass, %0d fail, %0d top-level skipped",
             Label, n_line, n_pass, n_fail, n_skip);
    n_pass_o = n_pass;
    n_fail_o = n_fail;
    n_skip_o = n_skip;
  end

endmodule : vc_replay


module tb_vc_scrambler;

  int unsigned tp_pass, tp_fail, tp_skip;
  int unsigned mp_pass, mp_fail, mp_skip;

  // config/terapool.mk: 4x4 groups, 16 tiles/group, 16 banks/tile, 1 KiB banks
  vc_replay #(
    .GoldenFile        ("vc_golden_terapool_v1.txt"),
    .Label             ("terapool"),
    .NumX              (4 ),
    .NumY              (4 ),
    .NumTilesPerGroup  (16),
    .NumBanksPerTile   (16),
    .BankSizeBytes     (1024)
  ) i_terapool (
    .n_pass_o (tp_pass), .n_fail_o (tp_fail), .n_skip_o (tp_skip)
  );

  // The former dse_s1_8x8_winner: 8x8 groups, 4 tiles/group, 64 banks/tile,
  // 4 KiB banks (Track A finding S2 — NOT the 32 banks/tile of the brief).
  vc_replay #(
    .GoldenFile        ("vc_golden_monsterpool8x8_v1.txt"),
    .Label             ("monsterpool8x8"),
    .NumX              (8 ),
    .NumY              (8 ),
    .NumTilesPerGroup  (4 ),
    .NumBanksPerTile   (64),
    .BankSizeBytes     (4096)
  ) i_monsterpool8x8 (
    .n_pass_o (mp_pass), .n_fail_o (mp_fail), .n_skip_o (mp_skip)
  );

  initial begin : p_summary
    #1ms;
    $display("=== VC scrambler golden replay ===");
    $display("  terapool       : %0d pass, %0d fail, %0d skipped",
             tp_pass, tp_fail, tp_skip);
    $display("  monsterpool8x8 : %0d pass, %0d fail, %0d skipped",
             mp_pass, mp_fail, mp_skip);
    if (tp_fail != 0 || mp_fail != 0)
      $fatal(1, "VC REPLAY FAILED: %0d mismatches", tp_fail + mp_fail);
    if (tp_pass == 0 || mp_pass == 0)
      $fatal(1, "VC REPLAY VACUOUS: a geometry contributed no passing vectors");
    $display("VC REPLAY PASSED");
    $finish(0);
  end

endmodule : tb_vc_scrambler
