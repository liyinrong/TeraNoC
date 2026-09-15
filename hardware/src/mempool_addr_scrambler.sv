// Copyright 2021 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// SPM address scrambler. Single combinational module applying up to three
// independently-gated address transforms, in this order:
//   Stage 1 (EnableSeqInterleaveSwap, sequential region): swap the
//     {tile_id, scramble} fields so HW sees an interleaved-bank layout.
//   Stage 2 (EnableVirtualCluster, virtual-cluster windows): the level rule —
//     re-place a window-relative offset inside the requester's 2^k x 2^k
//     group rectangle.
//   Stage 3 (EnableTileIdRemap, non-sequential TCDM): spread accesses
//     across the NumTilesPerDma tiles of a DMA group.
//
// BRANCH ORDER IS LOAD-BEARING. The VC
// branch must sit between stage 1 and stage 3. Appended after stage 3 it
// would never fire on a config with EnableTileIdRemap=1, because a VC
// address is inside the decoded TCDM region and above the sequential region,
// so the stage-3 condition swallows it first — silently, and only on those
// configs.
//
// Used by snitch cores pre-tcdm_shim (stages 1+3, and stage 2 at the three
// COMPUTE sites) and by the DMA tile-id path in mempool_group.sv (stage 1
// off, stage 3 on, stage 2 deliberately OFF so the DMA is virtual-cluster
// blind by construction; caller then slices tile_id_remap out of address_o
// for the reqrsp_demux select).
// Replaces address_scrambler.sv and mempool_dma_tile_id_remapper.sv.
//
// NOTE: bank-id remap is intentionally NOT here — it lives in the tile's
// bank-side network (mempool_bank_id_remapper).

module mempool_addr_scrambler
#(
  parameter int unsigned          AddrWidth                = 32,
  parameter int unsigned          ByteOffset               = 2,
  parameter int unsigned          NumTiles                 = 2,
  parameter int unsigned          NumTilesPerDma           = 16,
  parameter int unsigned          NumBanksPerTile          = 2,
  parameter bit                   Bypass                   = 1'b0,
  parameter int unsigned          SeqMemSizePerTile        = 4*1024,
  parameter logic [AddrWidth-1:0] TCDMBaseAddr             = 32'b0,
  parameter logic [31:0]          TCDMMask                 = '1 << 28,
  parameter bit                   EnableSeqInterleaveSwap  = 1'b1,
  parameter bit                   EnableTileIdRemap        = 1'b0,
  // --- Virtual-cluster remap ------------------------------------------------
  // Total decoded global TCDM region. Replaces TCDMMask as the region test:
  // the global window plus the per-level VC windows is 1.33x L1 and never a
  // power of two, so one mask cannot express it.
  parameter logic [AddrWidth-1:0] TCDMSize                 = 0,
  // Geometry. NumTilesPerGroup, NumX and NumY are NEW and must NOT be
  // replaced by TileIdBits, which spans the FLAT {group,tile} field that
  // stage 1 moves (Track A finding S10).
  parameter int unsigned          NumTilesPerGroup         = 1,
  parameter int unsigned          NumX                     = 1,
  parameter int unsigned          NumY                     = 1,
  parameter bit                   EnableVirtualCluster            = 1'b0,
  // ONLY the standalone replay testbench may set this. The S9 branch-ordering
  // regression has to instantiate the VC branch together with the tile-id
  // remap -- that is the one configuration in which a VC branch appended last
  // would be swallowed. In a SYSTEM the pair is unsafe (see the fatal below):
  // a bare permuter has no global twin for the VC address to disagree with.
  parameter bit                   VcTileRemapTbOnly               = 1'b0,
  // End of the whole VC region as computed by config/config.mk. The module
  // derives the window map independently below; a mismatch means the Make
  // arithmetic and the RTL arithmetic have drifted, which would put software's
  // pointers in a different window from the one hardware decodes. 0 disables
  // the check, which is what the standalone testbench uses.
  parameter logic [AddrWidth-1:0] VcRegionEndChk           = '0,
  // Derived; do not override.
  parameter int unsigned          GroupIdWidth             =
      (NumX*NumY) > 1 ? $clog2(NumX*NumY) : 1
) (
  input  logic [AddrWidth-1:0]    address_i,
  // Requesting group's flat mesh id, {gx, gy} with gy in the LOW half
  // (hardware/src/mempool_cluster_floonoc_wrapper.sv). Unused unless
  // EnableVirtualCluster.
  input  logic [GroupIdWidth-1:0] group_id_i,
  // Asserted when this request was remapped by the VC branch, with the level
  // it matched. Carried to the request routers so the confinement assertion
  // can tell VC traffic from ordinary global traffic
  // AND know which rectangle to check against. Constant 0 when the VC branch
  // is disabled. Verification-only; nothing in the datapath reads them.
  output logic                    vc_hit_o,
  output logic [2:0]              vc_level_o,
  output logic [AddrWidth-1:0]    address_o
);

  `define max(a,b) (((a) > (b))? (a) : (b))

  localparam int unsigned BankOffsetBits    = $clog2(NumBanksPerTile);
  localparam int unsigned TileIdBits        = $clog2(NumTiles);
  localparam int unsigned TileIdBitsPerDma  = `max(1, $clog2(NumTilesPerDma));
  localparam int unsigned SeqPerTileBits    = $clog2(SeqMemSizePerTile);
  localparam int unsigned SeqTotalBits      = SeqPerTileBits + TileIdBits;
  localparam int unsigned ConstantBitsLSB   = ByteOffset + BankOffsetBits;
  localparam int unsigned ScrambleBits      = SeqPerTileBits - ConstantBitsLSB;

  // ---- Virtual-cluster geometry (all compile-time constants) ------------
  // Canonical physical layout, low to high:
  //   [ byte | bank | tile | gy | gx | word_in_bank ]
  localparam int unsigned VcTileBits = $clog2(NumTilesPerGroup);
  localparam int unsigned VcGyBits   = $clog2(NumY);
  localparam int unsigned VcGxBits   = $clog2(NumX);
  localparam int unsigned VcBankPos  = ByteOffset;
  localparam int unsigned VcTilePos  = VcBankPos + BankOffsetBits;
  localparam int unsigned VcGyPos    = VcTilePos + VcTileBits;
  localparam int unsigned VcGxPos    = VcGyPos + VcGyBits;
  localparam int unsigned VcWordPos  = VcGxPos + VcGxBits;
  localparam int unsigned VcAddrBits = TCDMSize > 0 ? $clog2(TCDMSize) : 1;
  localparam int unsigned VcWordBits = VcAddrBits > VcWordPos
                                     ? VcAddrBits - VcWordPos : 1;

  // Number of VC windows. Levels run k = 0 .. VcLevels-1; the TOP level
  // k = log2(min(NumX,NumY)) is deliberately absent because its permuter is
  // provably the identity, so the global region already is that window
  // (Track A finding S11).
  localparam int unsigned VcMeshLog2 =
      $clog2(NumX < NumY ? NumX : NumY);
  localparam int unsigned VcLevels   = EnableVirtualCluster ? VcMeshLog2 : 0;
  // Per-group L1 in bytes, and a level's aperture 4^k x that.
  localparam longint unsigned VcGroupBytes =
      (NumX*NumY) > 0 ? TCDMSize / (NumX*NumY) : 0;

  function automatic longint unsigned vc_window_size (input int unsigned k);
    vc_window_size = VcGroupBytes << (2*k);
  endfunction

  // Packed DESCENDING by size immediately above the global region, so every
  // window lands on its natural alignment with zero padding.
  function automatic longint unsigned vc_window_base (input int unsigned k);
    automatic longint unsigned acc = TCDMBaseAddr + TCDMSize;
    for (int unsigned j = VcLevels; j > k + 1; j--) begin
      acc += vc_window_size(j - 1);
    end
    vc_window_base = acc;
  endfunction

  // The level rule. `offset` is window-relative.
  function automatic logic [AddrWidth-1:0] vc_permute (
      input int unsigned          k,
      input logic [AddrWidth-1:0] offset,
      input logic [GroupIdWidth-1:0] group_id
  );
    // Morton (Z-order) group ids: the rectangle is a CONTIGUOUS id range, so
    // the per-axis base mask of INVARIANT #4 collapses into one flat mask and
    // the two k-bit vc_gy/vc_gx fields of the offset merge into one 2k-bit
    // group field. The strip-instead-of-square failure mode that invariant
    // guarded against is not expressible here: there is no per-axis step to
    // get wrong. mempool_pkg::MortonGroupId is the encoding, gid_of_xy and
    // group_xy_of are the two directions.
    automatic logic [AddrWidth-1:0] ngroups = 1 << (2*k);
    automatic logic [AddrWidth-1:0] gmask   = ngroups - 1;
    automatic int unsigned o_vcg = VcGyPos;   // the group field's base
    automatic int unsigned o_row = VcGyPos + 2*k;
    automatic logic [AddrWidth-1:0] vc_g, row, out_gid;
    vc_g    = (offset >> o_vcg) & gmask;
    row     = (offset >> o_row) & ((1 << VcWordBits) - 1);
    out_gid = (group_id & ~gmask) | vc_g;
    // byte|bank|tile occupy the same bits in the offset and in the physical
    // address, so they pass straight through.
    vc_permute = TCDMBaseAddr
               | (offset & ((1 << VcGyPos) - 1))
               | (out_gid << VcGyPos)
               | (row     << VcWordPos);
  endfunction

  // ---- Region decode ----------------------------------------------------
  // K+1 constant range compares, OR'd. This replaces the single power-of-two
  // TCDMMask test: global + per-level windows is 1.33x L1, which is never a
  // power of two, so one mask would round the region up to 2x L1 — at 32x32
  // exactly onto L2 (Track A finding S11).
  // The global region alone IS a power of two, so the existing mask expresses
  // it exactly; what S11 rules out is covering global+VC with a single mask.
  // Keeping the mask here also means an instantiation that does not set
  // TCDMSize (the DMA site) keeps its region decode unchanged.
  logic global_hit;
  assign global_hit = (address_i & TCDMMask) == TCDMBaseAddr;

  localparam int unsigned VcHitMsb = VcLevels > 0 ? VcLevels-1 : 0;
  logic [VcHitMsb:0]              vc_hit;
  logic                           vc_any;
  logic [AddrWidth-1:0]           vc_addr;

  if (VcLevels == 0) begin : gen_no_vc
    assign vc_hit  = '0;
    assign vc_any  = 1'b0;
    assign vc_addr = address_i;
  end else begin : gen_vc
    logic [AddrWidth-1:0] vc_addr_lvl [VcLevels];
    for (genvar k = 0; k < VcLevels; k++) begin : gen_vc_level
      localparam logic [AddrWidth-1:0] Base = vc_window_base(k);
      localparam logic [AddrWidth-1:0] Size = vc_window_size(k);
      assign vc_hit[k] = (address_i >= Base) && (address_i < Base + Size);
      assign vc_addr_lvl[k] = vc_permute(k, address_i - Base, group_id_i);
    end
    assign vc_any = |vc_hit;
    always_comb begin
      vc_addr = address_i;
      for (int unsigned k = 0; k < VcLevels; k++) begin
        if (vc_hit[k]) vc_addr = vc_addr_lvl[k];
      end
    end
  end

  logic not_io_address;
  assign not_io_address = global_hit || vc_any;

  assign vc_hit_o = EnableVirtualCluster && vc_any;
  always_comb begin
    vc_level_o = '0;
    for (int unsigned k = 0; k < VcLevels; k++) begin
      if (vc_hit[k]) vc_level_o = k[2:0];
    end
  end

  function automatic logic [TileIdBitsPerDma-1:0] spm_tile_id_remap (
      logic [TileIdBitsPerDma-1:0] data_in,
      logic [TileIdBitsPerDma-1:0] idx_i
  );
    if (NumX != NumY)
      $fatal(1, "virtual_cluster needs a square mesh (NumX=%0d, NumY=%0d): the top level is only the identity, and the Morton group id is only defined, when the axes match.",
             NumX, NumY);
    if (EnableTileIdRemap) begin
      spm_tile_id_remap = data_in + idx_i;
    end else begin
      spm_tile_id_remap = data_in;
    end
  endfunction

  if (Bypass || NumTiles < 2) begin : gen_bypass
    assign address_o = address_i;
  end else begin : gen_active
    logic [ScrambleBits-1:0] scramble;
    logic [TileIdBits-1:0]   tile_id;

    assign scramble = address_i[SeqPerTileBits-1:ConstantBitsLSB];
    assign tile_id  = address_i[SeqTotalBits-1:SeqPerTileBits];

    always_comb begin
      address_o = address_i;

      // Stage 1: sequential→interleaved field swap (sequential region only).
      if (EnableSeqInterleaveSwap && (address_i < (NumTiles * SeqMemSizePerTile))) begin
        address_o[SeqTotalBits-1:ConstantBitsLSB] = {scramble, tile_id};
      end
      // Stage 2: virtual-cluster level rule. MUST precede the tile-id branch
      // (finding S9) — a VC address is not_io_address and above the
      // sequential region, so stage 3 would otherwise swallow it.
      else if (EnableVirtualCluster && vc_any) begin
        address_o = vc_addr;
      end
      // Stage 3: tile-id remap, interleaved region only — never rewrite the
      // sequential region's offset bits (matches the old dma_tile_id_remapper).
      else if (EnableTileIdRemap && not_io_address &&
               (address_i >= (NumTiles * SeqMemSizePerTile))) begin
        address_o[ConstantBitsLSB +: TileIdBitsPerDma] =
          spm_tile_id_remap(
            address_i[ConstantBitsLSB +: TileIdBitsPerDma],
            address_i[(ConstantBitsLSB + TileIdBits) +: TileIdBitsPerDma]
          );
      end
    end
  end : gen_active

  // Check for unsupported configurations
  if (NumBanksPerTile < 2)
    $fatal(1, "NumBanksPerTile must be greater than 2. The special case '1' is currently not supported!");
  if (SeqMemSizePerTile % (2**ByteOffset*NumBanksPerTile) != 0)
    $fatal(1, "SeqMemSizePerTile must be a multiple of BankWidth*NumBanksPerTile!");

  // ---- Virtual-cluster checks -------------------------------------------
  if (EnableVirtualCluster) begin : gen_vc_checks
    // The range compares must decode exactly the region the old mask did,
    // so switching to them cannot move an existing config's global window.
    if (TCDMSize == 0)
      $fatal(1, "EnableVirtualCluster needs TCDMSize; the region decode is a range compare, not a mask.");
    if (TCDMMask !== ~(TCDMSize - 1))
      $fatal(1, "TCDMMask (%h) and TCDMSize (%h) disagree; the VC region decode would move the global window.",
             TCDMMask, TCDMSize);
    if (TCDMBaseAddr % TCDMSize != 0)
      $fatal(1, "TCDMBaseAddr must be TCDMSize aligned for the composition ORs to equal addition.");
    // Level <= mesh: the only illegal descriptor left, and it is caught at
    // elaboration.
    if (VcLevels > VcMeshLog2)
      $fatal(1, "VC level %0d exceeds the mesh (min(NumX,NumY)=%0d, log2=%0d).",
             VcLevels, (NumX < NumY ? NumX : NumY), VcMeshLog2);
    // The VC branch and the tile-id remap are arms of one if/else chain, so a
    // VC access correctly bypasses the remapper -- but the flat global alias of
    // the SAME words, which the DMA and every global pointer use, is remapped.
    // The two then land on different tiles and alias equivalence breaks.
    // Falling through into the remap is NOT the fix: that stage reads
    // address_i, the pre-permute window address, so it would remap the wrong
    // value. Forbid the pair instead.
    if (EnableTileIdRemap && !VcTileRemapTbOnly)
      $fatal(1, "virtual_cluster and tile_id_remap cannot both be enabled: the VC branch bypasses the tile-id remap that the same words receive through their global alias.");
    if (NumTilesPerGroup * NumX * NumY != NumTiles)
      $fatal(1, "NumTilesPerGroup*NumX*NumY (%0d) != NumTiles (%0d); the VC geometry disagrees with the cluster.",
             NumTilesPerGroup*NumX*NumY, NumTiles);
    if (VcGroupBytes << (2*VcMeshLog2) != TCDMSize)
      $fatal(1, "Per-group L1 x 4^log2(mesh) (%0d) != TCDMSize (%0d).",
             VcGroupBytes << (2*VcMeshLog2), TCDMSize);
    // config/config.mk vs this module: one number pins the whole map, because
    // the region end is the sum of the global region and every window.
    if (VcRegionEndChk != 0 &&
        VcRegionEndChk != vc_window_base(0) + vc_window_size(0))
      $fatal(1, "VC map drift: config.mk says the region ends at %h, the RTL derives %h.",
             VcRegionEndChk, vc_window_base(0) + vc_window_size(0));
  end

`ifndef SYNTHESIS
  // The TOP level is the identity map, which is why the address map has no
  // top-level window.
  // Checked at time 0 over every field corner rather than a token sample.
  if (EnableVirtualCluster && (NumX == NumY)) begin : gen_vc_identity_check
    initial begin : p_top_level_identity
      automatic int unsigned kt = VcMeshLog2;
      automatic logic [AddrWidth-1:0] off, got;
      for (int unsigned g = 0; g < NumX*NumY; g++) begin
        for (int unsigned f = 0; f < 6; f++) begin
          // one probe per canonical field, at its lowest and highest value
          for (int unsigned hi = 0; hi < 2; hi++) begin
            case (f)
              0: off = hi ? ((1 << ByteOffset)      - 1) << 0         : 0;
              1: off = hi ? ((1 << BankOffsetBits)  - 1) << VcBankPos : 0;
              2: off = hi ? ((1 << VcTileBits)      - 1) << VcTilePos : 0;
              3: off = hi ? ((1 << kt)              - 1) << VcGyPos   : 0;
              4: off = hi ? ((1 << kt)              - 1) << (VcGyPos + kt) : 0;
              5: off = hi ? ((1 << VcWordBits)      - 1) << VcWordPos : 0;
              default: off = 0;
            endcase
            got = vc_permute(kt, off, GroupIdWidth'(g));
            if (got !== (TCDMBaseAddr | off))
              $fatal(1, "Top-level VC permuter is not the identity: k=%0d group=%0d offset=%h -> %h, expected %h",
                     kt, g, off, got, TCDMBaseAddr | off);
          end
        end
      end
    end
  end
`endif

endmodule : mempool_addr_scrambler
