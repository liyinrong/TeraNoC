// Copyright 2021 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

package mempool_pkg;

  import cf_math_pkg::idx_width;

  /*********************
   *  TILE PARAMETERS  *
   *********************/

  `include "axi/assign.svh"
  `include "axi/typedef.svh"

  localparam integer unsigned NumCores                = `ifdef NUM_CORES `NUM_CORES `else 0 `endif;
  localparam integer unsigned NumCoresPerTile         = `ifdef NUM_CORES_PER_TILE `NUM_CORES_PER_TILE `else 0 `endif;
  localparam integer unsigned NumDivsqrtPerTile       = `ifdef NUM_DIVSQRT_PER_TILE `NUM_DIVSQRT_PER_TILE `else (snitch_pkg::XDIVSQRT) `endif;
  localparam integer unsigned NumGroups               = `ifdef NUM_GROUPS `NUM_GROUPS `else 0 `endif;
  localparam integer unsigned NumRMTiles              = `ifdef NUM_REDMULE_TILES `NUM_REDMULE_TILES `else 0 `endif;
  localparam integer unsigned NumRMTilesPerGroup      = NumGroups == 0 ? 0 : NumRMTiles / NumGroups;
  localparam integer unsigned MAX_NumGroups           = 64;
  localparam integer unsigned RemoteGroupLatencyCycle = `ifdef REMOTE_GROUP_LATENCY_CYCLES `REMOTE_GROUP_LATENCY_CYCLES `else 7 `endif;
  localparam integer unsigned NumTiles                = NumCores / NumCoresPerTile;
  localparam integer unsigned NumTilesPerGroup        = NumTiles / NumGroups;
  localparam integer unsigned NumCoresPerGroup        = NumCores / NumGroups;
  localparam integer unsigned NumCoresPerCache        = NumCoresPerTile;
  localparam integer unsigned AxiCoreIdWidth          = 1;
  localparam integer unsigned AxiTileIdWidth          = AxiCoreIdWidth+1; // + 1 for cache
  localparam integer unsigned AxiDataWidth            = `ifdef AXI_DATA_WIDTH `AXI_DATA_WIDTH `else 0 `endif;
  localparam integer unsigned AxiLiteDataWidth        = 32;

  /************************
   *  EXTENSION SUPPORT   *
   ************************/
  // Spatz (RVV) vector + FP-format support. All default to 0/1 for non-Spatz
  // builds so the derived sizes below degrade to the scalar case
  // (NumFUsPerCore = 1, NumDataPortsPerCore = 1).
  localparam bit RVV      = `ifdef RVV `RVV `else 0 `endif;
  localparam bit RVF      = `ifdef RVF `RVF `else 0 `endif;
  localparam bit RVD      = 0;  // not supported for the MemPool family
  localparam bit XFVEC    = 0;
  localparam bit XFDOTP   = RVF ? 1 : 0;
  localparam bit XFAUX    = 0;
  localparam bit XF16     = RVF ? 1 : 0;
  localparam bit XF16ALT  = 0;
  localparam bit XF8      = RVF ? 1 : 0;
  localparam bit XF8ALT   = 0;
  localparam bit XDivSqrt = 0;

  localparam integer unsigned NumIPUsPerCore      = `ifdef N_IPU `N_IPU `else 1 `endif;
  localparam integer unsigned NumFPUsPerCore      = `ifdef N_FPU `N_FPU `else 1 `endif;
  localparam integer unsigned NumFUsPerCore       = NumIPUsPerCore > NumFPUsPerCore ? NumIPUsPerCore : NumFPUsPerCore;
  localparam integer unsigned NumFUsPerTile       = NumFUsPerCore * NumCoresPerTile;
  localparam integer unsigned NumMemPortsPerSpatz = NumFUsPerCore;
  // Per-core TCDM master ports: 1 scalar/FP port + RVV vector-LSU ports.
  localparam integer unsigned NumDataPortsPerCore = 1 + RVV * NumMemPortsPerSpatz;

  /***********************
   *  MEMORY PARAMETERS  *
   ***********************/

  localparam integer unsigned AddrWidth        = 32;
  localparam integer unsigned DataWidth        = 32;
  localparam integer unsigned BeWidth          = DataWidth / 8;
  localparam integer unsigned ByteOffset       = $clog2(BeWidth);
  localparam integer unsigned BankingFactor    = `ifdef BANKING_FACTOR `BANKING_FACTOR `else 0 `endif;
  localparam bit              LrScEnable       = 1'b1;
  localparam integer unsigned TCDMSizePerBank  = `ifdef L1_BANK_SIZE `L1_BANK_SIZE `else 0 `endif;
  localparam integer unsigned NumBanks         = NumCores * NumFUsPerCore * BankingFactor;
  localparam integer unsigned NumBanksPerTile  = NumBanks / NumTiles;
  localparam integer unsigned NumBanksPerGroup = NumBanks / NumGroups;
  localparam integer unsigned TCDMAddrMemWidth = $clog2(TCDMSizePerBank / mempool_pkg::BeWidth);
  localparam integer unsigned TCDMAddrWidth    = TCDMAddrMemWidth + idx_width(NumBanksPerGroup);

  // L2
  localparam integer unsigned L2Size           = `ifdef L2_SIZE `L2_SIZE `else 0 `endif; // [B]
  localparam integer unsigned NumL2Banks       = `ifdef L2_BANKS `L2_BANKS `else 1 `endif;
  localparam integer unsigned L2BankSize       = L2Size / NumL2Banks;
  localparam integer unsigned L2BankWidth      = AxiDataWidth;
  localparam integer unsigned L2BankBeWidth    = L2BankWidth/8;
  localparam integer unsigned L2BankNumWords   = L2BankSize / L2BankBeWidth;
  localparam integer unsigned L2BankAddrWidth  = $clog2(L2BankNumWords);
  localparam integer unsigned L2Width          = L2BankWidth * NumL2Banks;
  localparam integer unsigned L2BeWidth        = L2Width/8;
  localparam integer unsigned L2ByteOffset     = $clog2(L2BeWidth);
  localparam integer unsigned L2AddrWidth      = $clog2(L2Size);
  localparam integer unsigned Interleave       = `ifdef AXI_WIDTH_INTERLEAVED `AXI_WIDTH_INTERLEAVED `else 16 `endif;
  localparam integer unsigned L2BankByteOffset = $clog2(L2BankBeWidth);

  typedef logic [AxiCoreIdWidth-1:0    ] axi_core_id_t;
  typedef logic [AxiTileIdWidth-1:0    ] axi_tile_id_t;
  typedef logic [AxiDataWidth-1:0      ] axi_data_t;
  typedef logic [AxiDataWidth/8-1:0    ] axi_strb_t;
  typedef logic [AxiLiteDataWidth-1:0  ] axi_lite_data_t;
  typedef logic [AxiLiteDataWidth/8-1:0] axi_lite_strb_t;
  typedef logic [AddrWidth-1:0         ] addr_t;
  typedef logic [DataWidth-1:0         ] data_t;
  typedef logic [BeWidth-1:0           ] strb_t;

  `ifdef DRAM
    localparam integer unsigned NumDrams         = `ifdef L2_BANKS `L2_BANKS `else 1 `endif;
    localparam integer unsigned L2DramWidth      = AxiDataWidth;
    localparam integer unsigned L2DramBeWidth    = L2DramWidth/8;
    localparam integer unsigned L2DramByteOffset = $clog2(L2DramBeWidth);

    // DRAM Interleaving Functions
    typedef struct packed {
      int                   dram_ctrl_id;
      logic [AddrWidth-1:0] dram_ctrl_addr;
    } dram_ctrl_interleave_t;

    function automatic dram_ctrl_interleave_t getDramCTRLInfo(addr_t addr);
      automatic dram_ctrl_interleave_t res;
      localparam int unsigned ConstantBits = $clog2(L2DramBeWidth * Interleave);
      localparam int unsigned ScrambleBits = (NumDrams == 1) ? 1 : $clog2(NumDrams);
      localparam int unsigned ReminderBits = AddrWidth - ScrambleBits - ConstantBits;
      automatic addr_t reminder_addr = addr[AddrWidth-1: AddrWidth-ReminderBits];

      res.dram_ctrl_id = addr[ScrambleBits + ConstantBits - 1: ConstantBits];
      res.dram_ctrl_addr = {reminder_addr, addr[ConstantBits-1:0]};
      return res;
    endfunction
  `else
    // SRAM Interleaving Functions
    typedef struct packed {
      int                   sram_ctrl_id;
      logic [AddrWidth-1:0] sram_ctrl_addr;
    } sram_ctrl_interleave_t;

    function automatic sram_ctrl_interleave_t getSramCTRLInfo(addr_t addr);
      automatic sram_ctrl_interleave_t res;
      localparam int unsigned ConstantBits = $clog2(L2BankBeWidth * Interleave);
      localparam int unsigned ScrambleBits = (NumL2Banks == 1) ? 1 : $clog2(NumL2Banks);
      localparam int unsigned ReminderBits = AddrWidth - ScrambleBits - ConstantBits;
      automatic addr_t reminder_addr = addr[AddrWidth-1: AddrWidth-ReminderBits];

      res.sram_ctrl_id = addr[ScrambleBits + ConstantBits - 1: ConstantBits];
      if (Interleave == 1) begin
        res.sram_ctrl_addr = addr[AddrWidth-1: L2BankByteOffset + ScrambleBits];
      end else begin
        res.sram_ctrl_addr = { addr[AddrWidth-1: AddrWidth-ReminderBits], addr[ConstantBits-1: L2BankByteOffset] };
      end
      return res;
    endfunction
  `endif

  localparam NumSystemXbarMasters = 1;
  localparam AxiSystemIdWidth = $clog2(NumSystemXbarMasters) + AxiTileIdWidth;
  typedef logic [AxiSystemIdWidth-1:0] axi_system_id_t;

  localparam AxiInterIdWidth = 1 + AxiTileIdWidth;
  typedef logic [AxiInterIdWidth-1:0] axi_inter_id_t;

  localparam NumTestbenchXbarMasters = 1;
  localparam AxiTestbenchIdWidth = $clog2(NumTestbenchXbarMasters) + AxiSystemIdWidth;
  typedef logic [AxiTestbenchIdWidth-1:0] axi_tb_id_t;

  `AXI_TYPEDEF_AW_CHAN_T(axi_core_aw_t, addr_t, axi_core_id_t, logic);
  `AXI_TYPEDEF_W_CHAN_T(axi_core_w_t, axi_data_t, axi_strb_t, logic);
  `AXI_TYPEDEF_B_CHAN_T(axi_core_b_t, axi_core_id_t, logic);
  `AXI_TYPEDEF_AR_CHAN_T(axi_core_ar_t, addr_t, axi_core_id_t, logic);
  `AXI_TYPEDEF_R_CHAN_T(axi_core_r_t, axi_data_t, axi_core_id_t, logic);
  `AXI_TYPEDEF_REQ_T(axi_core_req_t, axi_core_aw_t, axi_core_w_t, axi_core_ar_t);
  `AXI_TYPEDEF_RESP_T(axi_core_resp_t, axi_core_b_t, axi_core_r_t );

  `AXI_TYPEDEF_AW_CHAN_T(axi_tile_aw_t, addr_t, axi_tile_id_t, logic);
  `AXI_TYPEDEF_W_CHAN_T(axi_tile_w_t, axi_data_t, axi_strb_t, logic);
  `AXI_TYPEDEF_B_CHAN_T(axi_tile_b_t, axi_tile_id_t, logic);
  `AXI_TYPEDEF_AR_CHAN_T(axi_tile_ar_t, addr_t, axi_tile_id_t, logic);
  `AXI_TYPEDEF_R_CHAN_T(axi_tile_r_t, axi_data_t, axi_tile_id_t, logic);
  `AXI_TYPEDEF_REQ_T(axi_tile_req_t, axi_tile_aw_t, axi_tile_w_t, axi_tile_ar_t);
  `AXI_TYPEDEF_RESP_T(axi_tile_resp_t, axi_tile_b_t, axi_tile_r_t );

  `AXI_TYPEDEF_AW_CHAN_T(axi_inter_aw_t, addr_t, axi_inter_id_t, logic);
  `AXI_TYPEDEF_W_CHAN_T(axi_inter_w_t, axi_data_t, axi_strb_t, logic);
  `AXI_TYPEDEF_B_CHAN_T(axi_inter_b_t, axi_inter_id_t, logic);
  `AXI_TYPEDEF_AR_CHAN_T(axi_inter_ar_t, addr_t, axi_inter_id_t, logic);
  `AXI_TYPEDEF_R_CHAN_T(axi_inter_r_t, axi_data_t, axi_inter_id_t, logic);
  `AXI_TYPEDEF_REQ_T(axi_inter_req_t, axi_inter_aw_t, axi_inter_w_t, axi_inter_ar_t);
  `AXI_TYPEDEF_RESP_T(axi_inter_resp_t, axi_inter_b_t, axi_inter_r_t);

  `AXI_TYPEDEF_AW_CHAN_T(axi_system_aw_t, addr_t, axi_system_id_t, logic);
  `AXI_TYPEDEF_W_CHAN_T(axi_system_w_t, axi_data_t, axi_strb_t, logic);
  `AXI_TYPEDEF_B_CHAN_T(axi_system_b_t, axi_system_id_t, logic);
  `AXI_TYPEDEF_AR_CHAN_T(axi_system_ar_t, addr_t, axi_system_id_t, logic);
  `AXI_TYPEDEF_R_CHAN_T(axi_system_r_t, axi_data_t, axi_system_id_t, logic);
  `AXI_TYPEDEF_REQ_T(axi_system_req_t, axi_system_aw_t, axi_system_w_t, axi_system_ar_t);
  `AXI_TYPEDEF_RESP_T(axi_system_resp_t, axi_system_b_t, axi_system_r_t);

  // AXI to periph
  `AXI_TYPEDEF_W_CHAN_T(axi_periph_w_t, axi_lite_data_t, axi_lite_strb_t, logic);
  `AXI_TYPEDEF_R_CHAN_T(axi_periph_r_t, axi_lite_data_t, axi_system_id_t, logic);
  `AXI_TYPEDEF_REQ_T(axi_periph_req_t, axi_system_aw_t, axi_periph_w_t, axi_system_ar_t);
  `AXI_TYPEDEF_RESP_T(axi_periph_resp_t, axi_system_b_t, axi_periph_r_t);

  `AXI_TYPEDEF_AW_CHAN_T(axi_tb_aw_t, addr_t, axi_tb_id_t, logic);
  `AXI_TYPEDEF_W_CHAN_T(axi_tb_w_t, axi_data_t, axi_strb_t, logic);
  `AXI_TYPEDEF_B_CHAN_T(axi_tb_b_t, axi_tb_id_t, logic);
  `AXI_TYPEDEF_AR_CHAN_T(axi_tb_ar_t, addr_t, axi_tb_id_t, logic);
  `AXI_TYPEDEF_R_CHAN_T(axi_tb_r_t, axi_data_t, axi_tb_id_t, logic);
  `AXI_TYPEDEF_REQ_T(axi_tb_req_t, axi_tb_aw_t, axi_tb_w_t, axi_tb_ar_t);
  `AXI_TYPEDEF_RESP_T(axi_tb_resp_t, axi_tb_b_t, axi_tb_r_t);

  `AXI_LITE_TYPEDEF_AW_CHAN_T(axi_lite_slv_aw_t, addr_t)
  `AXI_LITE_TYPEDEF_W_CHAN_T(axi_lite_slv_w_t, axi_lite_data_t, axi_lite_strb_t)
  `AXI_LITE_TYPEDEF_B_CHAN_T(axi_lite_slv_b_t)
  `AXI_LITE_TYPEDEF_AR_CHAN_T(axi_lite_slv_ar_t, addr_t)
  `AXI_LITE_TYPEDEF_R_CHAN_T(axi_lite_slv_r_t, axi_lite_data_t)
  `AXI_LITE_TYPEDEF_REQ_T(axi_lite_slv_req_t, axi_lite_slv_aw_t, axi_lite_slv_w_t, axi_lite_slv_ar_t)
  `AXI_LITE_TYPEDEF_RESP_T(axi_lite_slv_resp_t, axi_lite_slv_b_t, axi_lite_slv_r_t)

  /***********************
   *  INSTRUCTION CACHE  *
   ***********************/

  localparam int unsigned ICacheSizeByte  = 512 * NumFUsPerCore * NumCoresPerCache;     // Total Size of instruction cache in bytes
  localparam int unsigned ICacheWays      = (NumCoresPerCache < 4) ? 2 : (NumCoresPerCache / 2);       // Number of ways (cluster_icache requires >= 2)
  localparam int unsigned ICacheLineWidth = 32 * 2 * NumFUsPerCore * NumCoresPerCache;  // Size of each cache line in bits

  /*********************
   *  READ-ONLY CACHE  *
   *********************/

  localparam int unsigned AxiHierRadix      = `ifdef AXI_HIER_RADIX `AXI_HIER_RADIX `else NumTilesPerGroup `endif;
  localparam int unsigned ROCacheLineWidth  = `ifdef RO_LINE_WIDTH `RO_LINE_WIDTH `else 0 `endif;
  localparam int unsigned ROCacheSizeByte   = 8192;
  localparam int unsigned ROCacheWays       = 2;

  localparam int unsigned ROCacheNumAddrRules = 4;
  typedef struct packed {
    logic enable;
    logic flush_valid;
    logic [ROCacheNumAddrRules-1:0][AddrWidth-1:0] start_addr;
    logic [ROCacheNumAddrRules-1:0][AddrWidth-1:0] end_addr;
  } ro_cache_ctrl_t;

  // RO cache reset value to avoid fatal warnings during reset in the address decoder
  localparam ro_cache_ctrl_t ro_cache_ctrl_default = '{
    enable: '0,
    flush_valid: '0,
    start_addr: {32'h18,32'h10,32'h08,32'h00},
    end_addr: {32'h1C,32'h14,32'h0C,32'h04}
  };

  /*********
   *  DMA  *
   *********/

  localparam int unsigned NumDmasPerGroup       = `ifdef DMAS_PER_GROUP `DMAS_PER_GROUP `else 4 `endif;
  localparam int unsigned NumTilesPerDma        = NumTilesPerGroup/NumDmasPerGroup;
  localparam int unsigned DmaDataWidth          = AxiDataWidth;
  localparam int unsigned DmaNumWords           = DmaDataWidth/DataWidth;
  localparam int unsigned NumSuperbanks         = NumBanksPerTile/DmaNumWords;
  localparam int unsigned DmaBurstLen           = (NumBanksPerGroup / NumDmasPerGroup) / DmaNumWords;
  localparam int unsigned NumAXIMastersPerGroup = `ifdef AXI_MASTERS_PER_GROUP `AXI_MASTERS_PER_GROUP `else 1 `endif;

  typedef logic [DmaNumWords*DataWidth-1:0] dma_data_t;
  typedef logic [DmaNumWords*DataWidth/8-1:0] dma_strb_t;

  typedef struct packed {
    axi_tile_id_t id;
    addr_t src;
    addr_t dst;
    logic [31:0] num_bytes;
    axi_pkg::cache_t cache_src;
    axi_pkg::cache_t cache_dst;
    axi_pkg::burst_t burst_src;
    axi_pkg::burst_t burst_dst;
    logic decouple_rw;
    logic deburst;
    logic serialize;
  } dma_req_t;

  typedef struct packed {
    logic backend_idle;
    logic trans_complete;
  } dma_meta_t;

  /***********************
   *  REDMULE PARAMETERS *
   ***********************/

  localparam integer unsigned ARRAY_HEIGHT = `ifdef ARRAY_HEIGHT `ARRAY_HEIGHT `else 4 `endif;
  localparam integer unsigned PIPE_REGS    = `ifdef PIPE_REGS `PIPE_REGS `else 3 `endif;
  localparam integer unsigned ARRAY_WIDTH  = `ifdef ARRAY_WIDTH `ARRAY_WIDTH `else (ARRAY_HEIGHT*PIPE_REGS) `endif;
  localparam integer unsigned ROB_DEPTH    = `ifdef ROB_DEPTH `ROB_DEPTH `else 16 `endif;

  localparam integer unsigned RMNumStreams              = 4;
  localparam integer unsigned RMOutstandingTransactions = ROB_DEPTH;
  localparam integer unsigned RMDataWidth               = 16 * ARRAY_HEIGHT * (PIPE_REGS + 1);
  localparam integer unsigned RMMasterPorts             = RMDataWidth / DataWidth;
  localparam integer unsigned RMRegSize                 = 256;

  // Wide TCDM plane (WP2). WideTcdmDataWidth is the configured payload width of
  // one single-beat wide flit. The current endpoint supports exactly one RedMulE
  // TCDM beat per flit, so enabled builds require this to equal RMDataWidth.
  // A wide flit spans NumSuperbankWords 32-bit banks (a "superbank").
  localparam integer unsigned WideTcdmDataWidth =
      `ifdef WIDE_TCDM_DATA_WIDTH `WIDE_TCDM_DATA_WIDTH `else RMDataWidth `endif;
  localparam integer unsigned NumSuperbankWords = WideTcdmDataWidth / DataWidth;

  // ------------------------------------------------------------------
  //  Wide-plane requesters on a tile
  // ------------------------------------------------------------------
  // Each wide client gets its OWN scrambler + tcdm_shim and its OWN local port
  // into the bank interconnect, so two clients hitting different superbanks of
  // the same tile proceed in the same cycle, and one may go own-tile while the
  // other goes remote. Only the tile's single physical wide master link is
  // shared; the existing stream_xbar (NumInp = NumWideLocalPorts) muxes it.
  //
  // The count is build-uniform, not per-tile: a tile without RedMulE ties its
  // slot off rather than changing the port dimensions of the group.
  //   slot 0 : RedMulE      (present when NumRMTiles > 0)
  //   slot 1 : Spatz burst adapter (present when spatz_burst is on)
  localparam bit SpatzWideBurst =
      `ifdef SPATZ_VLSU_BURST (`SPATZ_VLSU_BURST != 0) `else 1'b0 `endif;
  localparam integer unsigned NumWideRmPorts    = (NumRMTiles > 0) ? 1 : 0;
  // ONE wide slot per Spatz core: each core gets its own burst adapter and its own
  // shim, so two cores of a tile can hold wide accesses at once (different
  // superbanks, or one own-tile and one remote). They contend only at the tile's
  // single wide remote link, which the mux after the shims resolves.
  localparam integer unsigned NumWideSpatzPorts =
      SpatzWideBurst ? NumCoresPerTile : 0;
  // At least one slot: the arrays and the crossbars are declared unconditionally
  // and a zero-width dimension is not legal.
  localparam integer unsigned NumWideLocalPorts =
      (NumWideRmPorts + NumWideSpatzPorts) > 0 ? (NumWideRmPorts + NumWideSpatzPorts) : 1;
  localparam integer unsigned WideRmPortIdx    = 0;
  // Base slot of the Spatz block; core c owns WideSpatzPortIdx + c.
  localparam integer unsigned WideSpatzPortIdx = NumWideRmPorts;

  localparam integer unsigned RMBaseAddr = 32'h4002_0000;
  localparam integer unsigned RMMask     = ~((1 << idx_width(RMRegSize)) - 1);

  // Per-tile local TCDM master ports: the (Spatz-scaled) core data ports plus
  // the RedMulE master ports when present. Degrades to NumCoresPerTile for the
  // scalar case (NumDataPortsPerCore = 1, NumRMTiles = 0).
  localparam integer unsigned MaxLocalPortsPerTile =
      NumRMTiles > 0 ? RMMasterPorts + NumCoresPerTile * NumDataPortsPerCore
                     : NumCoresPerTile * NumDataPortsPerCore;

  // Shared core<->L1 meta_id. Every engine on a tile master port carries its
  // outstanding-transaction id in this one field, so it must be as wide as the
  // widest requirement over the engines THIS build instantiates -- where the
  // requirement is the engine's id plus whatever the tile packs alongside it:
  //
  //   scalar snitch     idx_width(NumIntOutstandingLoads)
  //   Spatz FP sequencer $clog2(NRVREG) -- a vector-REGISTER index
  //   Spatz vector LSU   idx_width(spatz_rob_depth) + 1  (store-ack flag)
  //   RedMulE            idx_width(RMNumStreams * RMOutstandingTransactions)
  //
  // The two Spatz requesters have DIFFERENT widths and must be counted separately.
  // snitch_pkg::MetaIdWidth covers port 0 (integer LSU + FP sequencer LSU). The
  // vector LSU ports are a second, independent requester: mempool_cc packs
  // {pad, write, id} into their meta_id, where id is the VLSU reorder-buffer id
  // (spatz_pkg::MemRspIdWidth, keyed off the same SPATZ_VLSU_ROB_DEPTH define
  // mirrored here) and the flag is the store ack that the TCDM response channel has
  // no line for. mempool_cc checks at elaboration that the two agree.
  // snitch_pkg::MetaIdWidth already maxes over every requester that rides a core
  // port, the vector LSU + store-ack flag included, because mempool_tile types its
  // per-core tcdm_shim with snitch_pkg::dreq_t and all core ports pass through it.
  localparam integer unsigned CoreMetaIdWidth = snitch_pkg::MetaIdWidth;
  localparam integer unsigned RMMetaIdWidth =
      NumRMTiles > 0 ? idx_width(RMNumStreams * RMOutstandingTransactions) : 0;
  localparam integer unsigned MetaIdWidth =
      CoreMetaIdWidth > RMMetaIdWidth ? CoreMetaIdWidth : RMMetaIdWidth;

  /**********************************
   *  TCDM INTERCONNECT PARAMETERS  *
   **********************************/

  typedef logic [TCDMAddrWidth-1:0] tcdm_addr_t;
  typedef logic [TCDMAddrMemWidth-1:0] bank_addr_t;
  typedef logic [TCDMAddrMemWidth+idx_width(NumBanksPerTile)-1:0] tile_addr_t;
  typedef logic [MetaIdWidth-1:0] meta_id_t;
  typedef logic [idx_width(MaxLocalPortsPerTile)-1:0] tile_core_id_t;
  typedef logic [idx_width(NumTilesPerGroup)-1:0] tile_group_id_t;
  typedef logic [idx_width(NumGroups)-1:0] group_id_t;
  typedef logic [3:0] amo_t;

  typedef struct packed {
    addr_t    addr;
    meta_id_t id;
    logic [3:0] amo;
    logic     write;
    data_t    data;
    strb_t    strb;
  } rm_dreq_t;

  typedef struct packed {
    data_t    data;
    meta_id_t id;
    logic     write;
    logic     error;
  } rm_dresp_t;

  typedef struct packed {
    meta_id_t meta_id;
    tile_core_id_t core_id;
    amo_t amo;
    data_t data;
  } tcdm_payload_t;

  typedef struct packed {
    tcdm_payload_t wdata;
    logic wen;
    strb_t be;
    group_id_t tgt_group_id; // FlooNoC Added
    tcdm_addr_t tgt_addr;
  } tcdm_master_req_t;

  typedef struct packed {
    tcdm_payload_t rdata;
  } tcdm_master_resp_t;

  typedef struct packed {
    tcdm_payload_t wdata;
    logic wen;
    strb_t be;
    tile_addr_t tgt_addr;
    tile_group_id_t ini_addr;
    group_id_t src_group_id; // FlooNoC Added
  } tcdm_slave_req_t;

  typedef struct packed {
    tcdm_payload_t rdata;
    tile_group_id_t ini_addr;
    group_id_t src_group_id; // FlooNoC Added
  } tcdm_slave_resp_t;

  /********************
   *  DMA PARAMETERS  *
   *******************/

  typedef struct packed {
    meta_id_t meta_id;
    tile_core_id_t core_id;
    amo_t amo;
    dma_data_t data;
  } dma_payload_t;

  typedef struct packed {
    dma_payload_t wdata;
    logic wen;
    dma_strb_t be;
    tile_addr_t tgt_addr;
  } tcdm_dma_req_t;

  typedef struct packed {
    dma_payload_t rdata;
  } tcdm_dma_resp_t;

  // Wide-engine (WP2 RedMulE) local superbank access: a WIDE_W-wide version of
  // the DMA wide req/resp, landing on NumSuperbankWords banks. Distinct from the
  // DMA wide type because WIDE_W (=RMDataWidth) need not equal DmaDataWidth.
  `ifdef USE_WIDE_TCDM_PLANE
  typedef logic [WideTcdmDataWidth-1:0]   wide_tcdm_data_t;
  typedef logic [WideTcdmDataWidth/8-1:0] wide_tcdm_strb_t;

  typedef struct packed {
    meta_id_t       meta_id;
    tile_core_id_t  core_id;
    amo_t           amo;
    wide_tcdm_data_t data;
  } wide_tcdm_payload_t;

  // Raw wide TCDM interfaces. As for the narrow TCDM interfaces above, FlooNoC
  // headers are added and removed only in mempool_group_floonoc_wrapper.
  typedef struct packed {
    wide_tcdm_payload_t wdata;
    logic              wen;
    wide_tcdm_strb_t    be;
    group_id_t         tgt_group_id;
    tcdm_addr_t        tgt_addr;
    tile_group_id_t    src_tile_id;
  } tcdm_wide_master_req_t;

  typedef struct packed {
    wide_tcdm_payload_t rdata;
    tile_group_id_t    tile_id;
  } tcdm_wide_master_resp_t;

  typedef struct packed {
    wide_tcdm_payload_t wdata;
    logic              wen;
    wide_tcdm_strb_t    be;
    tcdm_addr_t        tgt_addr;
    tile_group_id_t    ini_addr;
    group_id_t         src_group_id;
  } tcdm_wide_slave_req_t;

  typedef struct packed {
    wide_tcdm_payload_t rdata;
    tile_group_id_t    ini_addr;
    group_id_t         src_group_id;
    logic              is_write;
  } tcdm_wide_slave_resp_t;

  typedef struct packed {
    wide_tcdm_payload_t wdata;
    logic              wen;
    wide_tcdm_strb_t    be;
    tile_addr_t        tgt_addr;
    tile_group_id_t    ini_addr;      // source tile  — self-routing (mirror tcdm_slave_req_t)
    group_id_t         src_group_id;  // source group — routes the response back
  } tcdm_wide_req_t;

  typedef tcdm_wide_slave_resp_t tcdm_wide_resp_t;
  `endif

  /*************************************
   *  FlooNoC INTERCONNECT PARAMETERS  *
   *************************************/

  `include "floo_noc/typedef.svh"

  // TCDM feature-guard contract:
  //
  // * The narrow plane is always present. USE_RO_NARROW_REQ_CHANNEL adds its
  //   optional, meta-only RO request array.
  // * USE_WIDE_TCDM_PLANE owns every wide type and packed interface member.
  //   Its RW request and RW response channel arrays are mandatory.
  // * USE_RO_WIDE_REQ_CHANNEL and USE_WO_WIDE_RESP_CHANNEL are optional children
  //   of USE_WIDE_TCDM_PLANE. They add the meta-only RO request and WO response
  //   channel arrays, respectively.
  //
  // In this package, the optional guards are needed only where a zero channel
  // count would create an illegal packed array. The build derives them from the
  // corresponding channel counts; source code must not define them independently.

  // Narrow TCDM plane channel counts. The RO request class is meta-only; RW and
  // WO request allocations share the payload-bearing RW flit type. RD is the
  // data-bearing response class and also carries write acknowledgements.
  localparam integer unsigned NumNarrowRoReqPortsPerTile =
      `ifdef NOC_REQ_NARROW_RO_CHANNEL_NUM `NOC_REQ_NARROW_RO_CHANNEL_NUM `else 0 `endif;
  localparam integer unsigned NumNarrowRwReqPortsPerTile =
      `ifdef NOC_REQ_NARROW_RW_CHANNEL_NUM `NOC_REQ_NARROW_RW_CHANNEL_NUM `else 2 `endif;
  localparam integer unsigned NumNarrowWoReqPortsPerTile =
      `ifdef NOC_REQ_NARROW_WO_CHANNEL_NUM `NOC_REQ_NARROW_WO_CHANNEL_NUM `else 0 `endif;
  localparam integer unsigned NumNarrowRwRespPortsPerTile =
      `ifdef NOC_RESP_NARROW_RW_CHANNEL_NUM `NOC_RESP_NARROW_RW_CHANNEL_NUM `else 2 `endif;

  localparam integer unsigned NumNarrowPayloadReqPortsPerTile =
      NumNarrowRwReqPortsPerTile + NumNarrowWoReqPortsPerTile;

  // Wide TCDM plane (WP2) group-channel counts. A SEPARATE FlooNoC net with its own
  // routers/links, so these are NOT folded into NumNarrowRemoteReqPortsPerTile (which
  // sizes the 32-bit net). Default 0 => the wide plane is absent and the build
  // is bit-identical to one without it.
  //
  // The plane carries FOUR channel classes, mirroring the narrow TCDM plane's
  // RO/RW split and sized independently (roadmap S9 / stage-1 DSE plan §6.6).
  // Engine traffic is read-dominated and structurally asymmetric -- a request
  // carries no data, a read response carries a whole line -- so the cheap classes
  // are the ones worth multiplying:
  //
  //   class            payload            wires   role
  //   RO  req          meta only            ~46   read request (no data to send)
  //   RW  req          meta + WIDE_W + strb ~627  write request (also legal for reads)
  //   RW  resp         meta + WIDE_W       ~540   read data or a write ack
  //   WO  resp         meta only            ~28   write ack (NOT a data response)
  //
  // RW req and RW resp are mandatory whenever the plane is on. RO req and WO
  // resp are optional; the guard contract above keeps their zero-count arrays
  // out of packed interface types.
  localparam integer unsigned NumWideRoReqChannelsPerGroup =
      `ifdef NOC_REQ_WIDE_RO_CHANNEL_NUM `NOC_REQ_WIDE_RO_CHANNEL_NUM `else 0 `endif;
  localparam integer unsigned NumWideRwReqChannelsPerGroup =
      `ifdef NOC_REQ_WIDE_RW_CHANNEL_NUM `NOC_REQ_WIDE_RW_CHANNEL_NUM `else 0 `endif;
  localparam integer unsigned NumWideRwRespChannelsPerGroup =
      `ifdef NOC_RESP_WIDE_RW_CHANNEL_NUM `NOC_RESP_WIDE_RW_CHANNEL_NUM `else 0 `endif;
  localparam integer unsigned NumWideWoRespChannelsPerGroup =
      `ifdef NOC_RESP_WIDE_WO_CHANNEL_NUM `NOC_RESP_WIDE_WO_CHANNEL_NUM `else 0 `endif;

  // Logical wide ports at each tile. The first ports are dedicated same-group,
  // other-tile paths. The remaining indices are class-neutral inter-group ports;
  // the group maps them onto its physical RO/RW/WO channel classes.
  //
  // A tile's wide ports are SYMMETRIC: each kind is one count that sizes both
  // the request and the response side, so the two totals below are equal by
  // construction and cannot be configured apart.
  localparam integer unsigned NumWideIntraGroupPortsPerTile =
      `ifdef NUM_INTRA_GROUP_PORTS_PER_TILE
      `NUM_INTRA_GROUP_PORTS_PER_TILE
      `else 1 `endif;
  localparam integer unsigned NumWideInterGroupPortsPerTile =
      `ifdef NUM_INTER_GROUP_PORTS_PER_TILE
      `NUM_INTER_GROUP_PORTS_PER_TILE
      `else 1 `endif;
  localparam integer unsigned NumWideRemoteReqPortsPerTile =
      NumWideIntraGroupPortsPerTile + NumWideInterGroupPortsPerTile;
  localparam integer unsigned NumWideRemoteRespPortsPerTile =
      NumWideIntraGroupPortsPerTile + NumWideInterGroupPortsPerTile;

  // Totals: the physical wide channels carried by each group-to-group mesh direction.
  localparam integer unsigned NumWideRemoteReqChannelsPerGroup =
      NumWideRoReqChannelsPerGroup + NumWideRwReqChannelsPerGroup;
  localparam integer unsigned NumWideRemoteRespChannelsPerGroup =
      NumWideRwRespChannelsPerGroup + NumWideWoRespChannelsPerGroup;

  localparam integer unsigned NumNarrowRemoteReqPortsPerTile =
      1 + NumNarrowRoReqPortsPerTile + NumNarrowPayloadReqPortsPerTile;
  localparam integer unsigned NumNarrowRemoteRespPortsPerTile =
      1 + NumNarrowRwRespPortsPerTile;
  localparam integer unsigned NumDirections = `ifdef NUM_DIRECTIONS `NUM_DIRECTIONS `else 5 `endif;
  localparam integer unsigned NumX = `ifdef NUM_X `NUM_X `else 2 `endif;
  localparam integer unsigned NumY = NumGroups/NumX;
  localparam integer unsigned NocTopology = `ifdef NOC_TOPOLOGY `NOC_TOPOLOGY `else 0 `endif;
  localparam integer unsigned NocRoutingAlgorithm = `ifdef NOC_ROUTING_ALGORITHM `NOC_ROUTING_ALGORITHM `else 0 `endif;
  localparam integer unsigned NocRouterRemapping = `ifdef NOC_ROUTER_REMAPPING `NOC_ROUTER_REMAPPING `else 0 `endif;
  // router buffer configuration
  localparam integer unsigned NumRouterInFifoDepth  = `ifdef NOC_ROUTER_INPUT_FIFO_DEP   `NOC_ROUTER_INPUT_FIFO_DEP   `else 2 `endif;
  localparam integer unsigned NumRouterOutFifoDepth = `ifdef NOC_ROUTER_OUTPUT_FIFO_DEP  `NOC_ROUTER_OUTPUT_FIFO_DEP  `else 2 `endif;
  // Per-class router buffering. A request flit is metadata only; a response flit
  // carries the payload, so on the wide plane a response slot costs roughly 23x
  // a request slot and the two want different depths. Each defaults to the
  // global above, so a flavor that sets only the globals is unchanged.
  localparam integer unsigned NumNarrowReqRouterInFifoDepth =
      `ifdef NOC_ROUTER_NARROW_REQ_INPUT_FIFO_DEP
      `NOC_ROUTER_NARROW_REQ_INPUT_FIFO_DEP
      `else NumRouterInFifoDepth `endif;
  localparam integer unsigned NumNarrowReqRouterOutFifoDepth =
      `ifdef NOC_ROUTER_NARROW_REQ_OUTPUT_FIFO_DEP
      `NOC_ROUTER_NARROW_REQ_OUTPUT_FIFO_DEP
      `else NumRouterOutFifoDepth `endif;
  localparam integer unsigned NumNarrowRespRouterInFifoDepth =
      `ifdef NOC_ROUTER_NARROW_RESP_INPUT_FIFO_DEP
      `NOC_ROUTER_NARROW_RESP_INPUT_FIFO_DEP
      `else NumRouterInFifoDepth `endif;
  localparam integer unsigned NumNarrowRespRouterOutFifoDepth =
      `ifdef NOC_ROUTER_NARROW_RESP_OUTPUT_FIFO_DEP
      `NOC_ROUTER_NARROW_RESP_OUTPUT_FIFO_DEP
      `else NumRouterOutFifoDepth `endif;
  localparam integer unsigned NumWideReqRouterInFifoDepth =
      `ifdef NOC_ROUTER_WIDE_REQ_INPUT_FIFO_DEP
      `NOC_ROUTER_WIDE_REQ_INPUT_FIFO_DEP
      `else NumRouterInFifoDepth `endif;
  localparam integer unsigned NumWideReqRouterOutFifoDepth =
      `ifdef NOC_ROUTER_WIDE_REQ_OUTPUT_FIFO_DEP
      `NOC_ROUTER_WIDE_REQ_OUTPUT_FIFO_DEP
      `else NumRouterOutFifoDepth `endif;
  localparam integer unsigned NumWideRespRouterInFifoDepth =
      `ifdef NOC_ROUTER_WIDE_RESP_INPUT_FIFO_DEP
      `NOC_ROUTER_WIDE_RESP_INPUT_FIFO_DEP
      `else NumRouterInFifoDepth `endif;
  localparam integer unsigned NumWideRespRouterOutFifoDepth =
      `ifdef NOC_ROUTER_WIDE_RESP_OUTPUT_FIFO_DEP
      `NOC_ROUTER_WIDE_RESP_OUTPUT_FIFO_DEP
      `else NumRouterOutFifoDepth `endif;
  localparam integer unsigned SpmBankIdRemap = `ifdef SPM_BANK_ID_REMAP `SPM_BANK_ID_REMAP `else 0 `endif;
  localparam integer unsigned TileIdRemap = `ifdef TILE_ID_REMAP `TILE_ID_REMAP `else 0 `endif;
  localparam integer unsigned RouterRemapGroupSize = `ifdef NOC_ROUTER_REMAP_GROUP_SIZE `NOC_ROUTER_REMAP_GROUP_SIZE `else 2 `endif;
  // Wide-plane channel remapping, independent of the narrow plane's
  // NocRouterRemapping. Bit 0 rotates the RO request class, bit 1 the RW
  // response class; the other two classes do not alias and are never remapped.
  // config/config.mk documents the lever, hardware/Makefile validates the group
  // sizes against the class counts and the inter-group ports per tile.
  localparam integer unsigned WideRouterRemapping =
      `ifdef NOC_WIDE_ROUTER_REMAPPING `NOC_WIDE_ROUTER_REMAPPING `else 0 `endif;
  localparam integer unsigned WideRoReqRemapGroupSize =
      `ifdef NOC_WIDE_RO_REQ_REMAP_GROUP_SIZE
      `NOC_WIDE_RO_REQ_REMAP_GROUP_SIZE
      `else 2 `endif;
  localparam integer unsigned WideRwRespRemapGroupSize =
      `ifdef NOC_WIDE_RW_RESP_REMAP_GROUP_SIZE
      `NOC_WIDE_RW_RESP_REMAP_GROUP_SIZE
      `else 2 `endif;

  // FlooNoC group id types for XY routing
  typedef logic [idx_width(NumX)-1:0] group_x_id_t;
  typedef logic [idx_width(NumY)-1:0] group_y_id_t;
  typedef logic group_port_id_t;
  `FLOO_TYPEDEF_XY_NODE_ID_T(group_xy_id_t, group_x_id_t, group_y_id_t, group_port_id_t)

  // FlooNoC req types
  typedef struct packed {
    amo_t               amo;
    logic               wen;
    strb_t              be;
    data_t              data;
  } floo_tcdm_narrow_req_payload_t;

  typedef struct packed {
    meta_id_t           meta_id;
    tile_core_id_t      core_id;
    tile_group_id_t     src_tile_id;
    group_xy_id_t       src_id;
    group_xy_id_t       dst_id;
    tcdm_addr_t         tgt_addr;
    logic               last;
    logic               collective_op;
  } floo_tcdm_req_meta_t;

  typedef struct packed {
    floo_tcdm_req_meta_t  hdr;
  } floo_tcdm_narrow_ro_req_t;

  typedef struct packed {
    floo_tcdm_narrow_req_payload_t payload;
    floo_tcdm_req_meta_t           hdr;
  } floo_tcdm_narrow_rw_req_t;

  // --------------------------------------------------------------------------
  //  Wide TCDM plane (WP2) request flit + group-channel interface — a separate,
  //  single-beat FlooNoC net (WIDE_W wide). Present only when the plane is
  //  enabled (default off => bit-identical). Single-beat: one flit per engine
  //  request, hdr.last always 1'b1 (INVARIANT #1). It rides the same per-group
  //  boundary bundle as the 32-bit net but keeps its OWN routers/valid-ready/
  //  FIFOs, so it stays an isolated net (no HoL) while the inter-group mesh
  //  (whole-struct group-to-group wiring) carries it for free.
  // --------------------------------------------------------------------------
  `ifdef USE_WIDE_TCDM_PLANE
  typedef struct packed {
    amo_t         amo;
    logic         wen;
    wide_tcdm_strb_t  be;
    wide_tcdm_data_t  data;
  } floo_tcdm_wide_req_payload_t;

  // RW request: the full flit (write data + strobes). Also legal for reads, which
  // is what makes RO channels a pure optimisation rather than a requirement.
  // Reuse the 32-bit net's request meta: meta_id already covers RedMulE's ROB id
  // (MetaIdWidth), preserving the user-id end-to-end (INVARIANT #3).
  typedef struct packed {
    floo_tcdm_wide_req_payload_t payload;
    floo_tcdm_req_meta_t          hdr;
  } floo_tcdm_wide_rw_req_t;

  // RO request: meta only -- a read has nothing to send. Mirrors floo_tcdm_narrow_ro_req_t.
  typedef struct packed {
    floo_tcdm_req_meta_t hdr;
  } floo_tcdm_wide_ro_req_t;

  typedef struct packed {
    floo_tcdm_wide_rw_req_t req;
    logic                    valid;
    logic                    ready;
  } floo_tcdm_wide_rw_req_if_entry_t;

  typedef struct packed {
    floo_tcdm_wide_ro_req_t req;
    logic                    valid;
    logic                    ready;
  } floo_tcdm_wide_ro_req_if_entry_t;

  typedef struct packed {
    `ifdef USE_RO_WIDE_REQ_CHANNEL
    floo_tcdm_wide_ro_req_if_entry_t [NumWideRoReqChannelsPerGroup-1:0] wide_ro_req;
    `endif
    floo_tcdm_wide_rw_req_if_entry_t [NumWideRwReqChannelsPerGroup-1:0] wide_rw_req;
  } floo_tcdm_wide_req_if_group_entry_t;
  `endif

  typedef struct packed {
    floo_tcdm_narrow_ro_req_t req;
    logic                      valid;
    logic                      ready;
  } floo_tcdm_narrow_ro_req_if_entry_t;

  typedef struct packed {
    floo_tcdm_narrow_rw_req_t req;
    logic                      valid;
    logic                      ready;
  } floo_tcdm_narrow_rw_req_if_entry_t;

  typedef struct packed {
    `ifdef USE_RO_NARROW_REQ_CHANNEL
    floo_tcdm_narrow_ro_req_if_entry_t [NumNarrowRoReqPortsPerTile-1:0] narrow_ro_req;
    `endif
    floo_tcdm_narrow_rw_req_if_entry_t [NumNarrowPayloadReqPortsPerTile-1:0] narrow_rw_req;
  } floo_tcdm_req_if_per_tile_entry_t;

  typedef struct packed {
    floo_tcdm_req_if_per_tile_entry_t [NumTilesPerGroup-1:0] floo_tcdm_req;
    `ifdef USE_WIDE_TCDM_PLANE
    floo_tcdm_wide_req_if_group_entry_t floo_tcdm_wide_req;
    `endif
  } floo_tcdm_req_if_t;

  // FlooNoC resp types
  typedef struct packed {
    amo_t               amo;
    data_t              data;
  } floo_tcdm_narrow_resp_payload_t;

  typedef struct packed {
    meta_id_t           meta_id;
    tile_core_id_t      core_id;
    tile_group_id_t     tile_id;
    group_xy_id_t       src_id;
    group_xy_id_t       dst_id;
    logic               last;
    logic               collective_op;
  } floo_tcdm_resp_meta_t;

  typedef struct packed {
    floo_tcdm_narrow_resp_payload_t payload;
    floo_tcdm_resp_meta_t            hdr;
  } floo_tcdm_narrow_rw_resp_t;

  // Wide TCDM plane (WP2) response flits + group-channel interface (see request note).
  `ifdef USE_WIDE_TCDM_PLANE
  typedef struct packed {
    wide_tcdm_data_t data;
  } floo_tcdm_wide_resp_payload_t;

  // RD flit format: carries the line on an RW response channel.
  typedef struct packed {
    floo_tcdm_wide_resp_payload_t payload;
    floo_tcdm_resp_meta_t           hdr;
  } floo_tcdm_wide_rd_resp_t;

  // WR flit format: a meta-only ACK carried on a WO response channel. A write has
  // no data to return, so giving it a full-width response would burn WIDE_W wires
  // to carry nothing. The engine still needs the handshake -- RedMulE retires
  // stores through the same meta_id path.
  typedef struct packed {
    floo_tcdm_resp_meta_t hdr;
  } floo_tcdm_wide_wr_resp_t;

  typedef struct packed {
    floo_tcdm_wide_rd_resp_t resp;
    logic                     valid;
    logic                     ready;
  } floo_tcdm_wide_rw_rsp_if_entry_t;

  typedef struct packed {
    floo_tcdm_wide_wr_resp_t resp;
    logic                     valid;
    logic                     ready;
  } floo_tcdm_wide_wo_rsp_if_entry_t;
  `endif

  typedef struct packed {
    floo_tcdm_narrow_rw_resp_t resp;
    logic            valid;
    logic            ready;
  } floo_tcdm_narrow_rw_rsp_if_entry_t;

  typedef struct packed {
    floo_tcdm_narrow_rw_rsp_if_entry_t     [NumTilesPerGroup-1:0][NumNarrowRemoteRespPortsPerTile-1:1]    floo_tcdm_narrow_rw_resp;
    `ifdef USE_WIDE_TCDM_PLANE
    floo_tcdm_wide_rw_rsp_if_entry_t [NumWideRwRespChannelsPerGroup-1:0] floo_tcdm_wide_rw_resp;
    `ifdef USE_WO_WIDE_RESP_CHANNEL
    floo_tcdm_wide_wo_rsp_if_entry_t [NumWideWoRespChannelsPerGroup-1:0] floo_tcdm_wide_wo_resp;
    `endif
    `endif
  } floo_tcdm_rsp_if_t;

  /**********************
   *  QUEUE PARAMETERS  *
   **********************/

  // Size of xqueues in words (must be a power of two)
  localparam int unsigned XQueueSize = `ifdef XQUEUE_SIZE `XQUEUE_SIZE `else 0 `endif;

  /*****************
   *  ADDRESS MAP  *
   *****************/

  // TCDM Memory Region
  localparam addr_t TCDMSize = NumBanks * TCDMSizePerBank;
  localparam addr_t TCDMMask = ~(TCDMSize - 1);

  // Size in bytes of memory that is sequentially addressable per tile
  localparam int unsigned SeqMemSizePerCore = `ifdef SEQ_MEM_SIZE `SEQ_MEM_SIZE `else 0 `endif;
  localparam int unsigned SeqMemSizePerTile = NumCoresPerTile*SeqMemSizePerCore;

  typedef struct packed {
    int unsigned slave_idx;
    addr_t mask;
    addr_t value;
  } address_map_t;

  /***********************
   *  TRAFFIC GENERATOR  *
   ***********************/

  // Replaces core with a traffic generator
  parameter bit TrafficGeneration  = `ifdef TRAFFIC_GEN `TRAFFIC_GEN `else 0 `endif;

  //TeraPool Parameters
  `include "reqrsp_interface/typedef.svh"
  `REQRSP_TYPEDEF_ALL(reqrsp, addr_t, axi_data_t, axi_strb_t)

  // TeraPool PostLayout Control
  localparam bit PostLayoutGr = `ifdef POSTLAYOUTGR `POSTLAYOUTGR `else 0 `endif;

`ifndef TARGET_SYNTHESIS
  `ifndef TARGET_VERILATOR
  // tcdm memory pattern profile
  typedef struct {
    int unsigned initiated;
    int unsigned initial_cycle;
    int unsigned last_read_cycle;
    int unsigned last_write_cycle;
    int unsigned last_access_cycle;
    int unsigned access_read_number;
    int unsigned access_write_number;
    int unsigned access_number;
    int unsigned read_cycles[$];       // dynamic array to store cycles of read accesses
    int unsigned write_cycles[$];      // dynamic array to store cycles of write accesses
  } profile_t;

  `endif
`endif


endpackage : mempool_pkg
