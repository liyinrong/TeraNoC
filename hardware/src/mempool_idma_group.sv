// Copyright 2026 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// One mainstream iDMA v0.6.5 frontend per MemPool group. Every tile of the
// group reaches the same register context through mempool_group.sv's
// arbiter; accepted descriptors go to the configured backend streams.

module mempool_idma_group import mempool_pkg::*; #(
  parameter int unsigned NumRegs          = 1,
  parameter int unsigned NumStreams       = NumDmasPerGroup,
  parameter int unsigned ReqFifoDepth     = 16,
  parameter int unsigned NumAxInFlight    = 2,
  parameter int unsigned BufferDepth      = 2,
  parameter int unsigned MemSysDepth      = 0,
  parameter type         reg_req_t        = idma_reg_req_t,
  parameter type         reg_rsp_t        = idma_reg_rsp_t,
  parameter type         idma_req_t       = mempool_pkg::idma_req_t,
  parameter type         idma_rsp_t       = mempool_pkg::idma_rsp_t,
  parameter type         idma_nd_req_t    = mempool_pkg::idma_nd_req_t,
  parameter type         axi_req_t        = axi_tile_req_t,
  parameter type         axi_rsp_t        = axi_tile_resp_t,
  parameter type         read_meta_t      = idma_read_meta_channel_t,
  parameter type         write_meta_t     = idma_write_meta_channel_t,
  parameter type         error_payload_t  = idma_err_payload_t
) (
  input  logic                    clk_i,
  input  logic                    rst_ni,
  input  logic                    testmode_i,
  input  reg_req_t [NumRegs-1:0] ctrl_req_i,
  output reg_rsp_t [NumRegs-1:0] ctrl_rsp_o,
  output axi_req_t [NumStreams-1:0] axi_req_o,
  input  axi_rsp_t [NumStreams-1:0] axi_rsp_i,
  input  logic     [NumStreams-1:0] error_clear_i,
  output logic     [NumStreams-1:0] error_o,
  output error_payload_t [NumStreams-1:0] error_payload_o,
  output logic     [NumStreams-1:0] busy_o
);

  localparam int unsigned StreamWidth = NumStreams > 1 ? $clog2(NumStreams) : 1;
  typedef logic [StreamWidth-1:0] stream_idx_t;
  localparam logic [IdmaNumDim-1:0][31:0] RepWidths = '{default: 32};

  idma_nd_req_t frontend_req;
  logic         frontend_req_valid;
  logic         frontend_req_ready;
  stream_idx_t  frontend_stream;
  idma_counter_t frontend_next_id;

  logic [NumStreams-1:0] frontend_stream_valid;
  logic [NumStreams-1:0] frontend_stream_ready;

  idma_nd_req_t [NumStreams-1:0] nd_req;
  logic         [NumStreams-1:0] nd_req_valid;
  logic         [NumStreams-1:0] nd_req_ready;
  logic         [NumStreams-1:0][$clog2(ReqFifoDepth)-1:0] req_fifo_usage;
  logic         [NumStreams-1:0] req_fifo_busy;
  idma_rsp_t    [NumStreams-1:0] nd_rsp;
  logic         [NumStreams-1:0] nd_rsp_valid;
  logic         [NumStreams-1:0] nd_rsp_ready;
  logic         [NumStreams-1:0] nd_busy;

  idma_req_t [NumStreams-1:0] backend_req;
  logic      [NumStreams-1:0] backend_req_valid;
  logic      [NumStreams-1:0] backend_req_ready;
  idma_rsp_t [NumStreams-1:0] backend_rsp;
  logic      [NumStreams-1:0] backend_rsp_valid;
  logic      [NumStreams-1:0] backend_rsp_ready;
  idma_pkg::idma_busy_t [NumStreams-1:0] backend_busy;

  idma_counter_t [NumStreams-1:0] next_id;
  idma_counter_t [NumStreams-1:0] completed_id;
  logic          [NumStreams-1:0] issue_id;
  logic          [NumStreams-1:0] retire_id;

  axi_req_t [NumStreams-1:0] axi_read_req, axi_write_req;
  axi_rsp_t [NumStreams-1:0] axi_read_rsp, axi_write_rsp;
  reg_req_t [NumRegs-1:0] frontend_ctrl_req;
  reg_rsp_t [NumRegs-1:0] frontend_ctrl_rsp;
  logic     [NumStreams-1:0] csr_error_clear;
  logic     [NumStreams-1:0] error_clear;

  // The generated register frontend has no error CSRs. Overlay two compact
  // per-stream read windows after its last register: STATUS is W1C and returns
  // {cause, type, valid}; ADDRESS returns the failing AXI burst address.
  always_comb begin
    frontend_ctrl_req = ctrl_req_i;
    ctrl_rsp_o         = frontend_ctrl_rsp;
    csr_error_clear    = '0;
    for (int r = 0; r < NumRegs; r++) begin
      for (int s = 0; s < NumStreams; s++) begin
        if (ctrl_req_i[r].valid &&
            ctrl_req_i[r].addr == IdmaErrorStatusOffset + 4*s) begin
          frontend_ctrl_req[r].valid = 1'b0;
          ctrl_rsp_o[r].ready         = 1'b1;
          ctrl_rsp_o[r].error         = 1'b0;
          ctrl_rsp_o[r].rdata         = '0;
          ctrl_rsp_o[r].rdata[0]      = error_o[s];
          ctrl_rsp_o[r].rdata[2:1]    = error_payload_o[s].err_type;
          ctrl_rsp_o[r].rdata[4:3]    = error_payload_o[s].cause;
          if (ctrl_req_i[r].write && ctrl_req_i[r].wstrb[0] && ctrl_req_i[r].wdata[0])
            csr_error_clear[s] = 1'b1;
        end
        if (ctrl_req_i[r].valid &&
            ctrl_req_i[r].addr == IdmaErrorAddressOffset + 4*s) begin
          frontend_ctrl_req[r].valid = 1'b0;
          ctrl_rsp_o[r].ready         = 1'b1;
          ctrl_rsp_o[r].error         = ctrl_req_i[r].write;
          ctrl_rsp_o[r].rdata         = error_payload_o[s].burst_addr;
        end
      end
    end
  end

  assign error_clear = error_clear_i | csr_error_clear;

  always_comb begin
    frontend_stream_valid = '0;
    frontend_req_ready    = 1'b0;
    frontend_next_id      = '0;
    if (unsigned'(frontend_stream) < NumStreams) begin
      frontend_stream_valid[frontend_stream] = frontend_req_valid;
      frontend_req_ready = frontend_stream_ready[frontend_stream];
      frontend_next_id = next_id[frontend_stream];
    end
  end

  idma_reg32_3d #(
    .NumRegs        (NumRegs          ),
    .NumStreams     (NumStreams       ),
    .IdCounterWidth (IdmaIdWidth      ),
    .reg_req_t      (reg_req_t        ),
    .reg_rsp_t      (reg_rsp_t        ),
    .dma_req_t      (idma_nd_req_t    ),
    .cnt_width_t    (idma_counter_t   ),
    .stream_t       (stream_idx_t     )
  ) i_idma_reg32_3d (
    .clk_i,
    .rst_ni,
    .dma_ctrl_req_i (frontend_ctrl_req  ),
    .dma_ctrl_rsp_o (frontend_ctrl_rsp  ),
    .dma_req_o      (frontend_req       ),
    .req_valid_o    (frontend_req_valid ),
    .req_ready_i    (frontend_req_ready ),
    .next_id_i      (frontend_next_id   ),
    .stream_idx_o   (frontend_stream    ),
    .done_id_i      (completed_id       ),
    .busy_i         (backend_busy       ),
    .midend_busy_i  (nd_busy | req_fifo_busy)
  );

  for (genvar s = 0; s < NumStreams; s++) begin : gen_streams
    stream_fifo_optimal_wrap #(
      .Depth     (ReqFifoDepth  ),
      .type_t    (idma_nd_req_t ),
      .PrintInfo (1'b0          )
    ) i_req_fifo (
      .clk_i,
      .rst_ni,
      .testmode_i,
      .flush_i (1'b0                      ),
      .usage_o (req_fifo_usage[s]         ),
      .data_i  (frontend_req              ),
      .valid_i (frontend_stream_valid[s]  ),
      .ready_o (frontend_stream_ready[s]  ),
      .data_o  (nd_req[s]                 ),
      .valid_o (nd_req_valid[s]           ),
      .ready_i (nd_req_ready[s]           )
    );

    idma_nd_midend #(
      .NumDim        (IdmaNumDim    ),
      .addr_t        (addr_t        ),
      .idma_req_t    (idma_req_t    ),
      .idma_rsp_t    (idma_rsp_t    ),
      .idma_nd_req_t (idma_nd_req_t ),
      .RepWidths     (RepWidths     )
    ) i_nd_midend (
      .clk_i,
      .rst_ni,
      .nd_req_i          (nd_req[s]           ),
      .nd_req_valid_i    (nd_req_valid[s]     ),
      .nd_req_ready_o    (nd_req_ready[s]     ),
      .nd_rsp_o          (nd_rsp[s]           ),
      .nd_rsp_valid_o    (nd_rsp_valid[s]     ),
      .nd_rsp_ready_i    (nd_rsp_ready[s]     ),
      .burst_req_o       (backend_req[s]      ),
      .burst_req_valid_o (backend_req_valid[s]),
      .burst_req_ready_i (backend_req_ready[s]),
      .burst_rsp_i       (backend_rsp[s]      ),
      .burst_rsp_valid_i (backend_rsp_valid[s]),
      .burst_rsp_ready_o (backend_rsp_ready[s]),
      .busy_o            (nd_busy[s]          )
    );

    idma_backend_rw_axi #(
      .DataWidth            (AxiDataWidth                  ),
      .AddrWidth            (AddrWidth                     ),
      .UserWidth            (1                             ),
      .AxiIdWidth           (AxiTileIdWidth                ),
      .NumAxInFlight        (NumAxInFlight                 ),
      .BufferDepth          (BufferDepth                   ),
      .TFLenWidth           ($bits(idma_tf_len_t)          ),
      .MemSysDepth          (MemSysDepth                   ),
      .CombinedShifter      (1'b1                          ),
      .RAWCouplingAvail     (1'b1                          ),
      .MaskInvalidData      (1'b0                          ),
      .HardwareLegalizer    (1'b1                          ),
      .RejectZeroTransfers  (1'b1                          ),
      .ErrorCap             (idma_pkg::ERROR_HANDLING      ),
      .PrintFifoInfo        (1'b0                          ),
      .idma_req_t           (idma_req_t                    ),
      .idma_rsp_t           (idma_rsp_t                    ),
      .idma_eh_req_t        (idma_pkg::idma_eh_req_t       ),
      .idma_busy_t          (idma_pkg::idma_busy_t         ),
      .axi_req_t            (axi_req_t                     ),
      .axi_rsp_t            (axi_rsp_t                     ),
      .read_meta_channel_t  (read_meta_t                   ),
      .write_meta_channel_t (write_meta_t                  )
    ) i_backend (
      .clk_i,
      .rst_ni,
      .testmode_i,
      .idma_req_i      (backend_req[s]       ),
      .req_valid_i     (backend_req_valid[s] ),
      .req_ready_o     (backend_req_ready[s] ),
      .idma_rsp_o      (backend_rsp[s]       ),
      .rsp_valid_o     (backend_rsp_valid[s] ),
      .rsp_ready_i     (backend_rsp_ready[s] ),
      .idma_eh_req_i   (idma_pkg::ABORT       ),
      .eh_req_valid_i  (error_clear[s]        ),
      .eh_req_ready_o  (                      ),
      .axi_read_req_o  (axi_read_req[s]      ),
      .axi_read_rsp_i  (axi_read_rsp[s]      ),
      .axi_write_req_o (axi_write_req[s]     ),
      .axi_write_rsp_i (axi_write_rsp[s]     ),
      .busy_o          (backend_busy[s]      )
    );

    axi_rw_join #(
      .axi_req_t  (axi_req_t ),
      .axi_resp_t (axi_rsp_t )
    ) i_axi_rw_join (
      .clk_i,
      .rst_ni,
      .slv_read_req_i   (axi_read_req[s]  ),
      .slv_read_resp_o  (axi_read_rsp[s]  ),
      .slv_write_req_i  (axi_write_req[s] ),
      .slv_write_resp_o (axi_write_rsp[s] ),
      .mst_req_o        (axi_req_o[s]     ),
      .mst_resp_i       (axi_rsp_i[s]     )
    );

    idma_transfer_id_gen #(
      .IdWidth (IdmaIdWidth )
    ) i_transfer_id_gen (
      .clk_i,
      .rst_ni,
      .issue_i     (issue_id[s]     ),
      .retire_i    (retire_id[s]    ),
      .next_o      (next_id[s]      ),
      .completed_o (completed_id[s] )
    );

    assign nd_rsp_ready[s] = 1'b1;
    // The ID returned by the launch read belongs to the request accepted into
    // this FIFO in the same cycle. Counting only when the ND midend dequeued it
    // could return the same ID for several back-to-back asynchronous launches.
    assign issue_id[s] = frontend_stream_valid[s] & frontend_stream_ready[s];
    assign retire_id[s] = nd_rsp_valid[s] & nd_rsp_ready[s];
    assign req_fifo_busy[s] = |req_fifo_usage[s];
    assign busy_o[s] = (|backend_busy[s]) | nd_busy[s] | req_fifo_busy[s];

    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        error_o[s]         <= 1'b0;
        error_payload_o[s] <= '0;
      end else begin
        if (error_clear[s]) error_o[s] <= 1'b0;
        if (backend_rsp_valid[s] && backend_rsp_ready[s] && backend_rsp[s].error) begin
          error_o[s]         <= 1'b1;
          error_payload_o[s] <= backend_rsp[s].pld;
        end else if (nd_rsp_valid[s] && nd_rsp_ready[s] && nd_rsp[s].error) begin
          error_o[s]         <= 1'b1;
          error_payload_o[s] <= nd_rsp[s].pld;
        end
      end
    end
  end

  `ifndef SYNTHESIS
  initial begin
    if (NumRegs < 1) $fatal(1, "mempool_idma_group requires at least one register port");
    if (NumStreams < 1) $fatal(1, "mempool_idma_group requires at least one stream");
  end
  `endif

endmodule
