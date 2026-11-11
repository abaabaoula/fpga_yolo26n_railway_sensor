// Two address-selected Cin32 halo banks, ready for later DMA/compute overlap.
module input_buffer (
    input logic clk, resetn, wr_en, wr_bank, rd_en, rd_bank,
    input logic [12:0] wr_addr,rd_addr,
    input logic [63:0] wr_data,
    output logic [63:0] rd_data
);
    (* ram_style="block" *) logic [63:0] memory[0:9215];
    wire [13:0] wa={1'b0,wr_addr}+(wr_bank ? 14'd4608:14'd0);
    wire [13:0] ra={1'b0,rd_addr}+(rd_bank ? 14'd4608:14'd0);
    always_ff @(posedge clk) begin
        if (resetn && wr_en) memory[wa]<=wr_data;
        if (!resetn) rd_data<=0;
        else if (rd_en) rd_data<=memory[ra];
    end
endmodule
