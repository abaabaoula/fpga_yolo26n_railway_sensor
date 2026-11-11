// Same Co32 x 16x16 INT32 capacity; 16 values per read, 512 addresses.
module partial_sum_buffer (
    input logic clk,resetn,wr_en,rd_en,
    input logic [8:0] wr_addr,rd_addr,
    input logic [511:0] wr_data,
    output wire [511:0] rd_data
);
    for(genvar lane=0;lane<16;lane++)begin:bank
        (* ram_style="block" *) logic [31:0] memory[0:511];
        logic [31:0] q;
        always_ff @(posedge clk)begin
            if(resetn && wr_en)memory[wr_addr]<=wr_data[lane*32+:32];
            if(!resetn)q<=0;else if(rd_en)q<=memory[rd_addr];
        end
        assign rd_data[lane*32+:32]=q;
    end
endmodule
