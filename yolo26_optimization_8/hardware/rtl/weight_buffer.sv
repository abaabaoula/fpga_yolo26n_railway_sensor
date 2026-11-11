// Same persistent Co32/Cin512/K3 capacity, banked by output-group parity.
// Two 8-output banks feed 16 outputs simultaneously without duplicating weights.
module weight_buffer (
    input logic clk,resetn,wr_en,wr_bank,rd_en,rd_bank,
    input logic [2:0] wr_lane,
    input logic [10:0] wr_addr,rd_addr,
    input logic [63:0] wr_data,
    output wire [1023:0] rd_data
);
    for(genvar half=0;half<2;half++) begin:half_bank
        for(genvar lane=0;lane<8;lane++) begin:lane_bank
            (* ram_style="block" *) logic [63:0] memory[0:511];
            logic [63:0] raw_read;
            (* dont_touch="true" *) logic [63:0] q;
            always_ff @(posedge clk) begin
                if(resetn && wr_en && wr_bank==half && wr_lane==lane)memory[wr_addr[8:0]]<=wr_data;
                if(!resetn)begin raw_read<=0;q<=0;end
                else begin if(rd_en)raw_read<=memory[rd_addr[8:0]];q<=raw_read;end
            end
            assign rd_data[(half*8+lane)*64+:64]=q;
        end
    end
endmodule
