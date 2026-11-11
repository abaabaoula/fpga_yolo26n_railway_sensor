// O8: 128 KiB, compact Co32/tile16/group8 output byte layout.
// Two 32-bit banks provide full data capacity without using parity bits.
module local_feature_cache (
    input logic clk,
    input logic wr_en,
    input logic [14:0] wr_word,
    input logic [31:0] wr_data,
    input logic rd_en,
    input logic [14:0] rd_word,
    output logic [31:0] rd_data
);
    (* ram_style="block" *) logic [31:0] bank0[0:16383];
    (* ram_style="block" *) logic [31:0] bank1[0:16383];
    logic [31:0] q0,q1;
    logic select_bank;
    always_ff @(posedge clk) begin
        if(wr_en&&!wr_word[0])bank0[wr_word[14:1]]<=wr_data;
        if(wr_en&& wr_word[0])bank1[wr_word[14:1]]<=wr_data;
        if(rd_en&&!rd_word[0])q0<=bank0[rd_word[14:1]];
        if(rd_en&& rd_word[0])q1<=bank1[rd_word[14:1]];
        if(rd_en)select_bank<=rd_word[0];
    end
    assign rd_data=select_bank?q1:q0;
endmodule
