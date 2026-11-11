// Decoupled AW/W AXI-Lite slave; independent read and write response channels.
module config_registers (
    input logic clk,resetn,
    input logic [11:0] awaddr,araddr,
    input logic awvalid,wvalid,bready,arvalid,rready,
    input logic [31:0] wdata,
    input logic [3:0] wstrb,
    output wire awready,wready,arready,
    output logic bvalid,rvalid,
    output wire [1:0] bresp,rresp,
    output logic [31:0] rdata,
    output logic start,soft_reset,
    output logic [31:0] cfg[0:16],
    input logic busy,done,error,
    input logic [31:0] counters[0:10],
    input logic [31:0] state_counters[0:31],ddr_read_bytes,ddr_write_bytes,
    input logic [31:0] resident_counters[0:15],
    input logic [31:0] error_code
);
    logic aw_pending,w_pending;
    logic [11:0] saved_aw;
    logic [31:0] saved_w;
    logic [3:0] saved_strobe;
    assign awready=!aw_pending && !bvalid;
    assign wready=!w_pending && !bvalid;
    assign arready=!rvalid;
    assign bresp=0;assign rresp=0;
    always_ff @(posedge clk) begin
        if (!resetn) begin
            aw_pending<=0;w_pending<=0;bvalid<=0;rvalid<=0;rdata<=0;
            start<=0;soft_reset<=0;saved_aw<=0;saved_w<=0;saved_strobe<=0;
            for (int i=0;i<17;i++) cfg[i]<=0;
        end else begin
            start<=0;soft_reset<=0;
            if (bvalid && bready) bvalid<=0;
            if (rvalid && rready) rvalid<=0;
            if (awvalid && awready) begin aw_pending<=1;saved_aw<=awaddr;end
            if (wvalid && wready) begin w_pending<=1;saved_w<=wdata;saved_strobe<=wstrb;end
            if (aw_pending && w_pending && !bvalid) begin
                aw_pending<=0;w_pending<=0;bvalid<=1;
                if (saved_aw[7:0]==0) begin
                    if (saved_strobe[0]) begin start<=saved_w[0];soft_reset<=saved_w[4];end
                end else if (saved_aw>=12'h080 && saved_aw<12'h094 && saved_aw[1:0]==0) begin
                    for (int byte_index=0;byte_index<4;byte_index++)
                        if(saved_strobe[byte_index]) cfg[12+((saved_aw-12'h080)>>2)][byte_index*8+:8]<=saved_w[byte_index*8+:8];
                end else if (saved_aw>=12'h010 && saved_aw<12'h040 && saved_aw[1:0]==0) begin
                    for (int byte_index=0;byte_index<4;byte_index++)
                        if (saved_strobe[byte_index]) cfg[(saved_aw-12'h010)>>2][byte_index*8+:8]<=saved_w[byte_index*8+:8];
                end
            end
            if (arvalid && arready) begin
                rvalid<=1;
                if (araddr==0) rdata<={28'b0,error,busy,done,1'b0};
                else if (araddr==4) rdata<=32'h00080001;
                else if(araddr==12'h008)rdata<=32'h0008000f;
                else if(araddr==12'h00c)rdata<=32'd131072;
                else if(araddr>=12'h200&&araddr<12'h240&&araddr[1:0]==0)rdata<=resident_counters[(araddr-12'h200)>>2];
                else if (araddr>=12'h010 && araddr<12'h040 && araddr[1:0]==0)
                    rdata<=cfg[(araddr-12'h010)>>2];
                else if (araddr>=12'h040 && araddr<12'h064 && araddr[1:0]==0)
                    rdata<=counters[(araddr-12'h040)>>2];
                else if (araddr==12'h064) rdata<=error_code;
                else if (araddr==12'h068) rdata<=counters[9];
                else if (araddr==12'h06C) rdata<=counters[10];
                else if (araddr>=12'h080 && araddr<12'h094 && araddr[1:0]==0) rdata<=cfg[12+((araddr-12'h080)>>2)];
                else if(araddr>=12'h100&&araddr<12'h180&&araddr[1:0]==0)rdata<=state_counters[(araddr-12'h100)>>2];
                else if(araddr==12'h180)rdata<=ddr_read_bytes;
                else if(araddr==12'h184)rdata<=ddr_write_bytes;
                else rdata<=0;
            end
        end
    end
endmodule
