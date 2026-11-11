// Passive measurement at the PS HP0 64-bit AXI data interface.
// Read bytes count all accepted read-data lanes; write bytes count WSTRB.
module axi_traffic_monitor(input logic clk,resetn,clear,
    input logic read_valid,read_ready,write_valid,write_ready,
    input logic [7:0] write_strobe,
    output logic [31:0] read_bytes,write_bytes);
    logic [3:0] enabled_write_bytes;
    always_comb begin
        enabled_write_bytes=0;
        for(int i=0;i<8;i++)enabled_write_bytes=enabled_write_bytes+{3'b0,write_strobe[i]};
    end
    always_ff @(posedge clk)begin
        if(!resetn||clear)begin read_bytes<=0;write_bytes<=0;end
        else begin
            if(read_valid&&read_ready)read_bytes<=read_bytes+32'd8;
            if(write_valid&&write_ready)write_bytes<=write_bytes+{28'b0,enabled_write_bytes};
        end
    end
endmodule
