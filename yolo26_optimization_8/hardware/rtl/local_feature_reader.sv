// Pipelined address generation. No variable division or modulo.
// A bounded FIFO keeps data stable across arbitrary controller backpressure.
module local_feature_reader (
    input logic clk,resetn,active,
    input logic [9:0] channel_base,
    input logic [15:0] words,
    input logic [9:0] channels,
    input logic [8:0] height,width,
    input logic [5:0] tile_h,tile_w,
    input logic [8:0] origin_y,origin_x,
    input logic pad,
    input logic [7:0] zero_point,
    output wire valid,
    output wire [31:0] data,
    input logic ready,
    output wire rd_en,
    output wire [14:0] rd_word,
    input logic [31:0] rd_data,
    output logic bounds_error,
    output wire request_fire,
    output wire padding_fire
);
    logic was_active;
    logic [15:0] issued,consumed;
    logic [5:0] x,y;
    logic [2:0] group_index;
    logic half_word;
    logic [4:0] outstanding;
    logic [3:0] wp,rp;
    logic [4:0] filled;
    logic [31:0] fifo[0:15];
    wire pop=valid&&ready;
    wire issue=active&&was_active&&issued<words&&outstanding<12;
    wire signed [10:0] sy=$signed({2'b0,origin_y})+$signed({5'b0,y})-$signed({10'b0,pad});
    wire signed [10:0] sx=$signed({2'b0,origin_x})+$signed({5'b0,x})-$signed({10'b0,pad});
    wire [9:0] channel=channel_base+{4'b0,group_index,3'b0};
    wire is_pad=sy<0||sx<0||sy>=$signed({2'b0,height})||sx>=$signed({2'b0,width})||channel>=channels;
    logic v0,v1,v2,v3,p0,p1,p2,p3;
    logic [7:0] z0,z1,z2,z3;
    logic half0,half1,half2;
    logic [4:0] cy0,cx0,ty0,tx0;
    logic [4:0] th0,tw0;
    logic [5:0] co0;
    logic [4:0] cb0;
    logic [1:0] cg0;
    (* use_dsp="no" *) logic [17:0] plane0;
    (* use_dsp="no" *) logic [13:0] row_stride0;
    (* use_dsp="no" *) logic [9:0] col_stride0;
    (* use_dsp="no" *) logic [8:0] group_stride0;
    wire [5:0] request_co=channels-{channel[9:5],5'b0}>=32?6'd32:channels-{channel[9:5],5'b0};
    wire [4:0] request_th=height-{sy[8:4],4'b0}>=16?5'd16:height-{sy[8:4],4'b0};
    wire [4:0] request_tw=width-{sx[8:4],4'b0}>=16?5'd16:width-{sx[8:4],4'b0};
    (* use_dsp="no" *) logic [17:0] term0,term1,term2,term3;
    (* use_dsp="no" *) logic [17:0] address2;
    wire push=v3;
    assign valid=active&&filled!=0;
    assign data=fifo[rp];
    assign rd_en=v2&&!p2;
    assign rd_word=address2[16:2];
    assign request_fire=issue;
    assign padding_fire=issue&&is_pad;
    always_ff @(posedge clk)begin
        if(!resetn||!active)begin
            was_active<=0;issued<=0;consumed<=0;x<=0;y<=0;group_index<=0;half_word<=0;
            outstanding<=0;wp<=0;rp<=0;filled<=0;v0<=0;v1<=0;v2<=0;v3<=0;bounds_error<=0;
        end else begin
            was_active<=1;
            v0<=issue;v1<=v0;v2<=v1;v3<=v2;
            if(issue)begin
                issued<=issued+1;p0<=is_pad;z0<=zero_point;half0<=half_word;
                cb0<=channel[9:5];cg0<=channel[4:3];
                co0<=request_co;
                plane0<=height*width;row_stride0<=width*request_co;
                col_stride0<=request_th*request_co;group_stride0<=request_th*request_tw;
                ty0<=sy[8:4];tx0<=sx[8:4];cy0<={1'b0,sy[3:0]};cx0<={1'b0,sx[3:0]};
                th0<=height-{sy[8:4],4'b0}>=16?5'd16:height-{sy[8:4],4'b0};
                tw0<=width-{sx[8:4],4'b0}>=16?5'd16:width-{sx[8:4],4'b0};
                half_word<=!half_word;
                if(half_word)begin
                    if(x==tile_w-1)begin x<=0;if(y==tile_h-1)begin y<=0;group_index<=group_index+1;end else y<=y+1;end
                    else x<=x+1;
                end
            end
            if(v0)begin
                p1<=p0;z1<=z0;half1<=half0;
                term0<=(cb0*plane0)<<5;
                term1<=(ty0*row_stride0)<<4;
                term2<=(tx0*col_stride0)<<4;
                term3<=(cg0*group_stride0+cy0*tw0+cx0)<<3;
            end
            if(v1)begin
                address2<=term0+term1+term2+term3+(half1?4:0);
                p2<=p1;z2<=z1;half2<=half1;
            end
            if(v2)begin
                p3<=p2;z3<=z2;
                if(!p2&&(address2>=131072||address2[1:0]!=0))bounds_error<=1;
            end
            if(push)begin fifo[wp]<=p3?{4{z3}}:rd_data;wp<=wp+1;end
            if(pop)begin rp<=rp+1;consumed<=consumed+1;end
            case({push,pop})2'b10:filled<=filled+1;2'b01:filled<=filled-1;default:begin end endcase
            case({issue,pop})2'b10:outstanding<=outstanding+1;2'b01:outstanding<=outstanding-1;default:begin end endcase
        end
    end
endmodule
