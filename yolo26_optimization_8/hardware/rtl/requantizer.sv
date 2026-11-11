// Four explicitly pipelined DSP products implement a signed32 x positive31
// multiply. Rounding is symmetric nearest/ties-even; clamp precedes optional LUT.
module requantizer(
    input logic clk,resetn,
    input logic in_valid,in_last,
    output wire in_ready,
    input logic signed [31:0] accumulator,
    input logic [30:0] coefficient,
    input logic [5:0] shift,
    input logic signed [7:0] zero_point,
    input logic enable_lut,
    input logic lut_write,
    input logic [5:0] lut_write_addr,
    input logic [31:0] lut_write_data,
    output wire out_valid,out_last,
    input logic out_ready,
    output logic [7:0] out_data
);
    logic [7:0] valid_pipe,last_pipe;
    logic [5:0] shift_pipe[0:4];
    logic signed [31:0] acc_reg;
    logic [30:0] coefficient_reg;
    (* use_dsp="yes" *) logic [31:0] p0,p1;
    (* use_dsp="yes" *) logic signed [32:0] p2,p3;
    logic signed [33:0] cross_sum;
    logic signed [63:0] base,product;
    logic [63:0] magnitude;
    logic negative;
    logic [63:0] quotient_reg,remainder_reg,half_reg;
    logic round_negative,shift_zero;
    logic [7:0] quantized;
    (* ram_style="distributed" *) logic [7:0] lut0[0:63],lut1[0:63],lut2[0:63],lut3[0:63];
    wire advance=!valid_pipe[7] || out_ready;
    assign in_ready=advance;
    assign out_valid=valid_pipe[7];assign out_last=last_pipe[7];
    wire increment=!shift_zero && (remainder_reg>half_reg || (remainder_reg==half_reg && quotient_reg[0]));
    wire [8:0] rounded_small={1'b0,quotient_reg[7:0]}+increment;
    wire signed [9:0] signed_small=round_negative ? -$signed({1'b0,rounded_small}):$signed({1'b0,rounded_small});
    wire signed [9:0] offset_result=signed_small+{{2{zero_point[7]}},zero_point};
    wire saturated_magnitude=|quotient_reg[63:8];
    always_ff @(posedge clk) begin
        if(lut_write) begin
            lut0[lut_write_addr]<=lut_write_data[7:0];lut1[lut_write_addr]<=lut_write_data[15:8];
            lut2[lut_write_addr]<=lut_write_data[23:16];lut3[lut_write_addr]<=lut_write_data[31:24];
        end
        if(!resetn) begin
            valid_pipe<=0;last_pipe<=0;out_data<=0;p0<=0;p1<=0;p2<=0;p3<=0;
            cross_sum<=0;base<=0;product<=0;magnitude<=0;negative<=0;quantized<=0;
            acc_reg<=0;coefficient_reg<=0;quotient_reg<=0;remainder_reg<=0;half_reg<=0;round_negative<=0;shift_zero<=0;
            for(int i=0;i<5;i++)shift_pipe[i]<=0;
        end else if(advance) begin
            valid_pipe<={valid_pipe[6:0],in_valid};last_pipe<={last_pipe[6:0],in_last};
            if(in_valid) begin
                acc_reg<=accumulator;coefficient_reg<=coefficient;shift_pipe[0]<=shift;
            end
            if(valid_pipe[0]) begin
                p0<=acc_reg[15:0]*coefficient_reg[15:0];
                p1<=acc_reg[15:0]*{1'b0,coefficient_reg[30:16]};
                p2<=$signed(acc_reg[31:16])*$signed({1'b0,coefficient_reg[15:0]});
                p3<=$signed(acc_reg[31:16])*$signed({2'b0,coefficient_reg[30:16]});
                shift_pipe[1]<=shift_pipe[0];
            end
            if(valid_pipe[1]) begin
                cross_sum<=$signed({2'b0,p1})+$signed({p2[32],p2});
                base<=($signed({{31{p3[32]}},p3})<<<32)+$signed({32'b0,p0});
                shift_pipe[2]<=shift_pipe[1];
            end
            if(valid_pipe[2]) begin
                product<=base+($signed({{30{cross_sum[33]}},cross_sum})<<<16);
                shift_pipe[3]<=shift_pipe[2];
            end
            if(valid_pipe[3]) begin
                negative<=product[63];magnitude<=product[63] ? (~product+64'd1):product;
                shift_pipe[4]<=shift_pipe[3];
            end
            if(valid_pipe[4]) begin
                quotient_reg<=magnitude>>shift_pipe[4];
                remainder_reg<=magnitude & ((64'd1<<shift_pipe[4])-64'd1);
                half_reg<=(shift_pipe[4]==0) ? 64'd0:(64'd1<<(shift_pipe[4]-1));
                shift_zero<=shift_pipe[4]==0;round_negative<=negative;
            end
            if(valid_pipe[5]) begin
                quantized<=saturated_magnitude ? (round_negative ? 8'd128:8'd127):
                           offset_result>127 ? 8'd127:offset_result< -128 ? 8'd128:offset_result[7:0];
            end
            if(valid_pipe[6]) begin
                if(!enable_lut) out_data<=quantized;
                else case(quantized[1:0])
                    0:out_data<=lut0[quantized[7:2]];1:out_data<=lut1[quantized[7:2]];
                    2:out_data<=lut2[quantized[7:2]];3:out_data<=lut3[quantized[7:2]];
                endcase
            end
        end
    end
endmodule
