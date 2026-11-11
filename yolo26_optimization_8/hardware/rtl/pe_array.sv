// 8 input lanes x 16 output lanes (128 MACs). Registered 9x9 multipliers, pipelined
// reduction, 48-bit internal sums; INT32 output is checked, never wrapped.
module pe_array #(parameter integer CONTEXTS=4) (
    input logic clk, resetn, clear, in_valid, in_last,
    input logic [63:0] activation,
    input logic [1023:0] weights,
    input logic signed [7:0] input_zp, weight_zp,
    input logic [511:0] initial_sums,
    input logic in_first,
    input logic [2:0] in_context,
    input logic [8:0] in_address,
    output logic [8:0] result_address,
    output logic result_valid, overflow,
    output logic [511:0] result
);
    logic [4:0] valid_pipe, last_pipe;
    logic signed [8:0] a[0:7], w[0:15][0:7];
    (* use_dsp = "yes" *) logic signed [17:0] product[0:15][0:7];
    (* use_dsp = "no" *) logic signed [18:0] sum1[0:15][0:3];
    (* use_dsp = "no" *) logic signed [19:0] sum2[0:15][0:1];
    (* use_dsp = "no" *) logic signed [20:0] sum3[0:15];
    (* use_dsp = "no" *) logic signed [47:0] accumulator[0:CONTEXTS-1][0:15];
    logic [2:0] context_pipe[0:4];
    logic [8:0] address_pipe[0:4];
    logic [4:0] first_pipe;
    logic [511:0] seed_pipe[0:4];
    wire signed [47:0] next_sum[0:15];
    for (genvar o=0;o<16;o++) begin
        assign next_sum[o] = (first_pipe[4] ? {{16{seed_pipe[4][o*32+31]}},seed_pipe[4][o*32+:32]} : accumulator[context_pipe[4]][o])+{{27{sum3[o][20]}},sum3[o]};
    end
    always_ff @(posedge clk) begin
        if (!resetn) begin
            valid_pipe<=0;last_pipe<=0;first_pipe<=0;result_valid<=0;overflow<=0;result<=0;result_address<=0;
            for(int p=0;p<5;p++)begin context_pipe[p]<=0;address_pipe[p]<=0;seed_pipe[p]<=0;end
            for (int i=0;i<8;i++) a[i]<=0;
            for (int i=0;i<16;i++) begin
                for(int c=0;c<CONTEXTS;c++)accumulator[c][i]<=0;sum3[i]<=0;
                for (int j=0;j<8;j++) begin w[i][j]<=0;product[i][j]<=0;end
                for (int j=0;j<4;j++) sum1[i][j]<=0;
                for (int j=0;j<2;j++) sum2[i][j]<=0;
            end
        end else begin
            result_valid<=0;
            valid_pipe<={valid_pipe[3:0],in_valid};
            last_pipe<={last_pipe[3:0],in_last};
            first_pipe<={first_pipe[3:0],in_first};
            context_pipe[0]<=in_context;address_pipe[0]<=in_address;seed_pipe[0]<=initial_sums;
            for(int p=1;p<5;p++)begin context_pipe[p]<=context_pipe[p-1];address_pipe[p]<=address_pipe[p-1];seed_pipe[p]<=seed_pipe[p-1];end
            if (clear) begin
                valid_pipe<=0;last_pipe<=0;overflow<=0;
                for (int o=0;o<16;o++)
                    for(int c=0;c<CONTEXTS;c++)accumulator[c][o]<=0;
            end else begin
                if (in_valid) begin
                    for (int i=0;i<8;i++)
                        a[i]<=$signed({activation[i*8+7],activation[i*8+:8]})-
                              $signed({input_zp[7],input_zp});
                    for (int o=0;o<16;o++) for (int i=0;i<8;i++)
                        w[o][i]<=$signed({weights[o*64+i*8+7],weights[o*64+i*8+:8]})-
                                 $signed({weight_zp[7],weight_zp});
                end
                if (valid_pipe[0]) for (int o=0;o<16;o++) for (int i=0;i<8;i++)
                    product[o][i]<=a[i]*w[o][i];
                if (valid_pipe[1]) for (int o=0;o<16;o++) for (int j=0;j<4;j++)
                    sum1[o][j]<=$signed({product[o][2*j][17],product[o][2*j]})+
                                $signed({product[o][2*j+1][17],product[o][2*j+1]});
                if (valid_pipe[2]) for (int o=0;o<16;o++) for (int j=0;j<2;j++)
                    sum2[o][j]<=$signed({sum1[o][2*j][18],sum1[o][2*j]})+
                                $signed({sum1[o][2*j+1][18],sum1[o][2*j+1]});
                if (valid_pipe[3]) for (int o=0;o<16;o++)
                    sum3[o]<=$signed({sum2[o][0][19],sum2[o][0]})+
                             $signed({sum2[o][1][19],sum2[o][1]});
                if (valid_pipe[4]) begin
                    for (int o=0;o<16;o++) begin
                        accumulator[context_pipe[4]][o]<=next_sum[o];
                        if (last_pipe[4]) begin
                            result[o*32+:32]<=next_sum[o][31:0];
                            if (next_sum[o][47:31]!={17{next_sum[o][31]}}) overflow<=1;
                        end
                    end
                    if (last_pipe[4])begin result_valid<=1;result_address<=address_pipe[4];end
                end
            end
        end
    end
endmodule
