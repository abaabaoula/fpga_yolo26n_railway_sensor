module conv_controller(
    input logic clk,resetn,start,
    input logic [31:0] cfg[0:16],
    input logic [31:0] s_data,input logic [3:0] s_keep,input logic s_valid,s_last,
    output wire s_ready,
    output wire [31:0] m_data,output wire [3:0] m_keep,output wire m_valid,m_last,input logic m_ready,
    output logic busy,done,error,
    output logic [31:0] error_code,counters[0:10],
    output logic [31:0] state_counters[0:31],
    output wire local_activation_phase,
    output wire [9:0] local_channel_base,
    output wire [15:0] local_activation_words
);
    typedef enum logic [4:0] {IDLE,CONFIG_CHECK,WAIT_BANK,READ_PSUM,CLEAR_PE,ISSUE,WAIT_PE,
        STORE_PSUM,NEXT_PIXEL,NEXT_CHUNK,OUTPUT_SETUP,OUTPUT_READ,OUTPUT_WAIT,OUTPUT_SEND,OUTPUT_FINAL,
        ERROR_DRAIN,ERROR_SEND,ERROR_FINAL} state_t;
    typedef enum logic [3:0] {L_IDLE,L_BIAS,L_MULTIPLIER,L_SHIFT,L_LUT,L_WAIT_SLOT,L_SETUP,L_WEIGHTS,L_ACTIVATIONS,L_DONE} load_state_t;
    state_t state;load_state_t loader;
    logic [9:0] cin_total,remaining_cin,load_remaining;
    logic [5:0] cout_tile,hin,win,chunk_cin,load_cin,bank_cin[0:1];
    logic [4:0] oh,ow,ox,oy;
    logic [1:0] kernel,stride,kx,ky,out_group;
    logic [2:0] cin_groups,cout_groups,in_group,out_lane,load_groups;
    logic signed [7:0] izp,wzp,output_zp;
    logic [1:0] output_mode;
    logic overlap_enable,compute_bank,load_bank,first_chunk;
    logic [1:0] bank_ready;
    logic [6:0] global_cin_groups,chunk_group_base,load_group_base;
    logic reuse_weights,cache_valid;
    logic [31:0] requested_cache_tag,cache_tag;
    logic [9:0] cache_cin;
    logic [5:0] cache_cout;
    logic [1:0] cache_kernel,load_w_out_group,load_w_ky,load_w_kx;
    logic [2:0] load_w_in_group;
    (* use_dsp="no" *) logic [11:0] load_w_address;
    logic [3:0] kernel_area;
    logic [8:0] pixel_count;
    logic [11:0] plane_size;
    logic [31:0] expected_words;
    logic config_invalid;
    localparam integer CONTEXTS=4;
    logic [4:0] base_x,base_y;
    logic [8:0] batch_base;
    logic [3:0] batch_count,context_slot;
    wire [4:0] following_x=ox==ow-1?5'd0:ox+5'd1;
    wire [4:0] following_y=ox==ow-1?oy+5'd1:oy;
    wire final_context=context_slot==batch_count-1;
    wire final_batch=batch_base+batch_count>=pixel_count;
    logic [2:0] context_tags[0:3];
    logic [8:0] address_tags[0:3];
    logic [3:0] first_tags;
    logic [1:0] seed_output_group;
    logic [511:0] seed_delay[0:2];
    wire [8:0] result_address;
    logic [9:0] committed_results;
    wire [9:0] target_results=pixel_count*((cout_groups+1)>>1);
    wire [6:0] prior_h={2'b0,oh}-7'd1,prior_w={2'b0,ow}-7'd1;
    wire [6:0] required_hi=(stride==2?(prior_h<<1):prior_h)+kernel;
    wire [6:0] required_wi=(stride==2?(prior_w<<1):prior_w)+kernel;
    wire invalid=config_invalid || {1'b0,hin}<required_hi || {1'b0,win}<required_wi;
    logic [15:0] word_index,weight_words,input_words;
    logic [31:0] low_word;
    logic signed [31:0] bias[0:31];
    logic [30:0] multiplier[0:31];
    logic [5:0] quant_shift[0:31];
    assign local_activation_phase=loader==L_ACTIVATIONS;
    assign local_channel_base={load_group_base,3'b0};
    assign local_activation_words=input_words;
    wire load_receiving=loader==L_BIAS||loader==L_MULTIPLIER||loader==L_SHIFT||loader==L_LUT||loader==L_WEIGHTS||loader==L_ACTIVATIONS;
    assign s_ready=resetn && (state==ERROR_DRAIN || (load_receiving && state!=ERROR_SEND && state!=ERROR_FINAL));
    wire rx_fire=s_valid&&s_ready,tx_fire=m_valid&&m_ready;
    wire compute_active=state>=READ_PSUM&&state<=NEXT_CHUNK;
    wire loader_active=loader>=L_BIAS&&loader<=L_ACTIVATIONS;
    wire buf_rd=state==ISSUE;
    wire issue_last=kernel==1 ? in_group==cin_groups-1 : in_group==cin_groups-1&&ky==kernel-1&&kx==kernel-1;
    logic read_valid,read_last,address_parts_valid,address_parts_last,address_valid,address_last;
    logic [12:0] activation_channel_base,activation_row,activation_address;
    (* use_dsp="no" *) logic [11:0] weight_output_base,weight_kernel_offset,weight_address;
    logic memory_valid,memory_last;
    logic [63:0] delayed_activation;
    wire [5:0] yy=(stride==2?{oy[3:0],1'b0}:{1'b0,oy})+ky;
    wire [5:0] xx=(stride==2?{ox[3:0],1'b0}:{1'b0,ox})+kx;
    (* use_dsp="no" *) wire [8:0] p_addr=(out_group>>1)*pixel_count+oy*ow+ox;
    wire w_write=rx_fire&&loader==L_WEIGHTS&&word_index[0]&&state!=ERROR_DRAIN;
    wire a_write=rx_fire&&loader==L_ACTIVATIONS&&word_index[0]&&state!=ERROR_DRAIN;
    wire [63:0] read_activation;
    wire [1023:0] read_weights;
    wire [511:0] read_partial,result;
    wire result_valid,pe_overflow;
    logic [511:0] seed;
    always_comb begin
        seed=read_partial;
        if(first_chunk)for(int lane=0;lane<16;lane++)
            seed[lane*32+:32]=((seed_output_group*8)+lane<cout_tile)?bias[(seed_output_group*8)+lane]:32'b0;
    end
    input_buffer input_cache(.clk(clk),.resetn(resetn),.wr_en(a_write),.wr_bank(load_bank),
        .wr_addr(word_index[13:1]),.wr_data({s_data,low_word}),
        .rd_en(address_valid),.rd_bank(compute_bank),.rd_addr(activation_address),.rd_data(read_activation));
    weight_buffer weight_cache(.clk(clk),.resetn(resetn),.wr_en(w_write),.wr_bank(load_w_out_group[0]),
        .wr_lane(word_index[3:1]),.wr_addr(load_w_address[10:0]),.wr_data({s_data,low_word}),
        .rd_en(address_valid),.rd_bank(compute_bank),.rd_addr(weight_address[10:0]),.rd_data(read_weights));
    partial_sum_buffer partial_sums(.clk(clk),.resetn(resetn),
        .wr_en(result_valid),.rd_en(buf_rd||state==OUTPUT_READ),
        .wr_addr(result_address),.rd_addr(p_addr),.wr_data(result),.rd_data(read_partial));
    pe_array compute(.clk(clk),.resetn(resetn),.clear(state==CLEAR_PE),.in_valid(read_valid),.in_last(read_last),
        .activation(delayed_activation),.weights(read_weights),.input_zp(izp),.weight_zp(wzp),
        .initial_sums(seed_delay[2]),.in_first(first_tags[3]),.in_context(context_tags[3]),
        .in_address(address_tags[3]),.result_address(result_address),.result_valid(result_valid),.overflow(pe_overflow),.result(result));
    wire word_valid=state==OUTPUT_SEND;
    wire word_last=out_group==cout_groups-1&&oy==oh-1&&ox==ow-1&&
        (out_lane==7||out_group*8+out_lane+1>=cout_tile);
    wire [4:0] output_channel={out_group,out_lane};
    wire [31:0] word_data=read_partial[(out_group[0]*8+out_lane)*32+:32];
    wire debug_ready,rq_ready,byte_ready,rq_valid,rq_last;
    wire [7:0] rq_byte;
    wire [31:0] debug_data,packed_data;
    wire [3:0] debug_keep,packed_keep;
    wire debug_valid,debug_last,packed_valid,packed_last;
    wire error_response=state==ERROR_SEND||state==ERROR_FINAL;
    wire select_quant=output_mode!=0&&!error_response;
    wire word_ready=output_mode==0?debug_ready:rq_ready;
    requantizer requant(.clk(clk),.resetn(resetn),.in_valid(word_valid&&output_mode!=0),.in_last(word_last),
        .in_ready(rq_ready),.accumulator($signed(word_data)),.coefficient(multiplier[output_channel]),
        .shift(quant_shift[output_channel]),.zero_point(output_zp),.enable_lut(output_mode==2),
        .lut_write(rx_fire&&loader==L_LUT&&state!=ERROR_DRAIN),.lut_write_addr(word_index[5:0]),.lut_write_data(s_data),
        .out_valid(rq_valid),.out_last(rq_last),.out_ready(byte_ready),.out_data(rq_byte));
    axis_byte_packer byte_output(.clk(clk),.resetn(resetn),.in_valid(rq_valid),.in_last(rq_last),
        .in_data(rq_byte),.in_ready(byte_ready),.m_data(packed_data),.m_keep(packed_keep),
        .m_valid(packed_valid),.m_last(packed_last),.m_ready(m_ready&&select_quant));
    axis_output_adapter debug_output(.clk(clk),.resetn(resetn),
        .word_valid((word_valid&&output_mode==0)||state==ERROR_SEND),.word_ready(debug_ready),
        .word_data(error_response?(32'hBAD00000|error_code):word_data),.word_last(error_response||word_last),
        .m_axis_tdata(debug_data),.m_axis_tkeep(debug_keep),.m_axis_tvalid(debug_valid),.m_axis_tlast(debug_last),
        .m_axis_tready(m_ready&&!select_quant));
    assign m_data=select_quant?packed_data:debug_data;assign m_keep=select_quant?packed_keep:debug_keep;
    assign m_valid=select_quant?packed_valid:debug_valid;assign m_last=select_quant?packed_last:debug_last;

    always_ff @(posedge clk) begin
        if(!resetn) begin
            state<=IDLE;loader<=L_IDLE;busy<=0;done<=0;error<=0;error_code<=0;
            cin_total<=0;remaining_cin<=0;load_remaining<=0;cout_tile<=0;hin<=0;win<=0;chunk_cin<=0;load_cin<=0;
            bank_cin[0]<=0;bank_cin[1]<=0;bank_ready<=0;compute_bank<=0;load_bank<=0;
            oh<=0;ow<=0;ox<=0;oy<=0;kernel<=0;stride<=0;kx<=0;ky<=0;out_group<=0;
            cin_groups<=0;cout_groups<=0;in_group<=0;out_lane<=0;load_groups<=0;
            izp<=0;wzp<=0;output_zp<=0;output_mode<=0;overlap_enable<=0;kernel_area<=0;first_chunk<=0;
            pixel_count<=0;plane_size<=0;expected_words<=0;config_invalid<=0;
            word_index<=0;weight_words<=0;input_words<=0;low_word<=0;
            read_valid<=0;read_last<=0;address_parts_valid<=0;address_parts_last<=0;address_valid<=0;address_last<=0;
            activation_channel_base<=0;activation_row<=0;activation_address<=0;
            weight_output_base<=0;weight_kernel_offset<=0;weight_address<=0;
            global_cin_groups<=0;chunk_group_base<=0;load_group_base<=0;
            reuse_weights<=0;cache_valid<=0;requested_cache_tag<=0;cache_tag<=0;cache_cin<=0;cache_cout<=0;cache_kernel<=0;
            load_w_out_group<=0;load_w_in_group<=0;load_w_ky<=0;load_w_kx<=0;load_w_address<=0;
            memory_valid<=0;memory_last<=0;delayed_activation<=0;
            seed_output_group<=0;base_x<=0;base_y<=0;batch_base<=0;batch_count<=0;context_slot<=0;first_tags<=0;committed_results<=0;
            for(int p=0;p<4;p++)begin context_tags[p]<=0;address_tags[p]<=0;end
            for(int p=0;p<3;p++)seed_delay[p]<=0;
            for(int i=0;i<11;i++)counters[i]<=0;
            for(int i=0;i<32;i++)state_counters[i]<=0;
            for(int i=0;i<32;i++)begin bias[i]<=0;multiplier[i]<=0;quant_shift[i]<=0;end
        end else begin
            seed_output_group<=out_group;context_tags[0]<=context_slot[2:0];address_tags[0]<=p_addr;
            first_tags<={first_tags[2:0],buf_rd&&in_group==0&&ky==0&&kx==0};
            for(int p=1;p<4;p++)begin context_tags[p]<=context_tags[p-1];address_tags[p]<=address_tags[p-1];end
            seed_delay[0]<=seed;seed_delay[1]<=seed_delay[0];seed_delay[2]<=seed_delay[1];
            if(result_valid)committed_results<=committed_results+1;
            address_parts_valid<=buf_rd;address_parts_last<=buf_rd&&issue_last;
            if(buf_rd)begin
                activation_channel_base<=in_group*plane_size;activation_row<=yy*win+xx;
                weight_output_base<=kernel==1 ? (out_group>>1)*global_cin_groups : (out_group>>1)*global_cin_groups*kernel_area;
                weight_kernel_offset<=kernel==1 ? chunk_group_base+in_group : (chunk_group_base+in_group)*kernel_area+ky*kernel+kx;
            end
            address_valid<=address_parts_valid;address_last<=address_parts_last;
            if(address_parts_valid)begin
                activation_address<=activation_channel_base+activation_row;
                weight_address<=weight_output_base+weight_kernel_offset;
            end
            memory_valid<=address_valid;memory_last<=address_valid&&address_last;
            read_valid<=memory_valid;read_last<=memory_valid&&memory_last;
            if(memory_valid)delayed_activation<=read_activation;
            load_w_address<=(load_w_out_group>>1)*global_cin_groups*kernel_area+
                (load_group_base+load_w_in_group)*kernel_area+load_w_ky*kernel+load_w_kx;
            if(busy)begin
                for(int i=0;i<18;i++)if(state==i)state_counters[i]<=state_counters[i]+1;
                for(int i=0;i<10;i++)if(loader==i)state_counters[18+i]<=state_counters[18+i]+1;
                if(s_ready&&!s_valid)state_counters[28]<=state_counters[28]+1;
                if(m_valid&&!m_ready)state_counters[29]<=state_counters[29]+1;
                if(state==WAIT_BANK&&!bank_ready[compute_bank])state_counters[30]<=state_counters[30]+1;
                if(result_valid)state_counters[31]<=state_counters[31]+1;
                counters[0]<=counters[0]+1;
                if(loader==L_BIAS||loader==L_MULTIPLIER||loader==L_SHIFT||loader==L_LUT||loader==L_SETUP||loader==L_WEIGHTS)counters[1]<=counters[1]+1;
                if(loader==L_ACTIVATIONS)counters[2]<=counters[2]+1;
                if(compute_active)counters[3]<=counters[3]+1;
                if(state>=OUTPUT_SETUP&&state<=OUTPUT_FINAL)counters[4]<=counters[4]+1;
                if(buf_rd)counters[5]<=counters[5]+1;
                if(rx_fire)counters[6]<=counters[6]+1;
                if(tx_fire)counters[7]<=counters[7]+1;
                if(compute_active&&load_receiving)counters[9]<=counters[9]+1;
                if(state==WAIT_BANK&&!bank_ready[compute_bank])counters[10]<=counters[10]+1;
            end
            // Producer FSM owns DMA writes. A ready bank remains occupied until
            // its consumer finishes all output pixels for that Cin chunk.
            case(loader)
                L_BIAS:if(rx_fire)begin
                    bias[word_index[4:0]]<=s_data;word_index<=word_index+1;
                    if(word_index==cout_tile-1)begin word_index<=0;loader<=output_mode==0?L_WAIT_SLOT:L_MULTIPLIER;end
                end
                L_MULTIPLIER:if(rx_fire)begin
                    multiplier[word_index[4:0]]<=s_data[30:0];word_index<=word_index+1;
                    if(word_index==cout_tile-1)begin word_index<=0;loader<=L_SHIFT;end
                end
                L_SHIFT:if(rx_fire)begin
                    quant_shift[word_index[4:0]]<=s_data[5:0];word_index<=word_index+1;
                    if(word_index==cout_tile-1)begin word_index<=0;loader<=output_mode==2?L_LUT:L_WAIT_SLOT;end
                end
                L_LUT:if(rx_fire)begin
                    word_index<=word_index+1;if(word_index==63)begin word_index<=0;loader<=L_WAIT_SLOT;end
                end
                L_WAIT_SLOT:if(!bank_ready[load_bank]&&(overlap_enable||(state==WAIT_BANK&&compute_bank==load_bank)))begin
                    load_cin<=load_remaining>32?6'd32:load_remaining[5:0];
                    load_groups<=load_remaining>32?3'd4:(load_remaining+7)>>3;loader<=L_SETUP;
                end
                L_SETUP:begin
                    weight_words<=cout_groups*load_groups*kernel_area*16;input_words<=load_groups*plane_size*2;
                    word_index<=0;loader<=reuse_weights?L_ACTIVATIONS:L_WEIGHTS;
                    load_w_out_group<=0;load_w_in_group<=0;load_w_ky<=0;load_w_kx<=0;
                end
                L_WEIGHTS:if(rx_fire)begin
                    if(!word_index[0])low_word<=s_data;word_index<=word_index+1;
                    if(word_index[3:0]==15)begin
                        if(load_w_kx==kernel-1)begin
                            load_w_kx<=0;
                            if(load_w_ky==kernel-1)begin
                                load_w_ky<=0;
                                if(load_w_in_group==load_groups-1)begin load_w_in_group<=0;load_w_out_group<=load_w_out_group+1;end
                                else load_w_in_group<=load_w_in_group+1;
                            end else load_w_ky<=load_w_ky+1;
                        end else load_w_kx<=load_w_kx+1;
                    end
                    if(word_index==weight_words-1)begin word_index<=0;loader<=L_ACTIVATIONS;end
                end
                L_ACTIVATIONS:if(rx_fire)begin
                    if(!word_index[0])low_word<=s_data;word_index<=word_index+1;
                    if(word_index==input_words-1)begin
                        bank_ready[load_bank]<=1;bank_cin[load_bank]<=load_cin;word_index<=0;
                        load_remaining<=load_remaining-load_cin;load_group_base<=load_group_base+load_groups;
                        if(load_remaining>load_cin)begin load_bank<=!load_bank;loader<=L_WAIT_SLOT;end
                        else begin
                            loader<=L_DONE;
                            if(!reuse_weights)begin cache_valid<=1;cache_tag<=requested_cache_tag;cache_cin<=cin_total;cache_cout<=cout_tile;cache_kernel<=kernel;end
                        end
                    end
                end
                default:begin end
            endcase
            // Consumer FSM owns PE/partial sums and final output.
            case(state)
                IDLE:if(start)begin
                    busy<=1;done<=0;error<=0;error_code<=0;
                    for(int i=0;i<32;i++)state_counters[i]<=0;
                    for(int i=0;i<11;i++)if(i!=8)counters[i]<=0;
                    counters[8]<=counters[8]+1;
                    cin_total<=cfg[0][9:0];remaining_cin<=cfg[0][9:0];load_remaining<=cfg[0][9:0];
                    cout_tile<=cfg[1][5:0];hin<=cfg[2][5:0];win<=cfg[3][5:0];
                    oh<=cfg[4][4:0];ow<=cfg[5][4:0];kernel<=cfg[6][1:0];stride<=cfg[7][1:0];
                    izp<=cfg[8][7:0];wzp<=cfg[9][7:0];expected_words<=cfg[10];output_zp<=cfg[12][7:0];
                    output_mode<=cfg[13][1:0];overlap_enable<=cfg[14][0];cout_groups<=(cfg[1]+7)>>3;
                    pixel_count<=cfg[4][4:0]*cfg[5][4:0];plane_size<=cfg[2][5:0]*cfg[3][5:0];
                    kernel_area<=cfg[6]==3?4'd9:4'd1;first_chunk<=1;compute_bank<=0;load_bank<=0;bank_ready<=0;
                    loader<=L_IDLE;word_index<=0;state<=CONFIG_CHECK;
                    global_cin_groups<=(cfg[0]+7)>>3;chunk_group_base<=0;load_group_base<=0;
                    reuse_weights<=cfg[15][0];requested_cache_tag<=cfg[16];
                    if(!cfg[15][0])cache_valid<=0;
                    config_invalid<=cfg[0]==0||cfg[0]>512||cfg[1]==0||cfg[1]>32||cfg[2]==0||cfg[2]>33||cfg[3]==0||cfg[3]>33||
                        cfg[4]==0||cfg[4]>16||cfg[5]==0||cfg[5]>16||(cfg[6]!=1&&cfg[6]!=3)||(cfg[7]!=1&&cfg[7]!=2)||
                        (cfg[6]==3&&((cfg[1]>16&&cfg[0]>224)||cfg[0]>448))||cfg[10]==0||cfg[11]>1||cfg[13]>2||cfg[14]>1||cfg[15]>1;
                end
                CONFIG_CHECK:begin
                    if(invalid)begin error<=1;error_code<=1;state<=ERROR_DRAIN;end
                    else if(reuse_weights&&(!cache_valid||cache_tag!=requested_cache_tag||cache_cin!=cin_total||cache_cout!=cout_tile||cache_kernel!=kernel))begin
                        error<=1;error_code<=6;state<=ERROR_DRAIN;
                    end else begin loader<=L_BIAS;state<=WAIT_BANK;end
                end
                WAIT_BANK:if(bank_ready[compute_bank])begin
                    chunk_cin<=bank_cin[compute_bank];cin_groups<=(bank_cin[compute_bank]+7)>>3;
                    ox<=0;oy<=0;out_group<=0;base_x<=0;base_y<=0;batch_base<=0;
                    batch_count<=pixel_count<CONTEXTS?pixel_count:CONTEXTS;
                    context_slot<=0;committed_results<=0;state<=READ_PSUM;
                end
                READ_PSUM:state<=CLEAR_PE;
                CLEAR_PE:begin in_group<=0;kx<=0;ky<=0;state<=ISSUE;end
                ISSUE:begin
                    if(!final_context)begin
                        context_slot<=context_slot+1;ox<=following_x;oy<=following_y;
                    end else begin
                        context_slot<=0;
                        if(issue_last)begin
                            in_group<=0;kx<=0;ky<=0;
                            if(!final_batch)begin
                                batch_base<=batch_base+batch_count;
                                batch_count<=pixel_count-batch_base-batch_count<CONTEXTS?pixel_count-batch_base-batch_count:CONTEXTS;
                                base_x<=following_x;base_y<=following_y;ox<=following_x;oy<=following_y;
                            end else if({1'b0,out_group}+3'd2<cout_groups)begin
                                out_group<=out_group+2;ox<=0;oy<=0;base_x<=0;base_y<=0;batch_base<=0;
                                batch_count<=pixel_count<CONTEXTS?pixel_count:CONTEXTS;
                            end else state<=WAIT_PE;
                        end else begin
                            ox<=base_x;oy<=base_y;
                            if(kernel==1)in_group<=in_group+1;
                            else if(kx==kernel-1)begin
                                kx<=0;if(ky==kernel-1)begin ky<=0;in_group<=in_group+1;end else ky<=ky+1;
                            end else kx<=kx+1;
                        end
                    end
                end
                WAIT_PE:if(committed_results==target_results)state<=NEXT_CHUNK;
                STORE_PSUM:state<=NEXT_CHUNK;
                NEXT_PIXEL:state<=NEXT_CHUNK;
                NEXT_CHUNK:begin
                    bank_ready[compute_bank]<=0;
                    if(remaining_cin>chunk_cin)begin
                        remaining_cin<=remaining_cin-chunk_cin;chunk_group_base<=chunk_group_base+cin_groups;compute_bank<=!compute_bank;first_chunk<=0;state<=WAIT_BANK;
                    end else state<=OUTPUT_SETUP;
                end
                OUTPUT_SETUP:begin out_group<=0;ox<=0;oy<=0;out_lane<=0;state<=OUTPUT_READ;end
                OUTPUT_READ:state<=OUTPUT_WAIT;
                OUTPUT_WAIT:state<=OUTPUT_SEND;
                OUTPUT_SEND:if(word_ready)begin
                    if(word_last)state<=OUTPUT_FINAL;
                    else if(out_lane==7||out_group*8+out_lane+1>=cout_tile)begin
                        out_lane<=0;state<=OUTPUT_READ;
                        if(ox<ow-1)ox<=ox+1;
                        else begin ox<=0;if(oy<oh-1)oy<=oy+1;else begin oy<=0;out_group<=out_group+1;end end
                    end else out_lane<=out_lane+1;
                end
                OUTPUT_FINAL:if(tx_fire&&m_last)begin state<=IDLE;busy<=0;done<=1;end
                ERROR_DRAIN:if(rx_fire&&s_last)state<=ERROR_SEND;
                ERROR_SEND:if(debug_ready)state<=ERROR_FINAL;
                ERROR_FINAL:if(tx_fire&&m_last)begin state<=IDLE;loader<=L_IDLE;busy<=0;done<=1;end
                default:state<=IDLE;
            endcase
            if(result_valid&&pe_overflow&&compute_active)begin
                error<=1;error_code<=4;cache_valid<=0;loader<=L_IDLE;
                state<=(loader==L_DONE||counters[6]==expected_words)?ERROR_SEND:ERROR_DRAIN;
            end
            if(rx_fire&&state!=ERROR_DRAIN&&state!=ERROR_SEND&&state!=ERROR_FINAL)begin
                if(s_keep!=4'hf)begin error<=1;error_code<=2;cache_valid<=0;loader<=L_IDLE;state<=s_last?ERROR_SEND:ERROR_DRAIN;end
                else if(s_last!=(counters[6]+1==expected_words))begin error<=1;error_code<=3;cache_valid<=0;loader<=L_IDLE;state<=s_last?ERROR_SEND:ERROR_DRAIN;end
                else if((loader==L_MULTIPLIER&&s_data[31])||(loader==L_SHIFT&&s_data[31:6]!=0))begin
                    error<=1;error_code<=5;cache_valid<=0;loader<=L_IDLE;state<=s_last?ERROR_SEND:ERROR_DRAIN;
                end
            end
        end
    end
endmodule
