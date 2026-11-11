module optimization_4_queue_top (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 aclk CLK" *)
    (* X_INTERFACE_PARAMETER = "FREQ_HZ 100000000, ASSOCIATED_BUSIF S_AXI:S_AXIS:M_AXIS, ASSOCIATED_RESET aresetn" *)
    input logic aclk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 aresetn RST" *)
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input logic aresetn,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TDATA" *)
    input logic [31:0] s_axis_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TKEEP" *)
    input logic [3:0] s_axis_tkeep,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TVALID" *)
    input logic  s_axis_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TREADY" *)
    output wire  s_axis_tready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TLAST" *)
    input logic  s_axis_tlast,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TDATA" *)
    output wire [31:0] m_axis_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TKEEP" *)
    output wire [3:0] m_axis_tkeep,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TVALID" *)
    output wire  m_axis_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TREADY" *)
    input logic  m_axis_tready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TLAST" *)
    output wire  m_axis_tlast,
    (* X_INTERFACE_PARAMETER = "PROTOCOL AXI4LITE, DATA_WIDTH 32, ADDR_WIDTH 12" *)
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWADDR" *)
    input logic [11:0] s_axi_awaddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWPROT" *)
    input logic [2:0] s_axi_awprot,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWVALID" *)
    input logic  s_axi_awvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWREADY" *)
    output wire  s_axi_awready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WDATA" *)
    input logic [31:0] s_axi_wdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WSTRB" *)
    input logic [3:0] s_axi_wstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WVALID" *)
    input logic  s_axi_wvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WREADY" *)
    output wire  s_axi_wready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BRESP" *)
    output wire [1:0] s_axi_bresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BVALID" *)
    output wire  s_axi_bvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BREADY" *)
    input logic  s_axi_bready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARADDR" *)
    input logic [11:0] s_axi_araddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARPROT" *)
    input logic [2:0] s_axi_arprot,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARVALID" *)
    input logic  s_axi_arvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARREADY" *)
    output wire  s_axi_arready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RDATA" *)
    output wire [31:0] s_axi_rdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RRESP" *)
    output wire [1:0] s_axi_rresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RVALID" *)
    output wire  s_axi_rvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RREADY" *)
    input logic  s_axi_rready,
    input logic [31:0] ddr_read_bytes,ddr_write_bytes,
    output wire traffic_clear
);


    // QCO4 legacy descriptors remain valid for C/D same-firmware controls.
    // QCO8 has 28 header words: original cfg + external count + tensor metadata.
    wire start,soft_reset;
    wire local_resetn=aresetn&&!soft_reset;
    logic busy,done,error;
    logic [31:0] error_code,counters[0:10],state_counters[0:31],resident_counters[0:15],cfg[0:16];
    wire [31:0] cfg_unused[0:16],cc[0:10],cs[0:31],ce_code;
    wire cb,cd,ce,cs_ready,cm_valid,cm_last;
    wire [31:0] cm_data;wire [3:0] cm_keep;
    typedef enum logic[3:0]{Q_IDLE,Q_HEADER,Q_META_PLANE,Q_META_BYTES,Q_VALIDATE,Q_START,Q_PAYLOAD,Q_WAIT,Q_DRAIN,Q_ERROR_WORD,Q_FINISH}qstate_t;
    qstate_t qs;
    logic [5:0] header_index;
    logic extended,input_ended,abort_after_drain;
    logic [31:0] flags,remaining,external_remaining,external_words;
    logic [31:0] tensor_id,tensor_c,tensor_h,tensor_w,origin_y,origin_x,tensor_bytes,frame_id;
    // Full-width bounds are checked independently. Bounded multiplication is
    // split into clock stages, never a two-multiply path into queue control.
    logic [17:0] checked_plane;
    logic [27:0] checked_bytes;
    logic cache_valid;
    logic [31:0] cache_id,cache_c,cache_h,cache_w,cache_bytes,cache_frame,write_words;
    wire local_in=extended&&flags[1],local_out=extended&&flags[2],mirror=extended&&flags[6];
    wire final_tile=flags[0];
    wire c_resetn=local_resetn&&qs!=Q_DRAIN&&qs!=Q_ERROR_WORD;
    wire local_phase;wire [9:0] local_base;wire [15:0] local_words;
    wire reader_valid,reader_bounds,reader_request,reader_padding;
    wire [31:0] reader_data,cache_data;wire [14:0] cache_read_word;
    wire cache_read_en;
    wire reader_active=qs==Q_PAYLOAD&&local_in&&local_phase;
    wire use_reader=reader_active;
    wire ctrl_valid=qs==Q_PAYLOAD&&(use_reader?reader_valid:s_axis_tvalid);
    wire [31:0]ctrl_data=use_reader?reader_data:s_axis_tdata;
    wire [3:0]ctrl_keep=use_reader?4'hf:s_axis_tkeep;
    wire ctrl_fire=ctrl_valid&&cs_ready;
    wire sink_ready=(local_out&&!mirror&&!ce)?1'b1:m_axis_tready;
    wire sink_fire=qs==Q_WAIT&&cm_valid&&sink_ready;
    wire cache_write=sink_fire&&local_out&&!ce&&write_words<32768&&write_words<cache_bytes/4;
    wire in_fire=s_axis_tvalid&&s_axis_tready;
    wire out_fire=m_axis_tvalid&&m_axis_tready;
    assign traffic_clear=start;
    assign s_axis_tready=(qs==Q_HEADER||qs==Q_DRAIN)?1'b1:qs==Q_PAYLOAD&&!use_reader?cs_ready:1'b0;
    assign m_axis_tvalid=qs==Q_ERROR_WORD?1'b1:qs==Q_WAIT&&(!local_out||mirror||ce)?cm_valid:1'b0;
    assign m_axis_tdata=qs==Q_ERROR_WORD?32'hBAD00801:cm_data;
    assign m_axis_tkeep=qs==Q_ERROR_WORD?4'hf:cm_keep;
    assign m_axis_tlast=qs==Q_ERROR_WORD?1'b1:cm_last&&(final_tile||ce);
    local_feature_cache feature_cache(.clk(aclk),.wr_en(cache_write),.wr_word(write_words[14:0]),.wr_data(cm_data),.rd_en(cache_read_en),.rd_word(cache_read_word),.rd_data(cache_data));
    local_feature_reader reader(.clk(aclk),.resetn(c_resetn),.active(reader_active),.channel_base(local_base),.words(local_words),
        .channels(cache_c[9:0]),.height(cache_h[8:0]),.width(cache_w[8:0]),.tile_h(cfg[2][5:0]),.tile_w(cfg[3][5:0]),
        .origin_y(origin_y[8:0]),.origin_x(origin_x[8:0]),.pad(cfg[11][0]),.zero_point(cfg[8][7:0]),
        .valid(reader_valid),.data(reader_data),.ready(cs_ready),.rd_en(cache_read_en),.rd_word(cache_read_word),.rd_data(cache_data),
        .bounds_error(reader_bounds),.request_fire(reader_request),.padding_fire(reader_padding));
    config_registers registers(.clk(aclk),.resetn(aresetn),.awaddr(s_axi_awaddr),.awvalid(s_axi_awvalid),.awready(s_axi_awready),
        .wdata(s_axi_wdata),.wstrb(s_axi_wstrb),.wvalid(s_axi_wvalid),.wready(s_axi_wready),.bresp(s_axi_bresp),.bvalid(s_axi_bvalid),.bready(s_axi_bready),
        .araddr(s_axi_araddr),.arvalid(s_axi_arvalid),.arready(s_axi_arready),.rdata(s_axi_rdata),.rresp(s_axi_rresp),.rvalid(s_axi_rvalid),.rready(s_axi_rready),
        .start(start),.soft_reset(soft_reset),.cfg(cfg_unused),.busy(busy),.done(done),.error(error),.counters(counters),.state_counters(state_counters),
        .resident_counters(resident_counters),.ddr_read_bytes(ddr_read_bytes),.ddr_write_bytes(ddr_write_bytes),.error_code(error_code));
    conv_controller controller(.clk(aclk),.resetn(c_resetn),.start(qs==Q_START),.cfg(cfg),
        .s_data(ctrl_data),.s_keep(ctrl_keep),.s_valid(ctrl_valid),.s_last(local_in&&ce?external_remaining==1:remaining==1),.s_ready(cs_ready),
        .m_data(cm_data),.m_keep(cm_keep),.m_valid(cm_valid),.m_last(cm_last),.m_ready(sink_ready&&qs==Q_WAIT),
        .busy(cb),.done(cd),.error(ce),.error_code(ce_code),.counters(cc),.state_counters(cs),
        .local_activation_phase(local_phase),.local_channel_base(local_base),.local_activation_words(local_words));
    wire bad_metadata=extended&&(
        flags[31:7]!=0||flags[1]&&flags[2]||flags[3]&&!flags[2]||flags[4]&&!flags[2]||flags[6]&&!flags[2]||flags[5]||
        external_words==0||external_words>cfg[10]||
        ((local_in||local_out)&&(tensor_c==0||tensor_c>512||tensor_c[2:0]!=0||tensor_h==0||tensor_h>256||tensor_w==0||tensor_w>256||
          tensor_bytes!={4'b0,checked_bytes}||tensor_bytes>131072||tensor_bytes[1:0]!=0))||
        (local_in&&(!cache_valid||cache_id!=tensor_id||cache_frame!=frame_id||cache_c!=tensor_c||cache_h!=tensor_h||cache_w!=tensor_w||cache_bytes!=tensor_bytes||cfg[0]!=tensor_c||origin_y>511||origin_x>511))||
        (local_out&&!flags[3]&&(cache_id!=tensor_id||cache_frame!=frame_id||cache_bytes!=tensor_bytes))||
        (local_out&&flags[3]&&cache_valid)||
        (!local_in&&external_words!=cfg[10]));
    always_ff @(posedge aclk)begin
        if(!local_resetn)begin
            qs<=Q_IDLE;busy<=0;done<=0;error<=0;error_code<=0;header_index<=0;extended<=0;flags<=0;
            input_ended<=0;abort_after_drain<=0;remaining<=0;external_remaining<=0;external_words<=0;
            tensor_id<=0;tensor_c<=0;tensor_h<=0;tensor_w<=0;origin_y<=0;origin_x<=0;tensor_bytes<=0;frame_id<=0;checked_plane<=0;checked_bytes<=0;
            cache_valid<=0;cache_id<=0;cache_c<=0;cache_h<=0;cache_w<=0;cache_bytes<=0;cache_frame<=0;write_words<=0;
            for(int i=0;i<17;i++)cfg[i]<=0;
            for(int i=0;i<11;i++)counters[i]<=0;
            for(int i=0;i<32;i++)state_counters[i]<=0;
            for(int i=0;i<16;i++)resident_counters[i]<=0;
        end else begin
            if(busy)counters[0]<=counters[0]+1;
            if(in_fire)counters[6]<=counters[6]+1;
            if(out_fire)counters[7]<=counters[7]+1;
            if(cache_write)begin write_words<=write_words+1;resident_counters[0]<=resident_counters[0]+4;end
            if(cache_read_en)resident_counters[1]<=resident_counters[1]+4;
            if(ctrl_fire&&use_reader)resident_counters[2]<=resident_counters[2]+4;
            if(reader_padding)resident_counters[3]<=resident_counters[3]+4;
            if(qs==Q_PAYLOAD&&use_reader&&!reader_valid)resident_counters[6]<=resident_counters[6]+1;
            if(qs==Q_WAIT&&cm_valid&&!sink_ready)resident_counters[7]<=resident_counters[7]+1;
            case(qs)
                Q_IDLE:if(start)begin
                    qs<=Q_HEADER;busy<=1;done<=0;error<=0;error_code<=0;header_index<=0;input_ended<=0;abort_after_drain<=0;cache_valid<=0;write_words<=0;
                    for(int i=0;i<11;i++)counters[i]<=0;
                    for(int i=0;i<32;i++)state_counters[i]<=0;
                    for(int i=0;i<16;i++)resident_counters[i]<=0;
                end
                Q_HEADER:if(in_fire)begin
                    if(s_axis_tlast||s_axis_tkeep!=15||
                       (header_index==0&&s_axis_tdata!=32'h51434f34&&s_axis_tdata!=32'h51434f38)||
                       (header_index==1&&!extended&&s_axis_tdata>1)||
                       (header_index==12&&(s_axis_tdata==0||s_axis_tdata>200000)))begin
                        error<=1;error_code<=32'h801;cache_valid<=0;input_ended<=s_axis_tlast;qs<=s_axis_tlast?Q_ERROR_WORD:Q_DRAIN;abort_after_drain<=1;
                    end else if(header_index==0)begin extended<=s_axis_tdata==32'h51434f38;header_index<=1;end
                    else if(header_index==1)begin flags<=s_axis_tdata;header_index<=2;end
                    else if(header_index<=18)begin
                        cfg[header_index-2]<=s_axis_tdata;
                        if(header_index==18&&!extended)begin external_words<=cfg[10];qs<=Q_META_PLANE;end else header_index<=header_index+1;
                    end else begin
                        case(header_index)
                            19:external_words<=s_axis_tdata;20:tensor_id<=s_axis_tdata;21:tensor_c<=s_axis_tdata;22:tensor_h<=s_axis_tdata;23:tensor_w<=s_axis_tdata;
                            24:origin_y<=s_axis_tdata;25:origin_x<=s_axis_tdata;26:tensor_bytes<=s_axis_tdata;27:frame_id<=s_axis_tdata;
                        endcase
                        if(header_index==27)qs<=Q_META_PLANE;else header_index<=header_index+1;
                    end
                end
                Q_META_PLANE:begin checked_plane<=tensor_h[8:0]*tensor_w[8:0];qs<=Q_META_BYTES;end
                Q_META_BYTES:begin checked_bytes<=checked_plane*tensor_c[9:0];qs<=Q_VALIDATE;end
                Q_VALIDATE:begin
                    if(bad_metadata)begin error<=1;error_code<=32'h802;cache_valid<=0;qs<=Q_DRAIN;abort_after_drain<=1;resident_counters[8]<=resident_counters[8]+1;end
                    else begin
                        if(local_out&&flags[3])begin cache_valid<=0;write_words<=0;cache_id<=tensor_id;cache_c<=tensor_c;cache_h<=tensor_h;cache_w<=tensor_w;cache_bytes<=tensor_bytes;cache_frame<=frame_id;end
                        if(local_in)resident_counters[5]<=resident_counters[5]+1;
                        if(local_out)resident_counters[4]<=resident_counters[4]+1;
                        qs<=Q_START;
                    end
                end
                Q_START:begin remaining<=cfg[10];external_remaining<=external_words;qs<=Q_PAYLOAD;end
                Q_PAYLOAD:begin
                    if(ctrl_fire)begin remaining<=remaining-1;if(remaining==1||(local_in&&ce&&external_remaining==1))qs<=Q_WAIT;end
                    if(in_fire)begin
                        external_remaining<=external_remaining-1;
                        if(s_axis_tkeep!=15||external_remaining==0||s_axis_tlast!=(final_tile&&external_remaining==1))begin
                            error<=1;error_code<=32'h803;cache_valid<=0;input_ended<=s_axis_tlast;abort_after_drain<=1;qs<=s_axis_tlast?Q_ERROR_WORD:Q_DRAIN;
                        end else if(s_axis_tlast)input_ended<=1;
                    end
                    if(ctrl_fire&&remaining==1&&external_remaining!=(in_fire?1:0))begin
                        error<=1;error_code<=32'h804;cache_valid<=0;abort_after_drain<=1;qs<=(input_ended||(in_fire&&s_axis_tlast))?Q_ERROR_WORD:Q_DRAIN;
                    end
                end
                Q_WAIT:if(sink_fire&&cm_last)begin
                    for(int i=0;i<32;i++)state_counters[i]<=state_counters[i]+cs[i]+(i==14?1:0);
                    for(int i=1;i<6;i++)counters[i]<=counters[i]+cc[i]+(i==4?1:0);
                    counters[8]<=counters[8]+1;counters[9]<=counters[9]+cc[9];counters[10]<=counters[10]+cc[10];
                    if(ce)begin error<=1;error_code<=ce_code;cache_valid<=0;abort_after_drain<=0;qs<=input_ended?Q_FINISH:Q_DRAIN;end
                    else if(local_out&&flags[4]&&write_words+1!=cache_bytes/4)begin
                        error<=1;error_code<=32'h805;cache_valid<=0;abort_after_drain<=1;qs<=input_ended?Q_ERROR_WORD:Q_DRAIN;
                    end else begin
                        if(local_out&&flags[4])cache_valid<=1;
                        if(final_tile)begin cache_valid<=0;qs<=Q_FINISH;end else begin qs<=Q_HEADER;header_index<=0;end
                    end
                end
                Q_DRAIN:if(in_fire&&s_axis_tlast)begin input_ended<=1;qs<=abort_after_drain?Q_ERROR_WORD:Q_FINISH;end
                Q_ERROR_WORD:if(out_fire)qs<=Q_FINISH;
                Q_FINISH:begin busy<=0;done<=1;cache_valid<=0;qs<=Q_IDLE;end
                default:qs<=Q_IDLE;
            endcase
            if(sink_fire&&local_out&&!ce&&(write_words>=32768||write_words>=cache_bytes/4))begin error<=1;error_code<=32'h806;cache_valid<=0;abort_after_drain<=1;qs<=input_ended?Q_ERROR_WORD:Q_DRAIN;end
            if(reader_active&&reader_bounds)begin error<=1;error_code<=32'h807;cache_valid<=0;abort_after_drain<=1;qs<=input_ended?Q_ERROR_WORD:Q_DRAIN;end
        end
    end
endmodule
