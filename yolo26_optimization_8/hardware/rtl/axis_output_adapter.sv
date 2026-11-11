module axis_output_adapter (
    input logic clk,resetn,word_valid,word_last,
    input logic [31:0] word_data,
    output wire word_ready,
    output logic [31:0] m_axis_tdata,
    output wire [3:0] m_axis_tkeep,
    output logic m_axis_tvalid,m_axis_tlast,
    input logic m_axis_tready
);
    assign word_ready=!m_axis_tvalid || m_axis_tready;
    assign m_axis_tkeep=4'hf;
    always_ff @(posedge clk) begin
        if (!resetn) begin m_axis_tvalid<=0;m_axis_tlast<=0;m_axis_tdata<=0;end
        else if (word_ready) begin
            m_axis_tvalid<=word_valid;
            if (word_valid) begin m_axis_tdata<=word_data;m_axis_tlast<=word_last;end
        end
    end
endmodule
