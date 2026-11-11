module axis_byte_packer(
    input logic clk,resetn,in_valid,in_last,
    input logic [7:0] in_data,
    output wire in_ready,
    output logic [31:0] m_data,
    output wire [3:0] m_keep,
    output logic m_valid,m_last,
    input logic m_ready
);
    logic [31:0] assembly;
    logic [1:0] fill;
    wire completes=(fill==3)||in_last;
    assign in_ready=!completes || !m_valid || m_ready;
    assign m_keep=4'hf; // Final partial word is zero padded; byte length is explicit.
    logic [31:0] next_word;
    always_comb begin
        next_word=assembly;
        next_word[fill*8+:8]=in_data;
    end
    always_ff @(posedge clk) begin
        if(!resetn) begin assembly<=0;fill<=0;m_data<=0;m_valid<=0;m_last<=0;end
        else begin
            if(m_valid && m_ready)m_valid<=0;
            if(in_valid && in_ready) begin
                if(completes) begin
                    m_data<=next_word;m_last<=in_last;m_valid<=1;assembly<=0;fill<=0;
                end else begin assembly<=next_word;fill<=fill+1;end
            end
        end
    end
endmodule
