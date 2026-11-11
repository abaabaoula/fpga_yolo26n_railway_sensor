set root [file normalize $::env(OPTIMIZATION_8_PROJECT_ROOT)]
set snapshot [file normalize $::env(OPTIMIZATION_8_SOURCE_SNAPSHOT)]
set run_id $::env(OPTIMIZATION_8_RUN_ID)
set out [file join $root output optimization_8 hardware $run_id]
set reports [file join $root reports optimization_8 hardware builds $run_id]
file mkdir $out $reports
set_param board.repoPaths [list [file join $snapshot board_repo]]
set board [get_board_parts tul.com.tw:pynq-z2:part0:1.0]
if {[llength $board] != 1} {error "PYNQ-Z2 board preset unavailable"}
create_project optimization_8_resident [file join $out project] -part xc7z020clg400-1
set_property board_part $board [current_project]
add_files -norecurse [glob [file join $snapshot rtl *.sv]]
create_bd_design design_1
set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0]
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 -config {make_external "FIXED_IO, DDR" apply_board_preset "1"} $ps
# The TUL PYNQ-Z2 preset has four negative UI DQS-to-clock delay entries.
# Vivado 2026.1 raises PSU-1..4 for these whenever HP0 is enabled. Keep the
# board preset and all package/trace data, but use the DDR controller's zero
# starting delay for these four UI entries (Zynq DDR write leveling calibrates
# the lane delays). This is deliberately a design setting, not a message waiver.
set_property -dict [list CONFIG.PCW_UIPARAM_DDR_DQS_TO_CLK_DELAY_0 {0.0} CONFIG.PCW_UIPARAM_DDR_DQS_TO_CLK_DELAY_1 {0.0} CONFIG.PCW_UIPARAM_DDR_DQS_TO_CLK_DELAY_2 {0.0} CONFIG.PCW_UIPARAM_DDR_DQS_TO_CLK_DELAY_3 {0.0} CONFIG.PCW_USE_M_AXI_GP0 {1} CONFIG.PCW_USE_S_AXI_HP0 {1} CONFIG.PCW_EN_CLK0_PORT {1}] $ps
for {set lane 0} {$lane < 4} {incr lane} {
    if {[get_property CONFIG.PCW_UIPARAM_DDR_DQS_TO_CLK_DELAY_$lane $ps] != 0.0} {error "DDR UI DQS delay override failed for lane $lane"}
}
if {[get_property CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ $ps] != 100} {error "FCLK_CLK0 is not 100 MHz"}
if {[get_property CONFIG.PCW_USE_S_AXI_HP0 $ps] != 1} {error "S_AXI_HP0 did not enable"}

set dma [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0]
set_property -dict [list CONFIG.C_SG_LENGTH_WIDTH {23} CONFIG.C_INCLUDE_SG {0} CONFIG.C_INCLUDE_MM2S {1} CONFIG.C_INCLUDE_S2MM {1} CONFIG.C_M_AXIS_MM2S_TDATA_WIDTH {32} CONFIG.C_S_AXIS_S2MM_TDATA_WIDTH {32} CONFIG.C_M_AXI_MM2S_DATA_WIDTH {32} CONFIG.C_M_AXI_S2MM_DATA_WIDTH {32} CONFIG.C_MICRO_DMA {0}] $dma
set accel [create_bd_cell -type module -reference optimization_4_queue_top conv_accel_0]
set ctrl [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_ctrl_interconnect]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {2}] $ctrl
set data [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_data_interconnect]
set_property -dict [list CONFIG.NUM_SI {2} CONFIG.NUM_MI {1}] $data
set rst [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 proc_sys_reset_0]
set locked [create_bd_cell -type inline_hdl -vlnv xilinx.com:inline_hdl:ilconstant:* dcm_locked_1]
set_property CONFIG.CONST_VAL {1} $locked

connect_bd_intf_net [get_bd_intf_pins $ps/M_AXI_GP0] [get_bd_intf_pins $ctrl/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins $ctrl/M00_AXI] [get_bd_intf_pins $dma/S_AXI_LITE]
connect_bd_intf_net [get_bd_intf_pins $ctrl/M01_AXI] [get_bd_intf_pins $accel/S_AXI]
connect_bd_intf_net [get_bd_intf_pins $dma/M_AXI_MM2S] [get_bd_intf_pins $data/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins $dma/M_AXI_S2MM] [get_bd_intf_pins $data/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins $data/M00_AXI] [get_bd_intf_pins $ps/S_AXI_HP0]
connect_bd_intf_net [get_bd_intf_pins $dma/M_AXIS_MM2S] [get_bd_intf_pins $accel/S_AXIS]
connect_bd_intf_net [get_bd_intf_pins $accel/M_AXIS] [get_bd_intf_pins $dma/S_AXIS_S2MM]

set clk [get_bd_pins $ps/FCLK_CLK0]
connect_bd_net $clk [get_bd_pins $ps/M_AXI_GP0_ACLK] [get_bd_pins $ps/S_AXI_HP0_ACLK] [get_bd_pins $dma/s_axi_lite_aclk] [get_bd_pins $dma/m_axi_mm2s_aclk] [get_bd_pins $dma/m_axi_s2mm_aclk] [get_bd_pins $accel/aclk] [get_bd_pins $rst/slowest_sync_clk] [get_bd_pins $ctrl/ACLK] [get_bd_pins $ctrl/S00_ACLK] [get_bd_pins $ctrl/M00_ACLK] [get_bd_pins $ctrl/M01_ACLK] [get_bd_pins $data/ACLK] [get_bd_pins $data/S00_ACLK] [get_bd_pins $data/S01_ACLK] [get_bd_pins $data/M00_ACLK]
connect_bd_net [get_bd_pins $ps/FCLK_RESET0_N] [get_bd_pins $rst/ext_reset_in]
connect_bd_net [get_bd_pins $locked/dout] [get_bd_pins $rst/dcm_locked]
connect_bd_net [get_bd_pins $rst/interconnect_aresetn] [get_bd_pins $ctrl/ARESETN] [get_bd_pins $ctrl/S00_ARESETN] [get_bd_pins $ctrl/M00_ARESETN] [get_bd_pins $ctrl/M01_ARESETN] [get_bd_pins $data/ARESETN] [get_bd_pins $data/S00_ARESETN] [get_bd_pins $data/S01_ARESETN] [get_bd_pins $data/M00_ARESETN]
connect_bd_net [get_bd_pins $rst/peripheral_aresetn] [get_bd_pins $dma/axi_resetn] [get_bd_pins $accel/aresetn]

# Passive bus measurement, no transactions are injected or stalled.
set traffic [create_bd_cell -type module -reference axi_traffic_monitor hp0_traffic]
connect_bd_net $clk [get_bd_pins $traffic/clk]
connect_bd_net [get_bd_pins $rst/peripheral_aresetn] [get_bd_pins $traffic/resetn]
connect_bd_net [get_bd_pins $accel/traffic_clear] [get_bd_pins $traffic/clear]
connect_bd_net [get_bd_pins $traffic/read_bytes] [get_bd_pins $accel/ddr_read_bytes]
connect_bd_net [get_bd_pins $traffic/write_bytes] [get_bd_pins $accel/ddr_write_bytes]
foreach {bus_signal monitor_signal} {RVALID read_valid RREADY read_ready WVALID write_valid WREADY write_ready WSTRB write_strobe} {
    set tap [get_bd_pins -quiet $ps/S_AXI_HP0_$bus_signal]
    if {[llength $tap] != 1} {error "HP0 signal missing: $bus_signal"}
    # Scalar probing overrides an interface-managed wire. Explicitly reconnect
    # both bus endpoints as well as the passive monitor, including master-driven
    # RREADY/WVALID/WSTRB. Otherwise BD may tie those inputs to zero.
    set master_tap [get_bd_pins -quiet $data/M00_AXI_$bus_signal]
    if {[llength $master_tap] != 1} {error "Interconnect monitor signal missing: $bus_signal"}
    connect_bd_net $tap $master_tap [get_bd_pins $traffic/$monitor_signal]
}
if {[get_property CONFIG.PCW_S_AXI_HP0_DATA_WIDTH $ps] != 64} {error "Traffic byte counter requires 64-bit HP0"}
assign_bd_address
set accel_segments [get_bd_addr_segs -of_objects [get_bd_intf_pins $accel/S_AXI]]
if {[llength $accel_segments] != 1} {error "Expected one accelerator register segment"}
assign_bd_address -offset 0x43C00000 -range 0x00001000 -target_address_space [get_bd_addr_spaces $ps/Data] $accel_segments -force
validate_bd_design
if {[get_property CONFIG.C_EXT_RESET_HIGH $rst] != 0} {
    error "proc_sys_reset must infer active-low FCLK_RESET0_N"
}
set fh [open [file join $reports dma_address_map.md] w]
puts $fh "# Vivado Address Editor: DMA stream test"
puts $fh ""
foreach master [list $ps/M_AXI_GP0 $dma/M_AXI_MM2S $dma/M_AXI_S2MM] {
    set spaces [get_bd_addr_spaces -of_objects [get_bd_intf_pins $master]]
    puts $fh "## $master"
    foreach space $spaces {
        foreach seg [get_bd_addr_segs -of_objects $space] {
            puts $fh "- $seg: OFFSET=[get_property OFFSET $seg], RANGE=[get_property RANGE $seg]"
        }
    }
    puts $fh ""
}
close $fh
save_bd_design
set bd [get_files -filter {FILE_TYPE == "Block Designs"}]
set_property synth_checkpoint_mode None $bd
generate_target all $bd
set wrapper [make_wrapper -files $bd -top]
add_files -norecurse $wrapper
set_property top design_1_wrapper [current_fileset]
update_compile_order -fileset sources_1
puts "DESIGN_PASS"
if {[llength $argv] > 0 && [lindex $argv 0] eq "design"} {exit 0}

launch_runs synth_1 -jobs 2
wait_on_run synth_1
if {[get_property STATUS [get_runs synth_1]] ne "synth_design Complete!"} {error "Synthesis failed: [get_property STATUS [get_runs synth_1]]"}
open_run synth_1
report_utilization -file [file join $reports synthesis_utilization.rpt]
report_utilization -hierarchical -hierarchical_depth 6 -file [file join $reports synthesis_hierarchy.rpt]
report_timing_summary -file [file join $reports synthesis_timing_summary.rpt]
close_design
puts "SYNTHESIS_PASS"

launch_runs impl_1 -to_step route_design -jobs 2
wait_on_run impl_1
set status [get_property STATUS [get_runs impl_1]]
if {![string match "route_design Complete*" $status]} {error "Implementation failed: $status"}
open_run impl_1
report_timing_summary -file [file join $reports post_route_timing_summary.rpt]
report_utilization -file [file join $reports post_route_utilization.rpt]
report_utilization -hierarchical -hierarchical_depth 6 -file [file join $reports post_route_hierarchy.rpt]
report_route_status -file [file join $reports route_status.rpt]
report_drc -file [file join $reports drc.rpt]
report_clock_utilization -file [file join $reports clock_utilization.rpt]
write_checkpoint -force [file join $out optimization_8_resident_routed.dcp]
puts "IMPLEMENTATION_PASS"

