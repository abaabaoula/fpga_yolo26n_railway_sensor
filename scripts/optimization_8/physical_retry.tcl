set root [file normalize $::env(OPTIMIZATION_8_PROJECT_ROOT)]
set run $::env(OPTIMIZATION_8_RUN_ID)
set out [file join $root output optimization_8 hardware $run]
set reports [file join $root reports optimization_8 hardware builds $run]
set_param board.repoPaths [list [file join $out source_snapshot board_repo]]
open_project [file join $out project optimization_8_resident.xpr]
if {[get_property PART [current_project]] ne "xc7z020clg400-1"} {error "Part mismatch"}
set flow [get_property FLOW [get_runs impl_1]]
create_run impl_physopt -parent_run synth_1 -flow $flow -strategy Performance_Explore
set_property STEPS.PHYS_OPT_DESIGN.IS_ENABLED true [get_runs impl_physopt]
set_property STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE AggressiveExplore [get_runs impl_physopt]
set_property STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE Explore [get_runs impl_physopt]
current_run -implementation [get_runs impl_physopt]
launch_runs impl_physopt -to_step route_design -jobs 2
wait_on_run impl_physopt
if {![string match "route_design Complete*" [get_property STATUS [get_runs impl_physopt]]]} {error "Physical route failed"}
open_run impl_physopt
if {[get_property PERIOD [get_clocks clk_fpga_0]] != 10} {error "100MHz constraint changed"}
report_timing_summary -file [file join $reports post_route_timing_summary.rpt]
report_utilization -file [file join $reports post_route_utilization.rpt]
report_utilization -hierarchical -hierarchical_depth 6 -file [file join $reports post_route_hierarchy.rpt]
report_route_status -file [file join $reports route_status.rpt]
report_drc -file [file join $reports drc.rpt]
report_clock_utilization -file [file join $reports clock_utilization.rpt]
write_checkpoint -force [file join $out optimization_8_resident_routed.dcp]
set wns [get_property SLACK [get_timing_paths -max_paths 1 -delay_type max]]
set whs [get_property SLACK [get_timing_paths -max_paths 1 -delay_type min]]
if {$wns < 0 || $whs < 0} {error "Timing failed WNS=$wns WHS=$whs"}
set violations [get_drc_violations -quiet -filter {SEVERITY == Error || SEVERITY == {Critical Warning}}]
if {[llength $violations] != 0} {error "Blocking DRC violations: $violations"}
close_design
launch_runs impl_physopt -to_step write_bitstream -jobs 2
wait_on_run impl_physopt
set bit [file join $out project optimization_8_resident.runs impl_physopt design_1_wrapper.bit]
if {![file exists $bit]} {error "No physical bitstream"}
file copy $bit [file join $out optimization_8_resident.bit]
open_run impl_physopt
write_hw_platform -fixed -include_bit -force [file join $out optimization_8_resident.xsa]
file copy [file join $out project optimization_8_resident.gen sources_1 bd design_1 hw_handoff design_1.hwh] [file join $out optimization_8_resident.hwh]
puts "O8_PHYSICAL_EXPORT_PASS"
exit
