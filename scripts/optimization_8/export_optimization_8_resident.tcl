set root [file normalize $::env(OPTIMIZATION_8_PROJECT_ROOT)]
set snapshot [file normalize $::env(OPTIMIZATION_8_SOURCE_SNAPSHOT)]
set run_id [lindex $argv 0]
if {$run_id eq ""} {error "Run ID required"}
set out [file join $root output optimization_8 hardware $run_id]
set_param board.repoPaths [list [file join $snapshot board_repo]]
open_project [file join $out project optimization_8_resident.xpr]
if {[get_property PART [current_project]] ne "xc7z020clg400-1"} {error "Unexpected FPGA part"}
launch_runs impl_1 -to_step write_bitstream -jobs 2
wait_on_run impl_1
set run_bit [file join $out project optimization_8_resident.runs impl_1 design_1_wrapper.bit]
if {![file exists $run_bit]} {error "Implementation-run bitstream not created"}
set bit [file join $out optimization_8_resident.bit]
file copy -force $run_bit $bit
open_run impl_1
set xsa [file join $out optimization_8_resident.xsa]
write_hw_platform -fixed -include_bit -force $xsa
if {![file exists $xsa]} {error "XSA not created"}
set hwh [file join $out project optimization_8_resident.gen sources_1 bd design_1 hw_handoff design_1.hwh]
if {![file exists $hwh]} {error "HWH not found"}
file copy -force $hwh [file join $out optimization_8_resident.hwh]
puts "EXPORT_PASS BIT=$bit XSA=$xsa HWH=[file join $out optimization_8_resident.hwh]"

