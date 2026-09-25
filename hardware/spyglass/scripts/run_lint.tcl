# Copyright 2022 ETH Zurich and University of Bologna.
# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51

proc env_or {name default} {
    if {[info exists ::env($name)] && $::env($name) ne ""} {
        return $::env($name)
    }
    return $default
}

set CONFIG [env_or config mempool]
set SCOPE  [env_or lint_scope cluster]
set FILES  [env_or SG_FILES tmp/files_${CONFIG}]
set NAME   [env_or SG_NAME ${CONFIG}_${SCOPE}]

# The system instantiates the TeraPool wrapper whenever `terapool=1`,
# whatever the flavor is called.
if {[env_or terapool 0] == 1} {
    set CLUSTER terapool_cluster_floonoc_wrapper
} else {
    set CLUSTER mempool_cluster_floonoc_wrapper
}
set GROUP mempool_group_floonoc_wrapper

# The SRAMs become macros in the backend; do not lint their behavioral model
set STOP {tc_sram tc_sram_impl}
switch $SCOPE {
    cluster    { set TOP $CLUSTER }
    cluster_bb { set TOP $CLUSTER; lappend STOP $GROUP }
    group      { set TOP $GROUP }
    default    { error "unknown lint_scope '$SCOPE' (cluster, cluster_bb, group)" }
}

set TIMESTAMP [exec date +%Y%m%d_%H%M%S]

new_project sg_projects/${NAME}_${TIMESTAMP}
current_methodology $env(SPYGLASS_HOME)/GuideWare/latest/block/rtl_handoff

# Read the RTL
read_file -type sourcelist ${FILES}

set_option enableSV09 yes
# cv32e40p (read in, but not part of the cluster) uses `assert final`
set_option non_lrm_options allow_assert_final
set_option language_mode mixed
set_option allow_module_override yes
set_option designread_disable_flatten no
# Above the largest flop-based memory (the RedMulE HCI FIFOs, ~95 kbit), so
# none of them is black-boxed
set_option mthresh 131072
set_option stop $STOP
set_option top ${TOP}
set_parameter handle_large_bus yes

# Read constraints
current_design ${TOP}
set_option sdc2sgdc yes
sdc_data -file sdc/func.sdc

# Link Design
compile_design

# Set lint_rtl goal and run
current_goal lint/lint_rtl
run_goal

# Create a link to the results
exec rm -rf sg_projects/${NAME}
exec ln -sf ${NAME}_${TIMESTAMP} sg_projects/${NAME}

# Ciao!
exit -save
