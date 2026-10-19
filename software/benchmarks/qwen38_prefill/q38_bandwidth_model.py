# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Isolated VLSU diagnostic: fixed cache, cores, banks and memory topology."""

from dataclasses import fields, replace

import gvsoc.runner as gvsoc
from gvrun.parameter import TargetParameter
from pulp.teranoc_v2.arch import TeranocConfig
from pulp.teranoc_v2.teranoc_system import TeranocSoc


GAPY_TARGET = True


class ProbeConfig(TeranocConfig):
    @property
    def icache_line_size(self):
        return 256

    @property
    def icache_size(self):
        return 16384


class BandwidthSoc(TeranocSoc):
    def prepare_arch(self, arch):
        arch = super().prepare_arch(arch)
        fpus = TargetParameter(self, name="f", value=8, cast=int,
                               description="FPUs and VLSU ports").get_value()
        vlen = TargetParameter(self, name="v", value=1024,
                               cast=int, description="VLEN bits").get_value()
        rob = TargetParameter(self, name="r", value=32, cast=int,
                              description="ROB entries per port").get_value()
        banks = 32 if arch.redmule.ce_height == 8 else 64
        vector = replace(arch.vector, nb_fpus=fpus, nb_ipus=4, vlen=vlen)
        result = replace(arch, vector=vector, nb_snitch_per_tile=4,
                         bank_factor=banks // (4 * vector.nb_lanes),
                         l2_axi_interleave=(banks * arch.nb_tiles_per_group *
                                            arch.l1_bank_width //
                                            arch.axi_data_width),
                         l1_bank_bytes=8192, spatz_burst=True,
                         wide=replace(arch.wide, data_width=vlen // 8),
                         spatz_max_burst_words=vlen // 32,
                         spatz_rob_depth=rob)
        result = ProbeConfig(**{f.name: getattr(result, f.name)
                                for f in fields(result)})
        self.add_property("bandwidth_contract", {
            "fpus": fpus, "vlen": vlen, "ports": result.vlsu_ports_per_core,
            "rob": rob, "block_alloc": 1, "burst_words": vlen // 32,
            "banks_per_tile": result.nb_banks_per_tile,
            "l1_bytes": result.l1_size, "cores": result.total_snitch,
            "icache_bytes": result.icache_size,
            "icache_line_bytes": result.icache_line_size})
        return result


class Target(gvsoc.Target):
    model = BandwidthSoc
    name = "q38_bandwidth_model"

    def __init__(self, parser, options=None, name=None):
        super().__init__(parser, options, model=BandwidthSoc,
                         description="Qwen isolated VLSU bandwidth probe",
                         name=name)
