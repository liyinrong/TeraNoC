# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in GVSoC target adapter; does not modify canonical winner profiles.

Use --target-dir=<this directory> and
--target=qwen38_model:system_config=dse_s1_winner:qwen_fpus=8:qwen_vlen=1024.
The resolved qwen38_contract is emitted in the simulator configuration.
"""

from dataclasses import replace

from pathlib import Path

from backing import boot_rom, validate_loads, memory_layout

import gvsoc.runner as gvsoc
from gvrun.parameter import TargetParameter
from pulp.teranoc_v2.teranoc_system import TeranocSoc


GAPY_TARGET = True


class Qwen38Soc(TeranocSoc):
    def configure(self):
        super().configure()
        sidecar = self.get_parameter("qwen_data_elf")
        if sidecar:
            binary = self.get_parameter("binary")
            validate_loads(binary, sidecar, 0x80000000, self.qwen_l2_end)
            self.loader.add_properties({"binary": [binary, sidecar],
                                        "entry": 0x80000000})

    def prepare_arch(self, arch):
        arch = super().prepare_arch(arch)
        TargetParameter(self, name="qwen_data_elf", value="", cast=str,
                        description="Data-only ELF loaded after executable")
        mib = TargetParameter(self, name="qwen_l2_mib", value=512, cast=int,
                              description="L2 MiB").get_value()
        if arch.nb_scalar_cores_per_tile:
            return self.native_arch(arch, mib)
        size, rom = memory_layout(mib)
        self.qwen_l2_end = arch.l2_base + size
        # Cores boot through the ROM, whose jump to L2 is PC-relative.
        image = Path(__file__).parent / f"qwen38_rom_{rom:08x}.bin"
        data = boot_rom(rom, arch.l2_base)
        if not image.exists() or image.read_bytes() != data:
            image.write_bytes(data)
        arch = replace(arch, l2_size=size, bootrom_base=rom,
                       bootrom_file=str(image))
        if arch.vector is None or arch.nb_snitch_per_tile != 2:
            raise ValueError(
                "Lane-first exploration requires two Spatz cores per tile")
        fpus = TargetParameter(self, name="qwen_fpus", value=4, cast=int,
                               description="FPUs per Spatz").get_value()
        vlen = TargetParameter(self, name="qwen_vlen", value=512, cast=int,
                               description="VLEN bits").get_value()
        spatz = TargetParameter(self, name="qwen_spatz_per_tile", value=2,
                                cast=int,
                                description="Spatz cores per tile").get_value()
        if (fpus not in (4, 8) or vlen not in (512, 1024)
                or spatz not in (2, 4)):
            raise ValueError("Supported sweep: four/eight FPUs, VLEN512/1024, "
                             "two/four Spatz per tile")
        vector = replace(arch.vector, nb_fpus=fpus, nb_ipus=4, vlen=vlen)
        banks = arch.nb_banks_per_tile
        # Fewer, larger banks preserve L1 bytes (dse_s1_winner_b32).
        want = TargetParameter(self, name="qwen_banks_per_tile", value=0,
                               cast=int,
                               description="Banks per tile at constant L1 "
                                           "(0 keeps the profile)").get_value()
        bank_bytes = arch.l1_bank_bytes
        if want:
            if want > banks or banks % want:
                raise ValueError("qwen_banks_per_tile must divide the "
                                 "profile's bank count")
            bank_bytes = arch.l1_bank_bytes * (banks // want)
            banks = want
        # More Spatz per tile keep one RedMulE per tile and the tile's banks:
        # the banking factor absorbs the extra cores as it absorbs lanes.
        divisor = spatz * vector.nb_lanes
        if banks % divisor:
            raise ValueError("Cannot preserve bank count with selected lanes")
        burst = TargetParameter(self, name="qwen_spatz_burst", value=0,
                                cast=int,
                                description="Spatz VLSU bursts on the wide "
                                            "plane").get_value()
        interleave = arch.l2_axi_interleave
        if want:
            # config/config.mk:310 derives axi_width_interleaved so one L2
            # granule is one group's bank row; fewer banks shrink the row.
            row = banks * arch.nb_tiles_per_group * arch.l1_bank_width
            interleave = row // arch.axi_data_width
        # RedMulE ROB depth and wide read-only request channels
        # (config/dse_s1_winner_b32_rob16*.mk).
        rob = TargetParameter(self, name="qwen_redmule_rob", value=0, cast=int,
                              description="RedMulE ROB depth (0 keeps the "
                                          "profile)").get_value()
        wide_ro = TargetParameter(self, name="qwen_wide_ro", value=0, cast=int,
                                  description="Wide RO request channels per "
                                              "group (0 keeps the "
                                              "profile)").get_value()
        redmule, wide = arch.redmule, arch.wide
        if rob:
            redmule = replace(redmule, rob_depth=rob)
        if wide_ro:
            wide = replace(wide, nb_ro_req_channels_per_group=wide_ro,
                           ro_remap_batch_size=min(wide.ro_remap_batch_size,
                                                   wide_ro))
        result = replace(arch, vector=vector, nb_snitch_per_tile=spatz,
                         redmule=redmule, wide=wide,
                         bank_factor=banks // divisor,
                         l1_bank_bytes=bank_bytes,
                         spatz_burst=bool(burst))
        if result.nb_banks_per_tile != banks:
            raise ValueError("Bank count did not reach the requested value")
        # The model derives the L2 fold granule from the bank geometry.
        if result.l2_axi_interleave != interleave:
            raise ValueError("L2 interleave differs from the RTL derivation")
        fixed = ("l1_size", "l2_size", "nb_l2_banks",
                 "axi_data_width", "dma_data_width",
                 "nb_inter_group_ports_per_tile",
                 "nb_intra_group_ports_per_tile")
        if not want:
            fixed = fixed + ("nb_banks_per_tile",)
        if not wide_ro:
            fixed = fixed + ("wide",)
        for field in fixed:
            if getattr(result, field) != getattr(arch, field):
                raise ValueError(f"Lane sweep changed fixed resource: {field}")
        # VLSU local ports and instruction-cache geometry scale with FU count
        # in RTL. Preserve that coupling and disclose it; this is not a pure
        # arithmetic-only experiment.
        self.add_property("qwen38_contract", {
            "spatz_per_tile": result.nb_snitch_per_tile,
            "spatz_burst": int(result.spatz_burst),
            # The native VLSU reserves full burst windows as blocks.
            "spatz_block_alloc": 1,
            "spatz_max_burst_words": result.spatz_effective_max_burst_words,
            "spatz_rob_depth": result.spatz_rob_depth,
            "n_fpu": fpus, "n_ipu": 4, "vlen": vlen,
            "banking_factor": result.bank_factor,
            "banks_per_tile": result.nb_banks_per_tile,
            "l1_bank_bytes": result.l1_bank_bytes,
            "l2_axi_interleave": result.l2_axi_interleave,
            "l1_bytes": result.l1_size, "cores": result.total_snitch,
            "l2_bytes": result.l2_size, "bootrom_base": result.bootrom_base,
            "icache_line_bytes": result.icache_line_size,
            "icache_bytes_per_tile": result.icache_size,
            "vlsu_ports_per_core": result.vlsu_ports_per_core,
            # A target parameter since the arch trim, not an arch field.
            "redmule_timing_only": str(self.get_parameter(
                "redmule_timing_only")).lower() == "true",
            "redmule_rob": result.redmule.rob_depth,
            "wide_channels": [result.wide.nb_ro_req_channels_per_group,
                              result.wide.nb_rw_req_channels_per_group,
                              result.wide.nb_rw_resp_channels_per_group,
                              result.wide.nb_wo_resp_channels_per_group]})
        return result

    def native_arch(self, arch, mib):
        """A profile with Snitch-only cores (the 1 + 3 S1 tile) runs as is.

        Its tile already is the target: no lane sweep, only a larger L2. The
        ROM stays at the profile's bootrom_base, below the L2 aperture, so
        extending L2 cannot collide with it.
        """
        if mib not in (512, 1024):
            raise ValueError("Supported L2 capacities: 512/1024 MiB")
        # The lane-sweep parameters may still be passed; they must describe
        # the profile, since nothing here overrides it.
        given = {name: TargetParameter(self, name=name, value=0, cast=int,
                                       description="Must match the profile "
                                                   "(0 = unset)").get_value()
                 for name in ("qwen_fpus", "qwen_vlen", "qwen_spatz_per_tile",
                              "qwen_banks_per_tile", "qwen_spatz_burst",
                              "qwen_redmule_rob", "qwen_wide_ro")}
        have = {"qwen_fpus": arch.vector.nb_fpus,
                "qwen_vlen": arch.vector.vlen,
                "qwen_spatz_per_tile": arch.nb_snitch_per_tile,
                "qwen_banks_per_tile": arch.nb_banks_per_tile,
                "qwen_spatz_burst": int(arch.spatz_burst),
                "qwen_redmule_rob": arch.redmule.rob_depth}
        for name, value in given.items():
            if value and value != have.get(name):
                raise ValueError(f"{name}={value} does not match the native "
                                 f"profile ({have.get(name)})")
        size = mib * 1024 * 1024
        self.qwen_l2_end = arch.l2_base + size
        rom = arch.bootrom_base
        image = Path(__file__).parent / f"qwen38_rom_{rom:08x}.bin"
        data = boot_rom(rom, arch.l2_base)
        if not image.exists() or image.read_bytes() != data:
            image.write_bytes(data)
        result = replace(arch, l2_size=size, bootrom_file=str(image))
        # Area what-ifs (GF12): integer units per Spatz, and Snitch-only cores
        # per tile (0 = every core Snitch-Spatz, for kernels not yet split).
        ipus = TargetParameter(self, name="qwen_ipus", value=0, cast=int,
                               description="Integer units per Spatz (0 keeps "
                                           "the profile)").get_value()
        scalar = TargetParameter(self, name="qwen_scalar", value=-1, cast=int,
                                 description="Snitch-only cores per tile "
                                             "(-1 keeps the profile)"
                                 ).get_value()
        if ipus:
            if ipus not in (1, 2, 4) or ipus > result.vector.nb_fpus:
                raise ValueError("qwen_ipus must be 1, 2 or 4, at most n_fpu")
            result = replace(result,
                             vector=replace(result.vector, nb_ipus=ipus))
        if scalar >= 0:
            result = replace(result, nb_scalar_cores_per_tile=scalar)
        icache = TargetParameter(self, name="qwen_icache", value=0, cast=int,
                                 description="Instruction cache bytes per "
                                             "tile (0 keeps the profile)"
                                 ).get_value()
        if icache:
            result = replace(result, icache_size_byte=icache)
        if TargetParameter(self, name="qwen_icache_stats", value=0, cast=int,
                           description="Print the icache counters"
                           ).get_value():
            result = replace(result, icache_stats=True)
        self.add_property("qwen38_contract", {
            "native": 1,
            "scalar_cores_per_tile": result.nb_scalar_cores_per_tile,
            "cores_per_tile": result.nb_snitch_per_tile,
            "spatz_burst": int(result.spatz_burst),
            "spatz_max_burst_words": result.spatz_effective_max_burst_words,
            "spatz_rob_depth": result.spatz_rob_depth,
            "n_fpu": result.vector.nb_fpus, "n_ipu": result.vector.nb_ipus,
            "vlen": result.vector.vlen,
            "banking_factor": result.bank_factor,
            "banks_per_tile": result.nb_banks_per_tile,
            "l1_bank_bytes": result.l1_bank_bytes,
            "l2_axi_interleave": result.l2_axi_interleave,
            "l1_bytes": result.l1_size, "cores": result.total_snitch,
            "l2_bytes": result.l2_size, "bootrom_base": result.bootrom_base,
            "icache_bytes_per_tile": result.icache_size,
            "redmule_timing_only": str(self.get_parameter(
                "redmule_timing_only")).lower() == "true",
            "redmule_rob": result.redmule.rob_depth})
        return result


class Target(gvsoc.Target):
    model = Qwen38Soc
    name = "qwen38_model"

    def __init__(self, parser, options=None, name=None):
        super().__init__(parser, options, model=Qwen38Soc,
                         description="Qwen full-layer lane-first exploration",
                         name=name)
