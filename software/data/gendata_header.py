#!/usr/bin/env python3

# Copyright 2022 ETH Zurich and University of Bologna.
# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51

# This script generates data.h files.
# Author: Marco Bertuletti <mbertuletti@iis.ee.ethz.ch>

import argparse
import os
import hjson
import ast
import numpy

import gendatalib as datalib
import pyflexfloat as ff


header = """\
// Copyright 2022 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// File generated with .data/print_header.py
// Author: Marco Bertuletti\n\n
"""


def format_fp16_literal(value):
    """Return an exact finite binary16 value as a C99 hex literal."""
    value_f16 = numpy.float16(value)
    if not numpy.isfinite(value_f16):
        raise ValueError("Cannot emit non-finite __fp16 data")
    return float(value_f16).hex()


def format_type(typ, value):
    """
    formats the type for printing in .h file.
    :param typ: Input type
    :param value: Input_value
    """
    typ_i32b = ["int32_t", "uint32_t"]
    typ_i16b = ["int16_t", "uint16_t"]
    typ_i8b = ["int8_t", "uint8_t"]

    if typ in typ_i32b:
        stringyfied_val = '({}) 0X{:08X}'.format(
            typ, value.astype(numpy.uint32) & 0xffffffff)
    elif typ in typ_i16b:
        stringyfied_val = '({}) 0X{:04X}'.format(
            typ, value.astype(numpy.uint32) & 0x0000ffff)
    elif typ in typ_i8b:
        stringyfied_val = '({}) 0X{:02X}'.format(
            typ, value.astype(numpy.uint32) & 0x000000ff)
    elif typ == 'float':
        stringyfied_val = '({}) {:+.8f}'.format(typ, value)
    elif typ == '__fp16':
        stringyfied_val = '({}) {}'.format(typ, format_fp16_literal(value))
    elif typ == '__fp8':
        if not isinstance(value, ff.FlexFloat):
            value = numpy.array([value]).astype(numpy.double)
            value = ff.FlexFloat("e5m2", value[0])
        value = value.bits()
        stringyfied_val = '({}) 0X{:02X}'.format(typ, value)
    else:
        raise Exception("ERROR: Unsupported data type!!!")

    return stringyfied_val


def print_array(arr, typ, name):
    """
    Converts arrays to a string.

    :param arr: Input array
    :param typ: Type of the array.
    :param name: Name of the array.
    """

    output_string = typ
    attr = " __attribute__((aligned(4 * NUM_BANKS), section(\".l2\"))) "
    if (arr.size > 1):
        output_string += attr
        output_string += name + '[{}] = {{\n'.format(arr.size)
        for (value, count) in zip(arr, range(arr.size)):
            output_string += (format_type(typ, value) + ', ')
            count += 1
            if count % 4 == 0:
                output_string += '\n'
        output_string = output_string[:-3]
        output_string += "};\n\n"
    else:
        output_string += attr
        output_string += (name + ' = ' + format_type(typ, arr[0]))
        output_string += ";\n\n"

    return output_string


def print_file(header, defines, arrays, filename):
    """
    Writes defines and arrays to a file.

    :param header: Header of the printed file
    :param defines: A tuple of (define_name, define_value).
    :param arrays: A tuple of (array_name, array_type, array_values).
    :param filename: The output file to write to.
    """

    # Initialize the output string
    output_string = header

    # Write the defines
    for def_key, def_value in defines.items():
        output_string += "#define {} ({})\n".format(def_key, def_value)
    output_string += "\n"  # Add space between defines and arrays

    # Write the arrays using print_array
    for array_values, array_type, array_name in arrays:
        output_string += print_array(array_values, array_type, array_name)

    # Write everything to the file
    with open(filename, "w") as file:
        file.write(output_string)

    print("Generate {}".format(filename))


def print_perf_file(header, defines, decls, filename):
    """Perf mode: emit shapes + uninitialized .l2_bss array declarations (no
    values, no golden) so a real-size layer costs L2 address space only."""
    output_string = header
    for def_key, def_value in defines.items():
        output_string += "#define {} ({})\n".format(def_key, def_value)
    output_string += "\n"
    for name, size in decls:
        output_string += (
            '__fp16 __attribute__((aligned(4 * NUM_BANKS), '
            'section(".l2_bss"))) {}[{}];\n\n'.format(name, size))
    with open(filename, "w") as file:
        file.write(output_string)
    print("Generate {} (perf)".format(filename))


def get_type(type_string):
    """
    Gets the numpy type from the type specifyied in the json
    :param type_string: type from json file.
    """
    if type_string == "int8":
        return numpy.int8
    elif type_string == "int16":
        return numpy.int16
    elif type_string == "int32":
        return numpy.int32
    elif type_string == "float32":
        return numpy.float32
    elif type_string == "float16":
        return numpy.float16
    elif type_string == "float8":
        return ff.FlexFloat('e5m2')
    else:
        raise Exception("Input type is not valid")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Generate data.h header files.')
    parser.add_argument('--app_name', type=str, help='Name of the app')
    parser.add_argument('--params', type=str, help='Name of the app')
    # Total usable L1 bytes and RedMulE PORT_WIDTH of the active config; the
    # LLM generators use them to derive the L1-fitting FFN tile (PANEL_I).
    parser.add_argument('--l1-bytes', type=int, default=0,
                        help='Total L1 bytes = NUM_BANKS*L1_BANK_SIZE')
    parser.add_argument('--port-width', type=int, default=32,
                        help='RedMulE PORT_WIDTH = REDMULE_H*(REDMULE_P+1)')

    # Parse the command-line arguments
    args = parser.parse_args()
    app_name = args.app_name

    # Stage-1 Qwen operation benchmarks declare perf-only tile buffers in their
    # shared C harness. They still participate in the normal Make dependency on
    # data_<app>.h, so emit a generated stub without requiring 28 duplicate
    # hjson blocks.
    if app_name.startswith("qwen_"):
        filename = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(filename, "data_{}.h".format(app_name))
        print_file(header, {}, [], filename)
        raise SystemExit(0)

    with open(args.params, 'r') as hjson_file:
        config_data = hjson.load(hjson_file)
    data_args = config_data.get(app_name)

    if data_args is not None:
        my_type = get_type(data_args.get("type"))
        defnes = dict([ast.literal_eval(defne)
                      for defne in data_args.get("defines")])
        raw_arrays = data_args.get("arrays")
        arrays = ([ast.literal_eval(array) for array in raw_arrays]
                  if raw_arrays is not None else [])
        # Feed the L1 budget to the dense-LLM generators (build-time tiling
        # solver). The generators pop these so they are not emitted as defines.
        if app_name.startswith("llm_"):
            defnes["L1_BYTES"] = args.l1_bytes
            defnes["PORT_WIDTH"] = args.port_width
        # Real-size shapes set LLM_PERF_ONLY=1: reserve uninitialized weights,
        # skip value/golden generation.
        perf = int(defnes.get("LLM_PERF_ONLY", 0))

    # Determine output file name
    filename = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(filename, "data_{}.h".format(app_name))

    # Define function mappings for each app_name
    function_map = {
        "sp_matmul_f32": {"func": datalib.generate_fmatmul},
        "sp_matmul_i32": {"func": datalib.generate_imatmul},
        "sp_axpy_f32": {"func": datalib.generate_faxpy},
        "sp_dotp_f32": {"func": datalib.generate_fdotp},
        "sp_gemv_f32": {"func": datalib.generate_fgemv},
        "sp_gemv_f16": {"func": datalib.generate_fgemv},
        "sp_fconv2d_f32": {"func": datalib.generate_fconv2d_padded},
        "sp_fft_f32": {"func": datalib.generate_fft},
        "sp_bandwidth_test_i32": {"func": datalib.generate_bandwidth},
        "axpy_i32": {"func": datalib.generate_iaxpy},
        "axpy_f8": {"func": datalib.generate_faxpy},
        "axpy_f16": {"func": datalib.generate_faxpy},
        "axpy_f32": {"func": datalib.generate_faxpy},
        "cfft_radix2_q16": {"func": datalib.generate_cfft_q16},
        "cfft_radix4_f16": {"func": datalib.generate_fcfft},
        "cfft_radix4_q16": {"func": datalib.generate_cfft_q16},
        "chest_q16": {"func": datalib.generate_qchest},
        "cholesky_q16": {"func": datalib.generate_qccholesky},
        "cholesky_q32": {"func": datalib.generate_qcholesky},
        "cmatmul_q16": {"func": datalib.generate_qcmatmul},
        "dotp_f16": {"func": datalib.generate_fdotp},
        "dotp_f32": {"func": datalib.generate_fdotp},
        "rm_gemm_f16": {"func": datalib.generate_rm_gemm},
        "rm_gemm_opt_f16": {"func": datalib.generate_rm_gemm},
        "rm_gemm_xshift_f16": {"func": datalib.generate_rm_gemm},
        "dotp_i32": {"func": datalib.generate_idotp},
        "conv2d_f16": {"func": datalib.generate_fconv2d},
        "conv2d_f32": {"func": datalib.generate_fconv2d},
        "gaussjordan_q32": {"func": datalib.generate_qgaussjordan},
        "matmul_f8": {"func": datalib.generate_fmatmul},
        "matmul_f16": {"func": datalib.generate_fmatmul},
        "matmul_f32": {"func": datalib.generate_fmatmul},
        "matmul_i32": {"func": datalib.generate_imatmul},
        "matmul_i16": {"func": datalib.generate_imatmul},
        "matmul_i8": {"func": datalib.generate_imatmul},
        "gemv_f32": {"func": datalib.generate_fgemv},
        "gemv_f16": {"func": datalib.generate_fgemv},
        "mimo_mmse_q16": {"func": datalib.generate_qmmse},
        "mimo_mmse_f32": {"func": datalib.generate_fmmse},
        "batchnorm_f8": {"func": datalib.generate_fbatchnorm},
        "batchnorm_f16": {"func": datalib.generate_fbatchnorm},
        "layernorm_f8": {"func": datalib.generate_flayernorm},
        "layernorm_f16": {"func": datalib.generate_flayernorm},
        "fence": {"func": datalib.generate_iarray},
        "memcpy": {"func": datalib.generate_iarray},
        "barriers_test": {"func": datalib.generate_barriers_test},
        "softmax_f8": {"func": datalib.generate_fsoftmax},
        "softmax_f16": {"func": datalib.generate_fsoftmax},
        "llm_gemm_ffn_f16": {"func": datalib.generate_llm_gemm_ffn},
        "llm_gemm_attention_f16": {
            "func": datalib.generate_llm_gemm_attention},
        "llm_ffn_prefill_f16": {"func": datalib.generate_llm_ffn},
        "llm_ffn_decode_f16": {"func": datalib.generate_llm_ffn},
        "llm_classic_ffn_prefill_f16": {
            "func": datalib.generate_llm_classic_ffn},
        "llm_classic_ffn_decode_f16": {
            "func": datalib.generate_llm_classic_ffn},
        "llm_attention_prefill_f16": {"func": datalib.generate_llm_attention},
        "llm_attention_decode_f16": {"func": datalib.generate_llm_attention},
        "llm_classic_attention_prefill_f16": {
            "func": datalib.generate_llm_classic_attention},
        "llm_classic_attention_decode_f16": {
            "func": datalib.generate_llm_classic_attention},
    }

    # Check if app_name exists in the function map
    if app_name in function_map and not perf:
        func_info = function_map[app_name]
        func = func_info["func"]
        # Call the function
        # The defnes dictionary is a function argument in case the generate
        # function adds new definitions.
        result, defnes = func(defines=defnes, my_type=my_type)
        # Print result to data header
        if len(arrays) == 1:
            arrays = [(result, *arrays[0])]
        else:
            arrays = [(result[i], *arrays[i]) for i in range(len(arrays))]
        print_file(header, defnes, arrays, filename)

    elif app_name in function_map and perf:
        # Real-size perf run: reserve uninitialized L2 weights, no golden.
        defnes = datalib.llm_perf_defines(app_name, defnes)
        decls = datalib.llm_perf_arrays(app_name, defnes)
        print_perf_file(header, defnes, decls, filename)

    else:
        print("Data generation is not defined.")
