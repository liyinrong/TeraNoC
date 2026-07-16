#!/usr/bin/env python3

# Copyright 2022 ETH Zurich and University of Bologna.
# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51

# This script generates data for the fp16 matmul.
# Author: Marco Bertuletti <mbertuletti@iis.ee.ethz.ch>

# The script generates random inputs for the C functions. The inputs are
# propagated though a python golden model. Golden models are from the
# numpy library or the qmath bit-true library.

import pyflexfloat as ff
import numpy as np
import math
import copy
import qmath

from scipy import signal
from scipy.linalg import solve_triangular


def select_maxval(my_type=np.int32):
    size = 8 * np.dtype(my_type).itemsize
    MAX = 2**(size - 2) - 1
    return MAX


def irandom(size, MAX, my_type=np.int16):
    """Generate random numbers.
    size (int or tuple): Size of the array to generate.
    mytype (np.dtype): Data type for the fixed-point representation.
    Defaults to np.int16.

    Returns:
        np.ndarray: Array of random fixed-point numbers.
    """
    return np.random.randint(-MAX, MAX - 1, size=size, dtype=my_type)


def icrandom(size, MAX, my_type=np.int16):
    """Generate random complex numbers.
    size (int or tuple): Size of the array to generate.
    mytype (np.dtype): Data type for the fixed-point representation.
    Defaults to np.int16.

    Returns:
        np.ndarray: Array of random complex fixed-point numbers.
    """
    real_part = np.random.randint(-MAX, MAX - 1, size=size, dtype=my_type)
    imag_part = np.random.randint(-MAX, MAX - 1, size=size, dtype=my_type)
    return real_part + 1j * imag_part


def generate_iarray(my_type=np.float32, defines={}):

    # Create random array of integers
    array_N = defines['array_N']
    MAX = select_maxval(my_type)
    A = irandom(MAX=MAX, size=(array_N), my_type=my_type)
    return A, defines


##############################################################################


def generate_barriers_test(my_type=np.int16, defines={}):

    array_N = defines['array_N']
    max_delay = defines['MAX_delay']
    delays = np.random.uniform(low=0.0, high=max_delay, size=array_N)
    delays = np.asarray(delays, dtype='int')
    return delays, defines


def generate_faxpy(my_type=np.float32, defines={}):

    # f8: Cast correct type
    if f"{my_type}" == f"{ff.FlexFloat('e5m2')}":

        # Create matrix (with type numpy)
        array_N = defines['array_N']
        A = (np.random.rand(1) - 0.5).astype(np.float16)
        X = (np.random.rand(array_N) - 0.5).astype(np.float16)
        Y = (np.random.rand(array_N) - 0.5).astype(np.float16)

        # Cast the correct type
        A = ff.array(A, 'e5m2')
        X = ff.array(X, 'e5m2')
        Y = ff.array(Y, 'e5m2')

        Z = Y + X * A

    # f16,f32: Use normal operation (FP type automatically retained)
    else:
        # Create matrix
        array_N = defines['array_N']
        A = (np.random.rand(1) - 0.5).astype(my_type)
        X = (np.random.rand(array_N) - 0.5).astype(my_type)
        Y = (np.random.rand(array_N) - 0.5).astype(my_type)
        Z = (Y + X * A).astype(my_type)

    return [A, X, Y, Z], defines


def generate_fdotp(my_type=np.float32, defines={}):

    # Create matrix
    array_N = defines['array_N']
    X = (np.random.rand(array_N) - 0.5).astype(my_type)
    Y = (np.random.rand(array_N) - 0.5).astype(my_type)
    Z = np.dot(X, Y).astype(my_type)
    Z = np.array(Z).astype(my_type)
    Z = np.resize(Z, 1)

    return [X, Y, Z], defines


def ftwiddleCoef(N, my_type=np.float32):
    PI = np.pi
    twiddleCoeff16 = np.zeros((int)(2 * 3 * N / 4), my_type)
    for i in range(0, int(3 * N / 4)):
        twiddleCoeff16_sin = np.sin(i * 2 * PI / N).astype(my_type)
        twiddleCoeff16_cos = np.cos(i * 2 * PI / N).astype(my_type)
        twiddleCoeff16[2 * i] = twiddleCoeff16_sin
        twiddleCoeff16[2 * i + 1] = twiddleCoeff16_cos
    return twiddleCoeff16


def generate_fcfft(my_type=np.float32, defines={}):

    N_CSAMPLES = defines['N_CSAMPLES']
    src_r = np.random.normal(0, 5, N_CSAMPLES).astype(np.float16)
    src_i = np.random.normal(0, 5, N_CSAMPLES).astype(np.float16)
    src = src_r + 1.j * src_i
    src = np.fft.ifft(src)
    dst = np.fft.fft(src)
    src = np.column_stack((src.imag, src.real)).astype(my_type).flatten()
    dst = np.column_stack((dst.imag, dst.real)).astype(my_type).flatten()

    twiddles = ftwiddleCoef(N_CSAMPLES, my_type)
    bitrever = qmath.bitreversal(N_CSAMPLES, 2)

    defines['LOG2'] = int(math.log2(N_CSAMPLES))
    defines['N_TWIDDLES'] = 3 * N_CSAMPLES // 4
    defines['BITREVINDEXTABLE_LENGTH'] = len(bitrever)
    defines['TOLERANCE'] = 0.1 * np.max(dst)

    return [src, dst, twiddles, bitrever], defines


def generate_fchest(my_type=np.float32, defines={}, division=False):

    nb_tx = defines['N_TX']
    nb_rx = defines['N_RX']
    nb_samples = defines['N_SAMPLES']

    H = np.random.randn(nb_rx, nb_tx)
    H = H + 1j * np.random.randn(nb_rx, nb_tx)

    vpilot_tx = []
    vpilot_rx = []
    vHest = []
    for k in range(nb_samples):
        if (division):
            # Compute data division
            pilot_tx = 1 * np.exp(1j * np.random.randn(nb_tx))
            pilot_rx = np.dot(H, pilot_tx)
            Hest = pilot_rx[:, np.newaxis] / pilot_tx[np.newaxis, :]
        else:
            # Compute data multiplication
            pilot_tx = np.exp(1j * np.random.randn(nb_tx))
            pilot_rx = np.dot(H, pilot_tx)
            pilot_tx = np.reciprocal(pilot_tx)
            Hest = pilot_rx[:, np.newaxis] * pilot_tx[np.newaxis, :]
            Hest = Hest.flatten()

        # Interleaved real and imaginary parts
        pilot_tx = np.column_stack((pilot_tx.imag, pilot_tx.real))
        pilot_rx = np.column_stack((pilot_rx.imag, pilot_rx.real))
        Hest = np.column_stack((Hest.imag, Hest.real))
        # Flatten arrays
        pilot_tx = pilot_tx.astype(my_type).flatten()
        pilot_rx = pilot_rx.astype(my_type).flatten()
        Hest = Hest.astype(my_type).flatten()
        # Output vectors
        vpilot_tx.append(pilot_tx)
        vpilot_rx.append(pilot_rx)
        vHest.append(Hest)

    vpilot_rx = np.concatenate(vpilot_rx, axis=0)
    vpilot_tx = np.concatenate(vpilot_tx, axis=0)
    vHest = np.concatenate(vHest, axis=0)

    return [vpilot_tx, vpilot_rx, vHest], defines


def generate_fccholesky(my_type=np.float32, defines={}):

    n_matrix = defines['matrix_N']
    n_samples = defines['N_SAMPLES']

    vector_G = []
    vector_L = []
    for k in range(n_samples):
        # Create hermitian matrix
        H = np.random.rand(n_matrix, n_matrix) + 1.j * \
            np.random.rand(n_matrix, n_matrix)
        # Matrix to be inverted
        # H_H = np.asmatrix(H).H
        G = np.matmul(H, np.asmatrix(H).H)
        # Cholesky decomposition
        L = np.linalg.cholesky(G)
        # Reshape
        G = np.reshape(np.asarray(G), (n_matrix * n_matrix), order='C')
        L = np.reshape(np.asarray(L), (n_matrix * n_matrix), order='C')
        G = np.column_stack((G.real, G.imag)).astype(my_type).flatten()
        L = np.column_stack((L.real, L.imag)).astype(my_type).flatten()
        # Output vectors
        vector_G.append(G)
        vector_L.append(L)

    vector_G = np.concatenate(vector_G, axis=0)
    vector_L = np.concatenate(vector_L, axis=0)
    return [vector_G, vector_L], defines


def generate_fconv2d(my_type=np.float32, defines={}):
    # Create matrix
    matrix_M = defines['matrix_M']
    matrix_N = defines['matrix_N']
    # Create kernel
    kernel_K = defines['kernel_K']
    A = (np.random.rand(matrix_N, matrix_M) - 0.5).astype(my_type)
    K = (np.random.rand(kernel_K, kernel_K) - 0.5).astype(my_type)
    if my_type == np.float32:
        C = signal.convolve2d(A, K, mode="same", boundary='fill')
    else:
        C = np.zeros((matrix_N, matrix_M), dtype=my_type)
    # Reshape
    A = np.reshape(A, (matrix_M * matrix_N), order='C').astype(my_type)
    K = np.reshape(K, (kernel_K * kernel_K), order='C').astype(my_type)
    C = np.reshape(C, (matrix_M * matrix_N), order='C').astype(my_type)
    # Output vectors
    return [A, K, C], defines


def generate_fcmatmul(my_type=np.float32, defines={}):

    # Create matrix
    matrix_M = defines['matrix_M']
    matrix_N = defines['matrix_N']
    matrix_P = defines['matrix_P']
    A = np.random.rand(matrix_M, matrix_N) + 1j * \
        np.random.rand(matrix_M, matrix_N)
    B = np.random.rand(matrix_N, matrix_P) + 1j * \
        np.random.rand(matrix_N, matrix_P)
    C = np.matmul(A, B)

    A = np.reshape(A, (matrix_M * matrix_N), order='C')
    B = np.reshape(B, (matrix_N * matrix_P), order='C')
    C = np.reshape(C, (matrix_M * matrix_P), order='C')

    A = np.column_stack((A.imag, A.real)).astype(my_type).flatten()
    B = np.column_stack((B.imag, B.real)).astype(my_type).flatten()
    C = np.column_stack((C.imag, C.real)).astype(my_type).flatten()

    return [A, B, C], defines


def generate_fgemm(my_type=np.float32, defines={}):

    # Create matrix
    matrix_M = defines['matrix_M']
    matrix_N = defines['matrix_N']
    matrix_P = defines['matrix_P']
    X = (np.random.rand(matrix_M, matrix_N) - 0.5).astype(my_type)
    W = (np.random.rand(matrix_N, matrix_P) - 0.5).astype(my_type)
    Y = (np.random.rand(matrix_M, matrix_P) - 0.5).astype(my_type)
    Z = np.matmul(X, W) + Y

    X = np.reshape(X, (matrix_M * matrix_N), order='C').astype(my_type)
    W = np.reshape(W, (matrix_N * matrix_P), order='C').astype(my_type)
    Y = np.reshape(Y, (matrix_M * matrix_P), order='C').astype(my_type)
    Z = np.reshape(Z, (matrix_M * matrix_P), order='C').astype(my_type)

    return [X, W, Y, Z], defines


def generate_fmatmul(my_type=np.float32, defines={}):

    # f8: Cast correct type
    if f"{my_type}" == f"{ff.FlexFloat('e5m2')}":

        # Define dimension
        matrix_M = defines['matrix_M']
        matrix_N = defines['matrix_N']
        matrix_P = defines['matrix_P']

        # Create matrix
        A = (np.random.rand(matrix_M, matrix_N) - 0.5).astype(np.float16)
        B = (np.random.rand(matrix_N, matrix_P) - 0.5).astype(np.float16)
        C = np.zeros((matrix_M, matrix_P), np.float16)

        # Cast the correct type
        A = ff.array(A, 'e5m2')
        B = ff.array(B, 'e5m2')
        C = ff.array(C, 'e5m2')
        for m in range(matrix_M):
            for n in range(matrix_N):
                for p in range(matrix_P):
                    C[m][p] += A[m][n] * B[n][p]

        # Flatten the matrices into 1D arrays
        A = np.reshape(A, (matrix_M * matrix_N), order='C')
        B = np.reshape(B, (matrix_N * matrix_P), order='C')
        C = np.reshape(C, (matrix_M * matrix_P), order='C')

    # f16,f32: Use normal operation (FP type automatically retained)
    else:
        # Create matrix
        matrix_M = defines['matrix_M']
        matrix_N = defines['matrix_N']
        matrix_P = defines['matrix_P']
        A = (np.random.rand(matrix_M, matrix_N) - 0.5).astype(my_type)
        B = (np.random.rand(matrix_N, matrix_P) - 0.5).astype(my_type)
        C = np.matmul(A, B)

        A = np.reshape(A, (matrix_M * matrix_N), order='C').astype(my_type)
        B = np.reshape(B, (matrix_N * matrix_P), order='C').astype(my_type)
        C = np.reshape(C, (matrix_M * matrix_P), order='C').astype(my_type)

    return [A, B, C], defines


def generate_fgemv(my_type=np.float32, defines={}):

    matrix_M = defines['matrix_M']
    matrix_N = defines['matrix_N']
    transpose_a = int(defines.get('transpose_a', 0))
    A = (np.random.rand(matrix_M, matrix_N) - 0.5).astype(my_type)
    X = (np.random.rand(matrix_N) - 0.5).astype(my_type)
    Y = np.matmul(A, X).astype(my_type)

    if transpose_a:
        A = A.T

    A = np.reshape(A, (matrix_M * matrix_N), order='C').astype(my_type)
    X = np.reshape(X, (matrix_N), order='C').astype(my_type)
    Y = np.reshape(Y, (matrix_M), order='C').astype(my_type)

    return [A, X, Y], defines


def generate_fmmse(my_type=np.float16, defines={}):

    N_tx = defines['N_TX']
    N_rx = defines['N_RX']
    N_itr = defines['N_ITR']
    vH = np.zeros([N_itr, N_tx * 2 * N_rx], dtype=my_type)
    vG = np.zeros([N_itr, N_tx * 2 * N_tx], dtype=my_type)
    vy = np.zeros([N_itr, 2 * N_rx], dtype=my_type)
    vN = np.zeros([N_itr, 2 * N_tx], dtype=my_type)
    vx = np.zeros([N_itr, 2 * N_tx], dtype=my_type)

    for k in range(N_itr):

        # Create input vector
        y = np.random.rand(N_rx).astype(my_type) + 1.j * \
            np.random.rand(N_rx).astype(my_type)

        # Create channel matrix
        H = np.random.rand(N_rx, N_tx).astype(my_type) + 1.j * \
            np.random.rand(N_rx, N_tx).astype(my_type)
        # Generate noise variance
        N = np.random.rand(1).astype(my_type)

        # Matrix to be inverted in MMSE estimator
        H_h = np.asmatrix(H).H
        G = np.matmul(H_h, H) + N * np.eye(H.shape[1])
        N = N * np.ones(N_tx)

        # Cholesky decomposition
        L = np.linalg.cholesky(G)
        # Linear system solution
        y1 = np.transpose(np.dot(H_h, y))
        y2 = solve_triangular(L, y1, lower=True)
        x = solve_triangular(np.asmatrix(L).H, y2)

        H = np.reshape(np.asarray(H), (N_tx * N_rx), order='C')
        G = np.reshape(np.asarray(G), (N_tx * N_tx), order='C')
        N = np.column_stack((N.real, N.imag)).astype(my_type).flatten()
        H = np.column_stack((H.real, H.imag)).astype(my_type).flatten()
        G = np.column_stack((G.real, G.imag)).astype(my_type).flatten()
        x = np.column_stack((x.real, x.imag)).astype(my_type).flatten()
        y = np.column_stack((y.real, y.imag)).astype(my_type).flatten()

        vH[k, :] = H
        vG[k, :] = G
        vy[k, :] = y
        vN[k, :] = N
        vx[k, :] = x

    vN = np.reshape(vN, (2 * N_tx * N_itr)).astype(my_type)
    vH = np.reshape(vH, (2 * N_rx * N_tx * N_itr)).astype(my_type)
    vG = np.reshape(vG, (2 * N_tx * N_tx * N_itr)).astype(my_type)
    vy = np.reshape(vy, (2 * N_rx * N_itr)).astype(my_type)
    vx = np.reshape(vx, (2 * N_tx * N_itr)).astype(my_type)

    return [vH, vG, vy, vN, vx], defines


def generate_fofdm(my_type=np.float32, defines={}):

    N_sc = defines['N_SC']
    N_rx = defines['N_RX']
    N_bs = defines['N_BEAMS']

    pFFT_src = (np.random.rand(2 * N_rx * N_sc)).astype(np.float16)
    pBF_coef = (np.random.rand(2 * N_rx * N_bs)).astype(np.float16)
    pBF_dst = (np.random.rand(2 * N_bs * N_sc)).astype(np.float16)
    twiddles = ftwiddleCoef(N_sc, my_type)
    bitrever = qmath.bitreversal(N_sc, 2)

    defines['LOG2'] = int(math.log2(N_sc))
    defines['N_TWIDDLES'] = 3 * N_sc // 4
    defines['BITREVINDEXTABLE_LENGTH'] = len(bitrever)

    return [pFFT_src, pBF_coef, pBF_dst, twiddles, bitrever], defines


##############################################################################


def generate_iaxpy(my_type=np.int32, defines={}):

    # Create matrix
    array_N = defines['array_N']
    MAX = select_maxval(my_type)
    A = np.random.randint(-MAX, MAX - 1, size=1, dtype=my_type)
    X = irandom(MAX=MAX, size=(array_N), my_type=my_type)
    Y = irandom(MAX=MAX, size=(array_N), my_type=my_type)
    Z = (Y + X * A).astype(my_type)

    return [A, X, Y, Z], defines


def generate_idotp(my_type=np.int32, defines={}):

    # Create matrix
    array_N = defines['array_N']
    MAX = select_maxval(my_type)
    X = irandom(MAX=MAX, size=(array_N), my_type=my_type)
    Y = irandom(MAX=MAX, size=(array_N), my_type=my_type)
    Z = np.dot(X, Y)
    Z = np.array(Z).astype(my_type)
    Z = np.resize(Z, 1)

    return [X, Y, Z], defines


def generate_iconv(my_type=np.int32, defines={}):

    # Create matrix
    matrix_M = defines['matrix_M']
    matrix_N = defines['matrix_N']
    kernel_N = defines['kernel_N']
    MAX = select_maxval(my_type)
    X = irandom(MAX=MAX, size=(matrix_M, matrix_N), my_type=my_type)
    K = irandom(MAX=MAX, size=(kernel_N, kernel_N), my_type=my_type)
    Y = signal.convolve2d(X, K, mode="same", boundary='fill')

    X = X.flatten().astype(my_type)
    K = K.flatten().astype(my_type)
    Y = Y.flatten().astype(my_type)

    return [X, K, Y], defines


def generate_imatmul(my_type=np.int32, defines={}):

    # Create matrix
    matrix_M = defines['matrix_M']
    matrix_N = defines['matrix_N']
    matrix_P = defines['matrix_P']
    MAX = select_maxval(my_type)
    A = irandom(MAX=MAX, size=(matrix_M, matrix_N), my_type=my_type)
    B = irandom(MAX=MAX, size=(matrix_N, matrix_P), my_type=my_type)
    C = np.matmul(A, B)

    A = np.reshape(A, (matrix_M * matrix_N), order='C').astype(my_type)
    B = np.reshape(B, (matrix_N * matrix_P), order='C').astype(my_type)
    C = np.reshape(C, (matrix_M * matrix_P), order='C').astype(np.int32)

    return [A, B, C], defines


##############################################################################


def generate_qcmatmul(defines={}, fixed_point=15, my_type=np.int32):
    MAX = 2**(fixed_point - 1)
    # Create matrix
    matrix_M = defines['matrix_M']
    matrix_N = defines['matrix_N']
    matrix_P = defines['matrix_P']
    A = np.random.randint(-MAX, MAX - 1, size=(matrix_M, matrix_N)) + 1j * \
        np.random.randint(-MAX, MAX - 1, size=(matrix_M, matrix_N))
    B = np.random.randint(-MAX, MAX - 1, size=(matrix_N, matrix_P)) + 1j * \
        np.random.randint(-MAX, MAX - 1, size=(matrix_N, matrix_P))
    [Cr, Ci] = qmath.qcmatmul(A.real, A.imag, B.real,
                              B.imag, fixed_point, my_type)

    A = np.reshape(A, (matrix_M * matrix_N), order='C')
    B = np.reshape(B, (matrix_N * matrix_P), order='C')
    A = np.column_stack((A.imag, A.real)).astype(my_type).flatten()
    B = np.column_stack((B.imag, B.real)).astype(my_type).flatten()
    C = np.column_stack((Ci, Cr)).astype(my_type).flatten()

    return [A, B, C], defines


def generate_qchest(defines={}, fixed_point=15, my_type=np.int16):

    N_TX = defines['N_TX']
    N_RX = defines['N_RX']
    N_SAMPLES = defines['N_SAMPLES']

    qvector_pilot_tx = []
    qvector_pilot_rx = []
    qvector_Hest = []
    for k in range(N_SAMPLES):
        # Create pilots
        pilot_rx = icrandom(size=N_RX, MAX=2**7, my_type=np.int32)
        pilot_tx = icrandom(size=N_TX, MAX=2**7, my_type=np.int32)
        # Compute Hest
        Hest = qmath.qchest(pilot_rx, pilot_tx, fixed_point=8)

        pilot_tx = np.column_stack((pilot_tx.imag, pilot_tx.real))
        pilot_rx = np.column_stack((pilot_rx.imag, pilot_rx.real))
        qvector_pilot_tx.append(pilot_tx.astype(np.int16).flatten())
        qvector_pilot_rx.append(pilot_rx.astype(np.int16).flatten())
        qvector_Hest.append(Hest)

    qvector_pilot_tx = np.reshape(qvector_pilot_tx, [2 * N_TX * N_SAMPLES])
    qvector_pilot_rx = np.reshape(qvector_pilot_rx, [2 * N_RX * N_SAMPLES])
    qvector_Hest = np.reshape(qvector_Hest, [2 * N_TX * N_RX * N_SAMPLES])
    return [qvector_pilot_tx, qvector_pilot_rx, qvector_Hest], defines


def generate_qccholesky(defines={}, fixed_point=15, my_type=np.int32):

    matrix_N = defines['matrix_N']
    FIXED_POINT = defines['FIXED_POINT']
    N_SAMPLES = defines['N_SAMPLES']

    vA = np.zeros([N_SAMPLES, 2 * matrix_N * matrix_N], dtype=my_type)
    vL = np.zeros([N_SAMPLES, 2 * matrix_N * matrix_N], dtype=my_type)
    for k in range(N_SAMPLES):

        Ar = np.random.normal(0, 1, [matrix_N, matrix_N]).astype(np.float32)
        Ai = np.random.normal(0, 1, [matrix_N, matrix_N]).astype(np.float32)
        A = Ar + 1.j * Ai
        G = np.matmul(A.conj().T, A)
        MAX_A = max(np.abs(A.real).max(), np.abs(A.imag).max())
        MAX_G = max(np.abs(G.real).max(), np.abs(G.imag).max())
        MAX = max(MAX_A, MAX_G)

        Ar = np.round((Ar / MAX) * 2**FIXED_POINT).astype(int)
        Ai = np.round((Ai / MAX) * 2**FIXED_POINT).astype(int)
        Ar = Ar + np.eye(matrix_N, dtype=int) * 256
        Ai = Ai + np.eye(matrix_N, dtype=int) * 256

        Ar, Ai = qmath.qcmatmul(Ar.T, -Ai.T, Ar, Ai, FIXED_POINT, my_type)
        Lr, Li = qmath.qccholesky(
            Ar, Ai, fixed_point=FIXED_POINT, mytype=my_type)

        A = np.column_stack((Ar, Ai)).astype(my_type).flatten()
        L = np.column_stack((Lr, Li)).astype(my_type).flatten()
        vA[k, :] = np.reshape(A, (2 * matrix_N * matrix_N),
                              order='C').astype(my_type)
        vL[k, :] = np.reshape(L, (2 * matrix_N * matrix_N),
                              order='C').astype(my_type)

    vA = np.reshape(vA, (2 * matrix_N * matrix_N * N_SAMPLES)).astype(my_type)
    vL = np.reshape(vL, (2 * matrix_N * matrix_N * N_SAMPLES)).astype(my_type)
    return [vA, vL], defines


def generate_qcholesky(defines={}, fixed_point=15, my_type=np.int32):

    matrix_N = defines['matrix_N']
    FIXED_POINT = defines['FIXED_POINT']
    N_SAMPLES = defines['N_SAMPLES']

    vA = np.zeros([N_SAMPLES, matrix_N * matrix_N], dtype=my_type)
    vL = np.zeros([N_SAMPLES, matrix_N * matrix_N], dtype=my_type)
    vy = np.zeros([N_SAMPLES, matrix_N], dtype=my_type)
    for k in range(N_SAMPLES):
        A = irandom(size=(matrix_N, matrix_N), MAX=2**14, my_type=my_type)
        y = irandom(size=matrix_N, MAX=2**14, my_type=my_type)
        A = qmath.qmatmul(A.T, A, FIXED_POINT, my_type)
        L = qmath.qcholesky(A, FIXED_POINT, my_type)

        vA[k, :] = np.reshape(A, (matrix_N * matrix_N),
                              order='C').astype(my_type)
        vL[k, :] = np.reshape(L, (matrix_N * matrix_N),
                              order='C').astype(my_type)
        vy[k, :] = np.reshape(y, matrix_N, order='C').astype(my_type)

    vA = np.reshape(vA, (matrix_N * matrix_N * N_SAMPLES)).astype(my_type)
    vL = np.reshape(vL, (matrix_N * matrix_N * N_SAMPLES)).astype(my_type)
    vy = np.reshape(vy, (matrix_N * N_SAMPLES)).astype(my_type)

    return [vA, vL, vy], defines


def generate_qgaussjordan(my_type=np.int32, defines={}):

    matrix_N = defines['matrix_N']
    fixed_point = defines['FIXED_POINT']

    issingular = True
    while (issingular):
        src = np.random.uniform(-10, 10, size=(matrix_N, matrix_N))
        if np.linalg.det(src) != 0:  # Ensure it's invertible
            issingular = False

    dst = np.linalg.inv(src)
    src = np.round(src * (1 << fixed_point)).astype(my_type)
    dst = np.round(dst * (1 << fixed_point)).astype(my_type)
    src = np.reshape(src, (matrix_N * matrix_N)).astype(my_type)
    dst = np.reshape(dst, (matrix_N * matrix_N)).astype(my_type)

    return [src, dst], defines


def generate_qmmse(defines={}, fixed_point=15, my_type=np.int32):

    FIXED_POINT = defines['FIXED_POINT']
    N_tx = defines['N_TX']
    N_rx = defines['N_RX']
    N_itr = defines['N_ITR']

    vN = np.zeros([N_itr, 2 * N_tx], dtype=np.int16)
    vH = np.zeros([N_itr, 2 * N_tx * N_rx], dtype=np.int16)
    vG = np.zeros([N_itr, 2 * N_tx * N_tx], dtype=np.int16)
    vy = np.zeros([N_itr, 2 * N_rx], dtype=np.int16)
    vx = np.zeros([N_itr, 2 * N_tx], dtype=np.int16)

    for k in range(N_itr):

        # Floating point inputs
        rH = np.random.normal(0, 1, [N_rx, N_tx]).astype(np.float32)
        iH = np.random.normal(0, 1, [N_rx, N_tx]).astype(np.float32)
        rN = np.random.normal(0, 1, [N_rx]).astype(np.float32)
        ry = np.random.normal(0, 1, [N_rx]).astype(np.float32)
        iy = np.random.normal(0, 1, [N_rx]).astype(np.float32)
        H = rH + 1j * iH
        y = ry + 1j * iy
        G = np.matmul(H.conj().T, H) + rN * np.eye(H.shape[1])
        y1 = np.dot(H.conj().T, y)

        # Rescale inputs
        H_max = max(np.abs(H.real).max(), np.abs(H.imag).max())
        G_max = max(np.abs(G.real).max(), np.abs(G.imag).max())
        y_max = max(np.abs(y.real).max(), np.abs(y.imag).max())
        y1_max = max(np.abs(y1.real).max(), np.abs(y1.imag).max())
        N_max = np.abs(rN).max()
        MAX = max(H_max, G_max, N_max, y_max, y1_max)
        SCALE_FACTOR = 2**FIXED_POINT
        rH = np.round((H.real / MAX) * SCALE_FACTOR).astype(int)
        iH = np.round((H.imag / MAX) * SCALE_FACTOR).astype(int)
        ry = np.round((y.real / MAX) * SCALE_FACTOR).astype(int)
        iy = np.round((y.imag / MAX) * SCALE_FACTOR).astype(int)
        rN = np.round((rN / MAX) * SCALE_FACTOR).astype(int) + 1024

        # Hermitian
        rG, iG = qmath.qcmatmul(rH.T, -iH.T, rH, iH, FIXED_POINT, my_type)
        ry1, iy1 = qmath.qcmvmul(rH.T, -iH.T, ry, iy, FIXED_POINT, my_type)
        np.fill_diagonal(rG, rG.diagonal() + rN)

        # Solve linear system
        rL, iL = qmath.qccholesky(rG, iG, FIXED_POINT, my_type)
        ry2, iy2 = qmath.qinvertLt(rL, iL, ry1, iy1, FIXED_POINT, my_type)
        rx, ix = qmath.qinvertUt(rL.T, -iL.T, ry2, iy2, FIXED_POINT, my_type)

        vN[k, :] = np.column_stack(
            (rN, np.zeros(np.size(rN)))).astype(my_type).flatten()
        vH[k, :] = np.column_stack((rH, iH)).astype(my_type).flatten()
        vG[k, :] = np.column_stack((rG, iG)).astype(my_type).flatten()
        vy[k, :] = np.column_stack((ry, iy)).astype(my_type).flatten()
        vx[k, :] = np.column_stack((rx, ix)).astype(my_type).flatten()

    vN = np.reshape(vN, (2 * N_tx * N_itr)).astype(my_type)
    vH = np.reshape(vH, (2 * N_rx * N_tx * N_itr)).astype(my_type)
    vG = np.reshape(vG, (2 * N_tx * N_tx * N_itr)).astype(my_type)
    vy = np.reshape(vy, (2 * N_rx * N_itr)).astype(my_type)
    vx = np.reshape(vx, (2 * N_tx * N_itr)).astype(my_type)
    return [vN, vH, vG, vy, vx], defines


def generate_cfft_q16(defines={}, fixed_point=15, my_type=np.int16):

    N_CSAMPLES = defines['N_CSAMPLES']
    src = icrandom(size=N_CSAMPLES, MAX=2**fixed_point, my_type=my_type)
    tolerance = {
        16: 16,
        32: 20,
        64: 24,
        128: 28,
        256: 32,
        512: 48,
        1024: 64,
        2048: 96,
        4096: 128}
    bit_shift_dict_q16 = {
        16: 11,
        32: 10,
        64: 9,
        128: 8,
        256: 7,
        512: 6,
        1024: 5,
        2048: 4,
        4096: 3}

    dst = np.fft.fft(src.astype(np.csingle) / (2**fixed_point))
    dst = dst * 2**(bit_shift_dict_q16[N_CSAMPLES])

    dst = (np.column_stack((dst.real, dst.imag))).flatten()
    src = (np.column_stack((src.real, src.imag))).flatten()
    dst = dst.astype(np.int16)
    src = src.astype(np.int16)

    twiddles = qmath.qtwiddleCoef(N_CSAMPLES)
    bitrever = qmath.bitreversal(N_CSAMPLES, 2)

    defines['LOG2'] = int(math.log2(N_CSAMPLES))
    defines['N_TWIDDLES'] = 3 * N_CSAMPLES // 4
    defines['BITREVINDEXTABLE_LENGTH'] = len(bitrever)
    defines['TOLERANCE'] = tolerance[N_CSAMPLES]

    return [src, dst, twiddles, bitrever], defines


def generate_fbatchnorm(my_type=np.float32, defines={}):

    # f8: Cast correct type
    if f"{my_type}" == f"{ff.FlexFloat('e5m2')}":

        # Define dimension
        matrix_M = defines['matrix_M']
        matrix_N = defines['matrix_N']

        # Create input matrix
        A = (np.random.rand(matrix_M, matrix_N) - 0.5).astype(np.float16)
        Sum = np.zeros(matrix_N, ).astype(np.float16)
        Squared_diff = np.zeros(matrix_N, ).astype(np.float16)
        # Cast the correct type
        A = ff.array(A, 'e5m2')
        Sum = ff.array(Sum, 'e5m2')
        Squared_diff = ff.array(Squared_diff, 'e5m2')

        # Normalize matrix A (using BatchNorm)
        for i in range(matrix_M):
            Sum += A[i]
        mean = Sum / matrix_M

        for i in range(matrix_M):
            diff = A[i] - mean
            Squared_diff += diff * diff
        var = Squared_diff / matrix_M

        B = (A - mean) / np.sqrt(var)

        # Flatten the matrices into 1D arrays
        A = np.reshape(A, (matrix_M * matrix_N), order='C')
        B = np.reshape(B, (matrix_M * matrix_N), order='C')

    # f16,f32: Use normal operation (FP type automatically retained)
    else:
        # Define dimension
        matrix_M = defines['matrix_M']
        matrix_N = defines['matrix_N']

        # Create input matrix
        A = (np.random.rand(matrix_M, matrix_N) - 0.5).astype(my_type)
        Sum = np.zeros(matrix_N, ).astype(my_type)
        Squared_diff = np.zeros(matrix_N, ).astype(my_type)

        # Normalize matrix A (using BatchNorm)
        for i in range(matrix_M):
            Sum += A[i]
        mean = Sum / matrix_M

        for i in range(matrix_M):
            diff = A[i] - mean
            Squared_diff += diff * diff
        var = Squared_diff / matrix_M

        B = (A - mean) / np.sqrt(var)

        # Flatten the matrices into 1D arrays
        A = np.reshape(A, (matrix_M * matrix_N), order='C')
        B = np.reshape(B, (matrix_M * matrix_N), order='C')

    return [A, B], defines


def generate_flayernorm(my_type=np.float32, defines={}):

    # f8: Cast correct type
    if f"{my_type}" == f"{ff.FlexFloat('e5m2')}":

        # Define dimension
        matrix_M = defines['matrix_M']
        matrix_N = defines['matrix_N']

        # Create input matrix
        A = (np.random.rand(matrix_M, matrix_N) - 0.25).astype(np.float16)
        B = np.zeros((matrix_M, matrix_N)).astype(np.float16)
        # Cast the correct type
        A = ff.array(A, 'e5m2')
        B = ff.array(B, 'e5m2')

        # Normalize matrix A (using LayerNorm)
        for i in range(matrix_M):  # Loop over each sample (row)
            row = A[i]
            # Compute: mean = sum / N
            row_sum = ff.FlexFloat("e5m2", 0)
            for el in row:
                row_sum += el
            mean = row_sum / matrix_N
            # Compute E[x^2]
            row_sq = row * row
            row_sum = ff.FlexFloat("e5m2", 0)
            for el in row_sq:
                row_sum += el
            expected = row_sum / matrix_N
            # Compute mean^2
            mean_sq = mean * mean
            # Compute: var = E[x^2] - mean^2
            var = expected - mean_sq
            # Compute: std = sqrt(var)
            std = np.sqrt(var)

            diff = row - mean
            B[i] = (diff) / std

        # Flatten the matrices into 1D arrays
        A = np.reshape(A, (matrix_M * matrix_N), order='C')
        B = np.reshape(B, (matrix_M * matrix_N), order='C')

    # f16,f32: Use normal operation (FP type automatically retained)
    else:
        # Define dimension
        matrix_M = defines['matrix_M']
        matrix_N = defines['matrix_N']

        # Create input matrix
        A = (np.random.rand(matrix_M, matrix_N) - 0.5).astype(my_type)
        B = np.zeros((matrix_M, matrix_N)).astype(my_type)

        # Normalize matrix A (using LayerNorm)
        for i in range(matrix_M):  # Loop over each sample (row)
            row = A[i]
            mean = np.sum(row) / matrix_N
            diff = row - mean
            var = np.sum(diff * diff) / matrix_N
            std = np.sqrt(var)
            B[i] = (diff) / std

        # Flatten the matrices into 1D arrays
        A = np.reshape(A, (matrix_M * matrix_N), order='C')
        B = np.reshape(B, (matrix_M * matrix_N), order='C')

    return [A, B], defines


def generate_fsoftmax(my_type=np.float32, defines={}):

    # f8: Cast correct type
    if f"{my_type}" == f"{ff.FlexFloat('e5m2')}":

        # Define dimension
        matrix_M = defines['matrix_M']
        matrix_N = defines['matrix_N']

        # Create input matrix
        A = (np.random.rand(matrix_M, matrix_N) - 0.5).astype(np.float16)
        B = np.zeros((matrix_M, matrix_N)).astype(np.float16)
        # Cast the correct type
        A = ff.array(A, 'e5m2')
        B = ff.array(B, 'e5m2')

        # Calculate the row-wise softmax
        for i in range(matrix_M):
            a_max = np.max(A[i])
            # Approximate exp with Taylor series
            diff = A[i] - a_max
            diff_2 = diff * diff
            diff_3 = diff * diff * diff
            numerator = 1 + diff + 0.5 * diff_2 + 0.167 * diff_3
            numerator_sum = ff.FlexFloat("e5m2", 0)
            for el in numerator:
                numerator_sum += el
            denominator = numerator_sum
            B[i] = numerator / denominator

        # Flatten the matrices into 1D arrays
        A = np.reshape(A, (matrix_M * matrix_N), order='C')
        B = np.reshape(B, (matrix_M * matrix_N), order='C')

    # f16,f32: Use normal operation (FP type automatically retained)
    else:
        # Define dimension
        matrix_M = defines['matrix_M']
        matrix_N = defines['matrix_N']

        # Create input matrix
        A = (np.random.rand(matrix_M, matrix_N) - 0.5).astype(my_type)
        B = np.zeros((matrix_M, matrix_N)).astype(my_type)

        # Calculate the row-wise softmax
        for i in range(matrix_M):
            a_max = np.max(A[i])
            # Approximate exp with Taylor series
            diff = A[i] - a_max
            numerator = 1 + diff + 0.5 * np.square(diff) + 0.167 * np.power(
                diff, 3)
            denominator = np.sum(numerator)
            B[i] = numerator / denominator

        # Flatten the matrices into 1D arrays
        A = np.reshape(A, (matrix_M * matrix_N), order='C')
        B = np.reshape(B, (matrix_M * matrix_N), order='C')

    return [A, B], defines


def generate_fconv2d_padded(defines={}, my_type=np.float32):
    """Padded 2D FP convolution data, CH x FxF.

    Faithful port of the padded fconv2d data script. The (R+F-1, C+F-1) padded
    input I is filled with random data, its (F-1)/2 border rows/cols are zeroed
    (zero_pad), the FxF filter is transposed on its (1,2) axes, and the golden
    output is sum_ch correlate2d(I[ch], F_T[ch], 'valid') -> (R, C).

    Only the mathematical input/filter/output data is generated here. Row
    partitioning and active-core selection are handled by the app.
    """
    CH = int(defines['CH'])
    R = int(defines['R'])
    C = int(defines['C'])
    F = int(defines['F'])

    np.random.seed(42)

    # Padded input feature map: (CH, R+F-1, C+F-1)
    ih = R + F - 1
    iw = C + F - 1
    vec_I = np.random.randn(CH, ih, iw).astype(my_type)
    # Filter: (CH, F, F)
    vec_F = np.random.randn(CH, F, F).astype(my_type)

    # zero_pad: zero the (F-1)/2 border columns/rows of the padded input
    pad = (F - 1) // 2
    for ch in range(CH):
        for j in range(pad):
            vec_I[ch][:, j] = 0
            vec_I[ch][:, iw - 1 - j] = 0
        for r in range(pad):
            vec_I[ch][r, :] = 0
            vec_I[ch][ih - 1 - r, :] = 0

    # Transpose the filter on its (1, 2) axes (matches gen_data.py
    # vec_F.transpose(1,2))
    vec_F_T = np.transpose(vec_F, (0, 2, 1))

    # Golden: sum over channels of valid 2D correlation -> (R, C)
    vec_GR = np.zeros((R, C), dtype=my_type)
    for ch in range(CH):
        vec_GR = vec_GR + signal.correlate2d(vec_I[ch], vec_F_T[ch],
                                             mode='valid')
    vec_GR = vec_GR.astype(my_type)

    # Reset output buffer (DMA'd into omtx by main.c): all zeros (R, C)
    vec_R = np.zeros((R, C), dtype=my_type)

    return [
        vec_I.reshape(CH * ih * iw),
        vec_F_T.reshape(CH * F * F),
        vec_R.reshape(R * C),
        vec_GR.reshape(R * C)
    ], defines


def _fft_serialize_cmplx(vector, NFFT, dtype):
    serial_vec = np.empty(2 * NFFT, dtype=dtype)
    serial_vec[0::2] = np.real(vector)
    serial_vec[1::2] = np.imag(vector)
    return serial_vec


def _fft_setup_input(samples):
    with np.nditer(samples, op_flags=['readwrite']) as it:
        for samp in it:
            samp[...]['re'] = np.random.randn(1)
            samp[...]['im'] = np.random.randn(1)


def _fft_setup_twiddles_lut(twiddles_vec, nfft):
    theta = (2 * np.pi) / nfft
    with np.nditer(twiddles_vec, op_flags=['readwrite']) as it:
        for idx, twi in enumerate(it):
            phi = 2 * np.pi - theta * (idx)
            twi[...]['re'] = np.cos(phi)
            twi[...]['im'] = np.sin(phi)


def _fft_setup_twiddles_lut_dif_vec(twiddles_vec, nfft):
    stages = int(np.log2(nfft))
    theta = (2 * np.pi) / nfft
    twi = [[np.cos((2 * np.pi) - i * theta), np.sin((2 * np.pi) - i * theta)]
           for i in range(int(nfft / 2))]
    for s in range(stages):
        for t in range(int(nfft / 2)):
            twiddles_vec[int(s * nfft / 2 + t)
                         ]['re'] = twi[int((2**(s) * t) % int(nfft / 2))][0]
            twiddles_vec[int(s * nfft / 2 + t)
                         ]['im'] = twi[int((2**(s) * t) % int(nfft / 2))][1]
    return twiddles_vec


def _fft_gen_bitrev_idx(nfft):
    fmt = '{:0' + str(int(np.log2(nfft))) + 'b}'
    bitrev = np.zeros(nfft)
    for n in np.arange(nfft):
        bitrev[n] = int(fmt.format(n)[::-1], 2)
    return bitrev


def _fft_gen_core_offset(ncore):
    max_cid = ncore - 1
    num_bits = max_cid.bit_length()
    return [int(format(cid, f'0{num_bits}b')[::-1], 2) for cid in range(ncore)]


def _fft_gen_store_idx(nfft):
    ibuf = []
    dbuf = []
    a = [*range(nfft)]
    b = np.zeros(nfft)
    old_b = a
    nffth = nfft >> 1
    for bf in range(int(np.log2(nffth))):
        stride = nffth >> (bf + 1)
        for h in range(2):
            for i in range(int(nffth / (stride << 1))):
                for j in range(stride):
                    b[h * (nffth >> 1) + i * stride + j] = \
                        old_b[h * nffth + j + 2 * i * stride]
                    b[h * (nffth >> 1) + i * stride + j + nffth] = \
                        old_b[h * nffth + j + 2 * i * stride + stride]
        delta = [[i for i in b].index(n) for n in old_b]
        ibuf += [[i for i in b]]
        dbuf += [delta]
        old_b = [n for n in copy.deepcopy(b)]
        b = np.zeros(nfft)
    idx_list = sum(ibuf, [])
    delta_list = sum(dbuf, [])
    return [idx_list, delta_list]


def _fft_twiddle_reim(nfft, dtype=np.float32):
    count = int(np.log2(nfft) * nfft / 2)
    dtype_cplx = np.dtype([('re', dtype), ('im', dtype)])
    twiddle_v = np.empty(count, dtype=dtype_cplx)
    twiddle_v = _fft_setup_twiddles_lut_dif_vec(twiddle_v, nfft)
    twiddle_s = _fft_serialize_cmplx(
        twiddle_v['re'] + 1j * twiddle_v['im'], count, dtype)
    return twiddle_s[0::2].astype(dtype), twiddle_s[1::2].astype(dtype)


def _fft_store_delta(nfft):
    if nfft <= 2:
        return np.zeros(0, dtype=np.uint16)

    _, store_delta = _fft_gen_store_idx(nfft)
    store_delta = [n * np.dtype(np.uint32).itemsize for n in store_delta]
    compact = []
    for i in range(len(store_delta) // nfft):
        compact += store_delta[i * nfft:i * nfft + nfft // 2]
    return np.array(compact, dtype=np.uint16)


def generate_fft(defines={}, my_type=np.float32):
    """FFT data independent of the execution core split."""
    NFFT = int(defines['npoints'])
    dtype = np.float32
    dtype_cplx = np.dtype([('re', dtype), ('im', dtype)])

    np.random.seed(42)
    samples = np.empty(NFFT, dtype=dtype_cplx)
    _fft_setup_input(samples)
    gold_out = np.fft.fft(samples['re'] + 1j * samples['im'])

    samples_s = _fft_serialize_cmplx(
        samples['re'] + 1j * samples['im'], NFFT, dtype)
    gold_out_s = _fft_serialize_cmplx(gold_out, NFFT, dtype)

    samples_reim = np.empty(2 * NFFT, dtype=dtype)
    samples_reim[0:NFFT] = samples_s[0::2]
    samples_reim[NFFT:2 * NFFT] = samples_s[1::2]

    twiddle_re_parts = []
    twiddle_im_parts = []
    store_idx_parts = []
    for sub_nfft in (1 << p for p in range(1, int(np.log2(NFFT)) + 1)):
        re, im = _fft_twiddle_reim(sub_nfft, dtype)
        twiddle_re_parts.append(re)
        twiddle_im_parts.append(im)
        store_idx_parts.append(_fft_store_delta(sub_nfft))

    twiddle_re = np.concatenate(twiddle_re_parts).astype(dtype)
    twiddle_im = np.concatenate(twiddle_im_parts).astype(dtype)
    store_idx = np.concatenate(store_idx_parts).astype(np.uint16)
    buffer_dram = np.zeros(2 * NFFT, dtype=dtype)

    defines['NFFT'] = NFFT
    defines['FFT_LOG2_NFFT'] = int(np.log2(NFFT))
    defines['FFT_TWIDDLE_TABLE_SIZE'] = int(twiddle_re.size)
    defines['FFT_STORE_IDX_TABLE_SIZE'] = int(store_idx.size)

    return [
        samples_reim.astype(dtype),
        buffer_dram,
        twiddle_re,
        twiddle_im,
        store_idx,
        gold_out_s.astype(dtype)
    ], defines


def generate_bandwidth(defines={}, my_type=np.int32):
    """Bandwidth-test data payload.

    Faithful port of the bandwidth-test payload generator.
    Offset/round/step traffic scheduling is an execution choice, so it is
    generated by the app instead of this data generator.
    """
    M = int(defines['M'])
    np.random.seed(42)
    return np.random.randint(-100, 100, size=M).astype(my_type), defines


# ---------------------------------------------------------------------------
# Dense-LLM benchmark data generators (FFN SwiGLU + attention, f16).
#
# These mirror the exact on-chip layouts and dtype flow of the baremetal
# kernels in software/kernels/baremetal/mempool_llm_{ffn,attention}_f16.h so
# the emitted golden output matches what the RedMulE + Spatz cluster computes:
#   * f16 tensors, f32 accumulation inside RMSNorm/softmax/RoPE.
#   * Wg/Wu are stored PANEL-MAJOR over the intermediate columns ([NUM_IP][D x
#     PANEL_I]); Wd is NATURAL [I x D]. Wq/Wk/Wv are head-major [heads][D x
#     HEAD_DIM]; Wo is [heads][HEAD_DIM x D].
#   * RoPE uses the adjacent-pair (GPT-J/interleaved) convention with a
#     precomputed cos/sin table; decode positions offset by the past length.
#   * The K cache is stored transposed [n_kv][HEAD_DIM][len] for RedMulE Q@K^T.
#
# PANEL_I (the FFN weight-streaming tile) is the one tiling knob derived from
# the L1 budget: pass L1_BYTES / PORT_WIDTH (the data generator acts as the
# build-time tiling solver, see mempool_llm_tiling.h). If PANEL_I is given
# explicitly it is honoured verbatim.
# ---------------------------------------------------------------------------
_LLM_F16 = np.float16
_LLM_F32 = np.float32
_LLM_F64 = np.float64
_LLM_U32 = np.uint32
_LLM_I32 = np.int32

# tensorpool256_spatz4 has VLEN=512. RMSNorm uses e16,m1 for its inputs and
# e32,m2 for its widening accumulator, so both views contain 32 elements.
_LLM_RMS_LANES = 32


def _llm_f32_literal(bits):
    """Return one binary32 value from its instruction-stream bit pattern."""
    return np.asarray(bits, dtype=_LLM_U32).view(_LLM_F32)[()]


# Constants as encoded in the LLVM 14 toy-LLM ELFs.
_LLM_ZERO = _llm_f32_literal(0x00000000)
_LLM_HALF = _llm_f32_literal(0x3f000000)
_LLM_ONE = _llm_f32_literal(0x3f800000)
_LLM_ONE_AND_HALF = _llm_f32_literal(0x3fc00000)
_LLM_TWO = _llm_f32_literal(0x40000000)
_LLM_EXP_HI = _llm_f32_literal(0x42a00000)
_LLM_EXP_LO = _llm_f32_literal(0xc2a00000)
_LLM_LOG2E = _llm_f32_literal(0x3fb8aa3b)
_LLM_NEG_LN2 = _llm_f32_literal(0xbf317218)
_LLM_EXP_A3 = _llm_f32_literal(0x3e2aaaab)
_LLM_EXP_A4 = _llm_f32_literal(0x3d2aaaab)
_LLM_EXP_A5 = _llm_f32_literal(0x3c088889)


def _llm_add_f32(a, b):
    """One finite binary32 add, rounded once to RNE binary32."""
    return (np.asarray(a, dtype=_LLM_F64) +
            np.asarray(b, dtype=_LLM_F64)).astype(_LLM_F32)


def _llm_mul_f32(a, b):
    """One finite binary32 multiply, rounded once to RNE binary32."""
    return (np.asarray(a, dtype=_LLM_F64) *
            np.asarray(b, dtype=_LLM_F64)).astype(_LLM_F32)


def _llm_fma_f32(a, b, c):
    """One finite binary32 fused multiply-add with RNE binary32 output.

    Binary64 is sufficient scratch precision for the bounded, normal toy-LLM
    operands. Keeping this helper explicit prevents NumPy evaluation order
    from silently replacing an instruction-level FMA with two operations.
    """
    return (np.asarray(a, dtype=_LLM_F64) *
            np.asarray(b, dtype=_LLM_F64) +
            np.asarray(c, dtype=_LLM_F64)).astype(_LLM_F32)


def _llm_mul_f16(a, b):
    """One finite binary16 multiply with RNE binary16 output."""
    return (np.asarray(a, dtype=_LLM_F64) *
            np.asarray(b, dtype=_LLM_F64)).astype(_LLM_F16)


def _llm_seed_f32(magic, x):
    x32 = np.asarray(x, dtype=_LLM_F32)
    bits = x32.view(_LLM_U32).astype(np.uint64)
    seed = (np.uint64(magic) - bits).astype(_LLM_U32)
    return seed.view(_LLM_F32)


def _llm_rsqrt(x):
    """The compiled divide-free llm_rsqrt: seed plus three Newton steps."""
    x32 = np.asarray(x, dtype=_LLM_F32)
    xhalf = _llm_mul_f32(x32, _LLM_HALF)
    shifted = x32.view(_LLM_U32) >> _LLM_U32(1)
    seed = (np.uint64(0x5f3759df) -
            shifted.astype(np.uint64)).astype(_LLM_U32)
    y = seed.view(_LLM_F32)
    for _ in range(3):
        y2 = _llm_mul_f32(y, y)
        correction = _llm_fma_f32(-xhalf, y2, _LLM_ONE_AND_HALF)
        y = _llm_mul_f32(correction, y)
    return y


def _llm_recip_pos(x):
    """The compiled positive-normal reciprocal seed plus four refinements."""
    x32 = np.asarray(x, dtype=_LLM_F32)
    y = _llm_seed_f32(0x7ef311c3, x32)
    for _ in range(4):
        correction = _llm_fma_f32(-x32, y, _LLM_TWO)
        y = _llm_mul_f32(correction, y)
    return y


def _llm_exp_range_reduce(x):
    """Shared scalar llm_expf clamp/range reduction from the target ELF."""
    x32 = np.asarray(x, dtype=_LLM_F32)
    x32 = np.maximum(np.minimum(x32, _LLM_EXP_HI),
                     _LLM_EXP_LO).astype(_LLM_F32)
    t = _llm_mul_f32(x32, _LLM_LOG2E)
    bias = np.where(t >= _LLM_ZERO, _LLM_HALF, -_LLM_HALF)
    k = np.trunc(_llm_add_f32(t, bias)).astype(_LLM_I32)
    r = _llm_fma_f32(k.astype(_LLM_F32), _LLM_NEG_LN2, x32)
    exponent = ((k + _LLM_I32(127)).astype(_LLM_U32) <<
                _LLM_U32(23))
    return r, exponent.view(_LLM_F32)


def _llm_exp_ffn_parts(x):
    """Return the FFN call context's Horner polynomial and power of two."""
    r, power = _llm_exp_range_reduce(x)
    poly = _llm_fma_f32(r, _LLM_EXP_A5, _LLM_EXP_A4)
    poly = _llm_fma_f32(poly, r, _LLM_EXP_A3)
    poly = _llm_fma_f32(poly, r, _LLM_HALF)
    poly = _llm_fma_f32(poly, r, _LLM_ONE)
    poly = _llm_fma_f32(poly, r, _LLM_ONE)
    return poly, power


def _llm_expf_softmax(x):
    """llm_expf as reassociated in the materialized-softmax caller.

    LLVM 14 with -O3 -ffast-math does not emit the FFN Horner topology here:
    it builds p-1, then folds the final +1 into the power-of-two multiply.
    The separate helper makes that call-context-specific executable behavior
    visible and testable.
    """
    r, power = _llm_exp_range_reduce(x)
    poly = _llm_fma_f32(r, _LLM_EXP_A5, _LLM_EXP_A4)
    poly = _llm_fma_f32(poly, r, _LLM_EXP_A3)
    poly = _llm_fma_f32(poly, r, _LLM_HALF)
    poly = _llm_mul_f32(poly, r)
    poly_minus_one = _llm_fma_f32(poly, r, r)
    return _llm_fma_f32(poly_minus_one, power, power)


def _llm_rmsnorm(x, gamma, eps=1e-5):
    """Spatz VLEN=512 RMSNorm, including its lane reduction topology."""
    x16 = np.asarray(x, dtype=_LLM_F16)
    gamma16 = np.asarray(gamma, dtype=_LLM_F16)
    dmodel = x16.shape[-1]
    if gamma16.shape != (dmodel,):
        raise ValueError("_llm_rmsnorm gamma must match the final x dimension")
    if dmodel > _LLM_RMS_LANES and dmodel % _LLM_RMS_LANES != 0:
        raise ValueError("exact RMSNorm oracle requires full 32-lane chunks")

    rows = x16.reshape(-1, dmodel)
    lane_count = min(dmodel, _LLM_RMS_LANES)
    lanes = np.zeros((rows.shape[0], lane_count), dtype=_LLM_F32)
    for begin in range(0, dmodel, _LLM_RMS_LANES):
        values = rows[:, begin:begin + _LLM_RMS_LANES].astype(_LLM_F32)
        width = values.shape[1]
        lanes[:, :width] = _llm_fma_f32(
            values, values, lanes[:, :width])

    sumsq = np.zeros(rows.shape[0], dtype=_LLM_F32)
    for lane in range(lane_count):
        sumsq = _llm_add_f32(sumsq, lanes[:, lane])

    inv_dmodel = _llm_recip_pos(_LLM_F32(dmodel))
    mean_square = _llm_fma_f32(sumsq, inv_dmodel, _LLM_F32(eps))
    rinv = _llm_rsqrt(mean_square).astype(_LLM_F16)
    scaled = _llm_mul_f16(rows, rinv[:, None])
    normalized = _llm_mul_f16(scaled, gamma16[None, :])
    return normalized.reshape(x16.shape)


def _llm_silu_mul(g, u):
    """Compiled scalar SiLU(g)*u, including fast-math reassociation.

    The target forms 1+exp with an FMA, completes three reciprocal Newton
    steps, then distributes the fourth step across the caller's g*u product.
    """
    g32 = np.asarray(g, dtype=_LLM_F16).astype(_LLM_F32)
    u32 = np.asarray(u, dtype=_LLM_F16).astype(_LLM_F32)
    poly, power = _llm_exp_ffn_parts(-g32)
    denominator = _llm_fma_f32(poly, power, _LLM_ONE)
    reciprocal = _llm_seed_f32(0x7ef311c3, denominator)
    for _ in range(3):
        correction = _llm_fma_f32(
            -denominator, reciprocal, _LLM_TWO)
        reciprocal = _llm_mul_f32(correction, reciprocal)
    correction = _llm_fma_f32(-denominator, reciprocal, _LLM_TWO)
    result = _llm_mul_f32(g32, u32)
    result = _llm_mul_f32(result, reciprocal)
    result = _llm_mul_f32(result, correction)
    return result.astype(_LLM_F16)


def _llm_softmax_f16(scores):
    """Materialized scalar softmax with the kernel's exact cast points."""
    rows = np.asarray(scores, dtype=_LLM_F16)
    if rows.ndim != 2 or rows.shape[1] == 0:
        raise ValueError("_llm_softmax_f16 expects nonempty 2-D rows")
    rows = rows.copy()

    max_value = rows[:, 0].astype(_LLM_F32)
    for column in range(1, rows.shape[1]):
        max_value = np.maximum(
            max_value, rows[:, column].astype(_LLM_F32))

    total = np.zeros(rows.shape[0], dtype=_LLM_F32)
    for column in range(rows.shape[1]):
        delta = _llm_add_f32(
            rows[:, column].astype(_LLM_F32), -max_value)
        exp_value = _llm_expf_softmax(delta)
        rows[:, column] = exp_value.astype(_LLM_F16)
        total = _llm_add_f32(total, exp_value)

    inverse = _llm_recip_pos(total)
    for column in range(rows.shape[1]):
        probability = _llm_mul_f32(
            rows[:, column].astype(_LLM_F32), inverse)
        rows[:, column] = probability.astype(_LLM_F16)
    return rows


def _llm_fma_f16(a, b, c):
    """One finite binary16 fused multiply-add with RNE binary16 output.

    Binary64 is host scratch precision for the product-plus-add only. The
    returned binary16 value is the accumulator passed to the next RedMulE
    contraction step.
    """
    return (a.astype(_LLM_F64) * b.astype(_LLM_F64) +
            c.astype(_LLM_F64)).astype(_LLM_F16)


def _llm_mm(a, b, addend=None):
    """RedMulE binary16 product/GEMM in natural contraction order.

    Bias-less products start from binary16 zero. The packaged LLM software
    retains an explicit zero-destination GEMM for compatibility with older
    RTL; current RTL MATMUL also initializes each traversal from zero.
    ``addend`` supplies an existing binary16 output tile for true GEMM
    accumulation. RedMulE rounds each fused multiply-add to binary16 and feeds
    that value back, rather than accumulating a complete dot product in
    binary32.
    """
    a = np.asarray(a, dtype=_LLM_F16)
    b = np.asarray(b, dtype=_LLM_F16)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("_llm_mm expects compatible two-dimensional matrices")

    shape = (a.shape[0], b.shape[1])
    if addend is None:
        acc = np.zeros(shape, dtype=_LLM_F16)
    else:
        acc = np.asarray(addend, dtype=_LLM_F16)
        if acc.shape != shape:
            raise ValueError("_llm_mm addend has incompatible shape")
        acc = acc.copy()

    for k in range(a.shape[1]):
        acc = _llm_fma_f16(a[:, k:k + 1], b[k:k + 1, :], acc)
    return acc


def _llm_pick_panel_i(S, D, inter, l1_bytes, port_width, row_tiled=False):
    """Largest PANEL_I dividing `inter` (multiple of PORT_WIDTH) whose resident
    FFN working set fits ~90% of the L1 budget, preferring >=2 panels so the
    weight stream can be double-buffered. l1_bytes==0 skips the fit check.

    The legacy full-row path keeps x, h, out, and two weight-panel sets. The
    constrained-L1 row-tiled path reuses its row activation buffer for the
    partial output and holds one weight-panel set:
      legacy: fixed=3*S*D+D, per-panel=2*S+6*D
      tiled:  fixed=S*D+D,   per-panel=2*S+3*D
    """
    if row_tiled:
        fixed = S * D + D
        per_panel = 2 * S + 3 * D
    else:
        fixed = 3 * S * D + D
        per_panel = 2 * S + 6 * D
    # The C/link-time budget is 90% of L1 after the per-core stack region.
    # The generator receives only total L1 bytes, so reserve 15% here. This is
    # conservative relative to the exact per-core-stack-plus-10% calculation
    # for the current 256-core configurations.
    budget = int(0.85 * l1_bytes) // 2 if l1_bytes else (1 << 62)
    step = port_width if (port_width and inter % port_width == 0) else 1
    cands = [p for p in range(step, inter + 1, step) if inter % p == 0]
    cands = cands or [inter]
    fit = [p for p in cands if fixed + per_panel * p <= budget]
    if not fit:
        raise Exception(
            "FFN shape S={} D={} I={} does not fit the L1 budget "
            "({} bytes) even at PANEL_I={}".format(S, D, inter, l1_bytes,
                                                   min(cands)))
    multi = [p for p in fit if inter // p >= 2]
    return max(multi) if multi else max(fit)


def generate_llm_ffn(my_type=np.float16, defines={}):
    """Dense-LLM SwiGLU FFN block: h=RMSNorm(x)*gamma; g=h@Wg; u=h@Wu;
    a=SiLU(g)*u; out=a@Wd; y=out+x. Returns [x, gamma, Wg_pm, Wu_pm, Wd, y]."""
    np.random.seed(42)
    S = int(defines['S'])
    D = int(defines['D'])
    inter = int(defines['I'])
    row_tile = int(defines.get('ROW_TILE', S))
    l1_bytes = int(defines.pop('L1_BYTES', 0))
    port_width = int(defines.pop('PORT_WIDTH', 32))

    if row_tile <= 0 or row_tile > S or S % row_tile != 0:
        raise Exception("ROW_TILE={} must be a positive divisor of S={}".format(
            row_tile, S))

    if 'PANEL_I' in defines:
        PANEL = int(defines['PANEL_I'])
    else:
        PANEL = _llm_pick_panel_i(
            row_tile, D, inter, l1_bytes, port_width, row_tile != S)
        defines['PANEL_I'] = PANEL
    if inter % PANEL != 0:
        raise Exception("PANEL_I={} must divide I={}".format(PANEL, inter))
    NUM_IP = inter // PANEL
    defines['NUM_IP'] = NUM_IP

    x = (np.random.rand(S, D) - 0.5).astype(_LLM_F16)
    gamma = (np.random.rand(D) - 0.5).astype(_LLM_F16)
    Wg = (np.random.rand(D, inter) - 0.5).astype(_LLM_F16)
    Wu = (np.random.rand(D, inter) - 0.5).astype(_LLM_F16)
    Wd = (np.random.rand(inter, D) - 0.5).astype(_LLM_F16)

    h = _llm_rmsnorm(x, gamma)
    g = np.empty((S, inter), dtype=_LLM_F16)
    u = np.empty((S, inter), dtype=_LLM_F16)
    for panel in range(NUM_IP):
        begin = panel * PANEL
        end = begin + PANEL
        g[:, begin:end] = _llm_mm(h, Wg[:, begin:end])
        u[:, begin:end] = _llm_mm(h, Wu[:, begin:end])
    a = _llm_silu_mul(g, u)
    out = None
    for panel in range(NUM_IP):
        begin = panel * PANEL
        end = begin + PANEL
        out = _llm_mm(a[:, begin:end], Wd[begin:end, :], out)
    y = (out.astype(_LLM_F32) + x.astype(_LLM_F32)).astype(_LLM_F16)

    # panel-major [NUM_IP][D x PANEL] for gate/up; natural [I x D] for down.
    wg_pm = Wg.reshape(D, NUM_IP, PANEL).transpose(1, 0, 2).reshape(-1)
    wu_pm = Wu.reshape(D, NUM_IP, PANEL).transpose(1, 0, 2).reshape(-1)
    arrays = [x.reshape(-1), gamma, wg_pm, wu_pm, Wd.reshape(-1),
              y.reshape(-1)]
    return [a.astype(my_type) for a in arrays], defines


def _llm_pick_classic_ffn_tiles(S, D, inter, l1_bytes, port_width, defines):
    """Derive token/intermediate tiles for the classic two-linear FFN.

    The resident f16 set is two token tiles (input/output), one intermediate
    panel, and two double-buffered weight panels::

      2*ROW_TILE*D + ROW_TILE*PANEL_I + 4*D*PANEL_I

    Use a conservative 80% of physical L1.  This leaves room for per-core
    stacks and runtime allocations even though the generator only receives the
    physical L1 size.
    """
    budget = int(0.8 * l1_bytes) // 2 if l1_bytes else (1 << 62)
    step = port_width if port_width else 1

    row_cands = [r for r in range(step, S + 1, step) if S % r == 0]
    panel_cands = [p for p in range(step, inter + 1, step)
                   if inter % p == 0]
    row_cands = row_cands or [S]
    panel_cands = panel_cands or [inter]

    if 'ROW_TILE' in defines:
        row_cands = [int(defines['ROW_TILE'])]
    if 'PANEL_I' in defines:
        panel_cands = [int(defines['PANEL_I'])]

    choices = []
    for rows in row_cands:
        if rows <= 0 or S % rows != 0 or rows % step != 0:
            continue
        for panel in panel_cands:
            if (panel <= 0 or inter % panel != 0 or
                    panel % step != 0):
                continue
            elems = (2 * rows * D + rows * panel +
                     4 * D * panel)
            if elems <= budget:
                choices.append((rows, panel))

    if not choices:
        raise Exception(
            "Classic FFN S={} D={} I={} has no ROW_TILE/PANEL_I that fits "
            "the L1 budget ({} bytes)".format(S, D, inter, l1_bytes))

    # Maximize row parallelism first, then prefer at least two I panels so the
    # next weight set can be prefetched while the current panel computes.
    best_rows = max(rows for rows, _ in choices)
    panels = [p for rows, p in choices if rows == best_rows]
    multi = [p for p in panels if inter // p >= 2]
    best_panel = max(multi) if multi else max(panels)
    defines['ROW_TILE'] = best_rows
    defines['PANEL_I'] = best_panel
    defines['NUM_IP'] = inter // best_panel
    return defines


def _llm_pick_classic_attention_tiles(S, D, num_heads, head_dim, past_len,
                                      l1_bytes, port_width, defines):
    """Derive FlashAttention query/key tiles for the classic MHA.

    Projected K/V live in L2.  L1 retains one input/output row tile, one
    D-by-HD projection weight, Q/K/V/Kt/context tiles, scores, and float32
    online-softmax state.  Equal BM/BN keeps all RedMulE dimensions aligned.
    """
    del num_heads  # Heads are processed serially; they do not scale L1 use.
    budget = int(0.8 * l1_bytes) // 2 if l1_bytes else (1 << 62)
    step = port_width if port_width else 1
    total_len = past_len + S

    bm_cands = [r for r in range(step, S + 1, step) if S % r == 0]
    bn_cands = [r for r in range(step, total_len + 1, step)
                if total_len % r == 0]
    bm_cands = bm_cands or [S]
    bn_cands = bn_cands or [total_len]

    if 'LLM_FLASH_BM' in defines:
        bm_cands = [int(defines['LLM_FLASH_BM'])]
    if 'LLM_FLASH_BN' in defines:
        bn_cands = [int(defines['LLM_FLASH_BN'])]

    choices = []
    for bm in bm_cands:
        if bm <= 0 or S % bm != 0 or bm % step != 0:
            continue
        for bn in bn_cands:
            if (bn <= 0 or total_len % bn != 0 or
                    bn % step != 0 or bn < bm):
                continue
            elems = (2 * bm * D + D * head_dim +
                     2 * bm * head_dim + 3 * bn * head_dim +
                     bm * bn + 4 * bm)
            if elems <= budget:
                choices.append((bm, bn))

    if not choices:
        raise Exception(
            "Classic attention S={} L={} D={} has no BM/BN tile that fits "
            "the L1 budget ({} bytes)".format(
                S, past_len, D, l1_bytes))

    # Query rows determine RedMulE parallelism.  Among equal-BM choices use
    # the largest key tile to reduce online-softmax loop overhead.
    best_bm = max(bm for bm, _ in choices)
    best_bn = max(bn for bm, bn in choices if bm == best_bm)
    defines['LLM_FLASH_BM'] = best_bm
    defines['LLM_FLASH_BN'] = best_bn
    return defines


def generate_llm_classic_ffn(my_type=np.float16, defines={}):
    """Classic Transformer FFN: ``y = ReLU(x @ W1) @ W2``.

    W1 is emitted panel-major for L2 streaming; W2 remains row-major so every
    contraction panel is contiguous.  There is deliberately no normalization,
    gated activation, or residual addition.
    """
    np.random.seed(45)
    S = int(defines['S'])
    D = int(defines['D'])
    inter = int(defines['I'])
    l1_bytes = int(defines.pop('L1_BYTES', 0))
    port_width = int(defines.pop('PORT_WIDTH', 32))
    _llm_pick_classic_ffn_tiles(
        S, D, inter, l1_bytes, port_width, defines)
    panel = int(defines['PANEL_I'])
    npanel = int(defines['NUM_IP'])

    x = (np.random.rand(S, D) - 0.5).astype(_LLM_F16)
    w1 = (np.random.rand(D, inter) - 0.5).astype(_LLM_F16)
    w2 = (np.random.rand(inter, D) - 0.5).astype(_LLM_F16)

    y = None
    for p in range(npanel):
        begin = p * panel
        end = begin + panel
        hidden = _llm_mm(x, w1[:, begin:end])
        hidden = np.maximum(hidden, _LLM_F16(0.0)).astype(_LLM_F16)
        y = _llm_mm(hidden, w2[begin:end, :], y)

    w1_pm = w1.reshape(D, npanel, panel).transpose(1, 0, 2).reshape(-1)
    arrays = [x.reshape(-1), w1_pm, w2.reshape(-1), y.reshape(-1)]
    return [a.astype(my_type) for a in arrays], defines


def generate_llm_classic_attention(my_type=np.float16, defines={}):
    """Classic causal multi-head attention, with an optional decode cache.

    This intentionally excludes RMSNorm, RoPE, GQA, QK-normalization,
    sliding-window masking, and the residual.  Decode caches are token-major
    per head so a FlashAttention key tile is contiguous in L2.
    """
    np.random.seed(46)
    S = int(defines['S'])
    D = int(defines['D'])
    num_heads = int(defines['N_HEADS'])
    head_dim = int(defines['HEAD_DIM'])
    past_len = int(defines.get('KV_LEN', 0))
    l1_bytes = int(defines.pop('L1_BYTES', 0))
    port_width = int(defines.pop('PORT_WIDTH', 32))
    if D != num_heads * head_dim:
        raise Exception("Classic attention requires D == N_HEADS*HEAD_DIM")
    _llm_pick_classic_attention_tiles(
        S, D, num_heads, head_dim, past_len, l1_bytes, port_width, defines)

    x = (np.random.rand(S, D) - 0.5).astype(_LLM_F16)
    wq = (np.random.rand(num_heads, D, head_dim) - 0.5).astype(_LLM_F16)
    wk = (np.random.rand(num_heads, D, head_dim) - 0.5).astype(_LLM_F16)
    wv = (np.random.rand(num_heads, D, head_dim) - 0.5).astype(_LLM_F16)
    wo = (np.random.rand(num_heads, head_dim, D) - 0.5).astype(_LLM_F16)

    q = np.stack([_llm_mm(x, wq[h]) for h in range(num_heads)])
    k_new = np.stack([_llm_mm(x, wk[h]) for h in range(num_heads)])
    v_new = np.stack([_llm_mm(x, wv[h]) for h in range(num_heads)])
    arrays = [x.reshape(-1)]

    if past_len:
        k_cache = (np.random.rand(num_heads, past_len, head_dim) -
                   0.5).astype(_LLM_F16)
        v_cache = (np.random.rand(num_heads, past_len, head_dim) -
                   0.5).astype(_LLM_F16)
        k = np.concatenate((k_cache, k_new), axis=1)
        v = np.concatenate((v_cache, v_new), axis=1)
        ctx = _llm_attn_core(q, k, v, num_heads, num_heads, head_dim,
                             0, past_len, past_len + S, S)
        arrays += [k_cache.reshape(-1), v_cache.reshape(-1)]
    else:
        ctx = _llm_attn_core(q, k_new, v_new, num_heads, num_heads,
                             head_dim, 0, 0, S, S)

    out = None
    for h in range(num_heads):
        out = _llm_mm(ctx[h], wo[h], out)
    arrays += [wq.reshape(-1), wk.reshape(-1), wv.reshape(-1),
               wo.reshape(-1), out.reshape(-1)]
    return [a.astype(my_type) for a in arrays], defines


def _llm_rope_tables(positions, head_dim, theta):
    half = head_dim // 2
    exps = -(2.0 * np.arange(half)) / head_dim
    inv_freq = (theta ** exps).astype(_LLM_F32)
    ang = np.outer(positions.astype(_LLM_F32), inv_freq)
    return np.cos(ang).astype(_LLM_F16), np.sin(ang).astype(_LLM_F16)


def _llm_apply_rope(x_pm, cos, sin, nheads, S, head_dim):
    out = x_pm.astype(_LLM_F32).copy()
    c = cos.astype(_LLM_F32)
    s = sin.astype(_LLM_F32)
    # Stage both lanes before either overlapping output view is updated.
    x0 = out[:, :, 0:head_dim:2].copy()
    x1 = out[:, :, 1:head_dim:2].copy()
    x1_sin = _llm_mul_f32(x1, s[None])
    x1_cos = _llm_mul_f32(x1, c[None])
    out[:, :, 0:head_dim:2] = _llm_fma_f32(
        x0, c[None], -x1_sin)
    out[:, :, 1:head_dim:2] = _llm_fma_f32(
        x0, s[None], x1_cos)
    return out.astype(_LLM_F16)


def _llm_attn_core(q, k, v, num_q, num_kv, head_dim, first_key_pos, past_len,
                   attn_len, S):
    """Causal GQA attention with the kernel's exact scale/mask/softmax flow:
    llm_rsqrt scale, f16 score materialization, and scalar custom softmax."""
    group = num_q // num_kv
    scale = _llm_rsqrt(_LLM_F32(head_dim))
    ctx = np.zeros((num_q, S, head_dim), _LLM_F16)
    key_pos = first_key_pos + np.arange(attn_len)
    for qh in range(num_q):
        kvh = qh // group
        sc = _llm_mul_f32(
            _llm_mm(q[qh], k[kvh].T).astype(_LLM_F32), scale)
        sc = sc.astype(_LLM_F16)
        for i in range(S):
            sc[i, key_pos > past_len + i] = _LLM_F16(-80.0)
        p = _llm_softmax_f16(sc)
        ctx[qh] = _llm_mm(p, v[kvh])
    return ctx


def generate_llm_attention(my_type=np.float16, defines={}):
    """Dense-LLM Llama-style attention (prefill or decode). Decode is selected
    by the presence of KV_LEN; it adds a random post-RoPE past K^T/V cache and
    applies the sliding window WINDOW_LEN. KV_CACHE_TOKEN_MAJOR selects an
    appendable [head][KV_LEN + S][HEAD_DIM] L2 layout for tiled decode; zero
    preserves the legacy transposed-K, past-only layout."""
    np.random.seed(43)
    defines.pop('L1_BYTES', None)
    defines.pop('PORT_WIDTH', None)
    S = int(defines['S'])
    D = int(defines['D'])
    num_q = int(defines['N_Q_HEADS'])
    num_kv = int(defines['N_KV_HEADS'])
    head_dim = int(defines['HEAD_DIM'])
    qk_norm = int(defines['QK_NORM'])
    theta = float(defines['ROPE_THETA'])
    decode = 'KV_LEN' in defines
    past_len = int(defines['KV_LEN']) if decode else 0
    window_len = int(defines.get('WINDOW_LEN', 0))
    cache_token_major = int(defines.get('KV_CACHE_TOKEN_MAJOR', 0))

    x = (np.random.rand(S, D) - 0.5).astype(_LLM_F16)
    gamma = (np.random.rand(D) - 0.5).astype(_LLM_F16)
    q_gamma = (np.random.rand(head_dim) - 0.5).astype(_LLM_F16)
    k_gamma = (np.random.rand(head_dim) - 0.5).astype(_LLM_F16)
    Wq = (np.random.rand(num_q, D, head_dim) - 0.5).astype(_LLM_F16)
    Wk = (np.random.rand(num_kv, D, head_dim) - 0.5).astype(_LLM_F16)
    Wv = (np.random.rand(num_kv, D, head_dim) - 0.5).astype(_LLM_F16)
    Wo = (np.random.rand(num_q, head_dim, D) - 0.5).astype(_LLM_F16)

    positions = np.arange(past_len, past_len + S)
    cos, sin = _llm_rope_tables(positions, head_dim, theta)

    h = _llm_rmsnorm(x, gamma)
    q = np.stack([_llm_mm(h, Wq[i]) for i in range(num_q)])
    k = np.stack([_llm_mm(h, Wk[i]) for i in range(num_kv)])
    v = np.stack([_llm_mm(h, Wv[i]) for i in range(num_kv)])
    if qk_norm:
        q = _llm_rmsnorm(q.reshape(-1, head_dim), q_gamma).reshape(q.shape)
        k = _llm_rmsnorm(k.reshape(-1, head_dim), k_gamma).reshape(k.shape)
    q = _llm_apply_rope(q, cos, sin, num_q, S, head_dim)
    k = _llm_apply_rope(k, cos, sin, num_kv, S, head_dim)

    arrays = [x.reshape(-1), gamma, q_gamma, k_gamma, cos.reshape(-1),
              sin.reshape(-1)]

    if decode:
        total_len = past_len + S
        past_k = (np.random.rand(num_kv, past_len, head_dim) -
                  0.5).astype(_LLM_F16)
        past_v = (np.random.rand(num_kv, past_len, head_dim) -
                  0.5).astype(_LLM_F16)
        full_kt = np.empty((num_kv, head_dim, total_len), _LLM_F16)
        full_v = np.empty((num_kv, total_len, head_dim), _LLM_F16)
        full_kt[:, :, :past_len] = past_k.transpose(0, 2, 1)
        full_v[:, :past_len, :] = past_v
        for kvh in range(num_kv):
            full_kt[kvh, :, past_len:] = k[kvh].T
            full_v[kvh, past_len:, :] = v[kvh]

        if cache_token_major:
            # Tiled decode appends in place, so reserve the S future rows in
            # the generated object and keep K contiguous by token for DMA.
            k_cache = np.zeros((num_kv, total_len, head_dim), _LLM_F16)
            v_cache = np.zeros((num_kv, total_len, head_dim), _LLM_F16)
            k_cache[:, :past_len, :] = past_k
            v_cache[:, :past_len, :] = past_v
        else:
            k_cache = past_k.transpose(0, 2, 1)
            v_cache = past_v
        if window_len == 0:
            attn_len = total_len
        else:
            attn_len = min(total_len, window_len)
        first_key_pos = total_len - attn_len
        end = first_key_pos + attn_len
        kw = full_kt[:, :, first_key_pos:end].transpose(0, 2, 1)
        vw = full_v[:, first_key_pos:end, :]
        ctx = _llm_attn_core(q, kw, vw, num_q, num_kv, head_dim, first_key_pos,
                             past_len, attn_len, S)
        arrays += [k_cache.reshape(-1), v_cache.reshape(-1)]
    else:
        ctx = _llm_attn_core(q, k, v, num_q, num_kv, head_dim, 0, 0, S, S)

    out = None
    for hh in range(num_q):
        out = _llm_mm(ctx[hh], Wo[hh], out)
    y = (out.astype(_LLM_F32) + x.astype(_LLM_F32)).astype(_LLM_F16)

    arrays += [Wq.reshape(-1), Wk.reshape(-1), Wv.reshape(-1), Wo.reshape(-1),
               y.reshape(-1)]
    return [a.astype(my_type) for a in arrays], defines


# ---------------------------------------------------------------------------
# Perf-mode (real-size) dense-LLM data: derive the tiling and list the L2
# arrays WITHOUT materializing any values. gendata_header.py emits these as
# uninitialized .l2_bss declarations so a full Llama-3.1-8B layer (352 MB of
# FFN weights) costs L2 address space only -- no giant header, no golden check.
# ---------------------------------------------------------------------------
def llm_perf_defines(app_name, defines):
    """Perf-mode tiling defines: derive PANEL_I/NUM_IP for the FFN from the L1
    budget; drop the generator-only L1_BYTES/PORT_WIDTH hints."""
    l1 = int(defines.pop('L1_BYTES', 0))
    pw = int(defines.pop('PORT_WIDTH', 32))
    if app_name.startswith('llm_classic_ffn'):
        _llm_pick_classic_ffn_tiles(
            int(defines['S']), int(defines['D']), int(defines['I']),
            l1, pw, defines)
    elif app_name.startswith('llm_classic_attention'):
        _llm_pick_classic_attention_tiles(
            int(defines['S']), int(defines['D']), int(defines['N_HEADS']),
            int(defines['HEAD_DIM']), int(defines.get('KV_LEN', 0)),
            l1, pw, defines)
    elif app_name.startswith('llm_ffn'):
        S = int(defines['S'])
        D = int(defines['D'])
        inter = int(defines['I'])
        row_tile = int(defines.get('ROW_TILE', S))
        if row_tile <= 0 or row_tile > S or S % row_tile != 0:
            raise Exception(
                "ROW_TILE={} must be a positive divisor of S={}".format(
                    row_tile, S))
        if 'PANEL_I' not in defines:
            defines['PANEL_I'] = _llm_pick_panel_i(
                row_tile, D, inter, l1, pw, row_tile != S)
        defines['NUM_IP'] = inter // int(defines['PANEL_I'])
    return defines


def llm_perf_arrays(app_name, defines):
    """(name, count) of every L2 array a perf run reserves (no values)."""
    d = defines
    if app_name.startswith('llm_classic_ffn'):
        S = int(d['S'])
        D = int(d['D'])
        inter = int(d['I'])
        return [('l2_x', S * D), ('l2_w1', D * inter),
                ('l2_w2', inter * D)]
    if app_name.startswith('llm_classic_attention'):
        S = int(d['S'])
        D = int(d['D'])
        heads = int(d['N_HEADS'])
        head_dim = int(d['HEAD_DIM'])
        arr = [('l2_x', S * D)]
        if 'KV_LEN' in d:
            kv = int(d['KV_LEN'])
            arr += [('l2_k_cache', heads * kv * head_dim),
                    ('l2_v_cache', heads * kv * head_dim)]
        arr += [('l2_wq', heads * D * head_dim),
                ('l2_wk', heads * D * head_dim),
                ('l2_wv', heads * D * head_dim),
                ('l2_wo', heads * head_dim * D)]
        return arr
    if app_name.startswith('llm_ffn'):
        S = int(d['S'])
        D = int(d['D'])
        inter = int(d['I'])
        return [('l2_x', S * D), ('l2_gamma', D), ('l2_wg', D * inter),
                ('l2_wu', D * inter), ('l2_wd', inter * D)]
    S = int(d['S'])
    D = int(d['D'])
    nq = int(d['N_Q_HEADS'])
    nkv = int(d['N_KV_HEADS'])
    hd = int(d['HEAD_DIM'])
    half = hd // 2
    arr = [('l2_x', S * D), ('l2_gamma', D), ('l2_q_gamma', hd),
           ('l2_k_gamma', hd), ('l2_cos', S * half), ('l2_sin', S * half)]
    if 'KV_LEN' in d:
        kv = int(d['KV_LEN'])
        cache_len = kv + S if int(d.get('KV_CACHE_TOKEN_MAJOR', 0)) else kv
        arr += [('l2_k_cache_t', nkv * cache_len * hd),
                ('l2_v_cache', nkv * cache_len * hd)]
    arr += [('l2_wq', D * nq * hd), ('l2_wk', D * nkv * hd),
            ('l2_wv', D * nkv * hd), ('l2_wo', nq * hd * D)]
    return arr
