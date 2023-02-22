'''
https://github.com/daniellerch/stegolab/blob/master/J-UNIWARD/j-uniward.py
'''
import copy
import cv2
import imageio
import scipy.fftpack
import scipy.signal
import scipy.ndimage
import numpy as np
import jpegio as jio
from PIL import Image
import random
import string
from captcha.image import ImageCaptcha
PAYLOAD = 0.5


def dct2(a):
    return scipy.fftpack.dct(scipy.fftpack.dct(a, axis=0, norm='ortho'), axis=1, norm='ortho')


def idct2(a):
    return scipy.fftpack.idct(scipy.fftpack.idct(a, axis=0, norm='ortho'), axis=1, norm='ortho')


def ternary_entropyf(pP1, pM1):
    p0 = 1 - pP1 - pM1
    P = np.hstack((p0.flatten(), pP1.flatten(), pM1.flatten()))
    H = -P * np.log2(P)
    eps = 2.2204e-16
    H[P < eps] = 0
    H[P > 1 - eps] = 0
    return np.sum(H)


def calc_lambda(rho_p1, rho_m1, message_length, n):
    l3 = 1e+3
    m3 = float(message_length + 1)
    iterations = 0
    while m3 > message_length:
        l3 = l3 * 2
        pP1 = (np.exp(-l3 * rho_p1)) / (1 + np.exp(-l3 * rho_p1) + np.exp(-l3 * rho_m1))
        pM1 = (np.exp(-l3 * rho_m1)) / (1 + np.exp(-l3 * rho_p1) + np.exp(-l3 * rho_m1))
        m3 = ternary_entropyf(pP1, pM1)

        iterations += 1
        if iterations > 10:
            return l3
    l1 = 0
    m1 = float(n)
    lamb = 0
    iterations = 0
    alpha = float(message_length) / n
    # limit search to 30 iterations and require that relative payload embedded
    # is roughly within pipeline/1000 of the required relative payload
    while float(m1 - m3) / n > alpha / 1000.0 and iterations < 300:
        lamb = l1 + (l3 - l1) / 2
        pP1 = (np.exp(-lamb * rho_p1)) / (1 + np.exp(-lamb * rho_p1) + np.exp(-lamb * rho_m1))
        pM1 = (np.exp(-lamb * rho_m1)) / (1 + np.exp(-lamb * rho_p1) + np.exp(-lamb * rho_m1))
        m2 = ternary_entropyf(pP1, pM1)
        if m2 < message_length:
            l3 = lamb
            m3 = m2
        else:
            l1 = lamb
            m1 = m2
    iterations = iterations + 1
    return lamb


def embedding_simulator(x, rho_p1, rho_m1, m):
    n = x.shape[0] * x.shape[1]
    lamb = calc_lambda(rho_p1, rho_m1, m, n)
    pChangeP1 = (np.exp(-lamb * rho_p1)) / (1 + np.exp(-lamb * rho_p1) + np.exp(-lamb * rho_m1))
    pChangeM1 = (np.exp(-lamb * rho_m1)) / (1 + np.exp(-lamb * rho_p1) + np.exp(-lamb * rho_m1))
    y = x.copy()
    randChange = np.random.rand(y.shape[0], y.shape[1])
    y[randChange < pChangeP1] = y[randChange < pChangeP1] + 1
    y[(randChange >= pChangeP1) & (randChange < pChangeP1 + pChangeM1)] = y[(randChange >= pChangeP1) & (
                randChange < pChangeP1 + pChangeM1)] - 1
    return y


def getCost(spatial, jpg):
    # step1:构造Directional filter bank(此处为Daubechies)
    # hdpf = 1D high-pass decomposition filter
    hpdf = np.array([
        -0.0544158422, 0.3128715909, -0.6756307363, 0.5853546837,
        0.0158291053, -0.2840155430, -0.0004724846, 0.1287474266,
        0.0173693010, -0.0440882539, -0.0139810279, 0.0087460940,
        0.0048703530, -0.0003917404, -0.0006754494, -0.0001174768
    ])
    sign = np.array([-1 if i % 2 else 1 for i in range(len(hpdf))])
    # [::-pipeline]表示步长=-pipeline，即把hpdf倒序过来
    # ldpf = 1D low-pass decomposition filter
    lpdf = hpdf[::-1] * sign
    # np.outer:计算外积
    F = [np.outer(lpdf.T, hpdf), np.outer(hpdf.T, lpdf), np.outer(hpdf.T, hpdf)]
    pad_size = 16
    # (512, 512) -> (544, 544)
    spatial_padded = np.pad(spatial, (pad_size, pad_size), 'symmetric')
    RC = []
    for i in range(len(F)):
        # mode=same表示最终结果和第一个参数大小一致
        # f = reference cover wavelet coefficients(LH, HL, HH for each iter)
        f = scipy.signal.correlate2d(spatial_padded, F[i], mode='same', boundary='fill')
        RC.append(f)

    # step2
    # Pre-compute impact in spatial domain when a jpeg coefficient is changed by pipeline
    spatial_impact = {}
    for i in range(8):
        for j in range(8):
            test_coeffs = np.zeros((8, 8))
            test_coeffs[i, j] = 1
            # [0]表示读取的是标准亮度分量量化表,[pipeline]是色度分量量化表
            spatial_impact[i, j] = idct2(test_coeffs) * jpg.quant_tables[0][i, j]
    # Pre-compute impact in wavelet coefficients when a jpeg coefficient is changed by pipeline
    wavelet_impact = {}
    for f_index in range(len(F)):
        for i in range(8):
            for j in range(8):
                # correlate2d：矩阵（二维张量）的互相关函数
                wavelet_impact[f_index, i, j] = scipy.signal.correlate2d(spatial_impact[i, j], F[f_index], mode='full',
                                                                         boundary='fill', fillvalue=0.)  # XXX

    # step3:计算cost function: rho[][]
    coeffs = jpg.coef_arrays[0]
    k, l = coeffs.shape
    nzAC = np.count_nonzero(jpg.coef_arrays[0]) - np.count_nonzero(jpg.coef_arrays[0][::8, ::8])
    rho = np.zeros((k, l))
    tempXi = [0.] * 3
    sgm = 2 ** (-6)
    for row in range(k):
        for col in range(l):
            mod_row = row % 8
            mod_col = col % 8
            sub_rows = list(range(row - mod_row - 6 + pad_size - 1, row - mod_row + 16 + pad_size))
            sub_cols = list(range(col - mod_col - 6 + pad_size - 1, col - mod_col + 16 + pad_size))
            for f_index in range(3):
                RC_sub = RC[f_index][sub_rows][:, sub_cols]
                wav_cover_stego_diff = wavelet_impact[f_index, mod_row, mod_col]
                tempXi[f_index] = abs(wav_cover_stego_diff) / (abs(RC_sub) + sgm)
            rho_temp = tempXi[0] + tempXi[1] + tempXi[2]
            rho[row, col] = np.sum(rho_temp)

    wet_cost = 10 ** 13
    # rho(+)
    rho_p1 = rho.copy()
    # rho(-)
    rho_m1 = rho.copy()
    rho_p1[rho_p1 > wet_cost] = wet_cost
    rho_p1[np.isnan(rho_p1)] = wet_cost
    rho_p1[coeffs > 1023] = wet_cost
    rho_m1[rho_m1 > wet_cost] = wet_cost
    rho_m1[np.isnan(rho_m1)] = wet_cost
    rho_m1[coeffs < -1023] = wet_cost
    return rho_p1, rho_m1, coeffs, nzAC


if __name__ == '__main__':
    count = 20000
    for i in range(count):
        rho_p1, rho_m1, coeffs, nzAC = getCost('../steganalysis/train/0_' + str(i) + '.jpg')
        stego_coeffs = embedding_simulator(coeffs, rho_p1, rho_m1, round(PAYLOAD * nzAC))
        jpg = jio.read('../steganalysis/train/0_' + str(i) + '.jpg')
        jpg.coef_arrays[0] = stego_coeffs
        jio.write(jpg, '../steganalysis/train/1_' + str(i) + '.jpg')  # 1表示是隐写图像
        print(str(i) + '_steganography_done!')

