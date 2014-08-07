#!/usr/bin/python3.4


"""This module contains code to simulate BPSK modulation with channel disturbed
by AWGN and AWGN + Rayleigh Fading. The flow is described below:

data_in -> [bpsk_mod] -> [channel] ->[bpsk_demod] -> data_out

For more details, see:
http://wiki.scipy.org/Cookbook/CommTheory
http://www.dsplog.com/2007/08/05/bit-error-probability-for-bpsk-modulation/
"""


import numpy as np
from scipy.special import erfc
import matplotlib.pyplot as plt

import sys
import math


# Constants

SNR_MIN = 0
SNR_MAX = 9     # Possible to simulate with 10, but with 11 the system crashes
Eb_by_No_dB = np.arange(SNR_MIN, SNR_MAX + 1) # SNR (dB) array from 0 to 9 dB
SNR = 10**(Eb_by_No_dB/10.0) # Linear SNR; L_dB = 10*log_10(SNR)


# Generic functions

def random_data(length):
    """Return a random sequence of bits.
    """
    data = np.random.choice([0, 1], size=length)
    return data

def plot_curve(abscissa, ordinate_analytic, ordinate_simulation):
    plt.grid(True)
    plt.xticks(np.arange(SNR_MIN, SNR_MAX + 1, 1))
    plt.yscale('log')

    plt.plot(abscissa, ordinate_analytic, 'r', linewidth=2)
    plt.plot(abscissa, ordinate_simulation, '-s')

    plt.legend(('analytic','simulation'), loc='lower left')
    plt.xlabel('Eb/No (dB)')
    plt.ylabel('BER')

def magnitude(x):
    log = math.log10(x)
    if log < 0:
        return math.floor(log)
    else:
        return math.ceil(log)


# Codes for BPSK

def bpsk_mod(data):
    return (2*data - 1)

def bpsk_demod(data_mod):
    return (data_mod + 1) / 2

def run_bpsk(rayleigh_scale=None):
    # Memory allocation
    Pe = np.empty(len(SNR))
    BER = np.empty(len(SNR))

    snr_count = 0
    # Flow: [bit source] -> [bpsk_mod] -> [channel] -> [bpsk_demod] -> [error count]
    for snr in SNR:
        Pe[snr_count] = 0.5*erfc(math.sqrt(snr))  # Equivalent to the Q function
        data_len = 10**(math.fabs(magnitude(Pe[snr_count])) + 1)
        data = random_data(data_len)
        data_mod = bpsk_mod(data)

        # Noise from the channel
        No = 1.0/snr    # SNR = Eb/No; Eb is constant and equals to 1 (ONE)
        noise = math.sqrt(No/2) * np.random.randn(data_len)
        received = data_mod + noise
        if rayleigh_scale:
            fading = np.random.rayleigh(rayleigh_scale, size=data_len)
            received = received + fading

        # Classification
        classified = np.sign(received)
        output = bpsk_demod(classified)
        error = np.where(output != data)[0]
        BER[snr_count] = len(error)/data_len

        print('Eb/No = %d dB, BER = %4.4e, Pe = %4.4e' % (Eb_by_No_dB[snr_count],
                                                          BER[snr_count],
                                                          Pe[snr_count]))

        snr_count = snr_count + 1

    plot_curve(Eb_by_No_dB, Pe, BER)


# Codes for QPSK

def qpsk_mod(data):
    """Uses Gray labeling (counterclockwise 00, 01, 11, 10) to modulate a bit
    string divided in symbols of 2 bits.
    """
    data_mod = np.empty(len(data), dtype=float)

    i = 0
    while i < len(data):
        if data[i] == data[i + 1] == 0:         # 00
            data_mod[i] = 1/math.sqrt(2)
            data_mod[i + 1] = 1/math.sqrt(2)
        elif data[i] == 0 and data[i + 1] == 1: #01
            data_mod[i] = -1/math.sqrt(2)
            data_mod[i + 1] = 1/math.sqrt(2)
        elif data[i] == data[i + 1] == 1:       #11
            data_mod[i] = -1/math.sqrt(2)
            data_mod[i + 1] = -1/math.sqrt(2)
        elif data[i] == 1 and data[i + 1] == 0: #10
            data_mod[i] = 1/math.sqrt(2)
            data_mod[i + 1] = -1/math.sqrt(2)

        i = i + 2

    return data_mod

def qpsk_demod(data_mod):
    """Uses Gray labeling (counterclockwise 00, 01, 11, 10) to demodulate a bit
    string in symbols (represented by 2 bits).
    """
    data = np.empty(len(data_mod), dtype=int)

    i = 0
    while i < len(data_mod):
        if data_mod[i] > 0 and data_mod[i + 1] > 0:   #00
            data[i] = data[i + 1] = (0)
        elif data_mod[i] < 0 and data_mod[i + 1] > 0: #01
            data[i] = 0
            data[i + 1] = 1
        elif data_mod[i] < 0 and data_mod[i + 1] < 0: #11
            data[i] = data[i + 1] = 1
        elif data_mod[i] > 0 and data_mod[i + 1] < 0: #10
            data[i] = 1
            data[i + 1] = 0

        i = i + 2

    return data

def run_qpsk(rayleigh_scale=None):
    # Memory allocation
    Pe = np.empty(len(SNR))
    BER = np.empty(len(SNR))
    SER = np.empty(len(SNR) // 2)

    snr_count = 0
    # Flow: [bit source] -> [qpsk_mod] -> [channel] -> [qpsk_demod] -> [error count]
    for snr in SNR:
        Pe[snr_count] = 0.5*erfc(math.sqrt(snr))  # Equivalent to the Q function
        data_len = 10**(math.fabs(magnitude(Pe[snr_count])) + 1) * 2 # 2-bit-symbol
        data = random_data(data_len)
        data_mod = qpsk_mod(data)

        # Noise from the channel
        No = 1.0/snr    # SNR = Eb/No; Eb is constant and equals to 1 (ONE)
        noise = math.sqrt(No/2) * np.random.randn(data_len)
        received = data_mod + noise
        if rayleigh_scale:
            fading = np.random.rayleigh(rayleigh_scale, size=data_len)
            received = received + fading

        # Classification
        classified = np.sign(received)
        output = qpsk_demod(classified)
        error = np.where(output != data)[0]
        BER[snr_count] = len(error)/data_len

        print('Eb/No = %d dB, BER = %4.4e, Pe = %4.4e' % (Eb_by_No_dB[snr_count],
                                                          BER[snr_count],
                                                          Pe[snr_count]))

        snr_count = snr_count + 1

    plot_curve(Eb_by_No_dB, Pe, BER)


if __name__ == '__main__':
    param = sys.argv[1 : ]

    if len(param) < 1:
        print('Type "bpsk" or "qpsk"')
        sys.exit(-1)

    modname = ['bpsk', 'qpsk']
    modfunc = [run_bpsk, run_qpsk]
    if param[0] in modname:
        title = '%s + AWGN' % param[0]
        print(title)
        plt.suptitle(title)
        modfunc[modname.index(param[0])]()
        plt.figure()

        rayleigh_scale = 0.01
        if len(param) > 1:
            rayleigh_scale = float(param[1])
        title = '%s + AWGN + %.2f Rayleigh scale' % (param[0], rayleigh_scale)
        print(title)
        plt.suptitle(title)
        modfunc[modname.index(param[0])](rayleigh_scale)

        plt.show()