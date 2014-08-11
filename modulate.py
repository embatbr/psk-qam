#!/usr/bin/python3.4


"""This module contains code to simulate BPSK modulation with channel disturbed
by AWGN and AWGN + Rayleigh Fading. The flow is described below:

data_in -> [bpsk_mod] -> [channel] ->[bpsk_demod] -> data_out

For more details, see:
http://wiki.scipy.org/Cookbook/CommTheory
http://www.dsplog.com/2007/08/05/bit-error-probability-for-bpsk-modulation/
http://www.dsplog.com/2008/08/10/ber-bpsk-rayleigh-channel/
"""


import numpy as np
from scipy.special import erfc
import matplotlib.pyplot as plt

import math


FIGURES_DIR = 'images'


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

def plot_curve(abscissa, ord_analytic, ord_simulation, ord_simulation_symbol):
    plt.grid(True)
    plt.xticks(np.arange(SNR_MIN, SNR_MAX + 1, 1))
    plt.yscale('log')

    plt.plot(abscissa, ord_analytic, '-sr', linewidth=2)
    plt.plot(abscissa, ord_simulation, '-sb', linewidth=2)
    plt.plot(abscissa, ord_simulation_symbol, '-sg', linewidth=2)

    plt.legend(('analytic','simulation', 'simulation (symbol)'), loc='lower left')
    plt.xlabel('Eb/No (dB)')
    plt.ylabel('BER or SER')

def magnitude(x):
    log = math.log10(x)
    if log < 0:
        return math.floor(log)
    else:
        return math.ceil(log)

# BPSK

def bpsk_mod(data):
    return (2*data - 1)

def bpsk_demod(data_mod):
    return (data_mod + 1) / 2

# QPSK

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


def num_error_symbol(data, data_demod, k):
    if len(data) != len(data_demod):
        raise Exception('Bit strings "data" and "data_demod" must have the same length')

    error_symbol = 0

    i = 0
    while i < len(data):
        error_sum = np.sum(data[i : i + k] - data_demod[i : i + k])
        if error_sum != 0:
            error_symbol = error_symbol + 1
#        for j in range(i, i + k):
#            if data[j] != data_demod[j]:
#                error_symbol = error_symbol + 1

        i = i + k

    return error_symbol

def run(k, rayleigh_scale=None, printable=True, showable=True):
    """Execute the modulation and demodulation given the parameter 'k', which
    means the number of bases (M = 2**k). If 'rayleigh_scale' is not None, the
    Rayleigh fading is applied.
    """
    # Memory allocation
    Pe = np.empty(len(SNR))
    BER = np.empty(len(SNR))
    SER = np.empty(len(SNR))    # Symbol Error Rate

    modfuncs = [bpsk_mod, qpsk_mod]
    demodfuncs = [bpsk_demod, qpsk_demod]
    modfunc = modfuncs[k - 1]
    demodfunc = demodfuncs[k - 1]

    snr_count = 0
    # For each iteration, the flow below is executed:
    # [bit source] -> [modfunc] -> [channel] -> [demodfunc] -> [error count]
    for snr in SNR:
        Pe[snr_count] = 0.5*erfc(math.sqrt(snr))  # erfc is equivalent to the Q function
        if rayleigh_scale:
            Pe[snr_count] = 0.5*(1 - math.sqrt(snr / (snr + 1)))

        symbol_len = 10**(math.fabs(magnitude(Pe[snr_count])) + 1)
        data_len = k * symbol_len
        data = random_data(data_len)
        data_mod = modfunc(data)

        # Channel
        No = 1.0/snr    # SNR = Eb/No; Eb is constant and equals to 1 (ONE)
        noise = math.sqrt(No/2) * np.repeat(np.random.randn(symbol_len), k) # same noise value for all axes
        received = data_mod + noise
        if rayleigh_scale:
            fading = np.random.rayleigh(rayleigh_scale, size=data_len)
            received = (fading*data_mod + noise)/fading

        # Classification
        classified = np.sign(received)
        data_demod = demodfunc(classified)
        error_bit = np.where(data_demod != data)[0]
        BER[snr_count] = len(error_bit) / data_len
        SER[snr_count] = BER[snr_count]
        if k > 1:
            SER[snr_count] = num_error_symbol(data, data_demod, k) / symbol_len

        if printable:
            print('Eb/No = %d dB, BER = %4.4e, Pe = %4.4e' % (Eb_by_No_dB[snr_count],
                                                              BER[snr_count],
                                                              Pe[snr_count]))

        snr_count = snr_count + 1

    plot_curve(Eb_by_No_dB, Pe, BER, SER)
    if showable:
        plt.show()

def run_burst(k, rayleigh_scale=0.0, printable=True):
    M = 2**k

    title = '%d-PSK + AWGN + %.2f Rayleigh scale' % (M, rayleigh_scale)
    if printable:
        print(title)

    run(k, rayleigh_scale, printable=printable, showable=False)
    plt.suptitle(title)
    plt.savefig('%s/%d-psk_awgn_%2d_rayleigh.png' % (FIGURES_DIR, M, int(rayleigh_scale*100)))
    plt.clf()


if __name__ == '__main__':
    import sys
    import os.path


    if not os.path.exists(FIGURES_DIR):
        os.mkdir(FIGURES_DIR)

    param = sys.argv[1 : ]

    if len(param) < 1:
        for k in [1, 2]:
            M = 2**k
            print('%d-PSK + AWGN' % M)
            run(k, rayleigh_scale=None, printable=False, showable=False)
            plt.suptitle('%d-PSK + AWGN' % M)
            plt.savefig('%s/%d-psk_awgn.png' % (FIGURES_DIR, M))
            plt.clf()

            for rayleigh_scale in np.linspace(0.1, 1, 10):
                print('%d-PSK + AWGN + %.2f Rayleigh scale' % (M, rayleigh_scale))
                run_burst(k, rayleigh_scale, printable=False)
    else:
        k = int(param[0])
        rayleigh_scale = None
        if len(param) > 1:
            rayleigh_scale = float(param[1])

        run(k, rayleigh_scale=rayleigh_scale)