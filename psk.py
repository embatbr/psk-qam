"""This module contains code to simulate BPSK and QPSK modulations with channels
disturbed by AWGN and AWGN + Rayleigh Fading.

For a M-PSK, the flow is described below:

data_in -> [m-psk_mod] -> [channel] ->[m-psk_demod] -> data_out or #errors
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

    plt.show()

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

# TODO colocar o Rayleigh fading
def run_bpsk():
    # Memory allocation
    Pe = np.empty(np.shape(SNR))
    BER = np.empty(np.shape(SNR))

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

def run_qpsk():
    pass


if __name__ == '__main__':
    param = sys.argv[1 : ]

    if len(param) == 0:
        print('Type a options: "bpsk" or "qpsk"')
        sys.exit()

    if 'bpsk' in param:
        run_bpsk()

    if 'qpsk' in param:
        run_qpsk()