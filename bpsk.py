"""Este módulo simula o comportamento de um sinal BPSK através dos canais com
Ruído Gaussiano Branco Aditivo (AWGN) e com desvanecimento Rayleigh + AWGN.
"""


from numpy import *
from scipy.special import erfc
import matplotlib.pyplot as plt


SNR_MIN = 0
SNR_MAX = 9
Eb_No_dB = arange(SNR_MIN, SNR_MAX + 1) # array de SNR (Eb/No) de 0 a 9 db
SNR = 10**(Eb_No_dB/10.0) # SNR linear; L_dB = 10*log_10(SNR)

# Alocando memória
Pe = empty(shape(SNR)) # Probabilidade de erro teórico
BER = empty(shape(SNR)) # Taxa de Erro de Bit (Bit Error Rate)

snr_count = 0
for snr in SNR:
    No = 1.0/snr
    Pe[snr_count] = 0.5*erfc(sqrt(snr))
    VEC_SIZE = ceil(100/Pe[snr_count]) # o tamanho do sinal é uma função de Pe

    s = 2*random.randint(0,high=2,size=VEC_SIZE) - 1 # Vetor do sinal
    No = 1.0/snr # Potência do ruído; potência média do sinal = 1
    n = sqrt(No/2)*random.randn(VEC_SIZE) # Mesmo tamanho do sinal
    r = s + n # Sinal recebido = sinal enviado + AWGN

    y = sign(r) # Decisão binária
    error = where(y != s)[0]
    error_sum = len(error)
    BER[snr_count] = error_sum/VEC_SIZE

    print('Eb_No_dB=%4.2f, BER=%10.4e, Pe=%10.4e' % (Eb_No_dB[snr_count], BER[snr_count],
                                                     Pe[snr_count]))

    snr_count = snr_count + 1

plt.grid(True)
plt.semilogy(Eb_No_dB, Pe, 'r', linewidth=2)
print(BER)
plt.semilogy(Eb_No_dB, BER, '-s', linewidth=2)
plt.legend(('analítica','simulação'))
plt.xlabel('Eb/No (dB)')
plt.ylabel('BER')
plt.show()