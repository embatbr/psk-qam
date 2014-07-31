#!/usr/bin/python3.4
# apenas para aprendizado

import matplotlib.pyplot as mp
import numpy as np

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2) / 2 * np.power(sig, 2.))

for (mu, sig) in [(-1, 1), (0, 2), (2, 3)]:
    x = np.linspace(-3, 3, 120)
    gauss = gaussian(x, mu, sig)
    mp.plot(gauss)

mp.show()