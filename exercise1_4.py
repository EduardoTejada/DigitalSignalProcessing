import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def filtrev(b, a, x):
    y = np.flip(signal.lfilter(b, a, np.flip(x)))
    return y