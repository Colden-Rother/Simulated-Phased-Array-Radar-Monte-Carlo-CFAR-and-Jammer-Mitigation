import numpy as np
import matplotlib.pyplot as plt
from config import *

def rect(x, tau, x0=0):
    f = np.where(tau/2 >= np.abs(x - x0), 1, 0)
    return f

def gap(x, tau, x0=0):
    f = np.where(tau/2 >= np.abs(x - x0), 0, 1)
    return f

def impulse(x, x0=0, scale=1):
    idx = np.argmin(np.abs(x - x0))
    delta = np.zeros_like(x)
    delta[idx] = scale
    return delta

def FT(f, dx):
    F = np.fft.fft(f)
    F *= dx
    F = np.fft.fftshift(F)
    return F

def IFT(F, dx):
    F /= dx
    F = np.fft.ifftshift(F)
    f = np.fft.ifft(F)
    return f

def dB(f):
    f_dB = 10 * np.log10(f)
    return f_dB