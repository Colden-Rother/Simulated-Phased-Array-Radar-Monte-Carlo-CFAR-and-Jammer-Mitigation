import numpy as np
import matplotlib.pyplot as plt
from config import *

def beam_shifting(x, threshold=60):
    theta0 = np.linspace(-threshold, threshold, 2*threshold+1)
    theta0 = np.deg2rad(theta0)
    shift = np.array([np.exp(1j * k * x * np.sin(theta0[i])) for i in range(len(theta0))])
    return shift

def noise_gen(x, scale=1):
    N = x.size
    noise = np.random.rand(N)
    noise -= 0.5
    noise *= 2*scale
    return noise

def add_subplot(axes, section, title='', xlabel=''):
    plot, = axes[section].plot([], [])
    axes[section].set_xlim(-75, 75)
    axes[section].set_ylim(-1.1, 1.1)
    axes[section].grid(True)
    axes[section].set_title(title)
    axes[section].set_xlabel(xlabel)
    return plot