import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from formulas import *
from functions import *
from config import *

def sim():
# -----------------------------------------------------------------
# sim setup
# -----------------------------------------------------------------
    t_f = pulse_width*scan_angles
    dt = 1
    t = np.arange(0, t_f, dt)

    x = np.arange(N) * d
    x -= (N-1)*d/2
    u = np.fft.fftfreq(N, d=d)
    u = np.fft.fftshift(u)
    theta = np.arcsin(u * lambda0) * 180/np.pi

# -----------------------------------------------------------------
# swept radiation pattern
# -----------------------------------------------------------------
    window_func = rect(x, d*N)
    beam_shift = beam_shifting(x, threshold=angle_threshold)
    aperture_scalar = window_func * beam_shift

    rad_pattern = FT(aperture_scalar, d)
    max = np.max(rad_pattern)
    rad_pattern /= max

# -----------------------------------------------------------------
# system
# -----------------------------------------------------------------
    target_pos = impulse(theta, x0=pos1, scale=1)
    jammer_pos = impulse(theta, x0=pos2, scale=1)
    jammer_pos2 = impulse(theta, x0=pos2+10, scale=1)
    jammer_pos3 = impulse(theta, x0=pos2+40, scale=1)
    clutter = noise_gen(theta, scale=0.1)
    passive_system = target_pos + clutter

# -----------------------------------------------------------------
# run sim
# -----------------------------------------------------------------
    data_cube = []
    data_active = []

    # No progress-bar package is needed for this small simulated scan.
    for t_i in t:
        theta_i = np.floor(t_i/pulse_width)
        theta_i %= scan_angles
        theta_i = int(theta_i)

        rad_pattern_i = rad_pattern[theta_i, :]
        TX = np.abs(rad_pattern_i)

        noise_i = noise_gen(theta, scale=0.05)
        jammer_i = jammer_pos*noise_gen(theta, scale=1)
        jammer_i2 = jammer_pos2*noise_gen(theta, scale=1)
        jammer_i3 = jammer_pos3*noise_gen(theta, scale=1)
        active_system = jammer_i + jammer_i2 + jammer_i3 + noise_i

        RX = TX * passive_system + active_system

        data_cube.append(RX)
        data_active.append(active_system)

    data_cube = np.array(data_cube)
    data_active = np.array(data_active)

# -----------------------------------------------------------------
# save data
# -----------------------------------------------------------------
    np.save('sim_data/time_vals', t)
    np.save('sim_data/theta_vals', theta)
    np.save('sim_data/window_func', window_func)
    np.save('sim_data/beam_shift', beam_shift)
    np.save('sim_data/rad_pattern', rad_pattern)
    np.save('sim_data/rad_pattern_max', max)
    np.save('sim_data/passive_system', passive_system)
    np.save('sim_data/active_system', data_active)
    np.save('sim_data/data_cube', data_cube)
    return
