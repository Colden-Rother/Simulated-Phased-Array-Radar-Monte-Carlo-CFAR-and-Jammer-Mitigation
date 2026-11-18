import numpy as np
from formulas import *
from functions import *
from config import *

def jammer_nulls(jammers):
# -----------------------------------------------------------------
# import data
# -----------------------------------------------------------------
    theta = np.load('sim_data/theta_vals.npy')
    rad_pattern = np.load('sim_data/rad_pattern.npy')
    max = np.load('sim_data/rad_pattern_max.npy')

# -----------------------------------------------------------------
# nullification at jammer pos
# -----------------------------------------------------------------
    null = np.ones_like(theta)
    for idx in jammers:
        pos = int(theta[idx])
        null *= gap(theta, 1, x0=pos)

    new_rad_pattern = np.copy(rad_pattern)
    new_rad_pattern *= null

# -----------------------------------------------------------------
# IFT -> new window function
# -----------------------------------------------------------------
    rec_aperature_scalar = IFT(new_rad_pattern*max, d)
    rec_max = np.max(rec_aperature_scalar)
    rec_aperature_scalar /= rec_max

    x = np.arange(N) * d
    x -= (N-1)*d/2

# -----------------------------------------------------------------
# save data
# -----------------------------------------------------------------
    np.save('sim_data/rec_aperature_scalar', rec_aperature_scalar)
    np.save('sim_data/new_rad_pattern', new_rad_pattern)

    return