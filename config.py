import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------------------------------------------------
# universal constraints
# -----------------------------------------------------------------
eps0 = 8.854e-12
mu0 = np.pi*4e-7
c0 = 1/np.sqrt(eps0 * mu0)

# -----------------------------------------------------------------
# antenna parameters
# -----------------------------------------------------------------
N = 256
f0 = 10e9
pulse_width = 1
angle_threshold = 60
scan_angles = angle_threshold*2+1

lambda0 = c0/f0
k = 2*np.pi/lambda0
d = lambda0/2

# -----------------------------------------------------------------
# target locations
# -----------------------------------------------------------------
pos1 =  12        #12
pos2 = -20