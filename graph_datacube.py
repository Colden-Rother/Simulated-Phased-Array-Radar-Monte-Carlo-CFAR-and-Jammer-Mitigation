import numpy as np
import matplotlib.pyplot as plt
from formulas import *
from functions import *
from config import *

# -----------------------------------------------------------------
# import data
# -----------------------------------------------------------------
t = np.load('sim_data/time_vals.npy')
theta = np.load('sim_data/theta_vals.npy')
rad_pattern = np.load('sim_data/rad_pattern.npy')
passive_system = np.load('sim_data/passive_system.npy')
active_system = np.load('sim_data/active_system.npy')
data_cube = np.load('sim_data/data_cube.npy')
#testing
clean_data_cube = np.load('sim_data/data_cube_clean.npy')

# -----------------------------------------------------------------
# post processing
# -----------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

#Raw data cube before processing
im1 = ax1.imshow(np.abs(data_cube), cmap='viridis', origin='lower')
ax1.set_title("Data Cube Unfiltered Jammer")
ax1.imshow(data_cube, cmap='viridis', origin='lower')
fig.colorbar(im1, ax=ax1, label='Intensity')
ax1.set_xlabel('Element')
ax1.set_ylabel('t (dt)')

#cleaned data cube post-processing
im2 = ax2.imshow(np.abs(clean_data_cube), cmap='viridis', origin='lower')
ax2.set_title("Data Cube Filtered Jammer")
ax2.imshow(clean_data_cube, cmap='viridis', origin='lower')
fig.colorbar(im2, ax=ax2, label='Intensity')
ax2.set_xlabel('Element')
ax2.set_ylabel('t (dt)')

plt.tight_layout()
plt.show()