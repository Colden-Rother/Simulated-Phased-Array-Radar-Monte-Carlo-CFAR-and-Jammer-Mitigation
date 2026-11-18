import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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

# -----------------------------------------------------------------
# plot setup
# -----------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(12, 3))

line0 = add_subplot(
    axes,
    section=0,
    title='TX Rad. Pattern',
    xlabel='Steering Angle (deg)'
)

line1 = add_subplot(
    axes,
    section=1,
    title='Passive and Active Systems',
    xlabel='Pos. Angle (deg)'
)
line1b, = axes[1].plot([], [])
line1.set_data(theta, passive_system)

line2 = add_subplot(
    axes,
    section=2,
    title='RX Rad. Pattern',
    xlabel='Steering Angle (deg)'
)

plt.tight_layout()

# -----------------------------------------------------------------
# frame update
# -----------------------------------------------------------------
def update(t_i):
    theta_i = t_i % scan_angles
    TX = np.abs(rad_pattern[theta_i, :])

    active_i = active_system[t_i, :]

    RX = TX * passive_system + active_i

    line0.set_data(theta, TX)
    # line1.set_data(theta, passive_system)
    line1b.set_data(theta, active_i)
    line2.set_data(theta, RX)

    return line0, line1, line2
    return

# -----------------------------------------------------------------
# animate
# -----------------------------------------------------------------
animate = FuncAnimation(
    fig,
    update,
    frames=t,
    interval=10,
    blit=False)

plt.show()