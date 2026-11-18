from formulas import *
from functions import *
from config import *
from sim import *
from monte_carlo import *
from nullification import *
from jammer_detect import *

#-------------------------------------------------------------------
# run sim and plot
#-------------------------------------------------------------------
sim()

# Calibrate on fresh, target-free and jammer-free simulations. The calibration
# uses a separate random generator and never examines data_cube.npy.
alpha_mc, cfar_report = calibrate_cfar_from_simulation(pfa_target=1e-6)
print(f"CFAR alpha: {alpha_mc:.4f}")
print(f"Calibration cells: {cfar_report['calibration_cells']:,}")
print(f"Validation cell Pfa: {cfar_report['false_cells']}/{cfar_report['validation_cells']:,} "
      f"= {cfar_report['measured_cell_pfa']:.3g}")
print(f"Validation false jammer columns: {cfar_report['false_columns']}/"
      f"{cfar_report['validation_columns']:,}")

#-------------------------------------------------------------------
# run sim and plot
#-------------------------------------------------------------------
jammer_columns = jammer_detect(alpha_mc)

# -----------------------------------------------------------------
# jammer nullification
# -----------------------------------------------------------------
jammer_nulls(jammer_columns)

# -----------------------------------------------------------------
# load data
# -----------------------------------------------------------------
t = np.load('sim_data/time_vals.npy')
theta = np.load('sim_data/theta_vals.npy')
window_func = np.load('sim_data/window_func.npy')
beam_shift = np.load('sim_data/beam_shift.npy')
rad_pattern = np.load('sim_data/rad_pattern.npy')
max = np.load('sim_data/rad_pattern_max.npy')
passive_system = np.load('sim_data/passive_system.npy')
active_system = np.load('sim_data/active_system.npy')
data_cube = np.load('sim_data/data_cube.npy')
rec_aperature_scalar = np.load('sim_data/rec_aperature_scalar.npy')
new_rad_pattern = np.load('sim_data/new_rad_pattern.npy')

# -----------------------------------------------------------------
# setup data
# -----------------------------------------------------------------
aperture_scalar = window_func * beam_shift
new_window_func = rec_aperature_scalar / beam_shift
x = np.arange(N) * d
x -= (N-1)*d/2

# -----------------------------------------------------------------
# graph window functions
# -----------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(12, 3))

line0, = axes[0].plot([], [])
line0b,  = axes[0].plot([], [], '--')
axes[0].set_xlim(-75, 75)
axes[0].set_ylim(0, 1.1)

line1, = axes[1].plot([], [])
line1b, = axes[1].plot([], [], '--')
axes[1].set_xlim(-75, 75)
axes[1].set_ylim(-1.1, 1.1)

line2, = axes[2].plot(x, window_func)
line2b, = axes[2].plot([], [], '--')
axes[2].set_xlim(-255*d/2, 255*d/2)
axes[2].set_ylim(-1.1, 1.1)

for ax in axes:
    ax.grid(True)

plt.tight_layout()

# ---------------------------------------------------------
# frame update
# ---------------------------------------------------------
def update(t_i):
    theta_i = t_i % scan_angles

    rad_pattern_i = np.abs(rad_pattern[theta_i, :])
    new_rad_pattern_i = np.abs(new_rad_pattern[theta_i, :])

    aperture_scalar_i = aperture_scalar[theta_i, :]
    rec_aperature_scalar_i = rec_aperature_scalar[theta_i, :]

    new_window_func_i = new_window_func[theta_i, :]


    line0.set_data(theta, rad_pattern_i)
    line0b.set_data(theta, new_rad_pattern_i)

    line1.set_data(theta, aperture_scalar_i)
    line1b.set_data(theta, rec_aperature_scalar_i)

    line2b.set_data(x, new_window_func_i)

    return line0, line0b, line1, line1b, line2b

# ---------------------------------------------------------
# animate
# ---------------------------------------------------------
ani = FuncAnimation(
    fig,
    update,
    frames=t,
    interval=10,
    blit=False
)
plt.show()
