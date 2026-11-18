"""Animated 3D (frame, scan angle, angle bin) target/jammer experiment.

Run main.py to regenerate a random sequence and animate raw versus nulled scans.
The complete sequence is built first so persistent jammer columns can be found
without using the simulator's known jammer positions as the detection result.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from config import N, angle_threshold, d, lambda0
from functions import beam_shifting
from monte_carlo import _noise_estimates, calibrate_cfar_from_simulation


DATA_DIR = Path('sim_data')
JAMMER_ANGLES = (-20, -10, 20)
TARGET_WAYPOINTS = (-45, -20, -10, 20, 45, 20, -10, -20, -45)
FRAMES_PER_SEGMENT = 12
SEED = None  # A new random radar sequence each run. CFAR calibration has its own seed.


def _angles_and_beams():
    """Build the beam response, shifting only the angle-bin FFT axis.

The old FT helper shifted both axes of its 2D input, accidentally moving scan
rows. Here row 0 really is the -60-degree steering position.
    """
    x = (np.arange(N) - (N - 1) / 2) * d
    theta = np.rad2deg(np.arcsin(np.fft.fftshift(np.fft.fftfreq(N, d=d)) * lambda0))
    weights = beam_shifting(x, threshold=angle_threshold)
    pattern = np.fft.fftshift(np.fft.fft(weights, axis=-1), axes=-1)
    pattern /= np.max(np.abs(pattern))
    return theta, pattern, np.abs(pattern)


def _target_path():
    """Sweep through each jammer angle, turn around, and sweep back."""
    segments = [
        np.linspace(start, stop, FRAMES_PER_SEGMENT, endpoint=False)
        for start, stop in zip(TARGET_WAYPOINTS[:-1], TARGET_WAYPOINTS[1:])
    ]
    return np.concatenate([*segments, [TARGET_WAYPOINTS[-1]]])


def build_live_cube(seed=SEED):
    """Return target truth and a new 3D cube; save data needed by calibration."""
    rng = np.random.default_rng(seed)
    theta, pattern, beam = _angles_and_beams()
    target_angles = _target_path()
    target_bins = np.abs(theta[None, :] - target_angles[:, None]).argmin(axis=1)
    jammer_bins = np.abs(theta[None, :] - np.asarray(JAMMER_ANGLES)[:, None]).argmin(axis=1)

    # Calibrator models this exact beam plus fixed-per-frame clutter and fresh
    # receiver noise. Targets and jammers are added only to the radar cube.
    DATA_DIR.mkdir(exist_ok=True)
    np.save(DATA_DIR / 'live_rad_pattern.npy', pattern)
    np.save(DATA_DIR / 'live_theta_vals.npy', theta)

    cube = np.empty((len(target_angles), beam.shape[0], beam.shape[1]), dtype=float)
    for frame, target_bin in enumerate(target_bins):
        clutter = rng.uniform(-0.1, 0.1, beam.shape[1])
        noise = rng.uniform(-0.05, 0.05, beam.shape)
        target = np.zeros(beam.shape[1])
        target[target_bin] = 1.2
        received = beam * (clutter + target) + noise
        # Jammers are fixed in direction but fluctuate independently by scan.
        # Positive amplitudes keep them visible so persistence can be measured.
        for jammer_bin in jammer_bins:
            received[:, jammer_bin] += rng.uniform(0.65, 1.0, beam.shape[0])
        cube[frame] = np.abs(received)
    return cube, theta, beam, target_angles, target_bins, jammer_bins


def find_persistent_jammers(cube, alpha, minimum_frames=0.6):
    """CFAR each scan, then accept columns recurring across many frames.

Five consecutive scan-row hits are the v2 criterion. The extra persistence
test distinguishes fixed jammer directions from the moving target.
    """
    if not 0 < minimum_frames <= 1:
        raise ValueError('minimum_frames must lie in (0, 1]')
    candidates = np.zeros((cube.shape[0], cube.shape[2]), dtype=bool)
    for frame, scan in enumerate(cube):
        noise, eligible = _noise_estimates(scan, num_train=10, num_guard=2)
        hits = scan[:, eligible] > alpha * noise
        consecutive = np.zeros(hits.shape[1], dtype=int)
        declared = np.zeros(hits.shape[1], dtype=bool)
        for row in hits:
            consecutive = (consecutive + 1) * row
            declared |= consecutive >= 5
        candidates[frame, eligible] = declared
    persistence = candidates.mean(axis=0)
    return np.flatnonzero(persistence >= minimum_frames), persistence


def prepare_live_sim(seed=SEED, calibration_cubes=1000, validation_cubes=500):
    """Generate a run, calibrate CFAR independently, and filter found columns."""
    cube, theta, beam, target_angles, target_bins, true_jammer_bins = build_live_cube(seed)
    alpha, validation = calibrate_cfar_from_simulation(
        pfa_target=1e-6,
        calibration_cubes=calibration_cubes,
        validation_cubes=validation_cubes,
        rad_pattern_path=DATA_DIR / 'live_rad_pattern.npy',
    )
    jammer_bins, persistence = find_persistent_jammers(cube, alpha)
    clean_cube = cube.copy()
    clean_cube[:, :, jammer_bins] = 0

    # Keep the full time sequence available for inspection after the animation.
    np.save(DATA_DIR / 'live_cube_raw.npy', cube)
    np.save(DATA_DIR / 'live_cube_clean.npy', clean_cube)
    np.save(DATA_DIR / 'live_target_angles.npy', target_angles)
    np.save(DATA_DIR / 'live_jammer_bins.npy', jammer_bins)
    return {
        'raw': cube, 'clean': clean_cube, 'theta': theta, 'beam': beam,
        'target_angles': target_angles, 'target_bins': target_bins,
        'jammer_bins': jammer_bins, 'true_jammer_bins': true_jammer_bins,
        'persistence': persistence, 'alpha': alpha, 'validation': validation,
    }


def animate_live_sim(run, interval_ms=120):
    """Show one complete scan per frame, before and after column nulling."""
    raw = run['raw']
    clean = run['clean']
    theta = run['theta']
    target_bins = run['target_bins']
    beam = run['beam']
    target_angles = run['target_angles']
    jammer_bins = run['jammer_bins']
    # Shared fixed scale makes suppression visible rather than rescaling it away.
    upper = float(np.quantile(raw, 0.9995))
    extent = (-0.5, raw.shape[2] - 0.5, -angle_threshold - 0.5, angle_threshold + 0.5)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
    images = []
    markers = []
    for ax, first, title in zip(
        axes, (raw[0], clean[0]), ('Raw: moving target + fixed jammers', 'Filtered: persistent jammer columns nulled')
    ):
        image = ax.imshow(first, origin='lower', aspect='auto', extent=extent,
                          cmap='viridis', vmin=0, vmax=upper)
        marker, = ax.plot([], [], 'o', markersize=9, markerfacecolor='none',
                          markeredgecolor='cyan', markeredgewidth=2,
                          label='True target position')
        ax.set_title(title)
        ax.set_xlabel('Angle (degrees)')
        ax.legend(loc='upper right')
        images.append(image)
        markers.append(marker)
    axes[0].set_ylabel('Beam steering angle (degrees)')
    tick_angles = np.arange(-60, 61, 20)
    tick_bins = [int(np.argmin(np.abs(theta - value))) for value in tick_angles]
    for ax in axes:
        ax.set_xticks(tick_bins, [str(value) for value in tick_angles])
    fig.subplots_adjust(left=0.07, right=0.88, bottom=0.12, top=0.84, wspace=0.12)
    colorbar_ax = fig.add_axes((0.91, 0.17, 0.015, 0.62))
    fig.colorbar(images[0], cax=colorbar_ax, label='Received magnitude')

    def update(frame):
        bin_index = int(target_bins[frame])
        # The ring shows ground truth even when a jammer null erases the target.
        scan_row = int(np.argmax(beam[:, bin_index])) - angle_threshold
        for image, data, marker in zip(images, (raw[frame], clean[frame]), markers):
            image.set_data(data)
            marker.set_data([bin_index], [scan_row])
        overlap = bin_index in jammer_bins
        fig.suptitle(
            f'Frame {frame + 1}/{len(raw)}  |  target {target_angles[frame]:+.1f}°  |  '
            f'jammer-angle overlap: {"YES - target also nulled" if overlap else "no"}',
            fontsize=12,
        )
        return (*images, *markers)

    update(0)
    animation = FuncAnimation(fig, update, frames=len(raw), interval=interval_ms,
                              repeat=True, blit=False)
    return fig, animation


def run_live_sim(show=True, **kwargs):
    run = prepare_live_sim(**kwargs)
    print(f"3D cube shape (frame, scan row, angle bin): {run['raw'].shape}")
    print(f"CFAR alpha: {run['alpha']:.4f}; validation cell Pfa: "
          f"{run['validation']['measured_cell_pfa']:.3g}")
    print('Detected fixed jammer angles:',
          [round(float(run['theta'][bin_index]), 1) for bin_index in run['jammer_bins']])
    print('Frames where target enters a nulled column:',
          int(np.isin(run['target_bins'], run['jammer_bins']).sum()))
    if show:
        fig, animation = animate_live_sim(run)
        plt.show()
        return run, fig, animation
    return run


if __name__ == '__main__':
    run_live_sim()
