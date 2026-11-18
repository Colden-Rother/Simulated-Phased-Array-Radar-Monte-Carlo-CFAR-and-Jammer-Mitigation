"""Interactive range-by-angle radar with bouncing targets and measured tracks.

The cube is frame x range bin x angle bin. Dots and predictions use CFAR
measurements; simulated truth only evaluates whether a detection is correct.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.widgets import Button, Slider
import numpy as np

from config import N, d, lambda0
from live_sim import find_persistent_jammers
from monte_carlo import _noise_estimates, calibrate_cfar_from_simulation


DATA_DIR = Path('sim_data')
JAMMER_ANGLES = (-20, -10, 20)
FRAMES = 121
RANGE_BINS = 121
MAX_TARGETS = 100
MAX_TRAIL_LINES = 10  # Dots remain visible for all targets; trails would crowd the plot.
NULL_HALF_WIDTH_BINS = 5
MAX_PREDICT_FRAMES = 12
RANGE_LIMITS = (5.0, RANGE_BINS - 6.0)
ANGLE_LIMITS = (-55.0, 55.0)


def angle_grid():
    """Angle assigned to each FFT bin; the vertical axis is now range."""
    return np.rad2deg(np.arcsin(np.fft.fftshift(np.fft.fftfreq(N, d=d)) * lambda0))


def random_paths(rng, target_count, frames=FRAMES):
    """Pick independent 2D positions and headings, then bounce off all walls."""
    if not 1 <= target_count <= MAX_TARGETS:
        if target_count == 0:
            return np.empty((frames, 0, 2), dtype=float)
        raise ValueError(f'target_count must be 0 through {MAX_TARGETS}')
    # Relax initial spacing as the display fills. Movement can still bring
    # targets together later; the detector may then miss or merge them.
    min_separation = 12 if target_count <= 20 else (7 if target_count <= 50 else 4)
    positions = np.empty((target_count, 2), dtype=float)  # angle degrees, range bin
    for target in range(target_count):
        for _ in range(1000):
            candidate = np.array([rng.uniform(*ANGLE_LIMITS), rng.uniform(*RANGE_LIMITS)])
            # Start away from jammer blinds and other targets, so the detector
            # can initialize tracks from measurements rather than ground truth.
            clear = all(abs(candidate[0] - jammer) > 7 for jammer in JAMMER_ANGLES)
            separated = all(np.linalg.norm(candidate - prior) > min_separation
                            for prior in positions[:target])
            if clear and separated:
                positions[target] = candidate
                break
        else:
            raise RuntimeError('Could not place separated starting targets')

    headings = rng.uniform(0, 2 * np.pi, target_count)
    speeds = rng.uniform(0.9, 2.1, target_count)
    velocities = np.column_stack((np.cos(headings), np.sin(headings))) * speeds[:, None]
    path = np.empty((frames, target_count, 2), dtype=float)
    path[0] = positions
    for frame in range(1, frames):
        positions = positions + velocities
        for axis, bounds in enumerate((ANGLE_LIMITS, RANGE_LIMITS)):
            low, high = bounds
            below = positions[:, axis] < low
            positions[below, axis] = 2 * low - positions[below, axis]
            velocities[below, axis] *= -1
            above = positions[:, axis] > high
            positions[above, axis] = 2 * high - positions[above, axis]
            velocities[above, axis] *= -1
        path[frame] = positions
    return path


def simulate_sequence(rng, theta, target_count):
    """Build a fresh noisy range-angle image for every motion frame."""
    paths = random_paths(rng, target_count)
    columns = np.abs(theta[None, None, :] - paths[:, :, 0, None]).argmin(axis=2)
    rows = np.rint(paths[:, :, 1]).astype(int)
    truth = np.stack((columns, rows), axis=-1)  # displayed x, y pixel positions
    jammer_bins = np.abs(theta[None, :] - np.asarray(JAMMER_ANGLES)[:, None]).argmin(axis=1)
    cube = np.empty((FRAMES, RANGE_BINS, N), dtype=float)
    for frame in range(FRAMES):
        # Same noise-only model used by Monte Carlo: clutter fixed by angle
        # within one frame, plus independent receiver noise in each cell.
        clutter = rng.uniform(-0.1, 0.1, N)
        received = clutter[None, :] + rng.uniform(-0.05, 0.05, (RANGE_BINS, N))
        for col, row in truth[frame]:
            received[row, col] += 1.2
        for col in jammer_bins:
            received[:, col] += rng.uniform(0.65, 1.0, RANGE_BINS)
        cube[frame] = np.abs(received)
    return cube, paths, truth, jammer_bins


def detected_candidates(clean_frame, alpha):
    """Return measured (angle bin, range bin) points that pass CFAR."""
    noise, eligible = _noise_estimates(clean_frame, num_train=10, num_guard=2)
    examined = clean_frame[:, eligible]
    rows, local_cols = np.where((examined > alpha * noise) & (examined >= 0.20))
    points = np.column_stack((eligible[local_cols], rows)).astype(float)
    return points, examined[rows, local_cols]


def _matches_truth(point, truth):
    """Evaluate a measured dot; this result is never used for tracking."""
    return bool(np.any(np.linalg.norm(truth - point, axis=1) <= 1.5))


def _predict(track, x_limits):
    """Continue measured velocity and reflect from the known screen edges."""
    point = track['pos'] + track['vel']
    velocity = track['vel'].copy()
    for axis, (low, high) in enumerate((x_limits, RANGE_LIMITS)):
        if point[axis] < low:
            point[axis] = 2 * low - point[axis]
            velocity[axis] *= -1
        elif point[axis] > high:
            point[axis] = 2 * high - point[axis]
            velocity[axis] *= -1
    return point, velocity


def track_targets(clean_cube, alpha, target_count, truth, x_limits):
    """Associate 2D CFAR hits; coast on velocity when a blind band hides one."""
    tracks, history = [], []
    for frame, image in enumerate(clean_cube):
        points, strengths = detected_candidates(image, alpha)
        available = set(range(len(points)))
        if frame == 0:
            strongest = np.argsort(strengths)[::-1][:target_count]
            for k in sorted(strongest, key=lambda j: points[j, 0]):
                point = points[k].copy()
                tracks.append({'id': len(tracks), 'pos': point, 'vel': np.zeros(2),
                               'last_measured': point.copy(), 'last_frame': frame,
                               'missed': 0,
                               'status': 'detected' if _matches_truth(point, truth[frame]) else 'false'})
            available.difference_update(strongest.tolist())
        else:
            for track in tracks:
                prediction, predicted_velocity = _predict(track, x_limits)
                nearest = min(available, key=lambda k: np.linalg.norm(points[k] - prediction),
                              default=None)
                gate = 8 + min(track['missed'], 8) * 2
                if nearest is not None and np.linalg.norm(points[nearest] - prediction) <= gate:
                    point = points[nearest].copy()
                    elapsed = frame - track['last_frame']
                    observed_velocity = (point - track['last_measured']) / elapsed
                    track.update(pos=point, vel=0.4 * predicted_velocity + 0.6 * observed_velocity,
                                 last_measured=point.copy(), last_frame=frame, missed=0,
                                 status='detected' if _matches_truth(point, truth[frame]) else 'false')
                    available.remove(nearest)
                else:
                    track.update(pos=prediction, vel=predicted_velocity,
                                 missed=track['missed'] + 1)
                    track['status'] = ('predicted' if track['missed'] <= MAX_PREDICT_FRAMES
                                       else 'lost')
            # Reacquire an unmatched measured peak after a long gap. Identity
            # may switch when several targets approach each other.
            for track in tracks:
                if track['status'] != 'lost' or not available:
                    continue
                k = min(available, key=lambda j: np.linalg.norm(points[j] - track['pos']))
                point = points[k].copy()
                track.update(pos=point, vel=np.zeros(2), last_measured=point.copy(),
                             last_frame=frame, missed=0,
                             status='detected' if _matches_truth(point, truth[frame]) else 'false')
                available.remove(k)
            for k in sorted(available, key=lambda j: -strengths[j]):
                if len(tracks) >= target_count:
                    break
                point = points[k].copy()
                tracks.append({'id': len(tracks), 'pos': point, 'vel': np.zeros(2),
                               'last_measured': point.copy(), 'last_frame': frame,
                               'missed': 0,
                               'status': 'detected' if _matches_truth(point, truth[frame]) else 'false'})
        history.append([{**track, 'pos': track['pos'].copy(), 'vel': track['vel'].copy()}
                        for track in tracks])
    return history


def build_run(rng, theta, alpha, target_count):
    raw, paths, truth, true_jammers = simulate_sequence(rng, theta, target_count)
    jammer_centers, persistence = find_persistent_jammers(raw, alpha)
    nulled = np.zeros(N, dtype=bool)
    for center in jammer_centers:
        nulled[max(0, center - NULL_HALF_WIDTH_BINS):center + NULL_HALF_WIDTH_BINS + 1] = True
    clean = raw.copy()
    clean[:, :, nulled] = 0
    x_limits = tuple(float(np.argmin(np.abs(theta - angle))) for angle in ANGLE_LIMITS)
    history = track_targets(clean, alpha, target_count, truth, x_limits)
    DATA_DIR.mkdir(exist_ok=True)
    np.save(DATA_DIR / 'interactive_cube_raw.npy', raw)
    np.save(DATA_DIR / 'interactive_cube_clean.npy', clean)
    np.save(DATA_DIR / 'interactive_target_angles.npy', paths[:, :, 0])
    np.save(DATA_DIR / 'interactive_target_ranges.npy', paths[:, :, 1])
    np.save(DATA_DIR / 'interactive_jammer_bins.npy', jammer_centers)
    return {'raw': raw, 'clean': clean, 'paths': paths, 'truth': truth,
            'true_jammers': true_jammers, 'jammer_centers': jammer_centers,
            'nulled': nulled, 'history': history, 'persistence': persistence,
            'target_count': target_count}


class InteractiveRadar:
    """Persistent window, animation, and widget callbacks."""

    def __init__(self, calibration_cubes=1000, validation_cubes=500):
        self.theta = angle_grid()
        DATA_DIR.mkdir(exist_ok=True)
        # Unit gain makes calibration match the range-angle background above.
        np.save(DATA_DIR / 'interactive_background_gain.npy', np.ones((RANGE_BINS, N)))
        self.alpha, self.validation = calibrate_cfar_from_simulation(
            pfa_target=1e-6, calibration_cubes=calibration_cubes,
            validation_cubes=validation_cubes,
            rad_pattern_path=DATA_DIR / 'interactive_background_gain.npy',
        )
        self.count = 1
        self.pending_count = 1
        self.run = build_run(np.random.default_rng(), self.theta, self.alpha, self.count)
        self.frame, self.playing = 0, True
        self._make_figure()
        print(f'CFAR alpha {self.alpha:.4f}; validation cell Pfa '
              f'{self.validation["measured_cell_pfa"]:.3g}')

    def _make_figure(self):
        self.fig, self.axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
        self.fig.subplots_adjust(left=0.07, right=0.87, bottom=0.20, top=0.84, wspace=0.12)
        extent = (-0.5, N - 0.5, -0.5, RANGE_BINS - 0.5)
        self.images, self.dots, self.trails = [], [], []
        for ax, data, title in zip(self.axes,
                                   (self.run['raw'][0], self.run['clean'][0]),
                                   ('Raw radar', 'Jammer filtered + tracked targets')):
            image = ax.imshow(data, origin='lower', aspect='auto', extent=extent,
                              cmap='viridis', vmin=0, vmax=1.3)
            dot = ax.scatter([], [], s=65, c=[], edgecolors='white', linewidths=0.6,
                             zorder=5)
            ax.set_title(title)
            ax.set_xlabel('Angle (degrees)')
            tick_angles = np.arange(-60, 61, 20)
            tick_bins = [int(np.argmin(np.abs(self.theta - value))) for value in tick_angles]
            ax.set_xticks(tick_bins, [str(value) for value in tick_angles])
            self.images.append(image)
            self.dots.append(dot)
            self.trails.append([ax.plot([], [], '-', color='white', alpha=0.6,
                                        linewidth=1.2)[0] for _ in range(MAX_TRAIL_LINES)])
        self.axes[0].set_ylabel('Range bin')
        cax = self.fig.add_axes((0.90, 0.24, 0.014, 0.55))
        self.fig.colorbar(self.images[0], cax=cax, label='Received magnitude')
        legend = [Line2D([], [], marker='o', linestyle='', color=color,
                         markeredgecolor='white', label=label)
                  for color, label in [('lime', 'Detected target'),
                                       ('orange', 'Predicted, no detection'),
                                       ('red', 'False match / lost')]]
        self.fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(0.5, 0.93),
                        ncol=3, fontsize=9)
        ax_random = self.fig.add_axes((0.12, 0.055, 0.17, 0.065))
        ax_pause = self.fig.add_axes((0.32, 0.055, 0.12, 0.065))
        ax_slider = self.fig.add_axes((0.55, 0.065, 0.30, 0.04))
        self.random_button = Button(ax_random, 'New random movement')
        self.pause_button = Button(ax_pause, 'Pause')
        self.slider = Slider(ax_slider, 'Targets', 0, MAX_TARGETS, valinit=1, valstep=1)
        self.random_button.on_clicked(self._new_random)
        self.pause_button.on_clicked(self._toggle_pause)
        self.slider.on_changed(self._change_count)
        # Regenerate once on release, instead of at every integer while dragging.
        self.fig.canvas.mpl_connect('button_release_event', self._commit_count)
        self._draw_frame()
        self.animation = FuncAnimation(self.fig, self._tick, interval=140,
                                       cache_frame_data=False, blit=False)

    def _draw_frame(self):
        current = self.run['history'][self.frame]
        color = {'detected': 'lime', 'predicted': 'orange', 'false': 'red', 'lost': 'red'}
        offsets = np.array([track['pos'] for track in current], dtype=float).reshape(-1, 2)
        colors = [color[track['status']] for track in current]
        for image, data, dot, lines in zip(
            self.images, (self.run['raw'][self.frame], self.run['clean'][self.frame]),
            self.dots, self.trails,
        ):
            image.set_data(data)
            dot.set_offsets(offsets)
            dot.set_color(colors)
            recent = self.run['history'][max(0, self.frame - 12):self.frame + 1]
            for target_id, line in enumerate(lines):
                if self.count > MAX_TRAIL_LINES:
                    line.set_data([], [])
                    continue
                points = [snapshot[target_id]['pos'] for snapshot in recent
                          if len(snapshot) > target_id]
                line.set_data([point[0] for point in points],
                              [point[1] for point in points])
                line.set_linestyle('--' if target_id < len(current) and
                                   current[target_id]['status'] == 'predicted' else '-')
        counts = {status: sum(t['status'] == status for t in current)
                  for status in ('detected', 'predicted', 'false', 'lost')}
        self.fig.suptitle(
            f'Frame {self.frame + 1}/{FRAMES} | {self.count} moving target(s) | '
            f"detected {counts['detected']}  predicted {counts['predicted']}  "
            f"false/lost {counts['false'] + counts['lost']}", fontsize=12,
        )
        self.fig.canvas.draw_idle()

    def _tick(self, _):
        if self.playing:
            self.frame = (self.frame + 1) % FRAMES
            self._draw_frame()
        return (*self.images, *self.dots)

    def _new_random(self, _):
        # Reuse calibrated alpha; regenerate positions, headings, and noise.
        self.count = int(self.slider.val)
        self.pending_count = self.count
        self.run = build_run(np.random.default_rng(), self.theta, self.alpha, self.count)
        self.frame, self.playing = 0, True
        self.pause_button.label.set_text('Pause')
        self._draw_frame()
        print('New 2D random movement; jammer bins:', self.run['jammer_centers'].tolist())

    def _toggle_pause(self, _):
        self.playing = not self.playing
        self.pause_button.label.set_text('Pause' if self.playing else 'Play')
        self.fig.canvas.draw_idle()

    def _change_count(self, value):
        self.pending_count = int(value)

    def _commit_count(self, _):
        if self.pending_count != self.count:
            self._new_random(None)


def main():
    app = InteractiveRadar()
    plt.show()
    return app


if __name__ == '__main__':
    main()
