"""CA-CFAR calibrated on independent, jammer-free simulated data."""

import numpy as np


def _noise_estimates(magnitudes, num_train, num_guard):
    """Mean of the training cells around each eligible cell in a row or cube."""
    samples = np.asarray(magnitudes, dtype=float)
    if samples.ndim not in (1, 2) or not np.all(np.isfinite(samples)) or np.any(samples < 0):
        raise ValueError("CFAR requires a finite, nonnegative 1D or 2D magnitude array")
    if num_train < 2 or num_train % 2 or num_guard < 0 or num_guard % 2:
        raise ValueError("Training and guard counts must be even; training count must be positive")
    half_win = (num_train + num_guard) // 2
    if samples.shape[-1] <= 2 * half_win:
        raise ValueError("Signal is too short for the requested CFAR window")

    indices = np.arange(half_win, samples.shape[-1] - half_win)
    # Leading zero makes differences of cumulative sums match Python slices.
    cumulative = np.concatenate(
        [np.zeros((*samples.shape[:-1], 1)), np.cumsum(samples, axis=-1)], axis=-1
    )
    guard_side = num_guard // 2
    left = cumulative[..., indices - guard_side] - cumulative[..., indices - half_win]
    right = cumulative[..., indices + half_win + 1] - cumulative[..., indices + guard_side + 1]
    return (left + right) / num_train, indices


def _noise_only_cube(rng, beam_magnitudes):
    """Match sim.py's background: fixed clutter plus fresh receiver noise."""
    clutter = rng.uniform(-0.1, 0.1, beam_magnitudes.shape[1])
    receiver_noise = rng.uniform(-0.05, 0.05, beam_magnitudes.shape)
    return np.abs(beam_magnitudes * clutter + receiver_noise)


def calibrate_cfar_from_simulation(
    pfa_target=1e-6, num_train=10, num_guard=2,
    calibration_cubes=1000, validation_cubes=500, seed=2026,
    rad_pattern_path='sim_data/rad_pattern.npy',
):
    """Fit one alpha on noise-only cubes, then measure it on fresh cubes.

    pfa_target is per tested cell. A false jammer column is a different
    outcome, measured below using the same five-consecutive-hit rule.
    """
    if not 0 < pfa_target < 1 or calibration_cubes < 1 or validation_cubes < 1:
        raise ValueError("Require 0 < pfa_target < 1 and positive cube counts")
    # v3 can calibrate against its live beam without replacing the v2 static data.
    beam_magnitudes = np.abs(np.load(rad_pattern_path))
    if beam_magnitudes.ndim != 2:
        raise ValueError("Expected a two-dimensional radiation pattern")
    _, eligible = _noise_estimates(beam_magnitudes, num_train, num_guard)
    training_cells = calibration_cubes * beam_magnitudes.shape[0] * len(eligible)
    if training_cells * pfa_target < 10:
        raise ValueError("Too few calibration cells for this Pfa; increase calibration_cubes")

    rng = np.random.default_rng(seed)
    ratios = []
    for _ in range(calibration_cubes):
        cube = _noise_only_cube(rng, beam_magnitudes)
        noise, indices = _noise_estimates(cube, num_train, num_guard)
        # Do not silently discard zero estimates or large noise ratios.
        ratios.append(np.divide(cube[:, indices], noise, out=np.full_like(noise, np.inf), where=noise > 0).ravel())
    # The upper tail, including its extremes, sets the false alarm threshold.
    alpha = float(np.quantile(np.concatenate(ratios), 1 - pfa_target, method='higher'))
    del ratios

    false_cells = false_columns = tested_cells = tested_columns = 0
    for _ in range(validation_cubes):
        cube = _noise_only_cube(rng, beam_magnitudes)
        noise, indices = _noise_estimates(cube, num_train, num_guard)
        hits = cube[:, indices] > alpha * noise
        false_cells += int(hits.sum())
        tested_cells += hits.size
        consecutive = np.zeros(hits.shape[1], dtype=int)
        declared = np.zeros(hits.shape[1], dtype=bool)
        for row in hits:
            consecutive = (consecutive + 1) * row
            declared |= consecutive >= 5
        false_columns += int(declared.sum())
        tested_columns += len(indices)

    return alpha, {
        'calibration_cells': training_cells,
        'validation_cells': tested_cells,
        'false_cells': false_cells,
        'measured_cell_pfa': false_cells / tested_cells,
        'validation_columns': tested_columns,
        'false_columns': false_columns,
        'measured_column_false_rate': false_columns / tested_columns,
    }


def run_cfar_on_real_data(raw_signal, alpha_mc, num_train=10, num_guard=2):
    """Detect cells using alpha calibrated separately from this row."""
    if not np.isfinite(alpha_mc) or alpha_mc <= 0:
        raise ValueError("alpha_mc must be a positive finite calibrated multiplier")
    magnitudes = np.asarray(raw_signal, dtype=float)
    if magnitudes.ndim != 1:
        raise ValueError("Expected one row of magnitude data")
    local_noise, indices = _noise_estimates(magnitudes, num_train, num_guard)
    thresholds = alpha_mc * local_noise
    detections = indices[magnitudes[indices] > thresholds]
    return float(alpha_mc), detections.tolist(), thresholds.tolist(), indices


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    alpha, report = calibrate_cfar_from_simulation()
    data_cube = np.load('sim_data/data_cube.npy')
    test_data = np.abs(data_cube[11, :])
    _, dets_x, thr_line, idxs = run_cfar_on_real_data(test_data, alpha)
    print(f"Calibrated alpha: {alpha:.4f}; validation: {report}")
    plt.figure(figsize=(12, 6))
    plt.plot(test_data, color='silver', label='Measured signal')
    plt.plot(idxs, thr_line, 'g--', label=f'Independent MC threshold (alpha={alpha:.2f})')
    plt.scatter(dets_x, [test_data[i] for i in dets_x], color='green', label='Detections')
    plt.xlabel('Angle-bin index')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
