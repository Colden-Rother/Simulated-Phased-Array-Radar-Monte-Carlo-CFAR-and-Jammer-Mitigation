import numpy as np
from config import *
from monte_carlo import *


def jammer_detect(alpha_mc):
    # Use the single multiplier calibrated on independent noise-only cubes.
    # Estimating it from each row being tested would bias the false-alarm rate.
#-------------------------------------------------------------------
# collect all detections
#-------------------------------------------------------------------
    data_cube = np.load('sim_data/data_cube.npy')
    all_detections = []

    # go through all rows of the data cube to gather all detections
    for row_idx in range(data_cube.shape[0]):
        # Get magnitude of the looped row
        current_row = np.abs(data_cube[row_idx, :])

        # Run monte carlo function
        alpha, dets_x, thr_line, idxs = run_cfar_on_real_data(current_row, alpha_mc)

        # Store the detections with the row (y-axis) and index (x-axis) so you know where they came from
        all_detections.append({
            'row_y': row_idx,
            'indices_x': dets_x,
        })
#-------------------------------------------------------------------
# filter out jammed signals
#-------------------------------------------------------------------
    #copy of og data
    clean_data_cube = data_cube.copy()

    # Converts list of detections into a 2D Detection Map
    # Makes a grid of True/False based on detections
    detection_map = np.zeros(data_cube.shape, dtype=bool)
    for entry in all_detections:
        row = entry['row_y']
        for col in entry['indices_x']:
            detection_map[row, col] = True

    # Scan columns Bottom to Top then Left to Right
    num_rows, num_cols = data_cube.shape
    jammer_columns = []

    for c in range(num_cols):
        consecutive_count = 0
        for r in range(num_rows):
            if detection_map[r, c]:
                consecutive_count += 1
                # If 5 or more cells are together mark for nulling
                if consecutive_count >= 5:
                    jammer_columns.append(c)
                    break # Move to next column once jammer has been found
            else:
                consecutive_count = 0 # Reset if there is a gap

    # Null the jammer columns, set it = 0
    for col in jammer_columns:
        clean_data_cube[:, col] = 0

    #how many jammers taken out
    print(f"Jammer Mitigation: Nullified {len(jammer_columns)} columns.")
#-------------------------------------------------------------------
# data save
#-------------------------------------------------------------------
    #testing
    np.save('sim_data/data_cube_clean.npy', clean_data_cube)
    return jammer_columns
