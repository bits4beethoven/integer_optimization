import numpy as np

# Math properties
ORIGIN = np.array([0, 0])
b_1 = np.array([1, 2])
b_2 = np.array([3, 1])
BASIS_MATRIX = np.column_stack([b_1, b_2])
DUAL_BASIS_MATRIX = np.linalg.inv(BASIS_MATRIX).T

# Plotting window properties
LATTICE_PROPERTIES = {
    'color': 'blue',
    'linewidth': 8,
    'alpha': 0.5,
}

PLOTTING_BOUND = 6
PLOT_BOTTOM_LEFT_CORNER = np.array([-PLOTTING_BOUND, -PLOTTING_BOUND])
PLOT_TOP_RIGHT_CORNER = np.array([PLOTTING_BOUND, PLOTTING_BOUND])
