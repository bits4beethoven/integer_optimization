import numpy as np

# Math properties
ORIGIN = np.array([0, 0])
b_1 = np.array([1, 2])
b_2 = np.array([3, 1])
BASIS_MATRIX = np.column_stack([b_1, b_2])
DUAL_BASIS_MATRIX = np.linalg.inv(BASIS_MATRIX).T

PATH_ONE_PROPERTIES = {
    'color': 'b',
    'scale_units': 'xy',
    'scale': 1,
    'alpha': 1
}

PATH_TWO_PROPERTIES = {
    'color': 'g',
    'scale_units': 'xy',
    'scale': 1,
    'alpha': 1
}

PLOTTING_BOUND = 6
PLOT_BOTTOM_LEFT_CORNER = np.array([-1, -3])
PLOT_TOP_RIGHT_CORNER = np.array([10, 8])
