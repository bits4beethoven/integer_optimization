import numpy as np

# Math properties
ORIGIN = np.array([0, 0])
b_1 = np.array([1, 2])
b_2 = np.array([3, -3])
BASIS_MATRIX = np.column_stack([b_1, b_2])
DUAL_BASIS_MATRIX = np.linalg.inv(BASIS_MATRIX).T
POINT_X_COORDINATES = np.array([-1.5, 2.4])

POINT_X_PROPERTIES = {
    'color': 'r',
    'label': 'some point $x$'
}

VECTOR_PROPERTIES = {
    'color': 'r',
    'alpha': 0.2,
    'linewidth': 2,
    'scale_units': 'xy',
    'angles': 'xy',
    'scale': 1,
}

PLOTTING_BOUND = 6
PLOT_BOTTOM_LEFT_CORNER = np.array([-PLOTTING_BOUND, -PLOTTING_BOUND])
PLOT_TOP_RIGHT_CORNER = np.array([PLOTTING_BOUND, PLOTTING_BOUND])
