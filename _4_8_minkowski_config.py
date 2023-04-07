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

CIRCLE_PROPERTIES = {
    'color': 'yellow',
    'alpha': 0.25,
    'linewidth': 1,
    'fill': True
}

FUNDAMENTAL_PARALLELEPIPED_PROPERTIES = {
    'vertex_color': 'b',
    'body_color': 'b',
    'body_alpha': 0.2,
}

BASIS_VECTOR_PROPERTIES = {
    'color': 'r',
    'scale_units': 'xy',
    'scale': 1
}

PATH_ONE_PROPERTIES = {
    'color': 'r',
    'scale_units': 'xy',
    'scale': 1,
    'alpha': 0.5
}

PATH_TWO_PROPERTIES = {
    'color': 'g',
    'scale_units': 'xy',
    'scale': 1,
    'alpha': 0.5
}

PLOTTING_BOUND = 4
PLOT_BOTTOM_LEFT_CORNER = np.array([-PLOTTING_BOUND, -PLOTTING_BOUND])
PLOT_TOP_RIGHT_CORNER = np.array([PLOTTING_BOUND, PLOTTING_BOUND])
