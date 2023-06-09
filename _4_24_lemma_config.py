import numpy as np

# Math properties
ORIGIN = np.array([0, 0])
b_1 = np.array([-1, -2])
b_2 = np.array([-2, -1])
BASIS_MATRIX = np.column_stack([b_1, b_2])
DUAL_BASIS_MATRIX = np.linalg.inv(BASIS_MATRIX).T
LEMMA_BOUND = 2**(3/2) / 4

# Plotting window properties
LATTICE_PROPERTIES = {
    'color': 'blue',
    'linewidth': 8,
    'alpha': 0.5,
}

DUAL_LATTICE_PROPERTIES = {
    'color': 'red',
    'linewidth': 1,
    'alpha': 0.5,
}

GRAM_SCHMIDT_PROPERTIES = {
    'color': 'g',
    'alpha': 0.25,
    'scale_units': 'xy',
    'scale': 1,
}

CIRCLE_PROPERTIES = {
    'color': 'yellow',
    'alpha': 0.25,
    'linewidth': 1,
    'fill': True
}

PRODUCT_CIRCLE_PROPERTIES = {
    'color': 'green',
    'alpha': 0.25,
    'linewidth': 1,
    'fill': True
}

PLOTTING_BOUND = 5
PLOT_BOTTOM_LEFT_CORNER = np.array([-PLOTTING_BOUND, -PLOTTING_BOUND])
PLOT_TOP_RIGHT_CORNER = np.array([PLOTTING_BOUND, PLOTTING_BOUND])
