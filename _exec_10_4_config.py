import numpy as np

# Math properties
ORIGIN = np.array([0, 0])
b_1 = np.array([1, 0])
b_2 = np.array([0, 1])
BASIS_MATRIX = np.column_stack([b_1, b_2])
DUAL_BASIS_MATRIX = np.linalg.inv(BASIS_MATRIX).T

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

FUNDAMENTAL_PARALLELEPIPED_PROPERTIES = {
    'vertex_color': 'b',
    'body_color': 'b',
    'body_alpha': 0.2,
}

FIGURE_PROPERTIES = {
    'vertex_color': 'g',
    'body_color': 'g',
    'body_alpha': 0.2,
}

PATH_ONE_PROPERTIES = {
    'color': 'b',
    'scale_units': 'xy',
    'scale': 1,
    'alpha': 0.05
}

PATH_TWO_PROPERTIES = {
    'color': 'g',
    'scale_units': 'xy',
    'scale': 1,
    'alpha': 1
}

PATH_THREE_PROPERTIES = {
    'color': 'r',
    'scale_units': 'xy',
    'scale': 1,
    'alpha': 0.5
}

CONIC_POINT_PROPERTIES = {
    'color': 'b',
    'alpha': 0.2
}

CONVEX_POINT_PROPERTIES = {
    'color': 'r',
    'alpha': 0.2
}

HULL_PROPERTIES = {
    'vertex_color': 'g',
    'body_color': 'g',
    'body_alpha': 0.2,
}

CONE_PROPERTIES = {
    'vertex_color': 'r',
    'body_color': 'r',
    'body_alpha': 0.2,
}

POLYHEDRON_ONE_FILL_PROPERTIES = {
    'color': 'r',
    'alpha': 0.05
}

POLYHEDRON_TWO_FILL_PROPERTIES = {
    'color': 'g',
    'alpha': 0.05
}

POLYHEDRON_THREE_FILL_PROPERTIES = {
    'color': 'b',
    'alpha': 0.05
}

PLOTTING_BOUND = 8
PLOT_BOTTOM_LEFT_CORNER = np.array([-2, -2])
PLOT_TOP_RIGHT_CORNER = np.array([5, 5])
