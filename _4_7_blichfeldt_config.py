import numpy as np

# Math properties
ORIGIN = np.array([0, 0])
b_1 = np.array([2, 0])
b_2 = np.array([2, 4])
BASIS_MATRIX = np.column_stack([b_1, b_2])
DUAL_BASIS_MATRIX = np.linalg.inv(BASIS_MATRIX).T
# Lattice determinant is 8.
# Let S be a triangle spanned by the following points:
S = np.array([np.array([-4, 0]), np.array([2, 0]), np.array([-2, -4])])
# We split it into three parts. This is relevant for the proof.
S_LEFT = np.array([np.array([-4, 0]), np.array([-3, -2]), np.array([-2, 0])])
S_MID = np.array([np.array([-3, -2]), np.array([-2, 0]),
                 np.array([0, 0]), np.array([-2, -4])])
S_RIGHT = np.array([np.array([2, 0]), np.array([-2, -4]), np.array([0, 0])])

# The volume of S is 12.

S_LEFT_PROPERTIES = {
    'vertex_color': 'r',
    'vertex_marker': '',
    'body_color': 'r',
    'body_alpha': .5,
    'line_color': 'r',
    'line_linewidth': 1
}

S_MID_PROPERTIES = {
    'vertex_color': 'g',
    'vertex_marker': '',
    'body_color': 'g',
    'body_alpha': .5,
    'line_color': 'g',
    'line_linewidth': 1
}

S_RIGHT_PROPERTIES = {
    'vertex_color': 'b',
    'vertex_marker': '',
    'body_color': 'b',
    'body_alpha': .5,
    'line_color': 'b',
    'line_linewidth': 1
}


PARALLELEPIPED_PROPERTIES = {
    'vertex_color': 'b',
    'vertex_marker': '',
    'vertex_linewidth': 1,
    'body_color': 'b',
    'body_alpha': 0.05,
    'line_color': 'b',
    'line_linewidth': 1,
    'line_alpha': 0.1
}

P_PROPERTIES = {
    'color': 'red',
    'linewidth': 3,
    'alpha': 1,
}

BASIS_VECTOR_PROPERTIES = {
    'color': 'r',
    'scale_units': 'xy',
    'scale': 1
}


PLOTTING_BOUND = 5
PLOT_BOTTOM_LEFT_CORNER = np.array([-PLOTTING_BOUND, -PLOTTING_BOUND])
PLOT_TOP_RIGHT_CORNER = np.array([PLOTTING_BOUND, PLOTTING_BOUND])
