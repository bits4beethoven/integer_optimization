import numpy as np

# Math properties
ORIGIN = np.array([0, 0])
b_1 = np.array([1, 1])
b_2 = np.array([1, 2])
BASIS_MATRIX = np.column_stack([b_1, b_2])
DUAL_BASIS_MATRIX = np.linalg.inv(BASIS_MATRIX).T

# LATTICE BASIS VECTOR PROPERTIES
BASIS_VECTOR_PROPERTIES = {
    'scale_units': 'xy',
    'angles': 'xy',
    'scale': 1,
    'color': 'b'
}

# LATTICE PROPERTIES
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

LATTICE_PLUS_PROPERTIES = {
    'point_color': 'g',
    'point_alpha': 1,
    'point_s': 40,
    'vector_color': 'g',
    'vector_alpha': 0.5,
    'vector_scale': 1,
    'vector_scale_units': 'y',
}

GRAM_SCHMIDT_PROPERTIES = {
    'color': 'g',
    'alpha': 0.25,
    'scale_units': 'xy',
    'scale': 1,
}


# TEXT PROPERTIES FOR ALL PLOTS
TEXT_ANNOTATION_PROPERTIES = {
    'textcoords': "offset points",
    'xytext': (1, 10),
    'ha': 'center',
    'fontsize': 12
}

TEXT_COORDINATES_PROPERTIES = {
    'textcoords': "offset points",
    'xytext': (0, -10),
    'ha': 'center',
    'fontsize': 8
}

# SPECIFIC PROPERTIES FOR THE RADII
PACKING_RADIUS_PROPERTIES = {
    'each_circle_fill': False,
    'each_circle_color': 'gray',
    'each_circle_alpha': 0.25
}

COVERING_RADIUS_PROPERTIES = {
    'main_circle_fill': True,
    'main_circle_alpha': 0.1,
    'main_circle_color': 'r',
    'each_circle_fill': False,
    'each_circle_color': 'green',
    'each_circle_alpha': 0.25,
    'triangle_body_fill': False,
    'triangle_line_color': 'r'
}

# HALFSPACE PROPERTIES
HALFSPACES_PROPERTIES = {
    'color': 'r',
    'alpha': 0.1
}

# HULL PROPERTIES
HULL_PROPERTIES = {
    'body_fill': True,
    'body_alpha': 0.1,
    'body_color': 'r',
    'vertex_color': 'r'
}

# FP PROPERTIES
FUNDAMENTAL_PARALLELEPIPED_PROPERTIES = {
    'vertex_color': 'b',
    'body_color': 'b',
    'body_alpha': 0.2,
}

# POLYHEDRON TEMPLATES.
POLYHEDRON_ONE_FILL_PROPERTIES = {
    'color': 'r',
    'alpha': 0.25
}

POLYHEDRON_TWO_FILL_PROPERTIES = {
    'color': 'g',
    'alpha': 0.25
}

POLYHEDRON_THREE_FILL_PROPERTIES = {
    'color': 'b',
    'alpha': 0.25
}

# PATH TEMPLATES. USEFUL FOR VECTORS.
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
    'alpha': 0.25
}

# FIGURE TEMPLATES.
FIGURE_ONE_PROPERTIES = {
    'vertex_color': 'r',
    'body_color': 'r',
    'body_alpha': 0.2,
}

FIGURE_TWO_PROPERTIES = {
    'vertex_color': 'b',
    'body_color': 'b',
    'body_alpha': 0.2,
}

FIGURE_THREE_PROPERTIES = {
    'vertex_color': 'g',
    'body_color': 'g',
    'body_alpha': 0.2,
}

# CIRCLE TEMPLATES
CIRCLE_ONE_PROPERTIES = {
    'color': 'yellow',
    'alpha': 0.25,
    'linewidth': 1,
    'fill': True
}

CIRCLE_TWO_PROPERTIES = {
    'color': 'green',
    'alpha': 0.25,
    'linewidth': 1,
    'fill': True
}

CIRCLE_THREE_PROPERTIES = {
    'color': 'blue',
    'alpha': 0.25,
    'linewidth': 1,
    'fill': True
}

PLOTTING_BOUND = 10
PLOT_BOTTOM_LEFT_CORNER = np.array([-PLOTTING_BOUND, -PLOTTING_BOUND])
PLOT_TOP_RIGHT_CORNER = np.array([PLOTTING_BOUND, PLOTTING_BOUND])
