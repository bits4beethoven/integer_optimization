import numpy as np

# Math properties
ORIGIN = np.array([0, 0])
b_1 = np.array([-1, 2])
b_2 = np.array([0, 1])
BASIS_MATRIX = np.column_stack([b_1, b_2])
DUAL_BASIS_MATRIX = np.linalg.inv(BASIS_MATRIX).T

PLOTTING_BOUND = 6
PLOT_BOTTOM_LEFT_CORNER = np.array([-1, -1])
PLOT_TOP_RIGHT_CORNER = np.array([5, 5])
