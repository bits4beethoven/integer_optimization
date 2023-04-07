import numpy as np
import matplotlib.pyplot as plt
import _5_13_L_plus_config as local_config
import bib_config as global_config
from bib import *

# Create plot
fig, ax = plt.subplots()
setup_plot(ax, local_config.PLOT_BOTTOM_LEFT_CORNER,
           local_config.PLOT_TOP_RIGHT_CORNER)

# lattice_points = plot_lattice(ax, c.BASIS_MATRIX, c.PLOT_BOTTOM_LEFT_CORNER, c.PLOT_TOP_RIGHT_CORNER, False,**c.LATTICE_PROPERTIES)

v_1 = np.array([1, 1])
v_2 = np.array([-2, -1])
Y_with_column_vectors = np.column_stack([v_1, v_2])

plot_plus_lattice(ax, Y_with_column_vectors, global_config)

# Show the plot
plt.show()
