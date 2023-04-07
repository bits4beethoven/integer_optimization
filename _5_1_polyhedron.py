import numpy as np
import matplotlib.pyplot as plt
import _5_1_polyhedron_config as local_config
import bib_config as global_config
from bib import *

# Create plot
fig, ax = plt.subplots()
setup_plot(ax, local_config.PLOT_BOTTOM_LEFT_CORNER,
           local_config.PLOT_TOP_RIGHT_CORNER)

lattice_points = plot_lattice(ax, local_config.BASIS_MATRIX, local_config.PLOT_BOTTOM_LEFT_CORNER,
                              local_config.PLOT_TOP_RIGHT_CORNER, False, **global_config.LATTICE_PROPERTIES)

A = np.array([[-1, 1], [0, 1], [0.5, -1], [-1, -1],
             [1, 0], [-1, 0], [4/3, 1], [0, -1]])
b = np.array([0, 0, 2.5, 3, 2, 1, 2, 2.2])

plot_polyhedron(ax, A, b, config, **
                global_config.POLYHEDRON_TWO_FILL_PROPERTIES)
plot_integer_hull(ax, A, b, lattice_points)

# Show the plot
plt.show()
