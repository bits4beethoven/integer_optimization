import numpy as np
import matplotlib.pyplot as plt
import _5_5_convex_conic_hull_config as local_config
import bib_config as global_config
from bib import *

# Create plot
fig, ax = plt.subplots()
setup_plot(ax, local_config.PLOT_BOTTOM_LEFT_CORNER,
           local_config.PLOT_TOP_RIGHT_CORNER)

lattice_points = plot_lattice(ax, local_config.BASIS_MATRIX, local_config.PLOT_BOTTOM_LEFT_CORNER,
                              local_config.PLOT_TOP_RIGHT_CORNER, False, **global_config.LATTICE_PROPERTIES)

A = np.array([[-2, 1], [1/2, -1]])
b = np.array([0, 0])

plot_polyhedron(ax, A, b, config, **
                global_config.POLYHEDRON_TWO_FILL_PROPERTIES)
# plot_integer_hull(ax, A, b, lattice_points)

# Plot figure.
vertex_list = np.array([[1, 1], [2, 4], [4, 3], [3, 1.5]])
plot_figure(ax, vertex_list, **global_config.FIGURE_TWO_PROPERTIES)
# Show the plot
plt.show()
