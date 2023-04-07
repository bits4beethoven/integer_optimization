import numpy as np
import matplotlib.pyplot as plt
import _8_1_defi_split_cut_config as local_config
import bib_config as global_config
from bib import *

# Create plot
fig, ax = plt.subplots(figsize=(12, 8))
setup_plot(ax, local_config.PLOT_BOTTOM_LEFT_CORNER,
           local_config.PLOT_TOP_RIGHT_CORNER)

lattice_points = plot_lattice(ax, local_config.BASIS_MATRIX, local_config.PLOT_BOTTOM_LEFT_CORNER,
                              local_config.PLOT_TOP_RIGHT_CORNER, False, **global_config.LATTICE_PROPERTIES)

A = np.array([[-3, -4], [-1, 1], [4, 6], [4, -10]])
b = np.array([-8, 2, 27, 3])

plot_polyhedron(ax, A, b, local_config, **
                global_config.POLYHEDRON_TWO_FILL_PROPERTIES)
plot_integer_hull(ax, A, b, lattice_points)
pi = np.array([2, 1])
pi_0 = np.array([6])
# left part
plot_polyhedron(ax, np.array([pi]), pi_0, local_config, **
                global_config.POLYHEDRON_ONE_FILL_PROPERTIES)
# right part
plot_polyhedron(ax, np.array([-pi]), -(pi_0+1), local_config,
                **global_config.POLYHEDRON_ONE_FILL_PROPERTIES)

# plot the resulting convex set
vertex_list = np.array(
    [[0, 2], [4/3, 10/3], [15/8, 13/4], [4.5, 1.5], [2, 0.5]])
plot_figure(ax, vertex_list, **global_config.FIGURE_TWO_PROPERTIES)

plt.show()
