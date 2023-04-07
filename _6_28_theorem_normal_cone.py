import numpy as np
import matplotlib.pyplot as plt
import _6_28_theorem_normal_cone_config as local_config
import bib_config as global_config
from bib import *

# Create plot
fig, ax = plt.subplots()
setup_plot(ax, local_config.PLOT_BOTTOM_LEFT_CORNER,
           local_config.PLOT_TOP_RIGHT_CORNER)

A = np.array([[1, -1], [0, 1], [1, 0]])
b = np.array([2, 2, 2])

plot_polyhedron(ax, A, b, local_config, **
                global_config.POLYHEDRON_TWO_FILL_PROPERTIES)

# Normal cones:
vertex_list = np.array([[0, 0], [0, 10], [10, 10], [10, 0]])
plot_figure(ax, vertex_list, **global_config.FIGURE_TWO_PROPERTIES)
plot_vector(ax, local_config.ORIGIN, [1, 0],
            **global_config.PATH_TWO_PROPERTIES)
plot_vector(ax, local_config.ORIGIN, [0, 1],
            **global_config.PATH_TWO_PROPERTIES)
plot_plus_lattice(ax, np.array([[1, 0], [0, 1]]), global_config)

vertex_list = np.array([[0, 0], [9, -9], [9, 0]])
plot_figure(ax, vertex_list, **global_config.FIGURE_ONE_PROPERTIES)
plot_vector(ax, local_config.ORIGIN, [
            1, -1], **global_config.PATH_THREE_PROPERTIES)
plot_vector(ax, local_config.ORIGIN, [1, 0],
            **global_config.PATH_THREE_PROPERTIES)
plot_plus_lattice(ax, np.array([[1, 1], [0, -1]]), global_config)
plt.show()
