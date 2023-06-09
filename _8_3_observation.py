import numpy as np
import matplotlib.pyplot as plt
import _8_3_observation_config as local_config
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
# plot_polyhedron(ax,np.array([pi]),pi_0, c, **c.POLYHEDRON_ONE_FILL_PROPERTIES)
# right part
# plot_polyhedron(ax,np.array([-pi]),-(pi_0+1), c, **c.POLYHEDRON_ONE_FILL_PROPERTIES)

# plot the resulting convex set
vertex_list = np.array(
    [[0, 2], [4/3, 10/3], [15/8, 13/4], [4.5, 1.5], [2, 0.5]])
# plot_figure(ax, vertex_list, **c.FIGURE_PROPERTIES)

x_star = get_cutting_point_for_lines(A[1], b[1], A[2], b[2])
# pi = [1,0], pi_0 = [1,3.5]
pi = np.array([1, 0])
pi_0 = np.array([np.floor(x_star[0])])
plot_polyhedron(ax, np.array([pi]), pi_0, local_config, **
                global_config.POLYHEDRON_TWO_FILL_PROPERTIES)
plot_polyhedron(ax, np.array([-pi]), -(pi_0+1), local_config,
                **global_config.POLYHEDRON_TWO_FILL_PROPERTIES)

# pi = [1,0], pi_0 = [1.5,3]
pi = np.array([0, 1])
pi_0 = np.array([np.floor(x_star[1])])
plot_polyhedron(ax, np.array([pi]), pi_0, local_config, **
                global_config.POLYHEDRON_THREE_FILL_PROPERTIES)
plot_polyhedron(ax, np.array([-pi]), -(pi_0+1), local_config,
                **global_config.POLYHEDRON_THREE_FILL_PROPERTIES)

plt.show()
