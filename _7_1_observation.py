import numpy as np
import matplotlib.pyplot as plt
import _7_1_observation_config as local_config
import bib_config as global_config
from bib import *

# Create plot
fig, ax = plt.subplots()
setup_plot(ax, local_config.PLOT_BOTTOM_LEFT_CORNER,
           local_config.PLOT_TOP_RIGHT_CORNER)

lattice_points = plot_lattice(ax, local_config.BASIS_MATRIX, local_config.PLOT_BOTTOM_LEFT_CORNER,
                              local_config.PLOT_TOP_RIGHT_CORNER, False, **global_config.LATTICE_PROPERTIES)

A = np.array([[-2, 1], [1, -2], [-1, -3]])
b = np.array([0, 0, -15])
plot_polyhedron(ax, A, b, local_config, **
                global_config.POLYHEDRON_TWO_FILL_PROPERTIES)
# Plot the integer hull
# plot_integer_hull(ax, A, b, lattice_points)

F = np.array([15/7, 30/7])
plot_point(ax, F, '$F$', True)

# Plot the inequality
A = np.array([[-2, -3]])
b = np.array([-120/7])
plot_polyhedron(ax, A, b, local_config, **
                global_config.POLYHEDRON_ONE_FILL_PROPERTIES)
# Plot the integer hull
# plot_integer_hull(ax, A, b, lattice_points)

# Make cut
b = np.floor(b)
plot_polyhedron(ax, A, b, local_config, **
                global_config.POLYHEDRON_THREE_FILL_PROPERTIES)

plt.show()
