import numpy as np
import matplotlib.pyplot as plt
import _5_19_theorem_config as local_config
import bib_config as global_config
from bib import *

# Create plot
fig, ax = plt.subplots()
setup_plot(ax, local_config.PLOT_BOTTOM_LEFT_CORNER,
           local_config.PLOT_TOP_RIGHT_CORNER)

lattice_points = plot_lattice(ax, local_config.BASIS_MATRIX, local_config.PLOT_BOTTOM_LEFT_CORNER,
                              local_config.PLOT_TOP_RIGHT_CORNER, False, **global_config.LATTICE_PROPERTIES)

# print(get_line_through_points([0.25,2],[0,0]))
# print(get_line_through_points([2,0.3],[0,0]))

# The polyhedron
A = np.array([[-8, 1], [0.15, -1], [1, 1]])
b = np.array([0, 0, 10])
plot_polyhedron(ax, A, b, local_config, **
                global_config.POLYHEDRON_TWO_FILL_PROPERTIES)
plot_integer_hull(ax, A, b, lattice_points)

# Char cone
plot_polyhedron(ax, A, np.array(
    [0, 0, 0]), local_config, **global_config.POLYHEDRON_THREE_FILL_PROPERTIES)

# Show the plot
plt.show()
