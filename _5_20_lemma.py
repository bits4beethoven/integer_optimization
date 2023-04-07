import numpy as np
import matplotlib.pyplot as plt
import _5_20_lemma_config as local_config
import bib_config as global_config
from bib import *

# Create plot
fig, ax = plt.subplots()
setup_plot(ax, local_config.PLOT_BOTTOM_LEFT_CORNER,
           local_config.PLOT_TOP_RIGHT_CORNER)

lattice_points = plot_lattice(ax, local_config.BASIS_MATRIX, local_config.PLOT_BOTTOM_LEFT_CORNER,
                              local_config.PLOT_TOP_RIGHT_CORNER, False, **local_config.LATTICE_PROPERTIES)

# Show the plot
Y = np.column_stack([[0.5, 1], [1, 1]])
plus_lattice_points = plot_plus_lattice(ax, Y, global_config)

vertex_list = np.array([[6, 1], [6, 2], [7, 3], [6.5, 1.5], [8, 2]])
plot_figure(ax, vertex_list, **global_config.FIGURE_ONE_PROPERTIES)

print(get_line_through_points([8, 2], [9, 3]))

vertex_list = np.array([[6, 1], [6, 2], [12, 14], [14, 14], [14, 8], [8, 2]])
plot_figure(ax, vertex_list, **global_config.HULL_PROPERTIES)

vertex_list = np.array([[0, 0], [15, 15], [7, 14]])
plot_figure(ax, vertex_list, **global_config.FIGURE_TWO_PROPERTIES)

# Solution to 6.4
z = np.array([8, 4])
# z is 1 * [6,2] + 2 * [1,1], so lambda_x=1, mu_y = 2
# M = 1 + 2 = 3
# t(mu) = 2/3, t(mu_null) = 1 - 2/3 = 1/3

plot_point(ax, z, '$z$', True)
plt.show()
