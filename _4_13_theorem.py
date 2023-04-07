import numpy as np
import matplotlib.pyplot as plt
import _4_13_theorem_config as local_config
import bib_config as global_config
from bib import *

# Create plot
fig, ax = plt.subplots()
setup_plot(ax, local_config.PLOT_BOTTOM_LEFT_CORNER,
           local_config.PLOT_TOP_RIGHT_CORNER)

# Plot the lattice
lattice_points = plot_lattice(ax, local_config.BASIS_MATRIX, local_config.PLOT_BOTTOM_LEFT_CORNER,
                              local_config.PLOT_TOP_RIGHT_CORNER, False, **global_config.LATTICE_PROPERTIES)

plot_figure(ax, np.array([[-1, 1], [1, 1], [1, -1],
            [-1, -1]]), **global_config.FIGURE_THREE_PROPERTIES)
plot_circle(ax, global_config.ORIGIN, 1, **global_config.CIRCLE_ONE_PROPERTIES)

plot_point(ax, [-1, 0], '$u_1$', False)
plot_point(ax, [0, 1], '$u_2$', False)
plot_point(ax, [1, 0], '$u_3$', False)
plot_point(ax, [0, -1], '$u_4$', False)

# Set all c_i to 1/2.

# Show the plot
plt.show()
