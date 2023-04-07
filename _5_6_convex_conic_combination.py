import numpy as np
import matplotlib.pyplot as plt
import _5_6_convex_conic_combination_config as local_config
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

# Plot conic combinations of two vectors
z_1 = np.array([1, 1])
z_2 = np.array([0.5, 2])

for lambda_1 in range(0, 20):
    for lambda_2 in range(0, 20):
        step = 0.2
        comb = step*lambda_1 * z_1 + step*lambda_2 * z_2
        plot_vector(ax, local_config.ORIGIN, comb, **
                    global_config.PATH_ONE_PROPERTIES)
        plot_point(ax, comb, '', False, **local_config.CONIC_POINT_PROPERTIES)

plot_vector(ax, local_config.ORIGIN, z_1, **global_config.PATH_TWO_PROPERTIES)
plot_vector(ax, local_config.ORIGIN, z_2, **global_config.PATH_TWO_PROPERTIES)

# Plot convex combinations of two vectors
z_1 = np.array([2.5, 2])
z_2 = np.array([4.5, 1])

for lambda_1 in range(0, 20):
    step = 1/20
    comb = step*lambda_1 * z_1 + step*(20-lambda_1) * z_2
    plot_vector(ax, local_config.ORIGIN, comb, **
                global_config.PATH_THREE_PROPERTIES)
    plot_point(ax, comb, '', False, **local_config.CONVEX_POINT_PROPERTIES)

plot_vector(ax, local_config.ORIGIN, z_1, **global_config.PATH_TWO_PROPERTIES)
plot_vector(ax, local_config.ORIGIN, z_2, **global_config.PATH_TWO_PROPERTIES)

# Show the plot
plt.show()
