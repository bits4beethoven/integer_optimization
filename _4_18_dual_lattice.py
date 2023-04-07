import numpy as np
import matplotlib.pyplot as plt
import _4_18_dual_lattice_config as local_config
import bib_config as global_config
from bib import *

# Create plot
fig, ax = plt.subplots()
setup_plot(ax, local_config.PLOT_BOTTOM_LEFT_CORNER,
           local_config.PLOT_TOP_RIGHT_CORNER)

lattice_points = plot_lattice(ax, local_config.BASIS_MATRIX, local_config.PLOT_BOTTOM_LEFT_CORNER,
                              local_config.PLOT_TOP_RIGHT_CORNER, False, **local_config.LATTICE_PROPERTIES)
dual_lattice_points = plot_dual_lattice(
    ax, local_config.BASIS_MATRIX, local_config.PLOT_BOTTOM_LEFT_CORNER, local_config.PLOT_TOP_RIGHT_CORNER, False, **local_config.DUAL_LATTICE_PROPERTIES)

# Check if the product of two vectors in an integer.
# We cannot just call is_integer() because of the rounding errors
epsilon = 10**(-10)
for lattice_point in lattice_points:
    for dual_point in dual_lattice_points:
        product = np.dot(lattice_point, dual_point)
        assert (np.round(product) - product < epsilon)

# Plot slices, as by page 35
plot_slices(ax, lattice_points,
            local_config.DUAL_BASIS_MATRIX[:, 1], local_config)
# Show the plot
plt.show()
