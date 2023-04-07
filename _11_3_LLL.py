import numpy as np
import matplotlib.pyplot as plt
import _11_3_LLL_config as c
from pylab import *
from bib import *

# Create plot
fig, ax = plt.subplots()
setup_plot(ax, c.PLOT_BOTTOM_LEFT_CORNER, c.PLOT_TOP_RIGHT_CORNER)

# Add lattice points
lattice_points = plot_lattice(ax, c.BASIS_MATRIX, c.PLOT_BOTTOM_LEFT_CORNER,
                              c.PLOT_TOP_RIGHT_CORNER, **c.LATTICE_PROPERTIES)
# plot_dual_lattice(ax, c.BASIS_MATRIX, c.PLOT_BOTTOM_LEFT_CORNER, c.PLOT_TOP_RIGHT_CORNER, **c.DUAL_LATTICE_PROPERTIES)

# Get Gram-Schmidt basis and plot it
g = get_gram_schmidt_basis(c.BASIS_MATRIX)
plt.quiver(c.ORIGIN[0], c.ORIGIN[1], g[0][0],
           g[0][1], **c.GRAM_SCHMIDT_PROPERTIES)
plt.quiver(c.ORIGIN[0], c.ORIGIN[1], g[1][0],
           g[1][1], **c.GRAM_SCHMIDT_PROPERTIES)

# Calculate norms and get minimum.
g_min = min(np.linalg.norm(g[0]), np.linalg.norm(g[1]))

# This minimum defines a circle around the origin.
plot_circle(ax, c.ORIGIN, g_min, **c.CIRCLE_PROPERTIES)

# Show the plot
plt.show()
