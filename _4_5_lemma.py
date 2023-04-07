import numpy as np
import matplotlib.pyplot as plt
import _4_5_lemma_config as local_config
import bib_config as global_config
from bib import *

# Create plot
fig, ax = plt.subplots()
setup_plot(ax, local_config.PLOT_BOTTOM_LEFT_CORNER,
           local_config.PLOT_TOP_RIGHT_CORNER)

lattice_points = plot_lattice(ax, local_config.BASIS_MATRIX, local_config.PLOT_BOTTOM_LEFT_CORNER,
                              local_config.PLOT_TOP_RIGHT_CORNER, False, **global_config.LATTICE_PROPERTIES)

# Fundamental parallelepiped from the basis vectors.
plot_fundamental_parallelepiped_from_point(
    ax, local_config.ORIGIN, local_config.BASIS_MATRIX, **global_config.FUNDAMENTAL_PARALLELEPIPED_PROPERTIES)

# Create the point x
plot_point(ax, local_config.POINT_X_COORDINATES, '$x$',
           True, **local_config.POINT_X_PROPERTIES)

# Find the combination of basis vectors for x:
solution = np.linalg.solve(local_config.BASIS_MATRIX,
                           local_config.POINT_X_COORDINATES)

# Define z and plot it
floored_solution = np.floor(solution).astype(int)
z = local_config.BASIS_MATRIX @ floored_solution
plot_point(ax, z, '$z$', True, **local_config.POINT_X_PROPERTIES)

# Define y and plot i
y = local_config.BASIS_MATRIX @ (solution - floored_solution)
# Round to avoid 3.0000000000004 errors:
y = np.round(y, 6)
plot_point(ax, y, '$y$', True, **local_config.POINT_X_PROPERTIES)

plot_vector(ax, local_config.ORIGIN, y, **local_config.VECTOR_PROPERTIES)
plot_vector(ax, local_config.ORIGIN, z, **local_config.VECTOR_PROPERTIES)
plot_vector(ax, z, y, **local_config.VECTOR_PROPERTIES)
plot_vector(ax, local_config.ORIGIN, local_config.POINT_X_COORDINATES,
            **local_config.VECTOR_PROPERTIES)

# Show the plot
plt.show()
