import numpy as np
import matplotlib.pyplot as plt
import _4_8_minkowski_config as local_config
import bib_config as global_config
from bib import *

# Create plot
fig, ax = plt.subplots()
setup_plot(ax, local_config.PLOT_BOTTOM_LEFT_CORNER,
           local_config.PLOT_TOP_RIGHT_CORNER)

lattice_points = plot_lattice(ax, local_config.BASIS_MATRIX, local_config.PLOT_BOTTOM_LEFT_CORNER,
                              local_config.PLOT_TOP_RIGHT_CORNER, False, **local_config.LATTICE_PROPERTIES)

# threshold for plotting parallelepiped
threshold = 5
for lattice_point in lattice_points:
    if np.linalg.norm(lattice_point) < threshold and np.linalg.norm(lattice_point + local_config.b_1 + local_config.b_2) < threshold:
        plot_fundamental_parallelepiped_from_point(
            ax, lattice_point, local_config.BASIS_MATRIX, **local_config.FUNDAMENTAL_PARALLELEPIPED_PROPERTIES)


# Get the lattice determinant
lattice_determinant = np.abs(np.linalg.det(local_config.BASIS_MATRIX))

# Get lower bound from the theorem.
bound = 4 * lattice_determinant
# Scale the volume by 1.5
bound *= 1.5

# Define a matrix defining an ellipsoid.
positive_definite_matrix = np.array([[1.0, 0.0], [0.0, 2.0]])

# Scale the ellipsoid until its volume exceeds the bounds.
while get_ellipsoid_volume(positive_definite_matrix) < bound:
    positive_definite_matrix *= 0.5

plot_ellipsoid(ax, local_config.ORIGIN,
               positive_definite_matrix, -np.pi / 4, False, True)
print(positive_definite_matrix)
# Get K
k_matrix = positive_definite_matrix * 4
plot_ellipsoid(ax, local_config.ORIGIN, k_matrix, -np.pi / 4, True, True)

# Check that vol(K) > det
assert (get_ellipsoid_volume(k_matrix) > lattice_determinant)

# Pick point x
x = np.array([-0.5, 0.5])
y = x - local_config.b_1 + local_config.b_2

plot_point(ax, x, '$x$', True)
plot_point(ax, y, '$y$', True)

# Get z
z = x - y
plot_point(ax, z, '$z$', True)

# Get poitns 2x, 2y
x_two = x * 2
y_two = y * 2
plot_point(ax, x_two, '$2x$', True)
plot_point(ax, y_two, '$2y$', True)
plot_point(ax, -y_two, '$-2y$', True)

# Visualize the equality
plot_vector(ax, local_config.ORIGIN, x, **local_config.PATH_ONE_PROPERTIES)
plot_vector(ax, x, z - x, **local_config.PATH_ONE_PROPERTIES)

plot_vector(ax, local_config.ORIGIN, x_two, **local_config.PATH_TWO_PROPERTIES)
plot_vector(ax, x_two, -y_two - x_two, **local_config.PATH_TWO_PROPERTIES)
# Show the plot
plt.show()
