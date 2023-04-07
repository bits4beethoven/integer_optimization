import numpy as np
import matplotlib.pyplot as plt
import _4_7_blichfeldt_config as local_config
import bib_config as global_config
from bib import *

# Create plot
fig, ax = plt.subplots()
setup_plot(ax, local_config.PLOT_BOTTOM_LEFT_CORNER,
           local_config.PLOT_TOP_RIGHT_CORNER)

# Plot the lattice
lattice_points = plot_lattice(ax, local_config.BASIS_MATRIX, local_config.PLOT_BOTTOM_LEFT_CORNER,
                              local_config.PLOT_TOP_RIGHT_CORNER, False, **global_config.LATTICE_PROPERTIES)

# The sets S_z. These are parts of the fundamental parallelepiped P, which when
# being translated cover some part of the figure. In thise case, there are three
# lattice points z such that S_z + z covers some part of S.
z_6_4 = local_config.S_LEFT + [6, 4]    # z = [-6,-4]
z_4_4 = local_config.S_MID + [4, 4]     # z = [-4,-4]
z_2_4 = local_config.S_RIGHT + [2, 4]   # z = [-2,-4]

plot_figure(ax, z_6_4, **local_config.S_LEFT_PROPERTIES)
plot_figure(ax, z_4_4, **local_config.S_MID_PROPERTIES)
plot_figure(ax, z_2_4, **local_config.S_RIGHT_PROPERTIES)

# Plot S.
plot_figure(ax, local_config.S_LEFT, **local_config.S_LEFT_PROPERTIES)
plot_figure(ax, local_config.S_MID, **local_config.S_MID_PROPERTIES)
plot_figure(ax, local_config.S_RIGHT, **local_config.S_RIGHT_PROPERTIES)


# Add missing pipeds, because our code plots a parallelepiped from the bottom left point.
lattice_points_copy_1, lattice_points_copy_2, lattice_points_copy_3 = [point.copy() for point in lattice_points], [point.copy() for point in lattice_points], [
    point.copy() for point in lattice_points]
lattice_determinant = np.round(np.linalg.det(local_config.BASIS_MATRIX))
lattice_points_copy_1 += np.array([0, -lattice_determinant])
lattice_points_copy_2 += np.array([-lattice_determinant, 0])
lattice_points_copy_3 += np.array([-lattice_determinant, -lattice_determinant])
merged_lattice = np.unique(np.concatenate(
    (lattice_points, lattice_points_copy_1, lattice_points_copy_2, lattice_points_copy_3), axis=0), axis=0)

# We cover the whole area with the parallelepipeds.
for lattice_point in merged_lattice:
    plot_fundamental_parallelepiped_from_point(
        ax, lattice_point, local_config.BASIS_MATRIX, **local_config.PARALLELEPIPED_PROPERTIES)

# Since vol(S) > vol(P), S_z cannot be disjoint. We take the point p = [3,3.5],
# which is in the intersection of S_z_6_4 and S_z_2_4. In fact, it is in the intersection
# of all three parts, but that does not matter for the proof.
p = np.array([3, 3.5])
# Set x and y as in the script
x = p - np.array([6, 4])
y = p - np.array([2, 4])

# Plot all points.
plot_point(ax, p, '$p$', True, **local_config.P_PROPERTIES)
plot_point(ax, x, '$x$', True, **local_config.P_PROPERTIES)
plot_point(ax, y, '$y$', True, **local_config.P_PROPERTIES)
plot_point(ax, x-y, '$x-y$', True, **local_config.P_PROPERTIES)

# Show the plot
plt.show()
