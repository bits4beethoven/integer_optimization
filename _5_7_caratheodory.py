import numpy as np
import matplotlib.pyplot as plt
import _5_7_caratheodory_config as local_config
import bib_config as global_config
from bib import *

# Create plot
fig, ax = plt.subplots()
setup_plot(ax, local_config.PLOT_BOTTOM_LEFT_CORNER,
           local_config.PLOT_TOP_RIGHT_CORNER)

lattice_points = plot_lattice(ax, local_config.BASIS_MATRIX, local_config.PLOT_BOTTOM_LEFT_CORNER,
                              local_config.PLOT_TOP_RIGHT_CORNER, False, **global_config.LATTICE_PROPERTIES)

shift = np.array([-0.3, 0.4])

# Create some figure.
vertex_list = np.array(
    [[1, 1], [2, 3], [4, 3.5], [2.5, 2.5], [4, 0.5], [2, 2]])

# Shift is away from the lattice points.
# vertex_list += shift

# Get the centroid
centroid = np.array([np.sum(vertex_list, axis=0)[
                    0]/len(vertex_list), np.sum(vertex_list, axis=0)[1] / len(vertex_list)])
plot_center = np.array([(local_config.PLOT_TOP_RIGHT_CORNER[0] - local_config.PLOT_BOTTOM_LEFT_CORNER[0])/2,
                       (local_config.PLOT_TOP_RIGHT_CORNER[1] - local_config.PLOT_BOTTOM_LEFT_CORNER[1]) / 2])
diff = plot_center - centroid
vertex_list += diff

m_1, b_1 = get_line_through_points([0, 0], vertex_list[1])
m_2, b_2 = get_line_through_points([0, 0], vertex_list[4])
cone_point_top = [local_config.PLOT_TOP_RIGHT_CORNER[0],
                  m_1 * local_config.PLOT_TOP_RIGHT_CORNER[0] + b_1]
cone_point_bottom = [local_config.PLOT_TOP_RIGHT_CORNER[0],
                     m_2 * local_config.PLOT_TOP_RIGHT_CORNER[0] + b_2]
cone_points = np.array(
    [local_config.ORIGIN, cone_point_top, cone_point_bottom])

plot_figure(ax, vertex_list, **global_config.FIGURE_ONE_PROPERTIES)
plot_figure(ax, cone_points, **global_config.FIGURE_THREE_PROPERTIES)


# Plot the convex hull.
hull_points = np.array([vertex_list[0], vertex_list[1],
                       vertex_list[2], vertex_list[4]])
plot_figure(ax, hull_points, **global_config.HULL_PROPERTIES)

# Now, Caratheodory says that we need at most 3 points from X,
# which convex combinations form the integer hull.

# Show the plot
plt.show()
