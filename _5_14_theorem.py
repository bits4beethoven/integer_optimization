import numpy as np
import matplotlib.pyplot as plt
import _5_14_theorem_config as local_config
import bib_config as global_config
from bib import *

# Create plot
fig, ax = plt.subplots()
setup_plot(ax, local_config.PLOT_BOTTOM_LEFT_CORNER,
           local_config.PLOT_TOP_RIGHT_CORNER)

lattice_points = plot_lattice(ax, local_config.BASIS_MATRIX, local_config.PLOT_BOTTOM_LEFT_CORNER,
                              local_config.PLOT_TOP_RIGHT_CORNER, False, **global_config.LATTICE_PROPERTIES)

# This is P.
A = np.array([[-2, 1], [0.5, -1], [-1, 0], [-0.5, -1]])
b = np.array([-4, 3, -2, 0])
plot_polyhedron(ax, A, b, local_config, **
                global_config.POLYHEDRON_ONE_FILL_PROPERTIES)

# This is ccone(Y) of P.
plot_polyhedron(ax, A, np.array(
    [0, 0, 0, 0]), local_config, **global_config.POLYHEDRON_TWO_FILL_PROPERTIES)

# We take Y as in _5_11_
# # y = 1/2x and y=2x. So we can pick [2,1] and [1,2] as vectors into Y.
Y = np.array([[1, 2], [2, 1]])
plot_vector(ax, local_config.ORIGIN, Y[0], **global_config.PATH_ONE_PROPERTIES)
plot_vector(ax, local_config.ORIGIN, Y[1], **global_config.PATH_ONE_PROPERTIES)
plot_point(ax, Y[0], '$\\in Y$', False)
plot_point(ax, Y[1], '$\\in Y$', False)

# We take the least common denominator a(y) of all vectors in Y in order to make them integer vectors.
# In this case we already have integer vectors.
# I write t(Y) for "tilde Y".
# So here we simply set t(Y) = Y. This is just scaling, nothing else.
# Now, let us define wtf Q is.
# We have that Y' is a subset of t(Y) with at most 2 elements.
# In this case, we have
# 1) Y'={}, 2) Y'={[1,2]}, 3) Y'={[2,1]}, 4) Y'=t(Y).
# For 1) there is nothing to do.
# For 2) we just have the vector [1,2] and all its points.
# For 3) we have the vecotr [2,1] with all its points.
# For 4) we have a geometric combination of these vectors.
# Either use this code to plot the combinations or take the figure after this:
"""
# create a grid of a and b values
a_vals = np.linspace(0, 1, num=100)
b_vals = np.linspace(0, 1, num=100)
a_grid, b_grid = np.meshgrid(a_vals, b_vals)
# compute the linear combinations
linear_combinations = a_grid[..., np.newaxis] * Y[0] + b_grid[..., np.newaxis] * Y[1]
# plot the linear combinations
for i in range(linear_combinations.shape[0]):
    plt.plot(linear_combinations[i,:,0], linear_combinations[i,:,1], 'b-', alpha=0.5)
"""

# So together we have this Q:
q_list = [local_config.ORIGIN, Y[0], Y[0] + Y[1], Y[1]]
plot_figure(ax, q_list, **global_config.FIGURE_ONE_PROPERTIES)

# Now, let us go back to X. X was defined as:
X = np.array([[2, 0], [2, -1], [3, -1.5]])
plot_figure(ax, np.array([X[0], X[1], X[2]]), **
            global_config.FIGURE_TWO_PROPERTIES)
plot_point(ax, X[0], '$\\in X$', False)
plot_point(ax, X[1], '$\\in X$', False)
plot_point(ax, X[2], '$\\in X$', False)

# We shift conv(X) by Q
conv_x_plus_q_list = [
    q_list[0] + X[0],
    q_list[1] + X[0],
    q_list[2] + X[0],
    q_list[2] + X[2],
    q_list[3] + X[2],
    X[2]]
plot_figure(ax, conv_x_plus_q_list, **global_config.FIGURE_THREE_PROPERTIES)

# Cut this thing with integers:
integer_points = np.array([[2, 0], [3, -1], [3, 0], [3, 1], [3, 2],
                          [4, -1], [4, 0], [4, 1], [4, 2], [5, 0], [5, 1], [5, 2], [5, 3]])
for point in integer_points:
    plot_point(ax, point, '', False, **{'color': 'red', 'alpha': 0.5})
# Now, t(X) = integer_points
# Get lattice plus points:
lattice_plus_points = plot_plus_lattice(
    ax, np.column_stack([Y[0], Y[1]]), global_config)

for integer_point in integer_points:
    new_dots = integer_point + lattice_plus_points
    for new_dot in new_dots:
        plot_point(ax, new_dot, '', False, **
                   {'color': 'green', 'alpha': 1, 's': 150})

# Show the plot
plt.show()
