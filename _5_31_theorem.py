import numpy as np
import matplotlib.pyplot as plt
import _5_31_theorem_config as local_config
import bib_config as global_config
from bib import *

# Create plot
fig, ax = plt.subplots()
setup_plot(ax, local_config.PLOT_BOTTOM_LEFT_CORNER,
           local_config.PLOT_TOP_RIGHT_CORNER)

lattice_points = plot_lattice(ax, local_config.BASIS_MATRIX, local_config.PLOT_BOTTOM_LEFT_CORNER,
                              local_config.PLOT_TOP_RIGHT_CORNER, False, **global_config.LATTICE_PROPERTIES)

# Plot the polyhedron and it's integer hull
A = np.array([[-2, 1], [0.5, -1], [1, 1]])
b = np.array([0, 0, 5])
plot_polyhedron(ax, A, b, local_config, **
                global_config.POLYHEDRON_TWO_FILL_PROPERTIES)
hull_points = plot_integer_hull(ax, A, b, lattice_points)

# Algorithm 5.32
# set c
c_vector = np.array([1, 2])
problem_max = -np.Inf
solution = []
# compute the maximum
for minimal_face in hull_points:
    product = np.dot(c_vector, minimal_face)
    if product > problem_max:
        problem_max = product
        solution = np.copy(minimal_face)

print(solution, problem_max)

# Plot the hyperplane through the maximum vertex
A = np.array([c_vector])
b = np.array([problem_max])
# plot_polyhedron(ax, A, b, c)

plt.show()
