import numpy as np
import matplotlib.pyplot as plt
import _6_21_theorem_config as local_config
import bib_config as global_config
from bib import *

# Create plot
fig, ax = plt.subplots()
setup_plot(ax, local_config.PLOT_BOTTOM_LEFT_CORNER,
           local_config.PLOT_TOP_RIGHT_CORNER)

lattice_points = plot_lattice(ax, local_config.BASIS_MATRIX, local_config.PLOT_BOTTOM_LEFT_CORNER,
                              local_config.PLOT_TOP_RIGHT_CORNER, False, **global_config.LATTICE_PROPERTIES)

A = np.array([[1, -1], [0, 1], [1, 0]])
b = np.array([2, 2, 2])

plot_polyhedron(ax, A, b, local_config, **
                global_config.POLYHEDRON_TWO_FILL_PROPERTIES)

points = np.array([[-2, 2], [-1, 2], [0, 2], [1, 2], [2, 2], [2, 1],
                  [2, 0], [1, -1], [0, -2], [-1, -3], [-2, -4], [-8, 0], [-8, -8]])
c_vector = np.array([1, 2])

max_product = -np.Inf
solution = []

for p in points:
    product = np.dot(c_vector, p)
    if product > max_product:
        max_product = product
        solution = p

print(solution, max_product)

# So we have maximum solution for x=[2,3] with the value delta=6.
# Plot the inequality:
A = np.array([[1, 2]])
b = np.array([6])
plot_polyhedron(ax, A, b, local_config, **
                global_config.POLYHEDRON_THREE_FILL_PROPERTIES)

# Denote the face.
plot_point(ax, solution, '$F$', True)
plt.show()
