import numpy as np
import matplotlib.pyplot as plt
import _6_31_theorem_tdi_config as local_config
import bib_config as global_config
from bib import *

# Create plot
fig, ax = plt.subplots()
setup_plot(ax, local_config.PLOT_BOTTOM_LEFT_CORNER,
           local_config.PLOT_TOP_RIGHT_CORNER)

A = np.array([[-2, 1], [0.25, 1], [1, 0]])
b = np.array([1, 1.5, 4])

plot_polyhedron(ax, A, b, local_config, **
                global_config.POLYHEDRON_TWO_FILL_PROPERTIES)

f_1 = np.array([2/9, 13/9])
plot_point(ax, f_1, '$F_1$', True)
f_2 = np.array([4, 1/2])
plot_point(ax, f_2, '$F_2$', True)

# Normal cone for f_1:
vertex_list = np.array([[0, 0], [-14, 7], [-14, 14], [5, 20]])
plot_figure(ax, vertex_list, **global_config.FIGURE_ONE_PROPERTIES)
print(get_line_through_points([0, 0], [-2, 1]))
print(get_line_through_points([0, 0], [1/4, 1]))
plot_plus_lattice(ax, np.column_stack([[-2, 1], [1, 4]]), global_config)
plot_plus_lattice(ax, np.column_stack([[-2, 1], [-1, 1]]), global_config)
plot_plus_lattice(ax, np.column_stack([[-2, 1], [0, 1]]), global_config)
plot_plus_lattice(ax, np.column_stack([[1, 4], [0, 1]]), global_config)
plot_plus_lattice(ax, np.column_stack([[1, 4], [-1, 1]]), global_config)
plot_plus_lattice(ax, np.column_stack([[0, 1], [-1, 1]]), global_config)
H_1 = np.array([[-2, 1], [-1, 1], [0, 1], [1, 4]])


# Normal cone for f_2:
print(get_line_through_points([0, 0], [1, 0]))
print(get_line_through_points([0, 0], [1/4, 1]))
vertex_list = np.array([[0, 0], [5, 20], [15, 20], [15, 0]])
plot_figure(ax, vertex_list, **global_config.FIGURE_TWO_PROPERTIES)
plot_plus_lattice(ax, np.column_stack([[1, 0], [1, 4]]), global_config)
plot_plus_lattice(ax, np.column_stack([[1, 0], [1, 1]]), global_config)
plot_plus_lattice(ax, np.column_stack([[1, 0], [1, 2]]), global_config)
plot_plus_lattice(ax, np.column_stack([[1, 0], [1, 3]]), global_config)
plot_plus_lattice(ax, np.column_stack([[1, 4], [1, 1]]), global_config)
plot_plus_lattice(ax, np.column_stack([[1, 4], [1, 2]]), global_config)
plot_plus_lattice(ax, np.column_stack([[1, 4], [1, 3]]), global_config)
plot_plus_lattice(ax, np.column_stack([[1, 1], [1, 2]]), global_config)
plot_plus_lattice(ax, np.column_stack([[1, 1], [1, 3]]), global_config)
plot_plus_lattice(ax, np.column_stack([[1, 2], [1, 3]]), global_config)
H_2 = np.array([[1, 0], [1, 1], [1, 2], [1, 3], [1, 4]])


# Constructing the system
A = []
b = []
for hilbert_basis_vector in H_1:
    A.append(hilbert_basis_vector)
    b.append(np.dot(hilbert_basis_vector, f_1))

for hilbert_basis_vector in H_2:
    A.append(hilbert_basis_vector)
    b.append(np.dot(hilbert_basis_vector, f_2))

print(A)
print(b)
A = np.array(A)
b = np.array(b)

plot_polyhedron(ax, A, b, local_config, **
                global_config.POLYHEDRON_TWO_FILL_PROPERTIES)
plt.show()
