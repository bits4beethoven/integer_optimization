import numpy as np
import matplotlib.pyplot as plt
import _exec_10_4_config as local_config
import bib_config as global_config
from bib import *
from matplotlib.animation import FuncAnimation, PillowWriter

# Create plot
fig, ax = plt.subplots()
setup_plot(ax, local_config.PLOT_BOTTOM_LEFT_CORNER,
           local_config.PLOT_TOP_RIGHT_CORNER)

lattice_points = plot_lattice(ax, local_config.BASIS_MATRIX, local_config.PLOT_BOTTOM_LEFT_CORNER,
                              local_config.PLOT_TOP_RIGHT_CORNER, False, **local_config.LATTICE_PROPERTIES)


def animate(i):
    # Main polyhedron
    A = np.array([[-2, 1], [2, 1], [0, -1]])
    b = np.array([0, 6, -1])

    # Take hyper plane H1. This is line y=2.
    H_1_A = np.array([[0, -1]])
    H_1_b = np.array([-0.5])

    # Take hyperplane H2.
    # print(get_line_through_points([5/3,3],[8/3,1]))
    H_2_A = np.array([[1, 0]])
    H_2_b = np.array([2.5])

    # Take hyperplane H3.
    # print(get_line_through_points([0.5,2],[3,8]))
    H_3_A = np.array([[-1, 0]])
    H_3_b = np.array([-0.5])

    if i == 0:
        # plot_figure(ax, vertex_list, **c.FIGURE_PROPERTIES)
        # Plot the integer hull
        # plot_integer_hull(ax, A, b, lattice_points)
        plot_polyhedron(ax, A, b, local_config, **
                        local_config.POLYHEDRON_TWO_FILL_PROPERTIES)

    elif i == 1:
        plot_integer_hull(ax, A, b, lattice_points)

    elif i == 2:
        plot_polyhedron(ax, H_1_A, H_1_b, local_config, **
                        local_config.POLYHEDRON_TWO_FILL_PROPERTIES)

    elif i == 3:
        plot_integer_hull(ax, H_1_A, H_1_b, lattice_points)

    elif i == 4:
        plot_polyhedron(ax, H_2_A, H_2_b, local_config, **
                        local_config.POLYHEDRON_TWO_FILL_PROPERTIES)

    elif i == 5:
        plot_integer_hull(ax, H_2_A, H_2_b, lattice_points)

    elif i == 6:
        plot_polyhedron(ax, H_3_A, H_3_b, local_config, **
                        local_config.POLYHEDRON_TWO_FILL_PROPERTIES)

    elif i == 7:
        plot_integer_hull(ax, H_3_A, H_3_b, lattice_points)


anim = FuncAnimation(fig, animate, frames=8, repeat=False, interval=2000)

plt.show()
