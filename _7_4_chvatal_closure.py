import numpy as np
import matplotlib.pyplot as plt
import _7_4_chvatal_closure_config as local_config
import bib_config as global_config
from bib import *
from matplotlib.animation import FuncAnimation, PillowWriter

# Create plot
fig, ax = plt.subplots()
setup_plot(ax, local_config.PLOT_BOTTOM_LEFT_CORNER,
           local_config.PLOT_TOP_RIGHT_CORNER)

lattice_points = plot_lattice(ax, local_config.BASIS_MATRIX, local_config.PLOT_BOTTOM_LEFT_CORNER,
                              local_config.PLOT_TOP_RIGHT_CORNER, False, **global_config.LATTICE_PROPERTIES)


def animate(i):
    # Main polyhedron
    A = np.array([[-2, 1], [1, -2], [-1, -3]])
    b = np.array([0, 0, -15])
    # Replace polyhedron by figure to avoid many lines on the plot.
    vertex_list = np.array([[15/7, 30/7], [4, 8], [8, 8], [8, 4], [6, 3]])

    # The polyhedron calculated by the lemma 6.22.
    F = np.array([15/7, 30/7])
    plot_point(ax, F, '$F$', True)
    # Plot the inequality
    L = np.array([[-2, -3]])
    m = np.array([-120/7])

    # Take hyper plane H1. This is line y=2.
    H_1_A = np.array([[0, -1]])
    H_1_b = np.array([-1.5])

    # Take hyperplane H2.
    print(get_line_through_points([1, 7], [2, 3]))
    H_2_A = np.array([[-4, -1]])
    H_2_b = np.array([-11])

    # Take hyperplane H3.
    print(get_line_through_points([0.5, 2], [3, 8]))
    H_3_A = np.array([[-2.4, 1]])
    H_3_b = np.array([0.8])

    if i == 0:
        plot_figure(ax, vertex_list, **global_config.FIGURE_THREE_PROPERTIES)

    elif i == 1:
        plot_polyhedron(ax, H_1_A, H_1_b, local_config, **
                        global_config.POLYHEDRON_ONE_FILL_PROPERTIES)

    elif i == 2:
        plot_integer_hull(ax, H_1_A, H_1_b, lattice_points)

    elif i == 3:
        plot_polyhedron(ax, H_2_A, H_2_b, local_config, **
                        global_config.POLYHEDRON_TWO_FILL_PROPERTIES)

    elif i == 4:
        plot_integer_hull(ax, H_2_A, H_2_b, lattice_points)

    elif i == 5:
        plot_polyhedron(ax, H_3_A, H_3_b, local_config, **
                        global_config.POLYHEDRON_THREE_FILL_PROPERTIES)

    elif i == 6:
        plot_integer_hull(ax, H_3_A, H_3_b, lattice_points)

    elif i == 7:
        plot_polyhedron(ax, L, m, local_config, **
                        global_config.POLYHEDRON_THREE_FILL_PROPERTIES)

    elif i == 8:
        plot_integer_hull(ax, L, m, lattice_points)


anim = FuncAnimation(fig, animate, frames=9, repeat=False, interval=2000)

# Save the animation
# writer = PillowWriter(fps=0.5)
# anim.save("chvatal_closure.gif", writer=writer)

plt.show()
