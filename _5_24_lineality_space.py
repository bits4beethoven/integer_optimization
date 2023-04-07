import numpy as np
import matplotlib.pyplot as plt
import _5_24_lineality_space_config as local_config
import bib_config as global_config
from bib import *

# Create plot
fig, ax = plt.subplots()
setup_plot(ax, local_config.PLOT_BOTTOM_LEFT_CORNER,
           local_config.PLOT_TOP_RIGHT_CORNER)

# lattice_points = plot_lattice(ax, c.BASIS_MATRIX, c.PLOT_BOTTOM_LEFT_CORNER, c.PLOT_TOP_RIGHT_CORNER, False,**c.LATTICE_PROPERTIES)

A = np.array([[-1, 1]])
b = np.array([-2])
plot_polyhedron(ax, A, b, local_config, **
                global_config.POLYHEDRON_TWO_FILL_PROPERTIES)

plot_line_through_points(ax, [0, 0], [1, 1])
plot_vector(ax, [0, 0], [1, 1], **config.BASIS_VECTOR_PROPERTIES)
plot_vector(ax, [0, 0], [-1, -1], **config.BASIS_VECTOR_PROPERTIES)

point = np.array([4, 2])
plot_point(ax, point, '$x^*$', False)
plot_vector(ax, local_config.ORIGIN, point, **
            global_config.PATH_THREE_PROPERTIES)
plot_vector(ax, point, [1, 1], **global_config.PATH_TWO_PROPERTIES)
plot_vector(ax, point, [-1, -1], **global_config.PATH_TWO_PROPERTIES)

vertex_list = np.array([[-6, 0], [-2, 2], [-4, 6], [0, 2]])
plot_figure(ax, vertex_list, **global_config.FIGURE_TWO_PROPERTIES)
# convex hull
plot_figure(ax, [vertex_list[0], vertex_list[2],
            vertex_list[3]], **global_config.HULL_PROPERTIES)


plt.show()
