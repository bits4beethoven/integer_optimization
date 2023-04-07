import numpy as np
import matplotlib.pyplot as plt
import _5_23_propo_config as local_config
import bib_config as global_config
from bib import *

# Create plot
fig, ax = plt.subplots()
setup_plot(ax, local_config.PLOT_BOTTOM_LEFT_CORNER,
           local_config.PLOT_TOP_RIGHT_CORNER)

# lattice_points = plot_lattice(ax, c.BASIS_MATRIX, c.PLOT_BOTTOM_LEFT_CORNER, c.PLOT_TOP_RIGHT_CORNER, False,**c.LATTICE_PROPERTIES)

# This is P.
A = np.array([[-2, 1], [0.5, -1], [-1, 0], [-0.5, -1]])
b = np.array([-4, 3, -2, 0])
plot_polyhedron(ax, A, b, local_config, **
                global_config.POLYHEDRON_TWO_FILL_PROPERTIES)

# This is ccone(Y) of P.
# plot_polyhedron(ax, A, np.array([0,0,0,0]),c)
# At this point we see that ccone(Y) is spanned by the vectors lying on the lines
# y = 1/2x and y=2x. So we can pick [2,1] and [1,2] as vectors into Y.
Y = np.array([[2, 1], [1, 2]])
plot_vector(ax, local_config.ORIGIN, Y[0], **local_config.PATH_ONE_PROPERTIES)
plot_vector(ax, local_config.ORIGIN, Y[1], **local_config.PATH_ONE_PROPERTIES)
plot_point(ax, Y[0], '$\\in Y$', False)
plot_point(ax, Y[1], '$\\in Y$', False)

# Then we obtain all vertices of P by solving the pairs of equalities.
# For this one, we have vertices [2,0], [2,-1], [3, -1.5]. We put them into X.
X = np.array([[2, 0], [2, -1], [3, -1.5]])
plot_figure(ax, np.array([X[0], X[1], X[2]]), **
            global_config.FIGURE_ONE_PROPERTIES)
plot_vector(ax, local_config.ORIGIN, X[0], **local_config.PATH_TWO_PROPERTIES)
plot_vector(ax, local_config.ORIGIN, X[1], **local_config.PATH_TWO_PROPERTIES)
plot_vector(ax, local_config.ORIGIN, X[2], **local_config.PATH_TWO_PROPERTIES)
plot_point(ax, X[0], '$\\in X$', False)
plot_point(ax, X[1], '$\\in X$', False)
plot_point(ax, X[2], '$\\in X$', False)

A = np.array([[1, 1], [2, 2]])
b = np.array([0, 0])
solution = A @ b

# Show the plot
plt.show()
