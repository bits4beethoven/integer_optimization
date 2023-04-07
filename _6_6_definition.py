import numpy as np
import matplotlib.pyplot as plt
import _6_6_definition_config as local_config
import bib_config as global_config
from bib import *

# Create plot
fig, ax = plt.subplots()
setup_plot(ax, local_config.PLOT_BOTTOM_LEFT_CORNER,
           local_config.PLOT_TOP_RIGHT_CORNER)

# P1 polyhedron
A = np.array([[1, -1], [0, 1], [1, 0]])
b = np.array([2, 2, 2])
plot_polyhedron(ax, A, b, local_config, **
                global_config.POLYHEDRON_TWO_FILL_PROPERTIES)

plot_figure(ax, [[0, -2], [0, 2], [2, 2], [2, 0]], **
            {'body_color': 'g', 'body_alpha': 0.4})
ax.text(0.5, 0.5, r'$(P_1)^\leq_+$')

plt.plot([0, 8], [-2, 6], **{'color': 'r', 'linewidth': 5})
ax.text(7, 6, r'$(P_2)^=_+$')

l = np.array([-6, -4])
u = np.array([-4, 0])
plot_point(ax, l, '$l$', True)
plot_point(ax, u, '$u$', True)
plot_figure(ax, [l, [l[0], u[1]], u, [u[0], l[1]]],
            **global_config.FIGURE_ONE_PROPERTIES)

# P2 polyhedron
plot_figure(ax, [[-8, -10], [10, 8], [10, -10]],
            **global_config.FIGURE_TWO_PROPERTIES)


plt.show()
