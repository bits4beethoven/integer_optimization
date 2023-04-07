import numpy as np
import matplotlib.pyplot as plt
import _4_23_packing_radius_config as local_config
from bib import *

# Create plot
fig, ax = plt.subplots()
setup_plot(ax, local_config.PLOT_BOTTOM_LEFT_CORNER,
           local_config.PLOT_TOP_RIGHT_CORNER)

lattice_points = plot_lattice(ax, local_config.BASIS_MATRIX, local_config.PLOT_BOTTOM_LEFT_CORNER,
                              local_config.PLOT_TOP_RIGHT_CORNER, False, **local_config.LATTICE_PROPERTIES)

plot_packing_radius(ax, lattice_points)
# Show the plot
plt.show()
