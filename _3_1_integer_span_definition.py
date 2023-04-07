import numpy as np
import matplotlib.pyplot as plt
import _3_1_integer_span_definition_config as local_config
import bib_config as global_config
from bib import *

# Setup plot.
fig, ax = plt.subplots()
setup_plot(ax, local_config.PLOT_BOTTOM_LEFT_CORNER,
           local_config.PLOT_TOP_RIGHT_CORNER)

# Plot a lattice.
plot_lattice(ax, local_config.BASIS_MATRIX, local_config.PLOT_BOTTOM_LEFT_CORNER,
             local_config.PLOT_TOP_RIGHT_CORNER, False, **global_config.LATTICE_PROPERTIES)

# Show the plot
plt.show()
