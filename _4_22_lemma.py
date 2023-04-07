import numpy as np
import matplotlib.pyplot as plt
import _4_22_lemma_config as local_config
import bib_config as global_config
from bib import *

# Create plot
fig, ax = plt.subplots()
setup_plot(ax, local_config.PLOT_BOTTOM_LEFT_CORNER,
           local_config.PLOT_TOP_RIGHT_CORNER)

lattice_points = plot_lattice(ax, local_config.BASIS_MATRIX, local_config.PLOT_BOTTOM_LEFT_CORNER,
                              local_config.PLOT_TOP_RIGHT_CORNER, False, **global_config.LATTICE_PROPERTIES)
# dual_lattice_points = plot_dual_lattice(ax,c.BASIS_MATRIX,c.PLOT_BOTTOM_LEFT_CORNER,c.PLOT_TOP_RIGHT_CORNER,False,**c.DUAL_LATTICE_PROPERTIES)

# Plot the bound
plot_circle(ax, local_config.ORIGIN, local_config.LEMMA_BOUND,
            **global_config.CIRCLE_TWO_PROPERTIES)

# Expression calculations
lattice_determinant = np.abs(np.linalg.det(local_config.BASIS_MATRIX))
cube_value = np.sqrt(lattice_determinant)

# Plot the cube
vertex_list = [[cube_value, cube_value], [cube_value, -cube_value],
               [-cube_value, -cube_value], [-cube_value, cube_value]]
plot_figure(ax, vertex_list, **global_config.FIGURE_ONE_PROPERTIES)

# Get the x inside the cube
point = []
for lattice_point in lattice_points:
    norm = np.linalg.norm(lattice_point)
    if norm <= cube_value and norm != 0:
        point = lattice_point
        inf_norm_sq = np.sqrt(2) * np.linalg.norm(lattice_point, np.inf)
        plot_circle(ax, local_config.ORIGIN, norm, **
                    global_config.CIRCLE_ONE_PROPERTIES)
        plot_circle(ax, local_config.ORIGIN, inf_norm_sq,
                    **global_config.CIRCLE_ONE_PROPERTIES)
        # Plot all the norms
        # plot_point(ax, [norm, 0], '$\Vert x \Vert_2$', False)
        plot_point(ax, [inf_norm_sq, 0],
                   '$\sqrt{2} \cdot \Vert x \Vert_\infty$', False)
        plot_point(ax, np.sqrt([0, local_config.LEMMA_BOUND]),
                   '$\sqrt{2}\cdot | det(A) |^{1/2}$', False)

        break

plot_point(ax, point, '$x$', True)

# Show the plot
plt.show()
