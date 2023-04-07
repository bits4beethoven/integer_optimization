import numpy as np
import matplotlib.pyplot as plt
import _4_24_lemma_config as local_config
import bib_config as global_config
from bib import *


def orthogonal_projection_onto_line(vector: np.ndarray, m, b):
    # Take a unit vector that lies on the line
    unit_vector = np.array([1, m + b])
    unit_vector /= np.linalg.norm(unit_vector)
    # Scalar projection of vector onto the unit vector
    scalar_projection = np.dot(vector, unit_vector)
    projection = unit_vector * scalar_projection
    return projection


def reflect_vector_accross_the_line(vector: np.ndarray, m, b):
    # Take a unit vector that lies on the line
    unit_vector = np.array([1, m + b])
    unit_vector /= np.linalg.norm(unit_vector)
    # Reflect the vector across the line
    reflection_matrix = np.array([
        [2*unit_vector[0]**2 - 1,           2 * unit_vector[0] * unit_vector[1]],
        [2*unit_vector[0]*unit_vector[1],   2 * unit_vector[1]**2 - 1]
    ])
    reflected_vector = reflection_matrix @ vector
    # Take the x coordinate as 1D representation
    result = reflected_vector[0]
    return result


def orthogonal_transformation_to_R(projected_vector: np.ndarray, m, b):
    # Take a unit vector that lies on the line
    unit_vector = np.array([1, m])
    unit_vector /= np.linalg.norm(unit_vector)

    # Calculate the distance along the line from the reference point
    # (projection of the origin onto the line) to the projected point
    reference_point = orthogonal_projection_onto_line(np.array([0, b]), m, b)
    position_on_line = np.dot(projected_vector - reference_point, unit_vector)

    return position_on_line


# Create plot
fig, ax = plt.subplots()
setup_plot(ax, local_config.PLOT_BOTTOM_LEFT_CORNER,
           local_config.PLOT_TOP_RIGHT_CORNER)

lattice_points = plot_lattice(ax, local_config.BASIS_MATRIX, local_config.PLOT_BOTTOM_LEFT_CORNER,
                              local_config.PLOT_TOP_RIGHT_CORNER, False, **local_config.LATTICE_PROPERTIES)
dual_lattice_points = plot_dual_lattice(
    ax, local_config.BASIS_MATRIX, local_config.PLOT_BOTTOM_LEFT_CORNER, local_config.PLOT_TOP_RIGHT_CORNER, False, **local_config.DUAL_LATTICE_PROPERTIES)


_, covering_radius = plot_covering_radius(
    ax, lattice_points, local_config.BASIS_MATRIX)
packing_radius_dual = plot_packing_radius(ax, dual_lattice_points)

# Plot right hand side
plot_circle(ax, local_config.ORIGIN, local_config.LEMMA_BOUND,
            **local_config.CIRCLE_PROPERTIES)
left_hand_side_product = covering_radius * packing_radius_dual
plot_circle(ax, local_config.ORIGIN, left_hand_side_product,
            **local_config.PRODUCT_CIRCLE_PROPERTIES)

# Left hand side is a rectangle defined by the radii
vertex_list = np.array([[0, 0], [0, packing_radius_dual], [
                       covering_radius, packing_radius_dual], [covering_radius, 0]])
plot_figure(ax, vertex_list, **
            global_config.FUNDAMENTAL_PARALLELEPIPED_PROPERTIES)

u = [-1, 1]
plot_point(ax, u, '$u$', True)
# Plot the orthogonal complement of u
m, b = get_line_through_points([0, 0], [1/3, 1/3])

projection = orthogonal_projection_onto_line([0, 1], m, b)
print(projection)
plot_line_through_points(ax, [0, 0], [1/3, 1/3])

projected_lattice_points = []
for lattice_point in lattice_points:
    projection = orthogonal_projection_onto_line(lattice_point, m, b)
    projected_lattice_points.append(projection)
    plot_vector(ax, lattice_point, projection -
                lattice_point, **global_config.PATH_ONE_PROPERTIES)

transformed_lattice_points = []
for projected_point in projected_lattice_points:
    transformed = orthogonal_transformation_to_R(projected_point, m, b)
    as_point = np.array([transformed, 0])
    transformed_lattice_points.append(as_point)
    plot_vector(ax, projected_point, as_point -
                projected_point, **global_config.PATH_THREE_PROPERTIES)

for transformed_point in transformed_lattice_points:
    plot_point(ax, transformed_point, r'$\in \Lambda_1$', False)

# Show the plot
plt.show()
