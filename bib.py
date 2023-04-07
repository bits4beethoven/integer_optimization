import numpy as np
import matplotlib.pyplot as plt
import bib_config as config
import fractions
import sys
from matplotlib import collections as mc
from pylab import *
from typing import List, Tuple


# Plotting methods
def plot_point(ax: plt.Axes, point: np.ndarray, annotation: str, want_coordinates: bool, **properties) -> None:
    """
    Plot a single point on a given matplotlib Axes object, with optional annotation and coordinate display.

    Args:
        ax (matplotlib.axes.Axes): The matplotlib axes on which to plot the point.
        point (np.ndarray): A 2D numpy array representing the coordinates of the point to be plotted.
        annotation (str): A string to annotate the plotted point.
        want_coordinates (bool): A boolean flag to determine whether or not to display the coordinates of the plotted point.
        **properties: Additional properties to be passed to the scatter and annotate functions.

    Returns:
        None

    Raises:
        None
    """
    ax.scatter(point[0], point[1], **properties)
    ax.annotate(annotation, (point[0], point[1]),
                **config.TEXT_ANNOTATION_PROPERTIES)
    if want_coordinates:
        ax.annotate(f"({pretty_format_float(point[0])}, {pretty_format_float(point[1])})", (
            point[0], point[1]-0.1), **config.TEXT_COORDINATES_PROPERTIES)


def plot_vector(ax: plt.Axes, start_point: np.ndarray, vector: np.ndarray, **properties) -> None:
    """
    Plot a vector given its starting point and displacement vector.

    Args:
        ax (matplotlib.axes.Axes): The matplotlib axes on which to plot the vector.
        start_point (np.ndarray): A 2D numpy array representing the starting point of the vector.
        vector (np.ndarray): A 2D numpy array representing the displacement vector.
        **properties: Additional properties to be passed to the quiver function.

    Returns:
        None
    """
    ax.quiver(start_point[0], start_point[1],
              vector[0], vector[1], **properties)


def plot_figure(ax: plt.Axes, vertex_list: np.ndarray, **properties: dict) -> None:
    """
    Plot a 2D figure with vertices, edges, and a filled polygon.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on.
    vertex_list : numpy.ndarray
        A list of 2D numpy arrays representing the vertices of the polygon. The points should be listed 
            consecutively in a specific order, where each point is connected to the next and the last point is connected 
            to the first to form a closed polygon.
    **properties : dict
        Arbitrary keyword arguments specifying the properties of the figure. Valid keys include:

        - 'vertex_color' : str, optional
            The color of the vertices.
        - 'vertex_marker' : str, optional
            The marker style of the vertices.
        - 'body_color' : str, optional
            The fill color of the polygon.
        - 'body_alpha' : float, optional
            The opacity of the polygon.
        - 'line_color' : str, optional
            The color of the edges.
        - 'line_linewidth' : float, optional
            The width of the edges.

    Returns
    -------
    None

    """

    # Extract vertex properties and plot the vertices.
    vertex_props = get_properties_for_prefix(properties, 'vertex_')
    for vertex in vertex_list:
        ax.scatter(vertex[0], vertex[1], **vertex_props)

    # Extract body properties and plot the body.
    body_props = get_properties_for_prefix(properties, 'body_')
    p = plt.Polygon(vertex_list, **body_props)
    ax.add_patch(p)

    # Extract line properties and plot the lines.
    line_props = get_properties_for_prefix(properties, 'line_')
    lines = []
    for i in range(0, len(vertex_list)):
        lines.append([(vertex_list[i][0], vertex_list[i][1]), (vertex_list[(
            i+1) % len(vertex_list)][0], vertex_list[(i+1) % len(vertex_list)][1])])
    lc = mc.LineCollection(lines, **line_props)
    ax.add_collection(lc)


def plot_lattice(ax: plt.Axes, basis_matrix: np.ndarray,
                 plot_bottom_left_corner: np.ndarray, plot_top_right_corner: np.ndarray,
                 want_coordinates: bool, color: str, linewidth: float, alpha: float) -> List[np.ndarray]:
    """
    Plot a 2D lattice given its basis matrix and a rectangular region.

    Args:
        ax (matplotlib.axes.Axes): The matplotlib axes on which to plot the lattice.
        basis_matrix (np.ndarray): A 2x2 numpy array representing the basis matrix of the lattice.
        plot_bottom_left_corner (np.ndarray): A 2D numpy array representing the lower left corner of the rectangular region.
        plot_top_right_corner (np.ndarray): A 2D numpy array representing the upper right corner of the rectangular region.
        want_coordinates (bool): A boolean flag indicating whether to display the coordinates of the plotted points.
        color (str): The color of the lattice points.
        linewidth (float): The linewidth of the plotted points.
        alpha (float): The transparency of the plotted points.

    Returns:
        List[np.ndarray]: A list of 2D numpy arrays representing the lattice points that were plotted.

    Raises:
        ValueError: If the rectangular region is invalid (e.g., if plot_bottom_left_corner is not lower than plot_top_right_corner).
    """

    # These are bounds for the loops.
    xmax, xmin, ymax, ymin = 0, 0, 0, 0

    # Solve the system of linear equations for the corner points.
    for p in [
        np.linalg.solve(basis_matrix, plot_top_right_corner),
        np.linalg.solve(basis_matrix, np.array(
            [plot_top_right_corner[0], plot_bottom_left_corner[1]])),
        np.linalg.solve(basis_matrix, plot_bottom_left_corner),
        np.linalg.solve(basis_matrix, np.array(
            [plot_bottom_left_corner[0], plot_top_right_corner[1]]))
    ]:
        px = 0
        py = 0
        # Round up the solution to the next larger bound. "Larger" in the sense of enlarging the plot.
        if p[0] >= 0:
            px = np.ceil(p[0])
            xmax = int(max(xmax, px))
        else:
            px = np.floor(p[0])
            xmin = int(min(xmin, px))
        if p[1] >= 0:
            py = np.ceil(p[1])
            ymax = int(max(ymax, py))
        else:
            py = np.floor(p[1])
            ymin = int(min(ymin, py))

    # list to hold the lattice points
    points = []

    for i in range(xmin, xmax+1):
        for j in range(ymin, ymax+1):
            # Make the linear combination of the basis vectors.
            p = i * basis_matrix[:, 0] + j * basis_matrix[:, 1]
            # If the point is within the plotting area, plot it and add the point to the list.
            if plot_bottom_left_corner[0] <= p[0] <= plot_top_right_corner[0] and plot_bottom_left_corner[1] <= p[1] <= plot_top_right_corner[1]:
                plot_point(ax, p, '', want_coordinates, color=color,
                           linewidth=linewidth, alpha=alpha)
                points.append(p)

    # Plot basis vectors
    properties = config.BASIS_VECTOR_PROPERTIES
    properties['alpha'] = alpha * 0.5
    properties['color'] = color
    plot_vector(ax, config.ORIGIN, basis_matrix[:, 0], **properties)
    plot_vector(ax, config.ORIGIN, basis_matrix[:, 1], **properties)

    return points


def plot_dual_lattice(ax: plt.Axes, original_lattice_basis_matrix: np.ndarray,
                      plot_bottom_left_corner: np.ndarray, plot_top_right_corner: np.ndarray, want_coordinates: bool,
                      **properties) -> List[np.ndarray]:
    """
    Plot the dual lattice of a given lattice.

    Args:
        ax (matplotlib.axes.Axes): The matplotlib axes on which to plot the dual lattice.
        original_lattice_basis_matrix (np.ndarray): A 2D numpy array representing the basis vectors of the original lattice.
        plot_bottom_left_corner (np.ndarray): A 2D numpy array representing the lower left corner of the rectangular region.
        plot_top_right_corner (np.ndarray): A 2D numpy array representing the upper right corner of the rectangular region.
        want_coordinates (bool): A boolean value that determines whether or not to display the coordinates of each point.
        **properties: Additional properties to be passed to the plot_lattice function.

    Returns:
        List[np.ndarray]: A list of 2D numpy arrays representing the lattice points that were plotted.

    Raises:
        ValueError: If the rectangular region is invalid (e.g., if plot_bottom_left_corner is not lower than plot_top_right_corner).
    """
    return plot_lattice(ax, np.linalg.inv(original_lattice_basis_matrix).T, plot_bottom_left_corner,
                        plot_top_right_corner, want_coordinates, **properties)


def plot_ellipsoid(ax: mpl.axes.Axes,
                   center: np.ndarray,
                   positive_definite_matrix: np.ndarray,
                   rotation_angle: float,
                   want_to_plot_eigenvectors: bool,
                   want_to_plot_circles: bool) -> None:
    """
    Plot an ellipsoid on the given matplotlib axes object `ax`.

    Parameters:
    -----------
    ax: matplotlib.axes.Axes
        The axes on which the ellipsoid will be plotted.
    center: np.ndarray
        A numpy array of shape (2,) representing the center of the ellipsoid.
    positive_definite_matrix: np.ndarray
        A 2x2 positive definite numpy array representing the ellipsoid.
    rotation_angle: float
        A float representing the rotation angle in radians.
    want_to_plot_eigenvectors: bool
        A boolean representing whether to plot the eigenvectors of the ellipsoid or not.

    Returns:
    --------
    None
    """
    # Generate points on the ellipse.
    theta = np.linspace(0, 2 * np.pi, 10000)
    eigenvalues, eigenvectors = np.linalg.eig(positive_definite_matrix)

    # Transpose the result to get eigenvectors in one array.
    eigenvectors = eigenvectors.T
    a = np.sqrt(1/eigenvalues[0])
    b = np.sqrt(1/eigenvalues[1])
    print("Eigenvalues of the ellipsoid: " + str(eigenvalues))
    print("Eigenvectors of the ellipsoid (row view): " + str(eigenvectors))
    print("Semi-axis of the ellipsoid: " + str(a) + ", " + str(b))

    # Get ellipse points.
    ellipse_points = a * np.cos(theta)[:, np.newaxis] * eigenvectors[:, 0] + \
        b * np.sin(theta)[:, np.newaxis] * eigenvectors[:, 1]

    # Rotate the points.
    rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)], [
                               np.sin(rotation_angle), np.cos(rotation_angle)]])
    rotated_points = np.dot(rotation_matrix, ellipse_points.T).T

    # Shift by center.
    rotated_points += center

    # Plot the ellipse
    ax.plot(rotated_points[:, 0], rotated_points[:, 1], 'b-')

    ax.scatter(center[0], center[1], c='b', s=100, marker='o', linewidths=1)

    # Show eigenvectors
    if want_to_plot_eigenvectors:
        # Rotate eigenvectors
        rotated_eigenvectors = np.dot(rotation_matrix, eigenvectors).T
        # Scale the eigenvectors according to the axis
        rotated_eigenvectors[0] = a * rotated_eigenvectors[0] / \
            np.linalg.norm(rotated_eigenvectors[0])
        rotated_eigenvectors[1] = b * rotated_eigenvectors[1] / \
            np.linalg.norm(rotated_eigenvectors[1])

        plot_vector(
            ax, center, rotated_eigenvectors[0], **config.BASIS_VECTOR_PROPERTIES)
        plot_vector(
            ax, center, rotated_eigenvectors[1], **config.BASIS_VECTOR_PROPERTIES)

    if want_to_plot_circles:
        plot_circle(ax, center, a, **config.CIRCLE_ONE_PROPERTIES)
        plot_circle(ax, center, b, **config.CIRCLE_ONE_PROPERTIES)


def plot_fundamental_parallelepiped_from_point(
    ax: plt.Axes,
    point: np.ndarray,
    basis_matrix: np.ndarray,
    **properties
) -> None:
    """
    Plots a fundamental parallelepiped on a given matplotlib Axes object, defined by a point and a basis matrix.

    Args:
        ax (matplotlib.axes.Axes): The matplotlib axes on which to plot the parallelepiped.
        point (np.ndarray): A 1D numpy array representing the corner point of the parallelepiped.
        basis_matrix (np.ndarray): A 2D numpy array representing the basis vectors of the parallelepiped.
        **properties: Additional properties to be passed to the plot_figure function.

    Returns:
        None

    Raises:
        None
    """
    points = np.array([config.ORIGIN, basis_matrix[:, 0], basis_matrix[:,
                      1] + basis_matrix[:, 0], basis_matrix[:, 1]]) + point
    plot_figure(ax, points, **properties)


def plot_circle(ax: plt.Axes, center: np.ndarray, radius: float, **properties) -> None:
    """
    Plots a circle in the given axis with the specified center and radius.

    Args:
        ax: A Matplotlib axis object.
        center: An ndarray specifying the (x,y) coordinates of the center of the circle.
        radius: A float representing the radius of the circle.
        **properties: Additional properties to be passed to the Circle patch constructor.

    Returns:
        None
    """
    circle = plt.Circle(center, radius, **properties)
    ax.add_patch(circle)


def plot_packing_radius(ax: object, lattice_points: List[np.ndarray]) -> float:
    """
    Computes the packing radius of a set of lattice points and plots a circle of the packing radius around each point.

    Args:
        ax: A matplotlib axis object on which to plot the circles.
        lattice_points: A list of numpy arrays, each representing a 3D lattice point.

    Returns:
        The computed packing radius as a float.

    Raises:
        None.
    """

    # Set packing raidus to a very big number.
    packing_radius = float("Inf")
    for lattice_point in lattice_points:
        # We do not consider the origin.
        if np.array_equal(lattice_point, config.ORIGIN):
            continue
        # If the distance to origin is smaller than the previous one:
        norm = np.linalg.norm(lattice_point)
        if norm < packing_radius:
            packing_radius = norm

    packing_radius = packing_radius * 1/2
    print("The packing radius is: " + str(packing_radius))

    # Plot the circle around each lattice points
    for lattice_point in lattice_points:
        plot_circle(ax, lattice_point, packing_radius, **
                    get_properties_for_prefix(config.PACKING_RADIUS_PROPERTIES, 'each_circle_'))

    return packing_radius


def plot_covering_radius(ax: object, lattice_points: List[np.ndarray], basis_matrix: np.ndarray) -> Tuple[Tuple[float, float], float]:
    """
    Plots the covering radius of a lattice with basis matrix basis_matrix using the lattice points in lattice_points.

    Args:
    - ax: A matplotlib Axes object where the plot will be drawn.
    - lattice_points: A List of np.arrays representing the lattice points.
    - basis_matrix: A np.array in the form np.array([np.array([b1x, b1y]), np.array([b2x, b2y])]) representing the basis matrix.

    Returns:
    - A Tuple containing the center of the circle as a Tuple of floats (circle_center_x, circle_center_y) and the radius of the circle as a float.
    """

    # Basis matrix is in the form np.array([np.array([b1x, b1y]), np.array([b2x, b2y])]).
    a = basis_matrix[:, 0]
    b = basis_matrix[:, 1]
    a_plus_b = a + b

    # Plot triangle between this points.
    plot_figure(ax, [a, b, a_plus_b], **get_properties_for_prefix(
        config.COVERING_RADIUS_PROPERTIES, 'triangle_'))

    # We look for a circle that goes through these points.
    squared_magnitude_b = b @ b
    squared_magnitude_a = a @ a
    squared_magnitude_a_plus_b = a_plus_b @ a_plus_b

    circumcenter_x = (squared_magnitude_a - squared_magnitude_b) / 2
    circumcenter_y = (squared_magnitude_b - squared_magnitude_a_plus_b) / 2
    det = b[0]*a[1] - a[0]*b[1]

    # If a,b and a+b are nearly collinear, the circumcenter cannot be uniquely determined.
    if abs(det) < 1.0e-6:
        return (None, np.inf)

    # Center of the cicle.
    circle_center_x = (
        circumcenter_x*(-a[1]) - circumcenter_y*(a[1] - b[1])) / det
    circle_center_y = (
        circumcenter_x*(a[0]) + circumcenter_y*(a[0] - b[0])) / det
    circle_center = np.array([circle_center_x, circle_center_y])

    radius = np.linalg.norm(circle_center - a)

    # Plot the circle in each lattice point.
    for lattice_point in lattice_points:
        plot_circle(ax, lattice_point, radius, **get_properties_for_prefix(
            config.COVERING_RADIUS_PROPERTIES, 'each_circle_',))

    # Plot the circle going through each of the points.
    plot_circle(ax, circle_center, radius, **
                get_properties_for_prefix(config.COVERING_RADIUS_PROPERTIES, 'main_circle_',))

    print("The covering radius is : " + str(radius))

    return ((circle_center_x, circle_center_y), radius)


def plot_equality(ax: plt.Axes, left_hand_side_coefficients: np.ndarray, right_hand_side: float) -> None:
    """
    Plots a linear equation in 2D space on a given matplotlib axis.

    Args:
        ax (matplotlib.pyplot.Axes): The axis object to plot on.
        left_hand_side_coefficients (numpy.ndarray): A 2-element array representing the coefficients of the left-hand side of the equation.
        right_hand_side (float): The right-hand side of the equation.

    Returns:
        None.

    Examples:
        >>> fig, ax = plt.subplots()
        >>> left_hand_side_coefficients = np.array([2, 3])
        >>> right_hand_side = 4
        >>> plot_equality(ax, left_hand_side_coefficients, right_hand_side)
    """
    print(left_hand_side_coefficients, right_hand_side)
    x = np.linspace(
        config.PLOT_BOTTOM_LEFT_CORNER[0], config.PLOT_TOP_RIGHT_CORNER[0], 1000)
    if left_hand_side_coefficients[1] == 0:
        bound = right_hand_side / left_hand_side_coefficients[0]
        plt.vlines(
            bound, config.PLOT_BOTTOM_LEFT_CORNER[1], config.PLOT_TOP_RIGHT_CORNER[1], colors=['b'])
    else:
        y = (right_hand_side -
             left_hand_side_coefficients[0] * x) / left_hand_side_coefficients[1]
        plt.plot(x, y, 'b-')


def plot_line_through_points(ax: plt.Axes, p1: np.ndarray, p2: np.ndarray) -> None:
    """
    Plots a line passing through two points in 2D space on a given matplotlib axis.

    Args:
        ax (matplotlib.pyplot.Axes): The axis object to plot on.
        p1 (numpy.ndarray): A 2-element array representing the (x,y) coordinates of the first point.
        p2 (numpy.ndarray): A 2-element array representing the (x,y) coordinates of the second point.

    Returns:
        None.
    """
    try:
        slope, shift = get_line_through_points(p1, p2)
    except ValueError as v:
        if v.args[0].startswith('Arguments'):
            plot_point(ax, p1, '', False)
    plot_point(ax, p1, '', False)
    plot_point(ax, p2, '', False)
    if slope == Inf:
        print(slope, shift)
        plot_equality(ax, np.array([1, 0]), p1[0])
    else:
        plot_equality(ax, np.array([-slope, 1]), shift)


def plot_slices(ax: plt.Axes, lattice_points: np.ndarray, vector: np.ndarray, config) -> None:
    """
    Plot slices of a lattice in a given direction on a matplotlib Axes object.

    Args:
        ax (matplotlib.axes.Axes): The matplotlib axes on which to plot the slices.
        lattice_points (np.ndarray): A 2D numpy array representing the lattice points to be plotted.
        vector (np.ndarray): A 1D numpy array representing the direction in which to plot the slices.
        config: A configuration object containing plot parameters.

    Returns:
        None

    Raises:
        None
    """
    # Get the plot parameters to simplify the readibility.
    left = config.PLOT_BOTTOM_LEFT_CORNER[0]
    right = config.PLOT_TOP_RIGHT_CORNER[0]
    top = config.PLOT_TOP_RIGHT_CORNER[1]
    bottom = config.PLOT_BOTTOM_LEFT_CORNER[1]
    # Get color cycle to synchronize the colors for vertical and horizontal lines.
    color_cycle = ax._get_lines.prop_cycler
    # Set the amount of samples for the x points here.
    x_array_size = (right-left)*100
    x = np.linspace(left, right, x_array_size)

    # Create partitions
    partitions = {}
    for lattice_point in lattice_points:
        product = round(np.dot(vector, lattice_point))
        if product in partitions:
            partitions[product].append(lattice_point)
        else:
            partitions[product] = [lattice_point]

    # Plot partitions
    slope, visited, single_point_keys = 0, 0, []
    for key in partitions.keys():
        if len(partitions[key]) >= 2:
            p1 = partitions[key][0]
            p2 = partitions[key][1]

            if p1[0] == p2[0] and p1[1] == p2[1]:
                # Same point.
                continue
            elif p1[0] == p2[0]:
                # x coordinates are the same, vertical line
                plt.vlines(p1[0], bottom, top, label=f"$x={p1[0]}$", color=next(
                    color_cycle)['color'])
            else:
                m = (p2[1] - p1[1]) / (p2[0] - p1[0])
                b = p1[1] - m * p1[0]
                y = m*x + b
                ax.plot(x, y, label=pretty_format_slice_as_label(
                    m, b, vector, key), color=next(color_cycle)['color'])
                if not visited:
                    slope = m
                    visited = 1
        else:
            single_point_keys.append(key)

    # Process single points
    for key in single_point_keys:
        single_point = partitions[key][0]
        # Check if the point is corner of a plot
        if single_point[0] in [left, right] and single_point[1] in [top, bottom]:
            continue
        b = single_point[1] - slope * single_point[0]
        y = slope * x + b
        ax.plot(x, y, label=pretty_format_slice_as_label(
            slope, b, vector, key), color=next(color_cycle)['color'])

    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


def plot_halfspace_defined_by_the_inequality(ax: plt.Axes, left_hand_side_coefficients: np.ndarray, right_hand_side: float, less_or_equal: bool, **properties) -> None:
    """
    Plots a half-space defined by a linear inequality in 2D space on a given matplotlib axis.

    Args:
        ax (matplotlib.pyplot.Axes): The axis object to plot on.
        left_hand_side_coefficients (numpy.ndarray): A 2-element array representing the coefficients of the left-hand side of the inequality.
        right_hand_side (float): The right-hand side of the inequality.
        less_or_equal (bool): Whether the inequality is less-than-or-equal-to (True) or greater-than (False).
        **properties: Optional keyword arguments to be passed to the `fill_between` method.

    Returns:
        None.

    Examples:
        >>> fig, ax = plt.subplots()
        >>> left_hand_side_coefficients = np.array([2, 3])
        >>> right_hand_side = 4
        >>> less_or_equal = True
        >>> plot_halfspace_defined_by_the_inequality(ax, left_hand_side_coefficients, right_hand_side, less_or_equal, facecolor='red', alpha=0.2)
    """
    # Plot the orthogonal vector.
    plot_vector(ax, config.ORIGIN, left_hand_side_coefficients,
                **config.BASIS_VECTOR_PROPERTIES)

    x = np.linspace(
        config.PLOT_BOTTOM_LEFT_CORNER[0], config.PLOT_TOP_RIGHT_CORNER[0], 1000)

    # If y == 0 and the inequality is a verticlal line
    if left_hand_side_coefficients[1] == 0:
        bound = right_hand_side / left_hand_side_coefficients[0]
        plt.vlines(
            bound, config.PLOT_BOTTOM_LEFT_CORNER[1], config.PLOT_TOP_RIGHT_CORNER[1])

        if less_or_equal:
            plt.fill_between(x, config.PLOT_BOTTOM_LEFT_CORNER[1], config.PLOT_TOP_RIGHT_CORNER[1], where=(
                x <= bound), **properties)
        else:
            plt.fill_between(x, config.PLOT_BOTTOM_LEFT_CORNER[1], config.PLOT_TOP_RIGHT_CORNER[1], where=(
                x >= bound), **properties)

    else:
        # Generate points on the line H^{=}(alpha, beta)
        y = (right_hand_side -
             left_hand_side_coefficients[0] * x) / left_hand_side_coefficients[1]

        if less_or_equal:
            limit = config.PLOT_BOTTOM_LEFT_CORNER[1]
        else:
            limit = config.PLOT_TOP_RIGHT_CORNER[1]

        # Plot the line H^{=}(alpha, beta)
        plt.plot(x, y, 'b-', label='H^{=}(alpha, beta)')

        # Fill the area below the line H^{=}(alpha, beta)
        plt.fill_between(x, y, limit, **properties)


def plot_polyhedron(ax: plt.Axes, A: np.ndarray, b: np.ndarray, config, **properties) -> None:
    """
    Plots the feasible set defined by the linear system of inequalities Ax <= b.

    Args:
        ax (matplotlib.axes.Axes): The matplotlib axes on which to plot the polyhedron.
        A (np.ndarray): A 2D numpy array representing the matrix A in Ax <= b.
        b (np.ndarray): A 1D numpy array representing the vector b in Ax <= b.
        config: A configuration object containing plot parameters.
        **properties: Additional properties to be passed to the plot function.

    Raises:
        ValueError: If both the x and y coordinates of any inequality are zero.
        AssertionError: If both the number of rows in A and the size of b are zero.

    Returns:
        None
    """

    # Get the plot parameters to simplify the readibility.
    left = config.PLOT_BOTTOM_LEFT_CORNER[0]
    right = config.PLOT_TOP_RIGHT_CORNER[0]
    top = config.PLOT_TOP_RIGHT_CORNER[1]
    bottom = config.PLOT_BOTTOM_LEFT_CORNER[1]
    # Get color cycle to synchronize the colors for vertical and horizontal lines.
    color_cycle = ax._get_lines.prop_cycler
    # Set the amount of samples for the x points here.
    x_array_size = (right-left)*100
    x = np.linspace(left, right, x_array_size)

    # Get the amount of inequalitites
    amount_of_inequalities = A.shape[0]

    # List of lists of function values for each matrix row.
    y_lines = []

    # x_bounds define the bounds for the area to be filled.
    x_less_or_equal_bounds = [right]
    x_greater_or_equal_bounds = [left]

    # Process each inequality.
    for i in range(amount_of_inequalities):
        x_coefficient, y_coefficient = A[i]
        if y_coefficient == 0:
            if x_coefficient == 0:
                raise ValueError("Both x,y coordinates are zero.")
            else:
                # We have a vertical line. This is not a function of x.
                bound = b[i] / x_coefficient
                plt.vlines(bound, bottom, top, label=pretty_format_row_as_label(
                    x_coefficient, y_coefficient, b[i]), color=next(color_cycle)['color'])
                # Determine if it is a <= or >= constraint depending on the sign.
                if x_coefficient > 0:
                    x_less_or_equal_bounds.append(bound)
                else:
                    x_greater_or_equal_bounds.append(bound)

                # Append dummy array for a vertical line.
                # We need this because vertical lines are not added into the y_lines[] list.
                # But, we want to keep the indices for inequalities consistent.
                dummy = np.array([0]*x_array_size)
                y_lines.append(dummy)
        else:
            # We have either a horizontal line, or a "normal" line.
            # Anyway, it is a function of x.
            y_line = (b[i]/y_coefficient) - (x_coefficient/y_coefficient) * x
            ax.plot(x, y_line, label=pretty_format_row_as_label(
                x_coefficient, y_coefficient, b[i]), color=next(color_cycle)['color'])
            y_lines.append(y_line)

    # Determine if it is a <= or >= constraint depending on the sign.
    # If the y_coefficient < 0, then it a greater or equal constraint.
    y_greater_or_equal_indices, y_less_or_equal_indices = [i for i in range(
        amount_of_inequalities) if A[i][1] < 0], [i for i in range(amount_of_inequalities) if A[i][1] > 0]

    # Get left and right bounds filling in the x axis.
    right_bound = np.min(x_less_or_equal_bounds)
    left_bound = np.max(x_greater_or_equal_bounds)

    # Determine the indices of the x array that are in these bounds.
    lower_index, upper_index = 0, x_array_size - 1

    while x[lower_index] < left_bound:
        lower_index += 1
    while x[upper_index] > right_bound:
        upper_index -= 1

    # Take the sublist of x for filling.
    x = x[lower_index:upper_index+1]

    # Reduce the y arrays into the x bounds.
    for j in range(amount_of_inequalities):
        y_lines[j] = y_lines[j][lower_index:upper_index+1]

    # Setup a new array size to fill the polyhedron.
    x_array_size = len(x)

    # Get top and bottom bounds for y.
    lower_bound = [config.PLOT_BOTTOM_LEFT_CORNER[1]] * x_array_size
    upper_bound = [config.PLOT_TOP_RIGHT_CORNER[1]] * x_array_size

    # Adjust the bounds.
    if y_greater_or_equal_indices:
        lower_bound = np.maximum.reduce(
            [y_lines[i] for i in y_greater_or_equal_indices])
    if y_less_or_equal_indices:
        upper_bound = np.minimum.reduce(
            [y_lines[i] for i in y_less_or_equal_indices])

    # Combine both lists.
    condition = np.greater(upper_bound, lower_bound)
    # If not condition is satisfied, the system does not have a solution.
    if not any(condition):
        print("No solution to the system!")

    # Fill the feasible set.
    ax.fill_between(x, lower_bound, upper_bound, where=condition, **properties)
    # Add legends.
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


def plot_integer_hull(ax: plt.Axes, A: np.ndarray, b: np.ndarray, lattice_points: List[np.ndarray]) -> None:
    """
    Plots the convex hull of the lattice points that lie inside the given polyhedron.

    Args:
        ax: The Matplotlib axes object to plot on.
        A: The constraint matrix of the polyhedron Ax <= b.
        b: The constraint vector of the polyhedron Ax <= b.
        lattice_points: A list of numpy arrays representing the lattice points.

    Returns:
        None
    """
    points_inside_the_polyhedron = get_lattice_points_inside_the_polyhedron(
        ax, A, b, lattice_points)
    convev_hull_points = perfrom_graham_scan(points_inside_the_polyhedron)
    plot_figure(ax, convev_hull_points, **config.HULL_PROPERTIES)

    return convev_hull_points


def plot_plus_lattice(ax: plt.Axes, Y_with_column_vectors: np.ndarray, config) -> List[np.ndarray]:
    """
    Plots a lattice defined by the column vectors in a given 2D plane, with additional points and vectors.

    Args:
        ax (matplotlib.axes.Axes): The matplotlib axes on which to plot the lattice.
        Y_with_column_vectors (np.ndarray): A 2D numpy array representing the column vectors of the lattice.
        config: A configuration object containing plot parameters.

    Raises:
        ValueError: If there are more than two vectors in Y_with_column_vectors or if there are fewer than two vectors.

    Returns:
        A list of 1D numpy arrays representing the plotted points.
    """
    amount_of_vectors = np.shape(Y_with_column_vectors)[1]
    if amount_of_vectors > 2:
        raise ValueError(
            'The code does not work for more than 2 vectors in Y.')
    if amount_of_vectors < 2:
        raise ValueError('Too less vectors in Y')

    left = config.PLOT_BOTTOM_LEFT_CORNER[0]
    bottom = config.PLOT_BOTTOM_LEFT_CORNER[1]
    right = config.PLOT_TOP_RIGHT_CORNER[0]
    top = config.PLOT_TOP_RIGHT_CORNER[1]

    bounds = [0]*amount_of_vectors

    for i in range(0, amount_of_vectors-1):
        for j in range(i+1, amount_of_vectors):
            # Take basis vectors
            b_i = Y_with_column_vectors[:, i]
            b_j = Y_with_column_vectors[:, j]
            basis_matrix = np.column_stack([b_i, b_j])

            # Solve the system of linear equations for the corner points.
            for p in [
                np.linalg.solve(basis_matrix, np.array([right, top])),
                np.linalg.solve(basis_matrix, np.array([right, bottom])),
                np.linalg.solve(basis_matrix, np.array([left, bottom])),
                np.linalg.solve(basis_matrix, np.array([left, top]))
            ]:
                # Quit if some negative lincomb is detected.
                if p[0] < 0 or p[1] < 0:
                    continue
                px = 0
                py = 0

                # Round up the solution to the next larger bound. "Larger" in the sense of enlarging the plot.
                px = np.ceil(p[0])
                bounds[i] = int(max(bounds[i], px))

                py = np.ceil(p[1])
                bounds[j] = int(max(bounds[j], py))

    # Step 2: get maximum along one vector
    temp_v_1 = np.copy(Y_with_column_vectors[:, 0])
    temp_v_2 = np.copy(Y_with_column_vectors[:, 1])
    index_v_1, index_v_2 = 0, 0
    while left <= temp_v_1[0] <= right and bottom <= temp_v_1[1] <= top:
        index_v_1 += 1
        temp_v_1 += Y_with_column_vectors[:, 0]
    while left <= temp_v_2[0] <= right and bottom <= temp_v_2[1] <= top:
        index_v_2 += 1
        temp_v_2 += Y_with_column_vectors[:, 1]

    bounds[0] = np.maximum(bounds[0], index_v_1)
    bounds[1] = np.maximum(bounds[1], index_v_2)

    # Get the points and plot the points
    points = []

    point_properties = get_properties_for_prefix(
        config.LATTICE_PLUS_PROPERTIES, 'point_')

    for i in range(0, bounds[0]+1):
        for j in range(0, bounds[1]+1):
            p = i * Y_with_column_vectors[:, 0] + \
                j * Y_with_column_vectors[:, 1]
            if (left - 1 <= p[0] <= right + 1) and (bottom - 1 <= p[1] <= top + 1):
                plot_point(ax, p, '', False, **point_properties)
                points.append(p)

    vector_properties = get_properties_for_prefix(
        config.LATTICE_PLUS_PROPERTIES, 'vector_')
    plot_vector(ax, config.ORIGIN,
                Y_with_column_vectors[:, 0], **vector_properties)
    plot_vector(ax, config.ORIGIN,
                Y_with_column_vectors[:, 1], **vector_properties)

    return points

# Calculations methods


def get_line_through_points(p1: np.ndarray, p2: np.ndarray) -> tuple:
    """
    Calculates the equation of the line passing through two points in 2D space.

    Args:
        p1 (numpy.ndarray): A 2-element array representing the (x,y) coordinates of the first point.
        p2 (numpy.ndarray): A 2-element array representing the (x,y) coordinates of the second point.

    Returns:
        A tuple of two values:
        - If the line is vertical, returns (inf, 0) to indicate that the slope is undefined and the x-intercept is 0.
        - If the points are the same, raises a ValueError with the message "Arguments are the same".
        - Otherwise, returns a tuple (slope, shift) representing the slope and y-intercept of the line, respectively.

    Raises:
        ValueError: If the arguments are the same (i.e., p1 == p2).

    Examples:
        >>> p1 = np.array([0, 0])
        >>> p2 = np.array([1, 1])
        >>> get_line_through_points(p1, p2)
        y = 1.0x + 0.0
        (1.0, 0.0)

        >>> p1 = np.array([0, 0])
        >>> p2 = np.array([0, 1])
        >>> get_line_through_points(p1, p2)
        y = infx + 0
        (inf, 0)

        >>> p1 = np.array([0, 0])
        >>> p2 = np.array([0, 0])
        >>> get_line_through_points(p1, p2)
        Traceback (most recent call last):
            ...
        ValueError: Arguments are the same.
    """
    slope, shift = 0, 0
    if p1[0] == p2[0]:
        if p1[1] == p2[1]:
            raise ValueError("Arguments are the same.")
        else:
            return Inf, 0
    slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
    shift = p1[1] - slope*p1[0]
    print("y = " + str(slope) + "x + " + str(shift))
    return slope, shift


def get_equation_through_points(p1: np.ndarray, p2: np.ndarray, less_or_equal: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the equation of the line through two points in a 2D plane.

    Args:
        p1 (np.ndarray): A 1D numpy array representing the coordinates of the first point.
        p2 (np.ndarray): A 1D numpy array representing the coordinates of the second point.
        less_or_equal (bool): A boolean flag indicating whether the line represents a less-than-or-equal-to constraint or a greater-than constraint.

    Returns:
        A tuple containing two 1D numpy arrays representing the coefficients of the equation of the line in the form ax + by <= c or ax + by > c.
    """
    # same point
    if p1[0] == p2[0] and p1[1] == p2[1]:
        raise ValueError("Points p1 and p2 are the same.")
    # x-s are the same, vertical line
    elif p1[0] == p2[0]:
        x = 1
        y = 0
        b = p1[0]

    # y-s are the same, horizontal line
    elif p1[1] == p2[1]:
        y = 1
        x = 0
        b = p1[1]

    # points are not on the same axis
    else:
        x = -(p2[1] - p1[1]) / (p2[0] - p1[0])
        y = 1
        b = p1[0]*x + p1[1]

    if not less_or_equal:
        x *= -1
        y *= -1
        b *= -1

    return np.array([x, y]), np.array([b])


def get_gram_schmidt_basis(basis_matrix: np.ndarray) -> List[np.ndarray]:
    """
    Applies the Gram-Schmidt algorithm to the given set of basis vectors to obtain an orthonormal basis.

    Args:
        basis_vectors: A numpy array of the basis vectors.

    Returns:
        A list of numpy arrays representing the Gram-Schmidt basis vectors.
    """
    n = len(basis_matrix)
    g = [None] * n
    g[0] = basis_matrix[:, 0].astype('float64')
    for i in range(1, n):
        g[i] = basis_matrix[:, i].astype('float64')
        for j in range(i):
            mu_ij = np.dot(basis_matrix[:, i], g[j]) / np.linalg.norm(g[j])**2
            g[i] -= mu_ij * g[j]
    return g


def perfrom_graham_scan(points: List[np.ndarray]) -> List[List[float]]:
    """
    Returns the set of points that define the convex hull for the given set of 2D points,
    using the Graham scan algorithm.

    Args:
    - points: A list of numpy arrays representing 2D points

    Returns:
    - A list of lists, where each sublist contains a pair of floats representing
      the coordinates of a point that defines the convex hull

    """

    if len(points) <= 1:
        return points

    points = [list(point) for point in points]  # Convert numpy arrays to lists
    # Sort the points by x-coordinate
    sorted_points = sorted(points)

    # Find the lower hull
    lower_hull = []
    for point in sorted_points:
        while len(lower_hull) >= 2 and get_orientation(lower_hull[-2], lower_hull[-1], point) != 2:
            lower_hull.pop()
        lower_hull.append(point)

    # Find the upper hull
    upper_hull = []
    for point in reversed(sorted_points):
        while len(upper_hull) >= 2 and get_orientation(upper_hull[-2], upper_hull[-1], point) != 2:
            upper_hull.pop()
        upper_hull.append(point)

    # Remove the first and last points from the upper hull to avoid duplicates
    upper_hull = upper_hull[1:-1]

    # Concatenate the lower and upper hulls to obtain the convex hull
    convex_hull_points = lower_hull + upper_hull
    return convex_hull_points


def get_lattice_points_inside_the_polyhedron(ax: Axes,
                                             A: np.ndarray,
                                             b: np.ndarray,
                                             lattice_points: List[np.ndarray]) -> List[np.ndarray]:
    """
    Finds the lattice points inside the polyhedron Ax <= b, where A is a numpy array matrix,
    b is a numpy array, and lattice_points is a list of numpy arrays.

    Args:
        ax: A matplotlib Axes object to plot the polyhedron on.
        A: A numpy array matrix defining the coefficients of the linear inequality Ax <= b.
        b: A numpy array defining the right-hand side of the linear inequality Ax <= b.
        lattice_points: A list of numpy arrays representing the lattice points to check.

    Returns:
        A list of numpy arrays representing the lattice points inside the polyhedron Ax <= b.
    """
    hull_points = []
    for p in lattice_points:
        if all(A @ p <= b):
            hull_points.append(p)
    return hull_points


def create_integer_positive_definite_matrix(dimension: int, minimal_entry: int, maximal_entry: int) -> np.ndarray:
    """
    Creates a dimension x dimension integer positive definite matrix with entries between minimal_entry and maximal_entry.

    Args:
        dimension (int): The dimension of the matrix.
        minimal_entry (int): The smallest possible entry in the matrix.
        maximal_entry (int): The largest possible entry in the matrix.

    Returns:
        np.ndarray: A dimension x dimension integer positive definite matrix with entries between minimal_entry and maximal_entry.
    """
    method_in_use = 4
    if method_in_use == 1:
        return __create_integer_positive_definite_matrix_wrong_result(dimension, minimal_entry, maximal_entry)
    elif method_in_use == 2:
        return __create_integer_positive_definite_matrix_with_eigenvalue_decomposition(dimension, minimal_entry, maximal_entry)
    elif method_in_use == 3:
        return __create_integer_positive_definite_matrix_with_transpose(dimension, minimal_entry, maximal_entry)
    elif method_in_use == 4:
        return __create_integer_positive_definite_matrix_with_cholesky(dimension, minimal_entry, maximal_entry)
    return


def get_ellipsoid_volume(positive_definite_matrix: np.ndarray) -> float:
    """
    Computes the volume of an ellipsoid in a 2D or 3D space defined by a positive definite matrix.

    Args:
        positive_definite_matrix (np.ndarray): A 2D numpy array representing a positive definite matrix that defines the ellipsoid.

    Returns:
        A float representing the volume of the ellipsoid.
    """
    return 4/3 * np.pi * np.sqrt(1/positive_definite_matrix[0][0]) * np.sqrt(1/positive_definite_matrix[1][1])


def get_cutting_point_for_lines(line_1_left_side: np.ndarray, line_1_right_side: np.ndarray,
                                line_2_left_side: np.ndarray, line_2_right_side: np.ndarray) -> np.ndarray:
    """
    Computes the intersection point of two lines in a 2D space given their equations in the form ax + by = c.

    Args:
        line_1_left_side (np.ndarray): A 1D numpy array representing the coefficients of the left-hand side of the equation for line 1 in the form ax + by.
        line_1_right_side (np.ndarray): A 1D numpy array representing the constant term of the equation for line 1 in the form c.
        line_2_left_side (np.ndarray): A 1D numpy array representing the coefficients of the left-hand side of the equation for line 2 in the form ax + by.
        line_2_right_side (np.ndarray): A 1D numpy array representing the constant term of the equation for line 2 in the form c.

    Returns:
        A 1D numpy array representing the coordinates of the intersection point of the two lines.
    """
    A = np.array([line_1_left_side, line_2_left_side])
    print(A)
    b = np.array([line_1_right_side, line_2_right_side])
    print(b)
    return np.linalg.solve(A, b)


def get_properties_for_prefix(properties: dict, prefix: str) -> dict:
    """
    Filter a dictionary of properties to only include those whose keys start with a given prefix.

    Args:
        properties: A dictionary of properties to be filtered.
        prefix: A string representing the prefix to match.

    Returns:
        A dictionary containing only the key-value pairs from `properties` whose keys start with `prefix`,
        with the prefix removed from the keys.

    Examples:
        >>> props = {'vertex_color': 'r', 'vertex_marker': 'o', 'body_color': 'b'}
        >>> get_properties_for_prefix(props, 'vertex_')
        {'color': 'r', 'marker': 'o'}
        >>> get_properties_for_prefix(props, 'body_')
        {'color': 'b'}
    """
    return {k.replace(prefix, ''): v for k, v in properties.items() if k.startswith(prefix)}


def get_orientation(p: List[float], q: List[float], r: List[float]) -> int:
    """
    Returns the orientation of the triplet (p, q, r) as follows:
    0 --> p, q and r are collinear
    1 --> Clockwise
    2 --> Counterclockwise

    Args:
    p (List[float]): A list of two float numbers representing the coordinates of point p
    q (List[float]): A list of two float numbers representing the coordinates of point q
    r (List[float]): A list of two float numbers representing the coordinates of point r

    Returns:
    int: An integer value indicating the orientation of the triplet (p, q, r).
    """

    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0:
        return 0
    elif val > 0:
        return 1
    else:
        return 2

# PRIVATE METHODS


def __create_integer_positive_definite_matrix_wrong_result(dimension: int, minimal_entry: int, maximal_entry: int) -> np.ndarray:
    """
    WARNING: This method does not work properly and should not be used.

    Generates a positive definite matrix with integer entries between the given
    minimal and maximal values using a method that produces incorrect results.

    Args:
        dimension (int): The dimension of the matrix to be generated.
        minimal_entry (int): The minimal entry value for the matrix.
        maximal_entry (int): The maximal entry value for the matrix.

    Returns:
        numpy.ndarray: A positive definite matrix with integer entries.
    """
    if maximal_entry < minimal_entry:
        raise ValueError("maximal_entry is less than minimal_entry")
    if maximal_entry >= 0:
        upper_bound = np.floor(np.sqrt(maximal_entry / dimension))
    else:
        pass

    if minimal_entry >= 0:
        lower_bound = np.ceil(np.sqrt(minimal_entry / dimension))
    else:
        pass

    A = np.random.randint(lower_bound, upper_bound + 1, (dimension, dimension))
    return np.dot(A, A.T)


def __create_integer_positive_definite_matrix_with_eigenvalue_decomposition(dimension: int, minimal_entry: int, maximal_entry: int) -> np.ndarray:
    """
    Generates a random integer positive definite matrix of dimension n x n, with entries between the specified bounds.
    This method uses the eigenvalue decomposition of a random matrix to generate the positive definite matrix.

    Args:
        dimension (int): The dimension of the matrix.
        minimal_entry (int): The minimal allowed value for the matrix entries.
        maximal_entry (int): The maximal allowed value for the matrix entries.

    Returns:
        np.ndarray: A randomly generated positive definite matrix with integer entries between minimal_entry and maximal_entry.

    Raises:
        RecursionError: If the maximal_entry and minimal_entry are too close, the method may not converge and cause a recursion error.

    Note:
        This method may hang indefinitely if the maximal_entry and minimal_entry bounds are too close to each other.
    """

    if maximal_entry < minimal_entry:
        raise ValueError("maximal_entry is less than minimal_entry")

    # Generate a random matrix with entries between -1 and 1
    A = np.random.rand(dimension, dimension) * 2 - 1

    # Make the matrix symmetric by setting the upper triangle equal to the lower triangle
    A = np.triu(A) + np.triu(A, 1).T

    # Take the eigenvalue decomposition of the matrix
    eigenvalues, eigenvectors = np.linalg.eigh(A)

    # Replace negative eigenvalues with their absolute values
    eigenvalues = np.abs(eigenvalues)

    # Reconstruct the matrix with the updated eigenvalues
    A = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    # Scale the matrix to have entries between minimal_entry and maximal_entry
    min_eigenvalue = np.min(eigenvalues)
    max_eigenvalue = np.max(eigenvalues)
    A = (A - min_eigenvalue) / (max_eigenvalue - min_eigenvalue)
    A = np.round(A * (maximal_entry - minimal_entry) +
                 minimal_entry).astype(int)

    # Check for infinite or NaN values in the matrix
    if np.any(np.isinf(A)) or np.any(np.isnan(A)) or not np.all(np.linalg.eigvals(A) > 0):
        return __create_integer_positive_definite_matrix_with_eigenvalue_decomposition(dimension, minimal_entry, maximal_entry)
    else:
        return A


def __create_integer_positive_definite_matrix_with_transpose(dimension, minimal_entry, maximal_entry):
    """
    Generates an integer-valued positive definite matrix using the transpose method.

    Parameters:
    dimension (int): The dimension of the matrix to be generated.
    minimal_entry (int): The smallest allowed entry in the matrix.
    maximal_entry (int): The largest allowed entry in the matrix.

    Returns:
    np.ndarray: An integer-valued positive definite matrix with entries between `minimal_entry` and `maximal_entry`.

    Note:
    This method may hang if `maximal_entry` and `minimal_entry` are close to each other. It is recommended to use this method with caution.
    """

    if maximal_entry < minimal_entry:
        raise ValueError("maximal_entry is less than minimal_entry")

    while True:
        # Generate a random matrix with entries between minimal_entry and maximal_entry
        A = np.random.randint(
            minimal_entry, maximal_entry + 1, (dimension, dimension))

        # Make a positive definite matrix by multiplying A with its transpose
        A = A @ A.T

        # Check if the matrix has entries within the desired range
        if np.min(A) >= minimal_entry and np.max(A) <= maximal_entry and all(np.linalg.eigvals(A) > 0):
            break

    return A


def __create_integer_positive_definite_matrix_with_cholesky(dimension, minimal_entry, maximal_entry):
    """
    Generates a random positive definite matrix with integer entries between minimal_entry and maximal_entry (both inclusive)
    using the Cholesky decomposition method.

    Parameters:
        dimension (int): The dimension of the matrix (a positive integer).
        minimal_entry (int): The minimum value of the entries in the matrix (an integer).
        maximal_entry (int): The maximum value of the entries in the matrix (an integer).

    Returns:
        np.ndarray: A randomly generated positive definite matrix with integer entries between minimal_entry and maximal_entry.

    Notes:
        This method is recommended to be used among the other four __create_integer_positive_definite_matrix methods as it is
        more reliable and efficient than the other methods.
    """

    if maximal_entry < minimal_entry:
        raise ValueError("maximal_entry is less than minimal_entry")

    while True:
        # Generate a random matrix with entries between minimal_entry and maximal_entry
        A = np.random.randint(
            minimal_entry, maximal_entry + 1, (dimension, dimension))

        # Make the matrix symmetric by setting the upper triangle equal to the lower triangle
        A = np.triu(A) + np.triu(A, 1).T

        # Attempt Cholesky decomposition
        try:
            L = np.linalg.cholesky(A)
            break
        except np.linalg.LinAlgError:
            continue

    return A

    pass

# TECHNICAL METHODS


def setup_plot(ax: plt.Axes, ldown: np.ndarray, rup: np.ndarray) -> None:
    """
    Setup the given Matplotlib axis for plotting.

    Args:
        ax (matplotlib.axes.Axes): The Matplotlib axis to be setup.
        ldown (np.ndarray): A 2D numpy array representing the lower left corner of the rectangular region.
        rup (np.ndarray): A 2D numpy array representing the upper right corner of the rectangular region.

    Returns:
        None
    """
    ax.set_aspect('equal')
    ax.grid(True, which='both')
    ax.set_xlim(ldown[0], rup[0])
    ax.set_ylim(ldown[1], rup[1])

    plt.axhline(0, color='black')
    plt.axvline(0, color='black')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    mngr = plt.get_current_fig_manager()
    mngr.resize(960 + 100, 1080)
    # mngr.full_screen_toggle()


def pretty_format_float(f: float) -> str:
    """
    Converts a floating point number to a string representation that displays it in a simplified form.

    Args:
        f (float): A floating point number to be formatted.

    Returns:
        A string representation of the input number that displays it in a simplified form using fractions.
    """
    a_f = fractions.Fraction(f).limit_denominator()
    return f"{a_f}"


def pretty_format_row_as_label(a1: float, a2: float, b: float) -> str:
    """
    Formats the given row coefficients as a string suitable for use as a label in a matplotlib plot.

    Parameters:
    a1 (float): The first coefficient.
    a2 (float): The second coefficient.
    b (float): The right-hand side constant.


    Returns:
    str: The formatted label string.
    """
    a1_f = fractions.Fraction(a1).limit_denominator()
    a2_f = fractions.Fraction(a2).limit_denominator()
    b_f = fractions.Fraction(b).limit_denominator()
    return f"$({a1_f}) \cdot x + $" + f"$({a2_f}) \cdot y \leq $" + f"${b_f}$"


def pretty_format_slice_as_label(m: float, b: float, v, k) -> str:
    """
    Formats the given row coefficients as a string suitable for use as a label in a matplotlib plot.

    Parameters:
    a1 (float): The first coefficient.
    a2 (float): The second coefficient.
    b (float): The right-hand side constant.

    Returns:
    str: The formatted label string.
    """
    m_f = fractions.Fraction(m).limit_denominator()
    b_f = fractions.Fraction(b).limit_denominator()
    v_0_f = fractions.Fraction(v[0]).limit_denominator()
    v_1_f = fractions.Fraction(v[1]).limit_denominator()
    k_f = fractions.Fraction(k).limit_denominator()
    return f"$y = ({m_f}) \cdot x + $" + f"$({b_f})$, " + f"$({v_0_f},{v_1_f})^Tx= {k_f}$"
