"""Planar cluster descriptors; inputs are already in a common Cartesian frame."""

from __future__ import annotations

import numpy as np


def minimum_area_bbox_perimeter(hull: np.ndarray) -> float:
    """Perimeter of the minimum-area enclosing rectangle, allowing rotation.

    ``hull`` contains ordered convex-hull vertices. An optimum rectangle has
    one side parallel to a hull edge. A line segment has width zero and
    perimeter twice its length; a single point has perimeter zero.
    """
    if len(hull) < 2:
        return 0.0
    centered = hull - hull[0]
    best_area = float("inf")
    best_perimeter = 0.0
    for edge in np.roll(hull, -1, axis=0) - hull:
        length = float(np.linalg.norm(edge))
        if length <= 1e-9:
            continue
        direction = edge / length
        normal = np.asarray([-direction[1], direction[0]])
        width = float(np.ptp(centered @ direction))
        height = float(np.ptp(centered @ normal))
        area = width * height
        perimeter = 2.0 * (width + height)
        if area < best_area or (area == best_area and perimeter < best_perimeter):
            best_area = area
            best_perimeter = perimeter
    return best_perimeter


def mean_diameter_line_distance(xy: np.ndarray, hull: np.ndarray) -> float:
    """Mean perpendicular distance to the line through the farthest two points.

    A farthest pair lies on the convex hull. Search hull pairs with O(H)
    temporary memory, not an N-by-N distance matrix over all cluster points.
    Distances are then averaged over *all* points, including interior points
    and repeated measurements. The reference is an infinite line, not a fit.
    """
    if len(hull) < 2:
        return 0.0
    longest_squared = 0.0
    first = second = hull[0]
    for index, point in enumerate(hull[:-1]):
        offsets = hull[index + 1 :] - point
        squared = np.einsum("ij,ij->i", offsets, offsets)
        farthest = int(np.argmax(squared))
        if squared[farthest] > longest_squared:
            longest_squared = float(squared[farthest])
            first = point
            second = hull[index + 1 + farthest]
    if longest_squared <= 1e-18:
        return 0.0
    direction = (second - first) / np.sqrt(longest_squared)
    offsets = xy - first
    distances = np.abs(offsets[:, 0] * direction[1] - offsets[:, 1] * direction[0])
    return float(np.mean(distances))
