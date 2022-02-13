import numpy as np
from typing import *


def equals(x, y):
    return np.max(np.abs(x-y)) == 0


def l2_squared(x):
    if len(x.shape) == 1:
        return np.sum(x**2)
    elif len(x.shape) == 2:
        return np.sum(x**2, axis=1)
    else:
        raise Exception("Unsupported type of array")


def update_center(
    points_coords: np.ndarray, clust_coords: np.ndarray, num_assign_to_clust: np.ndarray,
    point_moving_id: int, from_clust_id: int, to_clust_id: int
):
    xi = points_coords[point_moving_id]
    nl = num_assign_to_clust[from_clust_id]
    nj = num_assign_to_clust[to_clust_id]
    xlcenter = clust_coords[from_clust_id]
    xjcenter = clust_coords[to_clust_id]
    if nl > 1:
        new_xlcenter = (nl*xlcenter-xi)/(nl-1)
    else:
        # If we take a point from a cluster with a single point we arbitrarly set the center to the origin
        new_xlcenter = np.zeros(xlcenter.shape)
    new_xjcenter = (nj*xjcenter+xi)/(nj+1)
    return new_xlcenter, new_xjcenter


def update_cost(
    initial_cost: float,
    points_coords: np.ndarray, clust_coords: np.ndarray, num_assign_to_clust: np.ndarray,
    point_moving_id: int, from_clust_id: int, to_clust_id: int,
    updated_center_src: np.ndarray, updated_center_dst: np.ndarray
) -> float:
    xi = points_coords[point_moving_id]
    nl = num_assign_to_clust[from_clust_id]
    nj = num_assign_to_clust[to_clust_id]
    center_src = clust_coords[from_clust_id]
    center_dst = clust_coords[to_clust_id]

    part1 = nj/(nj+1)*np.sum((center_dst-xi)**2)
    part2 = None
    new_cost = initial_cost+part1
    if nl > 1:
        # As defined in M. Telgarsky et A. Vattani, « Hartigan’s Method: k-means Clustering without Voronoi », in Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics, mars 2010, p. 820‑827.
        # considering that a cluster with one point and an empty cluster have both  a cost of 0
        part2 = nl/(nl-1)*np.sum((center_src-xi)**2)
        new_cost -= part2
    # assert new_cost >= 0, f"Cost must be positive, {new_cost}={initial_cost}+{part1}-{part2}\n Details:\n {xi=}\n {nl=}\n {nj=}\n {updated_center_src=}\n {updated_center_dst=}"
    return new_cost


def non_opt_update_cost(
    points_coords: np.ndarray, points_init_assign: np.ndarray, clust_init_coords: np.ndarray, num_assign_to_clust: np.ndarray,
    point_moving_id: int, from_clust_id: int, to_clust_id: int,
    updated_center_src: np.ndarray, updated_center_dst: np.ndarray
) -> float:
    return cost(
        points_coords, *move(
            np.copy(points_init_assign), np.copy(clust_init_coords),
            point_moving_id, from_clust_id, to_clust_id,
            updated_center_src, updated_center_dst
        )
    )


def cost(points_coords: np.ndarray, points_assign: np.ndarray, clust_coords: np.ndarray):
    """
    !! Version non optimized designed to be used only at the start of the hmean algorithm. Elsewhere, use impact_on_cost to adjust the cost
    Compute the cost of one clustering proposition defined by the following 3 parameters :
    # Inputs: 
        points_coords (ndarray): [num_points, num_coordinates] for points coordinates 
        points_assign (ndarray): [num_points, ] with in each cell the centroid index associated with the point at the same position
        clust_coords (ndarray): [num_centroids, num_coordinates] for centroids coordinates
    # Output:
        cost (float): cost of the clustering proposition
    """
    cost = 0
    for i_pt in range(len(points_coords)):
        cost += np.sum((points_coords[i_pt] -
                       clust_coords[points_assign[i_pt]])**2
                       )
    return cost


def move(points_assign: np.ndarray, clust_coords: np.ndarray,
         pt_to_move_id: int, from_clust_id: int, to_clust_id: int,
         new_orig_clust_coord: np.ndarray, new_dest_clust_coord: np.ndarray
         ) -> Tuple[np.ndarray, np.ndarray]:
    """Create the clustering chosen with parameters :
    # Inputs: 
        points_assign (ndarray): [num_points, ] with in each cell the centroid index associated with the point at the same position
        clust_coords (ndarray): [num_centroids, num_coordinates] for centroids coordinates
    # Output:
        Tuple with 2 ndarrays : points_assign and clust_coords updated
    """
    points_assign[pt_to_move_id] = to_clust_id
    clust_coords[from_clust_id] = new_orig_clust_coord
    clust_coords[to_clust_id] = new_dest_clust_coord
    return points_assign, clust_coords

# From https://stackoverflow.com/questions/33987060/python-context-manager-that-measures-time
from time import perf_counter
from contextlib import contextmanager

@contextmanager
def catchtime() -> float:
    start = perf_counter()
    yield lambda: perf_counter() - start