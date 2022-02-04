"""Adaptation of the first vs best improvement approach. 
These algorithms are local search algorithms.
We will use the hmeans heuristic as describes in *[1] T. Pereira, D. Aloise, J. Brimberg, et N. Mladenovic, « Review of Basic Local Searches for Solving the Minimum Sum-of-Squares Clustering Problem », 2018, p. 249‑270.
* §2.2.2 H-Means Heuristic
"""


import numpy as np
from typing import *
from repr_partitions_cluster.src.hmeans.improvement_choice import AbstractCallbackImprovement, NoImprSolutionFound, move, cost
from repr_partitions_cluster.src.utils import *


def improvement_hmeans(points_coords: np.ndarray, points_init_assign: np.ndarray, clust_init_coords: np.ndarray, initial_cost: float,
                       callback_stop: AbstractCallbackImprovement) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Local search algorithm using HMeans method to get a neighborhood. This algorithm takes the best or first neighboor for the improvement step.
    # Inputs: 
        points_coords (ndarray): [num_points, num_coordinates] for points coordinates 
        points_init_assign (ndarray): [num_points, ] with in each cell the centroid index associated with the point at the same position
        clust_init_coords (ndarray): [num_centroids, num_coordinates] for centroids coordinates

    # Output: 
        points_final_assign (ndarray): [num_points, ] with in each cell the centroid index associated with the point at the same position
        clust_final_coords (ndarray): [num_centroids, num_coordinates] for centroids coordinates
        new_cost (float): cost of the new solution
    """
    clust_coords = clust_init_coords.copy()
    num_assign_to_clust = np.unique(points_init_assign, return_counts=True)[1]
    callback_stop.initialize(initial_cost)
    for point_moving_id in range(len(points_coords)):
        from_clust_id = points_init_assign[point_moving_id]
        for to_clust_id in (i for i in range(len(clust_init_coords)) if i != from_clust_id):
            new_xlcenter, new_xjcenter = update_center(
                points_coords, clust_coords, num_assign_to_clust,
                point_moving_id, from_clust_id, to_clust_id
            )
            new_data = move(
                np.copy(points_init_assign), np.copy(clust_init_coords),
                point_moving_id, from_clust_id, to_clust_id,
                new_xlcenter, new_xjcenter
            )
            new_cost = update_cost(
                points_coords, clust_coords, num_assign_to_clust,
                point_moving_id, from_clust_id, to_clust_id,
                new_xlcenter, new_xjcenter
            )
            assert new_cost >= 0, "Cost must be positive"
            if callback_stop.stop_loop(point_moving_id, from_clust_id, to_clust_id,
                                       new_xlcenter, new_xjcenter, new_cost):
                return callback_stop.get_new_clustering(points_init_assign, clust_init_coords)
    return callback_stop.get_new_clustering(points_init_assign, clust_init_coords)


def hmeans(points_coords: np.ndarray, points_init_assign: np.ndarray, clust_init_coords: np.ndarray, initial_cost: float,
           type_improvement: AbstractCallbackImprovement
           ):
    cost = initial_cost
    points_assign = points_init_assign
    clust_coords = clust_init_coords
    while True:
        try:
            points_assign, clust_coords, cost = improvement_hmeans(
                points_coords, points_assign, clust_coords, cost, type_improvement
            )
        except NoImprSolutionFound:
            break
    return points_assign, clust_coords, cost
