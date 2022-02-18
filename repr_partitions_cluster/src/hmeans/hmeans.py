"""Adaptation of the first vs best improvement approach. 
These algorithms are local search algorithms.
We will use the hmeans heuristic as describes in *[1] T. Pereira, D. Aloise, J. Brimberg, et N. Mladenovic, « Review of Basic Local Searches for Solving the Minimum Sum-of-Squares Clustering Problem », 2018, p. 249‑270.
* §2.2.2 H-Means Heuristic
"""


import numpy as np
from typing import *
from repr_partitions_cluster.src.hmeans.improvement_choice import (
    AbstractCallbackImprovement,
    NoImprSolutionFound,
    move,
    cost,
)
from repr_partitions_cluster.src.utils import *
from repr_partitions_cluster.src.visualize import (
    VisualizationCallback,
    VisualizeClusterList,
)
from repr_partitions_cluster.src.hmeans.iteration_order import (
    AbstractIterationOrder,
    BACK,
)
from numba import jit


def improvement_hmeans(
    points_coords: np.ndarray,
    points_init_assign: np.ndarray,
    clust_init_coords: np.ndarray,
    initial_cost: float,
    iteration_order: AbstractIterationOrder,
    callback_stop: AbstractCallbackImprovement,
    callback_visu: VisualizationCallback = None,
) -> Tuple[Tuple[np.ndarray, np.ndarray, float], int]:
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
    assignement = {
        index: num
        for index, num in zip(*np.unique(points_init_assign, return_counts=True))
    }
    num_assign_to_clust = np.array(
        [assignement[i] if i in assignement else 0 for i in range(len(clust_coords))]
    )
    callback_stop.initialize(initial_cost)
    if callback_visu is not None:
        callback_visu.register_points(points_coords)
        callback_visu.register_cluster(
            f"Initial clustering", points_init_assign, clust_init_coords
        )
    num_iter = 0
    for point_moving_id in range(len(points_coords)):
        from_clust_id = points_init_assign[point_moving_id]
        for to_clust_id in iteration_order.get_points_order(
            src_clust_id=from_clust_id, num_clusters=len(clust_coords)
        ):
            new_xlcenter, new_xjcenter = update_center(
                points_coords,
                clust_coords,
                num_assign_to_clust,
                point_moving_id,
                from_clust_id,
                to_clust_id,
            )
            if callback_visu is not None:
                tested_points_assign, tested_clust_coords = move(
                    np.copy(points_init_assign),
                    np.copy(clust_init_coords),
                    point_moving_id,
                    from_clust_id,
                    to_clust_id,
                    new_xlcenter,
                    new_xjcenter,
                )
                callback_visu.register_cluster(
                    f"Move point {points_coords[point_moving_id]} from {from_clust_id} to {to_clust_id}",
                    tested_points_assign,
                    tested_clust_coords,
                )
            new_cost = update_cost(
                initial_cost,
                points_coords,
                clust_coords,
                num_assign_to_clust,
                point_moving_id,
                from_clust_id,
                to_clust_id,
                new_xlcenter,
                new_xjcenter,
            )
            num_iter += 1
            if callback_stop.stop_loop(
                point_moving_id,
                from_clust_id,
                to_clust_id,
                new_xlcenter,
                new_xjcenter,
                new_cost,
            ):
                return (
                    callback_stop.get_new_clustering(
                        points_init_assign, clust_init_coords
                    ),
                    num_iter,
                )
    return (
        callback_stop.get_new_clustering(points_init_assign, clust_init_coords),
        num_iter,
    )


def hmeans(
    points_coords: np.ndarray,
    points_init_assign: np.ndarray,
    clust_init_coords: np.ndarray,
    initial_cost: float,
    type_improvement: AbstractCallbackImprovement,
    iteration_order: AbstractIterationOrder = None,
    callback_visu: VisualizationCallback = None,
    *args,
):
    if iteration_order is None:
        iteration_order = BACK()
    cost = initial_cost
    points_assign = points_init_assign
    clust_coords = clust_init_coords
    if callback_visu is not None:
        callback_visu.register_points(points_coords)
        callback_visu.register_cluster(
            f"Initial clustering (chosen)", points_init_assign, clust_init_coords
        )
    counter = 0
    counter_tot = 0
    while True:
        clustering, num_iter = improvement_hmeans(
            points_coords,
            points_assign,
            clust_coords,
            cost,
            iteration_order=iteration_order,
            callback_visu=None
            if callback_visu is None
            else VisualizationCallback(VisualizeClusterList(x0=0, x1=1)),
            callback_stop=type_improvement,
        )
        counter += 1
        counter_tot += num_iter
        if clustering is None:
            break
        else:
            points_assign, clust_coords, cost = clustering
            if callback_visu is not None:
                callback_visu.register_cluster(
                    f"Intermediate choice", points_assign, clust_coords
                )
    if counter == 0 and callback_visu is not None:
        callback_visu.show()
        i = input("Press enter to continue")
    return points_assign, clust_coords, cost, counter, counter_tot
