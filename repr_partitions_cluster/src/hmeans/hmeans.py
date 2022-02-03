"""Adaptation of the first vs best improvement approach. 
These algorithms are local search algorithms.
We will use the hmeans heuristic as describes in *[1] T. Pereira, D. Aloise, J. Brimberg, et N. Mladenovic, « Review of Basic Local Searches for Solving the Minimum Sum-of-Squares Clustering Problem », 2018, p. 249‑270.
* §2.2.2 H-Means Heuristic
"""


import numpy as np
from typing import *
from repr_partitions_cluster.src.hmeans.improvement_choice import AbstractCallbackImprovement, NoImprSolutionFound, move, cost

def update_center(
        points_coords: np.ndarray,clust_coords:np.ndarray,num_assign_to_clust:np.ndarray,
        point_moving_id:int,from_clust_id:int,to_clust_id:int
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
    return new_xlcenter,new_xjcenter
def update_cost(
        initial_cost:float,
        points_coords: np.ndarray,clust_coords:np.ndarray,num_assign_to_clust:np.ndarray,
        point_moving_id:int,from_clust_id:int,to_clust_id:int,
        updated_center_src: np.ndarray,updated_center_dst: np.ndarray
    ) -> float:
    xi = points_coords[point_moving_id]
    nl = num_assign_to_clust[from_clust_id]
    nj = num_assign_to_clust[to_clust_id]
    
    improv_of_cost = np.sum(nj/(nj+1)*(updated_center_dst-xi)**2) 
    if nl > 1:
        # As defined in M. Telgarsky et A. Vattani, « Hartigan’s Method: k-means Clustering without Voronoi », in Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics, mars 2010, p. 820‑827.
        # considering that a cluster with one point and an empty cluster have both  a cost of 0
        improv_of_cost -= np.sum(nl/(nl-1)*(updated_center_src-xi)**2)
    new_cost = initial_cost-improv_of_cost
    return new_cost

def non_opt_update_cost(
        points_coords: np.ndarray,points_init_assign:np.ndarray,clust_init_coords:np.ndarray,num_assign_to_clust:np.ndarray,
        point_moving_id:int,from_clust_id:int,to_clust_id:int,
        updated_center_src: np.ndarray,updated_center_dst: np.ndarray
    ) -> float:
    return cost(
        points_coords,*move(
            np.copy(points_init_assign), np.copy(clust_init_coords),
            point_moving_id,from_clust_id,to_clust_id,
            updated_center_src,updated_center_dst
            )
        )
    
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
            new_xlcenter,new_xjcenter = update_center(
                points_coords,clust_coords,num_assign_to_clust,
                point_moving_id,from_clust_id,to_clust_id
            )
            new_cost = update_cost(
                initial_cost,points_coords,clust_coords,num_assign_to_clust,
                point_moving_id,from_clust_id,to_clust_id,
                new_xlcenter,new_xjcenter
            )
            if callback_stop.stop_loop(point_moving_id,from_clust_id,to_clust_id,
                                       new_xlcenter,new_xjcenter,new_cost):
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
            points_assign,clust_coords,cost = improvement_hmeans(
                points_coords, points_assign, clust_coords, cost, type_improvement
            )
        except NoImprSolutionFound:
            break
    return points_assign,clust_coords,cost

