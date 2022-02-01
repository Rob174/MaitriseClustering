"""Adaptation of the first vs best improvement approach. 
These algorithms are local search algorithms.
We will use the hmeans heuristic as describes in *[1] T. Pereira, D. Aloise, J. Brimberg, et N. Mladenovic, « Review of Basic Local Searches for Solving the Minimum Sum-of-Squares Clustering Problem », 2018, p. 249‑270.
* §2.2.2 H-Means Heuristic
"""


import numpy as np
from typing import *
class AbstractCallbackImprovement:
    def __init__(self) -> None:
        self.best_id_pt_to_move: Optional[int] = None
        self.best_center_orig: Optional[int] = None
        self.best_center_dest: Optional[int] = None
        self.best_new_orig_clust_coord: Optional[np.ndarray] = None
        self.best_new_dest_clust_coord: Optional[np.ndarray] = None
        self.best_cost: Optional[float] = None
    def initialize(self):
        raise NotImplemented # Parent class to compute initial cost
    def stop_loop(self,point_moving_id,from_clust_id,to_clust_id,new_orig_clust_coord,new_dest_clust_coord,cost_improvement):
        raise NotImplemented
    def get_new_clustering(self):
        raise NotImplemented
    
class CallbackBestImprovement:
    def stop_loop(self,point_moving_id,from_clust_id,to_clust_id,new_orig_clust_coord,new_dest_clust_coord,cost_improvement):
        """Indicates if stop loop and save data if improvement is found."""
        # TODO compute cost
        # cost = 
        if cost < self.best_cost:
            self.best_id_pt_to_move = point_moving_id
            self.best_center_orig = from_clust_id
            self.best_center_dest = to_clust_id
            self.best_new_orig_clust_coord = new_orig_clust_coord
            self.best_new_dest_clust_coord = new_dest_clust_coord
            self.best_cost = cost
        return False

class CallbackFirstImprovement:
    def stop_loop(self,point_moving_id,from_clust_id,to_clust_id,new_orig_clust_coord,new_dest_clust_coord,cost_improvement):
        """Indicates if stop loop and save data if improvement is found. Stops as soon as an improvement is detected"""
        # TODO compute cost
        # cost = 
        if cost < self.best_cost:
            self.best_id_pt_to_move = point_moving_id
            self.best_center_orig = from_clust_id
            self.best_center_dest = to_clust_id
            self.best_new_orig_clust_coord = new_orig_clust_coord
            self.best_new_dest_clust_coord = new_dest_clust_coord
            self.best_cost = cost
            return True
        return False
            
        
def improvement_hmeans(points_coords: np.ndarray, points_init_assign: np.ndarray, clust_init_coords: np.ndarray, callback_stop: AbstractCallbackImprovement):
    """
    Local search algorithm using HMeans method to get a neighborhood. This algorithm takes the best or first neighboor for the improvement step.
    # Inputs: 
        points_coords (ndarray): [num_points, num_coordinates] for points coordinates 
        points_init_assign (ndarray): [num_points, ] with in each cell the centroid index associated with the point at the same position
        clust_init_coords (ndarray): [num_centroids, num_coordinates] for centroids coordinates
    
    # Output: 
        points_final_assign (ndarray): [num_points, ] with in each cell the centroid index associated with the point at the same position
        clust_final_coords (ndarray): [num_centroids, num_coordinates] for centroids coordinates
    """
    clust_coords = clust_init_coords.copy()
    num_assign_to_clust = np.unique(points_init_assign, return_counts=True)[1]
    callback_stop.initialize() # TODO compute initial cost and save
    for point_moving_id in range(len(points_coords)):
        from_clust_id = points_init_assign[point_moving_id]
        for to_clust_id in (i for i in range(len(clust_init_coords)) if i != from_clust_id):
            xi = points_coords[point_moving_id]
            nl = num_assign_to_clust[from_clust_id]
            nj = num_assign_to_clust[to_clust_id]
            xlcenter = clust_coords[from_clust_id]
            xjcenter = clust_coords[to_clust_id]
            new_xlcenter = (nl*xlcenter-xi)/(nl-1)
            new_xjcenter = (nj*xjcenter+xi)/(nj+1)
            #TODO To be checked  if new or not new
            impact_on_cost = nj/(nj+1)*(new_xjcenter-xi)**2 + nl/(nl-1)*(new_xlcenter-xi)**2
            if callback_stop.stop_loop(point_moving_id,from_clust_id,to_clust_id,new_xlcenter,new_xjcenter,impact_on_cost):
                return callback_stop.get_new_clustering()
    return callback_stop.get_new_clustering()
