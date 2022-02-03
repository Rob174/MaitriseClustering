
import numpy as np
from typing import *

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
        cost += np.sum((points_coords[i_pt] - clust_coords[points_assign[i_pt]])**2)
    return cost

def move(points_assign: np.ndarray, clust_coords: np.ndarray,
         pt_to_move_id: int,from_clust_id: int,to_clust_id: int,
         new_orig_clust_coord: np.ndarray,new_dest_clust_coord: np.ndarray
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
    return points_assign,clust_coords
    
class NoImprSolutionFound(Exception):
    """Exception raised when no improving solution is found"""
    pass

class AbstractCallbackImprovement:
    def __init__(self) -> None:
        self.best_pt_to_move_id: Optional[int] = None
        self.best_center_orig: Optional[int] = None
        self.best_center_dest: Optional[int] = None
        self.best_new_orig_clust_coord: Optional[np.ndarray] = None
        self.best_new_dest_clust_coord: Optional[np.ndarray] = None
        self.best_cost: Optional[float] = None
    def initialize(self,initial_cost: float):
        """Compute and register initial cost
        # Inputs: 
            points_coords (ndarray): [num_points, num_coordinates] for points coordinates 
            points_init_assign (ndarray): [num_points, ] with in each cell the centroid index associated with the point at the same position
            clust_init_coords (ndarray): [num_centroids, num_coordinates] for centroids coordinates
        """
        self.best_cost = initial_cost
    def stop_loop(self,point_moving_id,from_clust_id,to_clust_id,new_orig_clust_coord,new_dest_clust_coord,cost_improvement):
        raise NotImplemented
    def get_new_clustering(self,points_init_assign: np.ndarray, clust_init_coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Get the improving solution if one has been found else raise Exception
        # Input """
        # If one improving solution has been found return it
        if self.best_center_orig is not None:
            return (*move(
                        points_init_assign, clust_init_coords,
                        self.best_pt_to_move_id,self.best_center_orig,self.best_center_dest, # type: ignore
                        self.best_new_orig_clust_coord,self.best_new_dest_clust_coord # type: ignore
                    ), self.best_cost)
        raise NoImprSolutionFound
    
class CallbackBestImprovement(AbstractCallbackImprovement):
    def stop_loop(self,point_moving_id,from_clust_id,to_clust_id,new_orig_clust_coord,new_dest_clust_coord,cost):
        """Indicates if stop loop and save data if improvement is found."""
        if cost < self.best_cost:
            self.best_pt_to_move_id = point_moving_id
            self.best_center_orig = from_clust_id
            self.best_center_dest = to_clust_id
            self.best_new_orig_clust_coord = new_orig_clust_coord
            self.best_new_dest_clust_coord = new_dest_clust_coord
            self.best_cost = cost
        return False

class CallbackFirstImprovement(AbstractCallbackImprovement):
    def stop_loop(self,point_moving_id,from_clust_id,to_clust_id,new_orig_clust_coord,new_dest_clust_coord,cost):
        """Indicates if stop loop and save data if improvement is found. Stops as soon as an improvement is detected"""
        if cost < self.best_cost:
            self.best_pt_to_move_id = point_moving_id
            self.best_center_orig = from_clust_id
            self.best_center_dest = to_clust_id
            self.best_new_orig_clust_coord = new_orig_clust_coord
            self.best_new_dest_clust_coord = new_dest_clust_coord
            self.best_cost = cost
            return True # Stop as soon as an improvement is detected
        return False