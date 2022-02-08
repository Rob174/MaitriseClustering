
import numpy as np
from typing import *
from repr_partitions_cluster.src.utils import *


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
        self.initial_cost: Optional[float] = None

    def initialize(self, initial_cost: float):
        """Compute and register initial cost
        # Inputs: 
            points_coords (ndarray): [num_points, num_coordinates] for points coordinates 
            points_init_assign (ndarray): [num_points, ] with in each cell the centroid index associated with the point at the same position
            clust_init_coords (ndarray): [num_centroids, num_coordinates] for centroids coordinates
        """
        self.best_cost = initial_cost
        self.initial_cost = initial_cost

    def stop_loop(self, point_moving_id, from_clust_id, to_clust_id, new_orig_clust_coord, new_dest_clust_coord, cost_improvement):
        raise NotImplemented

    def get_new_clustering(self, points_init_assign: np.ndarray, clust_init_coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Get the improving solution if one has been found else raise Exception
        # Input """
        # If one improving solution has been found return it
        if self.best_center_orig is not None and self.best_cost < self.initial_cost:
            return (*move(
                points_init_assign, clust_init_coords,
                self.best_pt_to_move_id, self.best_center_orig, self.best_center_dest,  # type: ignore
                self.best_new_orig_clust_coord, self.best_new_dest_clust_coord  # type: ignore
            ), self.best_cost)
        raise NoImprSolutionFound


class CallbackBestImprovement(AbstractCallbackImprovement):
    def stop_loop(self, point_moving_id, from_clust_id, to_clust_id, new_orig_clust_coord, new_dest_clust_coord, cost):
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
    def stop_loop(self, point_moving_id, from_clust_id, to_clust_id, new_orig_clust_coord, new_dest_clust_coord, cost):
        """Indicates if stop loop and save data if improvement is found. Stops as soon as an improvement is detected"""
        if cost < self.best_cost:
            self.best_pt_to_move_id = point_moving_id
            self.best_center_orig = from_clust_id
            self.best_center_dest = to_clust_id
            self.best_new_orig_clust_coord = new_orig_clust_coord
            self.best_new_dest_clust_coord = new_dest_clust_coord
            self.best_cost = cost
            return True  # Stop as soon as an improvement is detected
        return False
