"""KMeans+ implementation: used to compute the initial centroids positions (given the points coordinates and the points' affectation)"""

import numpy as np
from typing import *

def kmean(points_coords: np.ndarray, points_assign: np.ndarray, clust_coords: np.ndarray) -> np.ndarray:
    """
    # Inputs:
        points_coords (np.ndarray) : [num_points, num_coordinates] 
        points_assign (np.ndarray) : [num_points, ] with in each cell the centroid index associated with the point at the same position
        clust_coords (np.ndarray) : [num_clusters, num_coordinates]
    # Output:
        (ndarray) [num_centroids, num_coordinates] for the centroids coordinates
    """
        
    change = True
    while change:
        change = False
        # Associate a centroid
        for i_pt in range(len(points_coords)):
            xi = points_coords[i_pt]
            best_dist, best_center = float('inf'), None
            for i_center in range(len(clust_coords)):
                dist = np.sum((xi-clust_coords[i_center])**2)
                if dist < best_dist:
                    best_dist = dist
                    best_center = i_center
            points_assign[i_pt] = best_center
        # Recompute centroids
        for i_center in range(len(clust_coords)):
            new_center = np.mean(points_coords[points_assign==i_center],axis=0)
            if new_center != clust_coords[i_center]:
                change = True
            clust_coords[i_center] = new_center
        
    # KMeans+ : Check if empty clusters
    empty_clusters = [i for i in range(len(clust_coords)) if i not in np.unique(points_assign)]
    if len(empty_clusters) > 0:
        # TODO sort by largest distance to centroid
        # TODO Take the farthest point to fill one of the empty clusters and repeat