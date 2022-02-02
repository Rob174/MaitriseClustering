"""
Module that contains functions that generates points associated with their optimal centroids for a provided number of centroids.
Result given as tuple of ndarrays with :
- [num_points, num_coordinates] for points coordinates 
- [num_points, ] with in each cell the centroid index associated with the point at the same position

The last par the centroids coordinates will be calculated as decided by the user and will output the result in a ndarray with
- [num_centroids, num_coordinates] for centroids coordinates
"""
import numpy as np
from typing import *

def generate_random(num_clusters: int,num_points: int, num_coordinates: int = 2) -> Tuple[np.ndarray,np.ndarray]:
    """Generates ndarray with dimensions [num_points, num_coordinates, pointcoord_or_centroidassociated]
    with random coordinates.

    # Inputs:
        num_clusters (int): number of clusters
        num_points (int): number of points per cluster
        num_coordinates (int): number of coordinates per point

    # Outputs
        [num_points, num_coordinates] for points coordinates 
        [num_points, ] with in each cell the centroid index associated with the point at the same position
    """
    points_coords = np.random.normal(loc=0,scale=1,size=(num_points, num_coordinates))
    points_init_assign = np.random.randint(0,num_clusters,size=num_points)
    return points_coords,points_init_assign

def generate_simple1():
    """Simple situation for tests only 
    Points placed as in 
    """
    points_coords = np.random.normal(loc=0,scale=1,size=(num_points, num_coordinates))
    points_init_assign = np.random.randint(0,num_clusters,size=num_points)