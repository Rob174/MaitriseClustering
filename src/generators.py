"""
Module that contains functions that generates points associated with their optimal centroids for a provided number of centroids.
Result given as tuple of ndarrays with :
- [num_points, num_coordinates] for points coordinates 
- [num_points, ] with in each cell the centroid index associated with the point at the same position
- [num_centroids, num_coordinates] for centroids coordinates
"""
import numpy as np

def generate_random(num_clusters: int,num_points: int, num_coordinates: int = 2) -> np.ndarray:
    """Generates ndarray with dimensions [num_points, num_coordinates, pointcoord_or_centroidassociated]
    with random coordinates.

    Args:
        num_clusters (int): number of clusters
        num_points (int): number of points per cluster
        num_coordinates (int): number of coordinates per point

    Returns:
        np.ndarray: ndarray with dimensions [num_points, num_coordinates, pointcoord_or_centroidassociated]
    """
    return np.random.rand(num_clusters, num_points, num_coordinates)