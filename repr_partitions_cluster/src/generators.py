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
    with random coordinates as defined in 
    P. Hansen et N. Mladenović, « First vs. best improvement: An empirical study », Discrete Applied Mathematics, vol. 154, nᵒ 5, p. 802‑817, avr. 2006, doi: 10.1016/j.dam.2005.05.020.


    # Inputs:
        num_clusters (int): number of clusters
        num_points (int): number of points per cluster
        num_coordinates (int): number of coordinates per point

    # Outputs
        [num_points, num_coordinates] for points coordinates 
        [num_points, ] with in each cell the centroid index associated with the point at the same position
    """
    points_coords = np.random.uniform(low=0,high=100,size=(num_points, num_coordinates))
    points_init_assign = np.random.randint(0,num_clusters,size=num_points)
    get_empty_clusters = lambda :np.array([i for i in range(num_clusters) if i not in np.unique(points_init_assign)])
    empty_clusters = get_empty_clusters()
    while len(empty_clusters) > 0:
        ids_pts_changed = np.random.permutation(np.arange(num_points))[:len(empty_clusters)]
        points_init_assign[ids_pts_changed] = empty_clusters
        empty_clusters = get_empty_clusters()
        
    return points_coords,points_init_assign

def generate_simple1():
    """Simple situation for tests only 
    Points placed as in ![](https://raw.githubusercontent.com/Rob174/MaitriseClustering/main/images/test_simple1.png?token=GHSAT0AAAAAABRFDVCKQZTGSTHRKJMCNYBGYQEB6SA)
    # Outputs
        [num_points, num_coordinates] for points coordinates 
        [num_points, ] with in each cell the centroid index associated with the point at the same position
    """
    points_coords = np.array([(0,0.),(0,1),(5,1),(5,0),(6,0)])
    points_init_assign = np.random.randint(0,2,size=points_coords.shape[0]).astype(int)
    return points_coords,points_init_assign


def random_centroids(points_coords,points_init_assign, num_clusters: int):
    """
    # Input
        [num_points, num_coordinates] for points coordinates 
        [num_points, ] with in each cell the centroid index associated with the point at the same position
    """
    pts_chosen = np.random.permutation(np.arange(len(points_init_assign)))[:num_clusters]
    clust_coords = points_coords[pts_chosen]
    return points_coords,points_init_assign,clust_coords