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
from repr_partitions_cluster.src.utils import *
from pathlib import Path
from h5py import File


def ward_random_clust_initialization(points_coords, num_clusters: int):
    clust_coords = np.copy(points_coords)
    curr_num_clusters = len(clust_coords)
    distance_matrix = np.zeros((curr_num_clusters, curr_num_clusters))
    num_pts_per_clust = np.ones((len(clust_coords)))
    for i in range(curr_num_clusters):
        for j in range(curr_num_clusters):
            if i < j:
                distance_matrix[i, j] = np.sum(
                    (clust_coords[i] - clust_coords[j])**2)
                distance_matrix[j, i] = distance_matrix[i, j]
    points_assign = np.arange(len(points_coords))
    while len(clust_coords) > num_clusters:
        # Pick two different random clusters
        clust_ids_perm = np.random.permutation(np.arange(len(distance_matrix)))
        merged_ids = np.sort(clust_ids_perm[:2])
        unchanged_ids = np.sort(clust_ids_perm[2:])
        center = (num_pts_per_clust[merged_ids[0]]*clust_coords[merged_ids[0]] + num_pts_per_clust[merged_ids[1]]
                  * clust_coords[merged_ids[1]])/(num_pts_per_clust[merged_ids[0]]+num_pts_per_clust[merged_ids[1]])

        for i in range(len(points_assign)):
            if points_assign[i] == merged_ids[1]:
                points_assign[i] = merged_ids[0]
            elif points_assign[i] > merged_ids[1]:
                points_assign[i] -= 1
            # else no modification

        # We put the new cluster at the place of the former first cluster merged
        new_clust_coords = np.zeros(
            (clust_coords.shape[0]-1, clust_coords.shape[1]))
        new_num_pts_per_clust = np.zeros(len(num_pts_per_clust))
        for i in range(len(clust_coords)-1):
            if i == merged_ids[0]:
                new_clust_coords[i] = center
                new_num_pts_per_clust[i] = num_pts_per_clust[merged_ids[0]
                                                             ] + num_pts_per_clust[merged_ids[1]]
            elif i >= merged_ids[1]:
                new_clust_coords[i] = clust_coords[i+1]
                new_num_pts_per_clust[i] = num_pts_per_clust[i+1]
            else:
                new_clust_coords[i] = clust_coords[i]
                new_num_pts_per_clust[i] = num_pts_per_clust[i]

        # First, update the distances related to the merged cluster (place at the position merged_ids[0])
        new_distance_matrix = np.copy(distance_matrix)
        i = merged_ids[0]
        for j in (range(len(clust_coords))):
            if i < j:
                new_distance_matrix[i, j] = np.sum(
                    (center - clust_coords[j])**2)
                new_distance_matrix[j, i] = new_distance_matrix[i, j]
        # Then delete useless rows and columns (previously associated with the cluster at position merged_ids[1] currently merged with merged_ids[0] at position merged_ids[0])
        new_distance_matrix = np.delete(
            new_distance_matrix, merged_ids[1], axis=0)
        new_distance_matrix = np.delete(
            new_distance_matrix, merged_ids[1], axis=1)

        # Update variables
        clust_coords = new_clust_coords
        num_pts_per_clust = new_num_pts_per_clust
        distance_matrix = new_distance_matrix
    return clust_coords, points_assign, num_pts_per_clust


def generate_random_points(num_points: int, num_coordinates: int = 2) -> np.ndarray:
    """Generates ndarray with dimensions [num_points, num_coordinates, pointcoord_or_centroidassociated]
    with random coordinates as defined in 
    P. Hansen et N. Mladenović, « First vs. best improvement: An empirical study », Discrete Applied Mathematics, vol. 154, nᵒ 5, p. 802‑817, avr. 2006, doi: 10.1016/j.dam.2005.05.020.


    # Inputs:
        num_points (int): number of points per cluster
        num_coordinates (int): number of coordinates per point

    # Outputs
        [num_points, num_coordinates] for points coordinates 
    """
    points_coords = np.random.uniform(
        low=0, high=100, size=(num_points, num_coordinates)
        )
    return points_coords


def random_clust_initialization(num_clusters: int, num_points: int, points_coords: Optional[np.ndarray] = None):
    points_assign = np.random.randint(
        0, num_clusters, size=(num_points))
    if points_coords is not None:
        missing_clusters = set(points_assign).difference(set(range(num_clusters)))
        if len(missing_clusters) > 0:
            print(f"Warning: {len(missing_clusters)} clusters are empty")
        clust_coords = np.zeros((num_clusters, points_coords.shape[1]))
        for i in range(num_clusters):
            points_ids = np.where(points_assign == i)
            clust_coords[i] = np.mean(points_coords[points_ids], axis=0)
        num_pts_per_clust = np.unique(points_assign, return_counts=True)
        return clust_coords, points_assign, num_pts_per_clust
    else:
        return None, points_assign, None


def generate_simple1():
    """Simple situation for tests only 
    Points placed as in ![](https://raw.githubusercontent.com/Rob174/MaitriseClustering/main/images/test_simple1.png?token=GHSAT0AAAAAABRFDVCKQZTGSTHRKJMCNYBGYQEB6SA)
    # Outputs
        [num_points, num_coordinates] for points coordinates 
        [num_points, ] with in each cell the centroid index associated with the point at the same position
    """
    points_coords = np.array([(0, 0.), (0, 1), (5, 1), (5, 0), (6, 0)])
    return points_coords


def random_centroids(points_coords, points_init_assign, num_clusters: int):
    """
    # Input
        [num_points, num_coordinates] for points coordinates 
        [num_points, ] with in each cell the centroid index associated with the point at the same position
    """
    pts_chosen = np.random.permutation(
        np.arange(len(points_init_assign)))[:num_clusters]
    clust_coords = points_coords[pts_chosen]
    return points_coords, points_init_assign, clust_coords


def centroids_from_points(points_coords: np.ndarray, points_init_assign: np.ndarray, num_clusters: int):
    centroids = np.unique(points_init_assign)
    Lcentroids = []
    for i in range(num_clusters):
        if i in centroids:
            Lcentroids.append(np.mean(points_coords[np.where(points_init_assign == i)],axis=0))
        else:
            Lcentroids.append(np.zeros((points_coords.shape[1],)))
    return np.stack(Lcentroids,axis=0)


def generate_init_solution(num_points: int, num_clusters: int, num_coordinates: int = 2):
    points_coords = generate_random_points(num_points, num_coordinates)
    clust_coords, points_assign, _ = random_clust_initialization(
        num_clusters,points_coords.shape[0],points_coords
    )
    return points_coords, points_assign, clust_coords

def balance_slicing_of_points_assign(points_assign_slicing: np.ndarray, num_clusters: int):
    np.random.seed(0)
    clusters_in = np.unique(points_assign_slicing)
    cluster_in_set = {c:i for i,c in enumerate(clusters_in)}
    # Reformat clusters between 0 and num_clusters-1
    for i in range(len(points_assign_slicing)):
        new_cluster = cluster_in_set[points_assign_slicing[i]]
        if new_cluster >= num_clusters:
            points_assign_slicing[i] = np.random.randint(0, num_clusters)
        else:
            points_assign_slicing[i] = new_cluster
    return points_assign_slicing
            
        
    