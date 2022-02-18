"""KMeans+ implementation: used to compute the initial centroids positions (given the points coordinates and the points' affectation)"""
from repr_partitions_cluster.src.utils import equals
from typing import *
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(".").resolve().joinpath("venv")))
print(str(Path(".").resolve().joinpath("venv")))


def kmean(
    points_coords: np.ndarray, points_assign: np.ndarray, clust_coords: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    # Inputs:
        points_coords (np.ndarray) : [num_points, num_coordinates]
        points_assign (np.ndarray) : [num_points, ] with in each cell the centroid index associated with the point at the same position
        clust_coords (np.ndarray) : [num_clusters, num_coordinates]
    # Output:
        (ndarray) [num_centroids, num_coordinates] for the centroids coordinates
    """

    def recompute_centroids(points_coords, points_assign, clust_coords):
        change = False
        for i_center in range(len(clust_coords)):
            points_ids = np.where(points_assign == i_center)[0]
            if len(points_ids) == 0:
                # If empty cluster, we put the centroid at the origin
                new_center = np.zeros(clust_coords.shape[1])
            elif len(points_ids) == 1:
                # If only one point in the cluster, we put the centroid at the point
                new_center = points_coords[points_ids]
            else:
                new_center = np.mean(points_coords[points_ids], axis=0)
                # assert not np.isnan(new_center).any(), "new_center is nan"
            clust_coords[i_center] = new_center
        return points_coords, points_assign, clust_coords

    change = True
    while change:
        change = False
        # Associate a centroid
        for i_pt in range(len(points_coords)):
            xi = points_coords[i_pt]
            best_dist, best_center = float("inf"), None
            for i_center in range(len(clust_coords)):
                dist = np.sum((xi - clust_coords[i_center]) ** 2)
                if dist < best_dist:
                    best_dist = dist
                    best_center = i_center
            if best_center != points_assign[i_pt]:
                change = True
            points_assign[i_pt] = best_center
        # Recompute centroids
        points_coords, points_assign, clust_coords = recompute_centroids(
            points_coords, points_assign, clust_coords
        )

    # KMeans+ : Check if empty clusters
    clusters_with_pts = np.unique(points_assign)
    empty_clusters = [i for i in range(len(clust_coords)) if i not in clusters_with_pts]
    if len(empty_clusters) > 0:
        # compute distance of each point to centroid
        # euclidian dist squared
        d_pts_clust = np.zeros((len(clust_coords), len(points_coords)))
        for i_pt in range(len(points_coords)):
            for i_clust in clusters_with_pts:
                d_pts_clust[i_clust, i_pt] = np.sum(
                    (points_coords[i_pt] - clust_coords[i_clust]) ** 2
                )
        # sort by largest distance to centroid : cf https://stackoverflow.com/questions/30577375/have-numpy-argsort-return-an-array-of-2d-indices
        sorted_coords = np.dstack(
            np.unravel_index(
                np.argsort(d_pts_clust.ravel()), (len(clust_coords), len(points_coords))
            )
        ).reshape(
            -1, 2
        )  # [num_pts, 2] in increasing order of dist
        # Take the farthest point to fill one of the empty clusters and repeat
        for i, empt_id in enumerate(empty_clusters):
            farthest_pt = sorted_coords[-1 - i, 1]
            points_assign[farthest_pt] = empt_id
        points_coords, points_assign, clust_coords = recompute_centroids(
            points_coords, points_assign, clust_coords
        )
    return points_assign, clust_coords
