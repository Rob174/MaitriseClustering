from repr_partitions_cluster.src.kmean.kmean import kmean
from repr_partitions_cluster.src.generators import generate_simple1, random_centroids
from repr_partitions_cluster.src.utils import equals
from repr_partitions_cluster.src.visualize import show_clustering
import numpy as np
from typing import *
from repr_partitions_cluster.src.kmean.kmean import kmean
import unittest


class KMeanTest(unittest.TestCase):

    def test_simple1(self):
        for _ in range(10):
            np.random.seed(0)
            points_coords = generate_simple1()
            clust_init_coords, points_init_assign, num_pts_per_clust = random_clust_initialization(
                2,points_coords.shape[0],points_coords
            )
            points_assign_impr, clust_coords_impr = kmean(
                points_coords, np.copy(
                    points_init_assign), np.copy(clust_init_coords)
            )


if __name__ == "__main__":
    unittest.main()
