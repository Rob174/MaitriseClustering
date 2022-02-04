from typing import *
import numpy as np
from repr_partitions_cluster.src.kmean.kmean import kmean
from repr_partitions_cluster.src.generators import generate_simple1, random_centroids, random_cluster_initialization, generate_random_points
from repr_partitions_cluster.src.hmeans.hmeans import hmeans
from repr_partitions_cluster.src.hmeans.improvement_choice import CallbackBestImprovement, CallbackFirstImprovement
from repr_partitions_cluster.src.utils import *


class HMeanTest:
    def test_hmeans_simple1(self):
        points_coords, points_init_assign, clust_init_coords = random_centroids(
            *generate_simple1(), num_clusters=2)
        points_assign_impr, clust_coords_impr = kmean(
            points_coords, np.copy(points_init_assign), np.copy(clust_init_coords))

        for i, ImprClass in enumerate([CallbackBestImprovement, CallbackFirstImprovement]):
            for params, init_type in list(zip([(points_coords, points_init_assign, clust_init_coords), (points_coords, points_assign_impr, clust_coords_impr)], ["random", "kmean+"]))[:i+1]:
                points_assign, clust_coords, fin_cost = hmeans(
                    *params, type_improvement=ImprClass(), initial_cost=cost(*params))
                print(f"{init_type=} {ImprClass.__name__=} {fin_cost=}")

    def test_hmeans_random(self):
        points_coords = generate_random_points(10, 2)
        clust_coords, points_assign, num_pts_per_clust = random_cluster_initialization(
            points_coords, 2)
        points_assign_impr, clust_coords_impr = kmean(
            points_coords, np.copy(points_assign), np.copy(clust_coords)
        )
        for i, ImprClass in enumerate([CallbackBestImprovement, CallbackFirstImprovement]):
            for params, init_type in list(zip([(points_coords, points_assign, clust_coords), (points_coords, points_assign_impr, clust_coords_impr)], ["random", "kmean+"]))[:i+1]:
                points_assign, clust_coords, fin_cost = hmeans(
                    *params, type_improvement=ImprClass(), initial_cost=cost(*params))
                print(f"{init_type=} {ImprClass.__name__=} {fin_cost=}")


if __name__ == "__main__":
    HMeanTest().test_hmeans_random()
