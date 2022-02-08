from typing import *
import numpy as np
import unittest
import inspect
from repr_partitions_cluster.src.kmean.kmean import kmean
from repr_partitions_cluster.src.generators import generate_simple1, random_clust_initialization, generate_random_points
from repr_partitions_cluster.src.hmeans.hmeans import hmeans
from repr_partitions_cluster.src.hmeans.improvement_choice import CallbackBestImprovement, CallbackFirstImprovement
from repr_partitions_cluster.src.utils import *
from repr_partitions_cluster.src.visualize import VisualizeClusterList, VisualizationCallback


class HMeanTest(unittest.TestCase):
    def test_hmeans_simple1(self):
        print(inspect.stack()[0][0].f_code.co_name)
        np.random.seed(0)
        points_coords = generate_simple1()
        clust_init_coords, points_init_assign, num_pts_per_clust = random_clust_initialization(
            2, points_coords.shape[0], points_coords
        )
        points_assign_impr, clust_coords_impr = kmean(
            points_coords, np.copy(points_init_assign), np.copy(clust_init_coords))

        for i, ImprClass in enumerate([CallbackBestImprovement, CallbackFirstImprovement]):
            for params, init_type in list(zip([(points_coords, points_init_assign, clust_init_coords), (points_coords, points_assign_impr, clust_coords_impr)], ["random", "kmean+"]))[:i+1]:
                points_assign, clust_coords, fin_cost = hmeans(
                    *params, type_improvement=ImprClass(), initial_cost=cost(*params))
                print(f"{init_type=} {ImprClass.__name__=} {fin_cost=}")

    def test_hmeans_random(self):
        print(inspect.stack()[0][0].f_code.co_name)
        NUM_CLUSTERS = 25
        NUM_POINTS = 100
        np.random.seed(0)
        points_coords = generate_random_points(NUM_POINTS, NUM_CLUSTERS)
        clust_coords, points_assign, num_pts_per_clust = random_clust_initialization(
            NUM_CLUSTERS, points_coords.shape[0], points_coords
        )
        points_assign_impr, clust_coords_impr = kmean(
            points_coords, np.copy(points_assign), np.copy(clust_coords)
        )
        for params, init_type in list(zip([(points_coords, points_assign, clust_coords), (points_coords, points_assign_impr, clust_coords_impr)], ["random", "kmean+"])):
            for ImprClass in [CallbackBestImprovement, CallbackFirstImprovement]:
                with catchtime() as t:
                    points_assign, clust_coords, fin_cost = hmeans(
                        *params, type_improvement=ImprClass(), initial_cost=cost(*params), callback_visu=VisualizationCallback(VisualizeClusterList(x0=0, x1=1))
                    )
                print(
                    f"{init_type=} {ImprClass.__name__=} {fin_cost=}, computed in {t():.4f} secs")


if __name__ == "__main__":
    unittest.main()
