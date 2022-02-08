from repr_partitions_cluster.src.generators import *
from repr_partitions_cluster.src.utils import *
from repr_partitions_cluster.src.visualize import *
import unittest


class TestUtils(unittest.TestCase):
    def test_update_cost(self):
        # callback_visu = VisualizationCallback(
        #             VisualizeClusterList(x0=0, x1=1)
        #         )
        for _ in range(10):
            points_coords = generate_random_points(10, 2)
            # callback_visu.register_points(points_coords)

            np.random.seed(0)
            clust_coords, points_assign, num_pts_per_clust = random_cluster_initialization(
                points_coords, 2)
            # callback_visu.register_cluster(
            #     f"Initial clustering (chosen)",
            #     points_assign, clust_coords
            # )
            initial_cost = cost(points_coords, points_assign, clust_coords)
            num_assign_to_clust = np.unique(
                points_assign, return_counts=True)[1]
            point_moving_id = np.random.permutation(len(points_coords))[0]
            from_clust_id = points_assign[point_moving_id]
            to_clust_id = [i for i in np.random.permutation(
                len(clust_coords)) if i != from_clust_id][0]
            new_xlcenter, new_xjcenter = update_center(
                points_coords, clust_coords, num_assign_to_clust,
                point_moving_id, from_clust_id, to_clust_id
            )
            new_cost = update_cost(
                initial_cost,
                points_coords, clust_coords, num_assign_to_clust,
                point_moving_id, from_clust_id, to_clust_id,
                new_xlcenter, new_xjcenter
            )
            tested_points_assign, tested_clust_coords = move(
                np.copy(points_assign), np.copy(clust_coords),
                point_moving_id, from_clust_id, to_clust_id,
                new_xlcenter, new_xjcenter
            )
            # callback_visu.register_cluster(
            #     f"New clustering (chosen) point {points_coords[point_moving_id]} from {from_clust_id} to {to_clust_id}",
            #     tested_points_assign, tested_clust_coords
            # )
            real_cost = cost(points_coords, tested_points_assign,
                             tested_clust_coords)
            # callback_visu.show()
            self.assertTrue(np.max(real_cost-new_cost) <
                            1e-5, "Cost must be equal")


if __name__ == "__main__":
    unittest.main()
