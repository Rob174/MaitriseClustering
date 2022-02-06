from repr_partitions_cluster.src.generators import *
from repr_partitions_cluster.src.utils import *


class TestUtils:
    def test_update_cost(self):
        points_coords = generate_random_points(10, 2)
        np.random.seed(0)
        clust_coords, points_assign, num_pts_per_clust = random_cluster_initialization(
            points_coords, 2)
        initial_cost = cost(points_coords, points_assign, clust_coords)
        num_assign_to_clust = np.unique(points_assign, return_counts=True)[1]
        point_moving_id = np.random.permutation(len(points_coords))[0]
        from_clust_id = points_assign[point_moving_id]
        to_clust_id = np.random.permutation(len(clust_coords))[0]
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
        real_cost = cost(points_coords, tested_points_assign,
                         tested_clust_coords)
        assert real_cost == new_cost, "Cost must be equal"


if __name__ == "__main__":
    TestUtils().test_update_cost()
