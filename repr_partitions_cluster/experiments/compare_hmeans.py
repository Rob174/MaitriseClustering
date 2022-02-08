import pandas as pd
from repr_partitions_cluster.src.kmean.kmean import kmean
from repr_partitions_cluster.src.generators import generate_simple1, random_clust_initialization, generate_random_points
from repr_partitions_cluster.src.hmeans.hmeans import hmeans
from repr_partitions_cluster.src.hmeans.improvement_choice import CallbackBestImprovement, CallbackFirstImprovement
from repr_partitions_cluster.src.utils import *
from repr_partitions_cluster.src.visualize import VisualizeClusterList, VisualizationCallback

if __name__ == "__main__":
    L_NUM_CLUSTERS = [2**i for i in range(1, 8)]
    L_NUM_POINTS = np.concatenate(
        (
            np.arange(20, 91, 10),
            np.arange(100, 1001, 100)
        ), axis=0)
    df = pd.DataFrame(columns=["num_clusters", "num_points",
                      "init_type", "ImprClass", "fin_cost", "time"])
    for num_points in L_NUM_POINTS:
        for num_clusters in L_NUM_CLUSTERS:
            points_coords = generate_random_points(num_points, num_clusters)
            clust_coords, points_assign, num_pts_per_clust = random_clust_initialization(
                num_clusters, len(points_coords), points_coords
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
