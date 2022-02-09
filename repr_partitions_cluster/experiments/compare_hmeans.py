import pandas as pd
from h5py import File
from pathlib import Path
from repr_partitions_cluster.src.kmean.kmean import kmean
from repr_partitions_cluster.src.generators import random_clust_initialization, balance_slicing_of_points_assign, centroids_from_points
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
    PATH = Path("./data/inputs_algorithms.hdf5")
    df = pd.DataFrame(columns=["num_clusters", "num_points",
                      "init_type", "ImprClass", "fin_cost", "time", "uuid_points_coords", "uuid_clust_coords"])
    with File(PATH, "r") as f:
        for uuid_points_coords, points_coords in f["points_coords"].items():
            points_coords = np.array(points_coords, dtype=np.float32)
            for num_points in L_NUM_POINTS:
                subset_points_coords = points_coords[:num_points]
                for num_clust, datasets_points_assign in f["points_assign"].items():
                    num_clusters = int(num_clust)
                    for uuid_points_assign, points_assign in datasets_points_assign.items():
                        points_assign = np.array(points_assign, dtype=np.int32)
                        subset_points_assign = points_assign[:num_points]
                        subset_points_assign = balance_slicing_of_points_assign(
                            subset_points_assign, num_clusters)
                        subset_clust_coords = np.stack(
                            centroids_from_points(
                                subset_points_coords, subset_points_assign
                            ), axis=0
                        )
                        points_assign_impr, clust_coords_impr = kmean(
                            subset_points_coords, np.copy(
                                subset_points_assign), np.copy(subset_clust_coords)
                        )
                        for params, init_type in list(zip([(subset_points_coords, subset_points_assign, subset_clust_coords), (subset_points_coords, points_assign_impr, clust_coords_impr)], ["random", "kmean+"])):
                            for ImprClass in [CallbackBestImprovement, CallbackFirstImprovement]:
                                with catchtime() as t:
                                    points_assign, clust_coords, fin_cost = hmeans(
                                        *params, type_improvement=ImprClass(), initial_cost=cost(*params), callback_visu=VisualizationCallback(VisualizeClusterList(x0=0, x1=1))
                                    )
                                time_elapsed = t()
                                df = pd.concat(
                                    [df, pd.DataFrame.from_dict(
                                        {
                                            "num_clusters": [num_clusters], "num_points": [num_points], "init_type": [init_type],
                                            "ImprClass": [ImprClass.__name__], "fin_cost": [fin_cost], "time": [time_elapsed],
                                            "uuid_points_coords": [uuid_points_coords], "uuid_points_assign": [uuid_points_assign]
                                        }
                                    )], join="inner"
                                )
                                df.to_csv(PATH.parent.joinpath(
                                    "algos_results.csv"))
                                print(
                                    f"{init_type=} {ImprClass.__name__=} {fin_cost=}, computed in {time_elapsed:.4f} secs")
