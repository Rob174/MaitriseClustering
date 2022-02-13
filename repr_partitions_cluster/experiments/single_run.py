import pandas as pd
from h5py import File
from pathlib import Path
from repr_partitions_cluster.src.kmean.kmean import kmean
from repr_partitions_cluster.src.generators import random_clust_initialization, balance_slicing_of_points_assign, centroids_from_points
from repr_partitions_cluster.src.hmeans.hmeans import hmeans
from repr_partitions_cluster.src.hmeans.improvement_choice import CallbackBestImprovement, CallbackFirstImprovement
from repr_partitions_cluster.src.utils import *
from repr_partitions_cluster.src.visualize import VisualizeClusterList, VisualizationCallback
from repr_partitions_cluster.src.hmeans.iteration_order import BACK, CURR, RANDOM
from rich.progress import track
import multiprocessing as mp
import time


def run_single(uuid_points_coords: str, num_points: int, num_clust: str, uuid_points_assign: str, init_type: str, imprClass, iteration_order):
    PATH = Path("./data/inputs_algorithms.hdf5")
    with File(PATH, "r") as f:
        points_coords = np.array(
            f["points_coords"][uuid_points_coords], dtype=np.float32)
        subset_points_coords = points_coords[:num_points]
        points_assign = f["points_assign"][num_clust][uuid_points_assign]
        num_clusters = int(num_clust)
        points_assign = np.array(points_assign, dtype=np.int32)
        subset_points_assign = points_assign[:num_points]
        # balance_slicing_of_points_assign(
        #     subset_points_assign, num_clusters
        # )
        subset_clust_coords = np.stack(
            centroids_from_points(
                subset_points_coords, subset_points_assign, num_clusters
            ), axis=0
        )
        if init_type == "kmean+":
            subset_points_assign, subset_clust_coords = kmean(
                subset_points_coords, np.copy(
                    subset_points_assign), np.copy(subset_clust_coords)
            )
        init_cost = cost(subset_points_coords,
                         subset_points_assign, subset_clust_coords)
        print(f"Initial cost: {init_cost}")
    points_assign, clust_coords, end_cost, num_iter = hmeans(
        subset_points_coords, subset_points_assign, subset_clust_coords,
        initial_cost=init_cost, type_improvement=imprClass, iteration_order=iteration_order,
        callback_visu=VisualizationCallback(
            VisualizeClusterList(x0=0, x1=1))
    )
    print(f"Final cost: {end_cost}")


if __name__ == "__main__":
    run_single(
        uuid_points_coords="00736491-c871-4a6a-90ad-9419eba0c134",
        num_points=30,
        num_clust=str(4),
        uuid_points_assign="003e64c9-febc-4f5b-a283-939d46ad95da",
        init_type="random",
        imprClass=CallbackFirstImprovement(),
        iteration_order=CURR()
    )
