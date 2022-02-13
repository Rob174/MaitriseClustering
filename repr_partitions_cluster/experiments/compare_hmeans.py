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


def generator():
    L_NUM_CLUSTERS = [2**i for i in range(1, 8)]
    L_NUM_POINTS = np.concatenate(
        (
            np.arange(20, 91, 10),
            np.arange(100, 1001, 100)
        ), axis=0)
    PATH = Path("./data/inputs_algorithms.hdf5")

    with File(PATH, "r") as f:
        for uuid_points_coords, points_coords in f["points_coords"].items():
            points_coords = np.array(points_coords, dtype=np.float32)
            for num_points in L_NUM_POINTS:
                print(f"{num_points=}")
                subset_points_coords: np.ndarray = points_coords[:num_points]
                for num_clust, datasets_points_assign in f["points_assign"].items():
                    num_clusters = int(num_clust)
                    if num_clusters >= num_points:
                        continue
                    for uuid_points_assign, points_assign in datasets_points_assign.items():
                        points_assign = np.array(points_assign, dtype=np.int32)
                        subset_points_assign = points_assign[:num_points]
                        subset_clust_coords = np.stack(
                            centroids_from_points(
                                subset_points_coords, subset_points_assign, num_clusters
                            ), axis=0
                        )
                        points_assign_impr, clust_coords_impr = kmean(
                            subset_points_coords, np.copy(
                                subset_points_assign), np.copy(subset_clust_coords)
                        )
                        for params, init_type in list(zip([(subset_points_coords, subset_points_assign, subset_clust_coords), (subset_points_coords, points_assign_impr, clust_coords_impr)], ["random", "kmean+"])):
                            init_cost = cost(*params)
                            for ImprClass in [CallbackBestImprovement, CallbackFirstImprovement]:
                                if ImprClass.__name__ == "CallbackFirstImprovement":
                                    iteration_orders = [
                                        CURR(), BACK(), RANDOM()]
                                else:
                                    iteration_orders = [BACK()]
                                for iteration_order in iteration_orders:
                                    # Important: make a copy of the parameters to avoid problems with multiprocessig
                                    yield np.copy(params[0]),np.copy(params[1]),np.copy(params[2]), init_cost, num_clusters, num_points, init_type, ImprClass, iteration_order, uuid_points_coords, uuid_points_assign


def wrapper(args):
    *args_hmeans_non_result, init_cost, num_clusters, num_points, init_type, ImprClass, iteration_order, uuid_points_coords, uuid_points_assign = args
        
    with catchtime() as t:
        points_assign, clust_coords, end_cost, num_iter = hmeans(
            *args_hmeans_non_result, initial_cost=init_cost, type_improvement=ImprClass(), iteration_order=iteration_order,
            # callback_visu=VisualizationCallback(
            #     VisualizeClusterList(x0=0, x1=1))
        )
    time = t()
    dico = {
        "num_clusters": num_clusters,
        "num_points": num_points,
        "init_type": init_type,
        "ImprClass": ImprClass.__name__,
        "iteration_order": iteration_order.__class__.__name__,
        "init_cost": init_cost,
        "end_cost": end_cost,
        "num_iter": num_iter,
        "time": time,
        "uuid_points_coords": uuid_points_coords,
        "uuid_points_assign": uuid_points_assign
    }
    PATH_RES = Path("./data/algos_results.csv")
    lock.acquire()
    with open(PATH_RES, 'a') as f:
        f.write(','.join(map(str, dico.values()))+"\n")
    lock.release()
    return


def wrapper_init(l):
    global lock
    lock = l


if __name__ == "__main__":
    PATH_RES = Path("./data/algos_results.csv")
    with open(PATH_RES, "w") as fp:
        fp.write(','.join([
            "num_clusters",
            "num_points",
            "init_type",
            "ImprClass",
            "iteration_order",
            "init_cost",
            "end_cost",
            "num_iter",
            "time",
            "uuid_points_coords",
            "uuid_points_assign",
        ])+"\n")
    # From https://stackoverflow.com/questions/25557686/python-sharing-a-lock-between-processes
    lock = mp.Lock()
    p = mp.Pool(None, wrapper_init, [lock])
    results = list(p.imap(wrapper, generator(), chunksize=100))
    p.close()
    p.join()

    # Version for debugging
    # for args in generator():
    #     wrapper(args)
