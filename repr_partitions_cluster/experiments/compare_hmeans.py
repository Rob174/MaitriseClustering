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
from repr_partitions_cluster.src.generators import generate_random_points


def generator():
    L_NUM_CLUSTERS = [2**i for i in range(1, 8)]
    L_NUM_POINTS = np.concatenate(
        (
            np.arange(20, 91, 10),
            np.arange(100, 1001, 100)
        ), axis=0)
    NUM_COORDINATES = 2
    np.random.seed(0)
    global_index = -1
    for num_points in L_NUM_POINTS:
        for num_clusters in L_NUM_CLUSTERS:
            if num_clusters >= num_points:
                continue
            for init_type in ["random", "kmean+"]:
                init_cost = cost(*params)
                for ImprClass in [CallbackBestImprovement, CallbackFirstImprovement]:
                    if ImprClass.__name__ == "CallbackFirstImprovement":
                        iteration_orders = [
                            CURR(), BACK(), RANDOM()]
                    else:
                        iteration_orders = [BACK()]
                    for iteration_order in iteration_orders:
                        for _ in range(1000):
                            points_coords = generate_random_points(
                                num_points, num_coordinates=NUM_COORDINATES,seed=None
                            )
                            points_assign = np.random.randint(
                                0, num_clusters, size=max(L_NUM_POINTS)
                            )
                            clust_coords = np.stack(
                                centroids_from_points(
                                    points_coords, points_assign, num_clusters
                                ), axis=0
                            )
                            if init_type == "kmean+":
                                points_assign, clust_coords = kmean(
                                    (points_coords), np.copy(points_assign), np.copy(clust_coords)
                                )
                            
                            init_cost = cost(points_coords, points_assign, clust_coords)
                            global_index += 1
                            # Important: make a copy of the parameters to avoid problems with multiprocessig
                            yield np.copy(points_coords),np.copy(points_assign),np.copy(clust_coords), init_cost, num_clusters, num_points, init_type, ImprClass, iteration_order,global_index 


def wrapper(args):
    *args_hmeans_non_result, init_cost, num_clusters, num_points, init_type, ImprClass, iteration_order, global_index = args
        
    with catchtime() as t:
        points_assign, clust_coords, end_cost, num_iter,num_iter_tot = hmeans(
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
        "num_iter_tot": num_iter_tot,
        "time": time,
        "global_index": global_index
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
            "num_iter_tot",
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
