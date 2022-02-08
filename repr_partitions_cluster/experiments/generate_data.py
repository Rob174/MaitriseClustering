import numpy as np
from h5py import File
from pathlib import Path
import uuid
import subprocess
from repr_partitions_cluster.src.generators import generate_random_points


def generate_input_algos():
    L_NUM_CLUSTERS = [2**i for i in range(1, 8)]
    L_NUM_POINTS = np.concatenate(
        (
            np.arange(20, 91, 10),
            np.arange(100, 1001, 100)
        ), axis=0)
    NUM_COORDINATES = 2
    NUM_INSTANCES = 1000
    PATH = Path("./data/inputs_algorithms.hdf5")

    with File(PATH, "w") as f:
        group_points_coords = f.create_group("points_coords")
        for _ in range(NUM_INSTANCES):
            data = generate_random_points(
                max(L_NUM_POINTS), num_coordinates=NUM_COORDINATES
            )
            dataset = group_points_coords.create_dataset(
                str(uuid.uuid4()), data=data, dtype="f", shape=data.shape
            )
            dataset.attrs["max_num_points"] = max(L_NUM_POINTS)
            dataset.attrs["num_coordinates"] = NUM_COORDINATES
            dataset.attrs["commit_id"] = subprocess.check_output(
                ['git', 'rev-parse', '--short', 'HEAD']
            ).decode("utf-8").strip()
        group_points_assign = f.create_group("points_assign")
        for num_clusters in L_NUM_CLUSTERS:
            group_num_clusters = group_points_assign.create_group(
                f"{num_clusters}"
            )
            np.random.seed(0)
            for _ in range(NUM_INSTANCES):
                # Generate random assignment
                points_assign = np.random.randint(
                    0, num_clusters, size=max(L_NUM_POINTS)
                )
                group_num_clusters.create_dataset(
                    str(uuid.uuid4()), data=points_assign, dtype='i', shape=points_assign.shape
                )


if __name__ == "__main__":
    generate_input_algos()
