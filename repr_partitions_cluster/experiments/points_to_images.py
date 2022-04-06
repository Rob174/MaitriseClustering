from pathlib import Path
from h5py import File
import numpy as np
import matplotlib.pyplot as plt
import json


def points_to_images(
    path: Path,
    dst_name: str,
    src_grid_max: float = 100.0,
    num_px: int = 100,
):

    num_clusters = 2
    dico = {
        "init_assignements": {},
        "points_coords": {},
        "metadata": {},
        "clusters_keys": {},
    }
    with File(path, "r") as f:
        for k, v in f["init_assignements"].items():
            dico["init_assignements"][k] = np.copy(v)
        for k, v in f["points_coords"].items():
            dico["points_coords"][k] = np.copy(v).reshape(-1, 2)
        for k, v in f["metadata"].items():
            arr = np.copy(v)
            dico["metadata"][k] = {
                "SEED": arr[0],
                "NUM_CLUST": arr[1],
                "NUM_POINTS": arr[2],
                "INIT_CHOICE": "random" if arr[3] == 0 else "kmeans+",
                "IMPR_CLASS": "BI" if arr[4] == 0 else "FI",
                "IT_ORDER": "BACK" if arr[5] == 0 else "other",
                "init_cost": arr[6],
                "final_cost": arr[7],
                "num_iter": arr[8],
                "num_iter_glob": arr[9],
                "duration": arr[10],
            }
            num_clust = int(arr[1])
            if arr[1] not in dico["clusters_keys"]:
                dico["clusters_keys"][num_clust] = []
            dico["clusters_keys"][num_clust].append(k)

    x_grid = np.linspace(0, src_grid_max, num_px + 1)
    y_grid = np.linspace(0, src_grid_max, num_px + 1)
    with open(Path(".") / "data" / "dico_best.json", "r") as f:
        dico_best = json.load(f)
    with File(path.parent / (dst_name + f"_grid_{num_px}px.hdf5"), "w") as f:
        f.create_group("input")
        f.create_group("output")
        dico_best = dico_best["final_cost"][str(num_clusters)]
        for k, best in dico_best.items():
            assignements = dico["init_assignements"][k]
            Lhist = []
            for channel in range(num_clusters):
                points = dico["points_coords"][k][assignements == channel, :]
                image, _, _ = np.histogram2d(
                    points[:, 0],
                    points[:, 1],
                    bins=(x_grid, y_grid),
                )
                Lhist.append(image)
            image = np.stack(Lhist, axis=-1)
            label = np.zeros((3,), dtype=np.float32)
            corresp = {
                "BI": 0,
                "FI": 1,
            }
            if best == "Equal":
                label = np.ones(label.shape, dtype=np.float32) * 1 / label.shape[-1]
            else:
                label[corresp[best]] = 1.0
            f["input"].create_dataset(k, data=image, dtype=np.float32)
            f["output"].create_dataset(k, data=label, dtype=np.float32)


if __name__ == "__main__":
    points_to_images(Path("data/dataset.hdf5"), "dataset_ia_2_clusters", num_px=512)
    points_to_images(Path("data/dataset.hdf5"), "dataset_ia_2_clusters", num_px=256)
    points_to_images(Path("data/dataset.hdf5"), "dataset_ia_2_clusters", num_px=128)
    points_to_images(Path("data/dataset.hdf5"), "dataset_ia_2_clusters", num_px=64)
