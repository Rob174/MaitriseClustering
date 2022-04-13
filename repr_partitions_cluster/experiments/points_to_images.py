from pathlib import Path
from h5py import File
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split


def points_to_images(
    path: Path,
    dst_name: str,
    src_grid_max: float = 100.0,
    num_px: int = 100,
    tr_size: float = 0.7,
    val_size: float = 0.2,
):
    dico_name = ""
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
            if arr.shape[0] == 11:
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
                dico_name = "dico_best.json"
            elif arr.shape[0] == 12:
                dico["metadata"][k] = {
                    "SEED_POINTS": arr[0],
                    "SEED_ASSIGNS": arr[1],
                    "NUM_CLUST": arr[2],
                    "NUM_POINTS": arr[3],
                    "INIT_CHOICE": "random" if arr[4] == 0 else "kmeans+",
                    "IMPR_CLASS": "BI" if arr[5] == 0 else "FI",
                    "IT_ORDER": "BACK" if arr[6] == 0 else "other",
                    "init_cost": arr[7],
                    "final_cost": arr[8],
                    "num_iter": arr[9],
                    "num_iter_glob": arr[10],
                    "duration": arr[11],
                }
                dico_name = "dico_best_diversified.json"
            num_clust = int(dico["metadata"][k]["NUM_CLUST"])
            if dico["metadata"][k]["NUM_CLUST"] not in dico["clusters_keys"]:
                dico["clusters_keys"][num_clust] = []
            dico["clusters_keys"][num_clust].append(k)

    x_grid = np.linspace(0, src_grid_max, num_px + 1)
    y_grid = np.linspace(0, src_grid_max, num_px + 1)
    with open(Path(".") / "data" / dico_name, "r") as f:
        dico_best = json.load(f)
    dico_best = dico_best["final_cost"][str(num_clusters)]
    keys = list(dico_best.keys())
    tr_keys, val_keys = train_test_split(
        keys, test_size=val_size, random_state=0, shuffle=False
    )
    # val_keys, tst_keys = train_test_split(
    #     tst_val_keys,
    #     test_size=tst_size / (tst_size + val_size),
    #     random_state=0,
    #     shuffle=False,
    # )
    for keys, name in zip([tr_keys, val_keys], ["tr", "val"]):
        with File(
            path.parent
            / "image_dataset"
            / (dst_name + f"_grid_{num_px}px_{name}.hdf5"),
            "w",
        ) as f:
            f.create_group("input")
            f.create_group("output")
            for k in keys:
                best = dico_best[k]
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
                label = np.zeros((2,), dtype=np.float32)
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
    # points_to_images(Path("data/dataset.hdf5"), "dataset_ia_2_clusters", num_px=512)
    points_to_images(Path("data/dataset_1000pts_40ksamples_diversified.hdf5"), "dataset_ia_2_clusters_1000pts_diversified", num_px=256)
    points_to_images(Path("data/dataset_1000pts_40ksamples_diversified.hdf5"), "dataset_ia_2_clusters_1000pts_diversified", num_px=128)
    points_to_images(Path("data/dataset_1000pts_40ksamples_diversified.hdf5"), "dataset_ia_2_clusters_1000pts_diversified", num_px=64)
