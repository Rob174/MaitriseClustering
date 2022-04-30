"""Manages the creation of the dataset (input/output true) for an ai that takes as input a top view of the initial clustering"""
from pathlib import Path
from h5py import File
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split

class Label:
    """Map labels to one-hot vectors"""
    def __init__(self,path:Path,num_clusters:int=2,attr:str = "final_cost"):
        with open(path,encoding="utf-8") as f:
            self.data = json.load(f)[attr][str(num_clusters)]
    def keys(self):
        return list(self.data.keys())
    def __getitem__(self,key):
        return np.array(self.data[key]).reshape(1)
class LabelOneHot(Label):
    def __getitem__(self,key):
        label = np.zeros((2,), dtype=np.float32)
        corresp = {
            "BI": 0,
            "FI": 1,
        }
        if self.data[key] == "Equal":
            label = np.ones(label.shape, dtype=np.float32) * 1 / label.shape[-1]
        else:
            label[corresp[self.data[key]]] = 1.0
        return label
def points_to_images(
    path: Path,
    dst_name: str,
    label_maker: Label,
    src_grid_max: float = 100.0,
    num_px: int = 100,
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
            if arr.shape[0] == 11: # Case of normal dataset
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
            elif arr.shape[0] == 12: # Case for diversified dat
                dico["metadata"][k] = {
                    "SEED_POINTS": arr[0],aset # Change here (2 seeds)
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
    # Train/test split
    keys = label_maker.keys()
    tr_keys, val_keys = train_test_split(
        keys, test_size=val_size, random_state=0, shuffle=False
    )
    # Prepare for grid projection
    x_grid = np.linspace(0, src_grid_max, num_px + 1)
    y_grid = np.linspace(0, src_grid_max, num_px + 1)
    for keys, name in zip([tr_keys, val_keys], ["tr", "val"]):
        with File(
            path.parent
            / "image_dataset"
            / (dst_name + f"_grid_{num_px}px_{name}.hdf5"),
            "w",
        ) as f:
            f.create_group("input")
            f.create_group("output")
            for k in keys: # Iterate over train/test keys
                assignements = dico["init_assignements"][k]
                Lhist = []
                for channel in range(num_clusters): # Projection separately for each initial cluster 
                    points = dico["points_coords"][k][assignements == channel, :]
                    image, _, _ = np.histogram2d( # Image projection
                        points[:, 0],
                        points[:, 1],
                        bins=(x_grid, y_grid),
                    )
                    Lhist.append(image)
                image = np.stack(Lhist, axis=-1)
                label = label_maker[k]
                # Write into hdf5 dataset
                f["input"].create_dataset(k, data=image, dtype=np.float32)
                f["output"].create_dataset(k, data=label, dtype=np.float32)


if __name__ == "__main__":
    label_maker = Label(Path("data/dico_diff.json"), num_clusters=2, attr="final_cost")
    # Create dataset for 3 projection sizes :( 64,128,256)
    points_to_images(Path("data/dataset_1000pts_40ksamples.hdf5"), "dataset_ia_2_clusters_1000pts_continuous", num_px=256,label_maker=label_maker)
    points_to_images(Path("data/dataset_1000pts_40ksamples.hdf5"), "dataset_ia_2_clusters_1000pts_continuous", num_px=128,label_maker=label_maker)
    points_to_images(Path("data/dataset_1000pts_40ksamples.hdf5"), "dataset_ia_2_clusters_1000pts_continuous", num_px=64,label_maker=label_maker)
