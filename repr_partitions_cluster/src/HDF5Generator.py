from pathlib import Path
from random import Random
from h5py import File
import numpy as np
from enum import Enum

from sklearn.cluster import KMeans


class FilterMode(Enum):
    Random = "random"
    KMeans = "kmeans+"
class HDF5Generator:
    def __init__(self, filename: Path):
        with File(filename, "r") as hf:
            self.keys = list(hf["input"].keys())
        self.filename = filename
    def __len__(self):
        return len(self.keys)
    def __getitem__(self, idx):
        with File(self.filename, "r") as hf:
            return np.copy(hf["input"][self.keys[idx]]), np.copy(hf["output"][self.keys[idx]])
         
    def __call__(self):
            for k in self.keys:
                with File(self.filename, "r") as hf:
                    yield np.copy(hf["input"][k]), np.copy(hf["output"][k])

class HDF5GeneratorFilter(HDF5Generator):
    def __init__(self, filename_data: Path,filename_metadata: Path, filter_mode: FilterMode):
        super().__init__(filename_data)
        self.keys = []
        with File(filename_metadata, "r") as cache:
            for k, v in cache["metadata"].items():
                arr = np.copy(v)
                init = "random" if arr[3] == 0 else "kmeans+"
                if init == filter_mode.value:
                    self.keys.append(k)
        
        with File(filename_data, "r") as hf:
            val_keys = set(list(hf["input"].keys()))
            self.keys = set(self.keys).intersection(val_keys)
        
