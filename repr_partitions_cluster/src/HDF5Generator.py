from pathlib import Path
from h5py import File
import numpy as np


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
