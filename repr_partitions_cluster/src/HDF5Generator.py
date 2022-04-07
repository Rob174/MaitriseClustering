from pathlib import Path
from h5py import File
import numpy as np
class HDF5Generator:
    def __call__(self,filename):
            with File(filename, 'r') as hf:
                for sample in zip(hf["inputs"], hf["outputs"]):
                    yield np.array(sample[0],dtype=np.float32),np.array(sample[1],dtype=np.float32)
