from pathlib import Path
import re
import numpy as np
from h5py import File

if __name__ == "__main__":
    # Path to the folder containing the TSPLIB instances filtered (only EUC_2D) with small script python (done in console mode)
    root = (Path(".") / "data" / "tsplib").resolve()
    files = {}
    GRID_COORD_MIN = 0.
    GRID_COORD_MAX = 100.
    for f in root.iterdir():
        if not f.is_file():
            continue
        with f.open("r") as fp:
            lines = fp.readlines()
        files[f.stem] = []
        for l in lines:
            l = l.strip()
            if re.match(r"^\d+( +\d+\.?\d*[eE]?[+-]?\d*){2}$", l):
                files[f.stem].append([float(e) for e in l.strip().split()if e != ""])
        # Convert to numpy array, remove unecessary index column
        files[f.stem] = np.array(files[f.stem],dtype=np.float32)[:,1:]
        assert len(files[f.stem].shape) == 2, "Error, wrong shape"
    with File(root.parent.joinpath("tsplib.h5"), "w") as cache:
        for k, v in files.items():
            cache.create_dataset(k, data=v)
    with File(root.parent.joinpath("tsplib_normalized.h5"), "w") as cache:
        for k, v in files.items():
            for i in [0,1]:
                v[:,i] = (v[:,i]-np.min(v[:,i]))/(np.max(v[:,i])-np.min(v[:,i]))*(GRID_COORD_MAX-GRID_COORD_MIN)+GRID_COORD_MIN
            cache.create_dataset(k, data=v)
    end=0