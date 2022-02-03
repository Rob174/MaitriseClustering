import numpy as np
def equals(x,y):
    return np.max(np.abs(x-y)) == 0