from itertools import product
from pathlib import Path
import os
import time
from subprocess import Popen,PIPE
if __name__ == '__main__':
    EXE_PATH = Path("./version_c") /"x64" / "Release" / "Project1.exe"
    NUM_POINTS = [20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000]
    NUM_DIM = [2]
    NUM_CLUST = [2,4,8,16,32,64,128,256]
    INIT_TYPE = [0,1]
    IMPR_CLASS = [0,1,2]
    IT_ORDER = [0,1,2]
    results = []
    i=0
    buffer = []
    init = time.perf_counter()
    num_poss = len(list(product(NUM_POINTS,NUM_DIM,NUM_CLUST,IT_ORDER,IMPR_CLASS,INIT_TYPE)))
    for pt,dim,clust,it_order,impr,init in product(NUM_POINTS,NUM_DIM,NUM_CLUST,IT_ORDER,IMPR_CLASS,INIT_TYPE):
        if pt<=clust:
            continue
        if impr == 0 and it_order > 0:
            continue
        # buffer.append(f"{EXE_PATH.resolve()} {pt} {dim} {clust} {it_order} {impr} {init}")
        print(f"{pt=} {clust=} {init=} {impr=} {it_order=}")
        i+=1
    
    print(f"performed {i} tests in {time.perf_counter()-init} seconds")