from itertools import product
from pathlib import Path
import os
import time
from subprocess import Popen,PIPE
if __name__ == '__main__':
    EXE_PATH = Path("./version_c") /"x64" / "Release" / "clustering_pack.exe"
    NUM_POINTS = [20,30,40,50,60,70,80,90]#,100,200,300,400,500,600,700,800,900,1000]
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
        buffer.append(f"{EXE_PATH.resolve()} {pt} {dim} {clust} {it_order} {impr} {init}")
        # print(f"{pt} {dim} {clust} {init} {impr} {it_order}")
        i+=1
        if len(buffer) == 16:
            print(f"{i}/{num_poss}={i/num_poss*100:.2f}%")
            procs = [ Popen(i, stdout=PIPE) for i in buffer ]
            for p in procs:
                out = p.communicate()[0].decode("utf-8")
                if "inf" in out:
                    b=0 # TODO debug cost inf unreproducible in single execution
                results.extend(out.strip().split("\r\n"))
            buffer = []
    
    if len(buffer) > 0:
        procs = [ Popen(i, stdout=PIPE) for i in buffer ]
        for p in procs:
            results.extend(p.communicate()[0].decode("utf-8").strip().split("\r\n"))
        buffer = []
    print(f"performed {i} tests in {time.perf_counter()-init} seconds")
    p = Path("./results.csv")
    with open(p.resolve(),"w") as f:
        header = ",".join([a.split(":")[0] for a in results[0].strip().split(",")])
        f.write(header+"\n")
        for r in results:
            f.write(",".join([a.split(":")[1] for a in r.strip().split(",")])+"\n")
            