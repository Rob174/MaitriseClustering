from itertools import product
from pathlib import Path
import os
from sqlite3 import Time
import time
from subprocess import (
    Popen,
    PIPE,
    TimeoutExpired,
    check_output,
    STDOUT,
    CalledProcessError,
)

if __name__ == "__main__":
    EXE_PATH = Path("./version_c") / "x64" / "Release" / "c.exe"
    NUM_POINTS = [
        # 20,
        # 30,
        # 40,
        # 50,
        # 60,
        # 70,
        # 80,
        # 90,
        # 100,
        # 200,
        # 300,
        # 400,
        # 500,
        # 600,
        # 700,
        # 800,
        # 900,
        1000,
    ]
    NUM_DIM = [2]
    NUM_CLUST = [2, 4, 8, 16, 32, 64, 128, 256]
    INIT_TYPE = [0, 1]
    # IMPR_CLASS = [0, 1 ,2]
    IMPR_CLASS = [1]
    # IMPR_CLASS = [2]
    IT_ORDER = [0, 1, 2]
    results = []
    i = 0
    buffer = []
    duration = 0
    num_poss = len(
        list(product(NUM_POINTS, NUM_DIM, NUM_CLUST, IT_ORDER, IMPR_CLASS, INIT_TYPE))
    )
    for pt, dim, it_order, impr, init, clust in product(
        NUM_POINTS, NUM_DIM, IT_ORDER, IMPR_CLASS, INIT_TYPE, NUM_CLUST
    ):
        if pt <= clust:
            continue
        if impr in [0, 2] and it_order > 0:
            continue
        if impr == 1 and it_order not in  [0]:
            continue
        init_t = time.perf_counter()
        args = (
            f"{int(pt)} {int(dim)} {int(clust)} {int(it_order)} {int(impr)} {int(init)}"
        )
        Lp = [
            f"{str(EXE_PATH.resolve())[:-4]}{c}.exe "
            + args
            + f" > out_{i*10+c}.txt"  # , stdout=PIPE,close_fds = True)
            for c in range(1000 // 100)
        ]
        for p in Lp:
            print(p)
            # results.extend(p.communicate()[0].decode("utf-8").strip().split("\n"))
        end = time.perf_counter()
        duration += end - init_t
        # print(f"{pt=} {clust=} {init=} {impr=} {it_order=} {(end-init_t)=}")
        i += 1

    print(f"performed {i} tests in {duration} seconds")
