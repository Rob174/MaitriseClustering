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
    EXE_PATH = Path("./version_c") / "x64" / "Release" / "Project1.exe"
    NUM_POINTS = [
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        100,
        200,
        300,
        400,
        500,
        600,
        700,
        800,
        900,
        1000,
    ]
    NUM_DIM = [2]
    NUM_CLUST = [2, 4, 8, 16, 32, 64, 128, 256]
    INIT_TYPE = [0, 1]
    # IMPR_CLASS = [0, 1 ,2]
    IMPR_CLASS = [0, 1]
    # IMPR_CLASS = [2]
    IT_ORDER = [0, 1, 2]
    results = []
    i = 0
    buffer = []
    duration = 0
    num_poss = len(
        list(product(NUM_POINTS, NUM_DIM, NUM_CLUST, IT_ORDER, IMPR_CLASS, INIT_TYPE))
    )
    for pt, dim, clust, it_order, impr, init in product(
        NUM_POINTS, NUM_DIM, NUM_CLUST, IT_ORDER, IMPR_CLASS, INIT_TYPE
    ):
        if pt <= clust:
            continue
        if impr == 0 and it_order > 0:
            continue
        cont = True
        while cont:
            try:
                init_t = time.perf_counter()
                args = f"{EXE_PATH.resolve()} {int(pt)} {int(dim)} {int(clust)} {int(it_order)} {int(impr)} {int(init)}"
                result = (
                    Popen(args, stdout=PIPE)
                    .communicate()[0]#timeout=60 * 5
                    .decode("utf-8")
                )
                end = time.perf_counter()
                duration += end - init_t
                cont = False
            except TimeoutExpired:
                print(
                    "timeout ",
                    pt,
                    dim,
                    clust,
                    it_order,
                    impr,
                    init,
                    "after ",
                    time.perf_counter() - init_t,
                )
        if "Wrong" in result:
            continue
        results.append(result)
        print(f"{pt=} {clust=} {init=} {impr=} {it_order=} {(end-init_t)=}")
        duration += end - init
        i += 1

    print(f"performed {i} tests in {duration} seconds")
    p = Path("./results.csv")
    with open(p.resolve(), "w+") as f:
        header = ",".join([a.split(":")[0] for a in results[0].strip().split(",")])
        f.write(header + "\n")
        for r in results:
            if len(r.strip().split(",")) > 1:
                f.write(
                    ",".join([a.split(":")[1] for a in r.strip().split(",")]) + "\n"
                )
