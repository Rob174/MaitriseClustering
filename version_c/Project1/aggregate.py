from pathlib import Path
import re
if __name__ == "__main__":
    folder = Path(".")
    results = []
    for p in folder.iterdir():
        if not re.match("^out_[0-9]+.txt",str(p.name)):
            continue
        with open(p, "r", encoding='utf-16-le') as f:
            text = f.read().strip()[1:].split("\n") #slicing to remove undesirable character
        for line in text:
            if line.strip() != "":
                splited = [a.split(":") for a in line.strip().split(",")]
                if len(results) == 0:
                    results.append(",".join([a[0] for a in splited]))
                results.append(",".join([a[1] for a in splited]))
    with open("results_fi_bi.csv", "w+") as f:
        f.write("\n".join(results))