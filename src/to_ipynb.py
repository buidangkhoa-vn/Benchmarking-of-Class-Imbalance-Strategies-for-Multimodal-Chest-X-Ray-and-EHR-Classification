import json, re, sys

def convert(src_path, out_path, gpu=False):
    src = open(src_path).read()
    parts = re.split(r'^(# %%.*)$', src, flags=re.M)
    cells = []
    i = 1
    while i < len(parts):
        header, body = parts[i], parts[i + 1]
        i += 2
        body = body.strip("\n")
        if not body.strip():
            continue
        if "[markdown]" in header:
            lines = [re.sub(r'^# ?', '', l) for l in body.splitlines()]
            cells.append({"cell_type": "markdown", "metadata": {}, "source": "\n".join(lines)})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": body})
    for c in cells:
        lines = c["source"].split("\n")
        c["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    meta = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}, "colab": {"provenance": []}}
    if gpu:
        meta["accelerator"] = "GPU"
    nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": meta, "cells": cells}
    json.dump(nb, open(out_path, "w"), indent=1, ensure_ascii=False)
    print(out_path, len(cells), "cells")

if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2], gpu=(len(sys.argv) > 3 and sys.argv[3] == "gpu"))
