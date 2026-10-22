import re, sys
import os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from to_ipynb import convert

def variant(src_path, out_py, overrides, intro_md):
    s = open(src_path).read()
    for name, val in overrides.items():
        pat = re.compile(rf'^({re.escape(name)}\s*=\s*)([^#\n]*?)(\s*#.*)?$', re.M)
        m = pat.search(s)
        assert m, name
        s = s[:m.start()] + f"{m.group(1)}{val}" + (("  " + m.group(3).strip()) if m.group(3) else "") + s[m.end():]
    md = "# %% [markdown]\n" + "\n".join("# " + l if l else "#" for l in intro_md.strip().splitlines()) + "\n\n"
    open(out_py, "w").write(md + s)
    return out_py

OUT = HERE + "/../notebooks"
import os
os.makedirs(OUT, exist_ok=True)
NB1 = HERE + "/nb1_src.py"; NB2 = HERE + "/nb2_src.py"

# ---------- NB2 full ----------
variant(NB2, HERE + "/v_nb2_full.py", {"RUN_MODE": '"full"'}, """
## STEP 1 — NB2 (GPU runtime; a T4 is enough) — preset to `RUN_MODE = "full"`
Runtime → Change runtime type → **GPU (T4)** → **Run all**. Nothing to edit.
Takes about 35–45 min (copying 29,083 images ~17 min + embedding extraction). If the session drops, Run all again; the notebook resumes.
Done when `raddino: saved (29083, 768)` and `xrv_chex: saved (29083, 1024)` appear.
Then continue with STEP 2 (three worker notebooks).
""")
convert(HERE + "/v_nb2_full.py", f"{OUT}/STEP1_NB2_embeddings_FULL.ipynb", gpu=True)

# ---------- NB1 workers ----------
for k in range(3):
    variant(NB1, HERE + f"/v_worker{k}.py",
            {"RUN_MODE": '"full"', "N_WORKERS": "3", "WORKER_INDEX": str(k), "ROLE": '"worker"', "REQUIRE_EMBEDDINGS": "True"},
            f"""
## STEP 2 — WORKER {k} / 3 (CPU session {k + 1}) — preconfigured, nothing to edit
Run only **after NB2 (STEP 1) has finished**.
Runtime → Change runtime type → **CPU, High-RAM** → **Run all**.
Open worker notebooks 0, 1 and 2 in three separate sessions and run them at the same time; each does one third of the tasks, without overlap.
If the session drops: reopen this notebook → Run all (finished tasks are skipped).
Done when `>>> WORKER {k} DONE` **and the final line `FLUSH DONE`** appear — close the session only after that line
(otherwise files that Google Drive has not finished syncing can be lost).
Each task is written to Drive as two files (instead of thousands of small files); earlier results are reused.
""")
    convert(HERE + f"/v_worker{k}.py", f"{OUT}/STEP2_NB1_worker{k}.ipynb")

# ---------- NB1 final ----------
variant(NB1, HERE + "/v_final.py",
        {"RUN_MODE": '"full"', "N_WORKERS": "3", "WORKER_INDEX": "0", "ROLE": '"final"', "REQUIRE_EMBEDDINGS": "True"},
        """
## STEP 3 — FINAL: merge results, tables and figures (any CPU session) — preconfigured
Run only **after all three workers report DONE**. Run all (~5 min, no training).
If the `jobs done …` line reports missing jobs, the notebook names the worker → reopen that worker notebook, Run all, then rerun FINAL.
Wait for the final line `FLUSH DONE` before closing the session.
Outputs: `MyDrive/data_aug/leakage_aware_v2/leakage_aware_v2/results_all.csv` and the `analysis/` folder (inputs of the analysis scripts).
""")
convert(HERE + "/v_final.py", f"{OUT}/STEP3_NB1_FINAL_merge_analysis.ipynb")
print("done")
