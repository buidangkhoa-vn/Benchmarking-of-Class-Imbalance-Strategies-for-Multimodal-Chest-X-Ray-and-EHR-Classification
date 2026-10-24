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
## STEP 1: NB2, full run (GPU)
GPU runtime (a T4 is enough), Run all. About 35-45 min; a rerun resumes.
Finished when `raddino: saved (29083, 768)` and `xrv_chex: saved (29083, 1024)` are printed.
""")
convert(HERE + "/v_nb2_full.py", f"{OUT}/STEP1_NB2_embeddings_FULL.ipynb", gpu=True)

# ---------- NB1 workers ----------
for k in range(3):
    variant(NB1, HERE + f"/v_worker{k}.py",
            {"RUN_MODE": '"full"', "N_WORKERS": "3", "WORKER_INDEX": str(k), "ROLE": '"worker"', "REQUIRE_EMBEDDINGS": "True"},
            f"""
## STEP 2: NB1, worker {k} of 3 (session {k + 1})
Start after STEP 1. CPU runtime with high RAM, Run all.
Run workers 0, 1 and 2 in three sessions at the same time; each takes a third of the tasks.
A rerun skips finished tasks. Finished when `>>> WORKER {k} DONE` and then `FLUSH DONE` are printed;
close the session only after that, so that Drive finishes syncing.
""")
    convert(HERE + f"/v_worker{k}.py", f"{OUT}/STEP2_NB1_worker{k}.ipynb")

# ---------- NB1 final ----------
variant(NB1, HERE + "/v_final.py",
        {"RUN_MODE": '"full"', "N_WORKERS": "3", "WORKER_INDEX": "0", "ROLE": '"final"', "REQUIRE_EMBEDDINGS": "True"},
        """
## STEP 3: NB1, merge and analysis
Run after all three workers have finished (about 5 min, no training). If jobs are reported missing, rerun that worker first.
Wait for `FLUSH DONE`. Outputs: `MyDrive/data_aug/leakage_aware_v2/leakage_aware_v2/results_all.csv` and `analysis/`.
""")
convert(HERE + "/v_final.py", f"{OUT}/STEP3_NB1_FINAL_merge_analysis.ipynb")
print("done")
