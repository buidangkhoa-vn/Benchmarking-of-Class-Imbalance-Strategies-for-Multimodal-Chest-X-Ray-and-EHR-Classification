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
## BƯỚC 1 — NB2 (tab GPU, T4 là đủ) — ĐÃ ĐẶT SẴN `RUN_MODE = "full"`
Runtime → Change runtime type → **GPU (T4)** → **Run all**. Không cần sửa gì.
Mất khoảng 35–45 phút (copy 29.083 ảnh ~17 phút + trích embedding). Rớt kết nối thì Run all lại, notebook chạy tiếp.
Xong khi thấy `raddino: saved (29083, 768)` và `xrv_chex: saved (29083, 1024)`.
Sau đó sang BƯỚC 2 (3 notebook worker).
""")
convert(HERE + "/v_nb2_full.py", f"{OUT}/STEP1_NB2_embeddings_FULL.ipynb", gpu=True)

# ---------- NB1 workers ----------
for k in range(3):
    variant(NB1, HERE + f"/v_worker{k}.py",
            {"RUN_MODE": '"full"', "N_WORKERS": "3", "WORKER_INDEX": str(k), "ROLE": '"worker"', "REQUIRE_EMBEDDINGS": "True"},
            f"""
## BƯỚC 2 — WORKER {k} / 3 (tab CPU thứ {k + 1}) — ĐÃ CẤU HÌNH SẴN, KHÔNG CẦN SỬA GÌ
Chỉ chạy **sau khi NB2 (BƯỚC 1) đã xong**.
Runtime → Change runtime type → **CPU, High-RAM** → **Run all**.
Mở 3 notebook worker 0, 1, 2 ở 3 tab khác nhau và cho chạy cùng lúc; mỗi tab làm 1/3 công việc, không trùng nhau.
Rớt kết nối: mở lại đúng notebook này → Run all (tự bỏ qua phần đã xong).
Xong khi thấy `>>> WORKER {k} XONG` **và dòng cuối cùng `FLUSH XONG`** — chỉ đóng tab sau dòng này
(lần trước kết quả worker 0 bị chậm/mất vì tab đóng khi Drive chưa đồng bộ xong).
Phiên bản này ghi mỗi task thành 2 file trên Drive (thay vì hàng nghìn file nhỏ) và vẫn dùng lại kết quả cũ.
""")
    convert(HERE + f"/v_worker{k}.py", f"{OUT}/STEP2_NB1_worker{k}.ipynb")

# ---------- NB1 final ----------
variant(NB1, HERE + "/v_final.py",
        {"RUN_MODE": '"full"', "N_WORKERS": "3", "WORKER_INDEX": "0", "ROLE": '"final"', "REQUIRE_EMBEDDINGS": "True"},
        """
## BƯỚC 3 — FINAL: gộp kết quả + bảng + hình (tab CPU bất kỳ) — ĐÃ CẤU HÌNH SẴN
Chỉ chạy **sau khi cả 3 worker đều báo XONG**. Run all (~5 phút, không train gì).
Nếu dòng `jobs done …` báo còn thiếu job, notebook sẽ ghi rõ worker nào thiếu → mở lại notebook worker đó, Run all, rồi chạy lại FINAL.
Chờ dòng cuối `FLUSH XONG` rồi mới đóng tab.
Kết quả: `MyDrive/data_aug/leakage_aware_v2/leakage_aware_v2/results_all.csv` và thư mục `analysis/` → chép vào `D:\\project_TTNT_Data_aug\\analysis\\`.
""")
convert(HERE + "/v_final.py", f"{OUT}/STEP3_NB1_FINAL_merge_analysis.ipynb")
print("done")
