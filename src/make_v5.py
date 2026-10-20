"""Build nb_v5_src.py (CheXpert report labels on the same radiographs; P1 record vs P2 patient split) from nb_v4_src.py."""
import os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from to_ipynb import convert

v4 = open(HERE + "/nb_v4_src.py").read()
lines = v4.splitlines(keepends=True)
cells = re.split(r'^(?=# %%)', v4, flags=re.M)
cells = [c for c in cells if c.strip()]
def cell(marker):
    hit = [c for c in cells if marker in c]
    assert len(hit) == 1, marker
    return hit[0]
AUTH = "".join(lines[184:219])          # v4 lines 185-219: bigquery import, auth (other_account / same_account), bq(), SESSION_USER check
assert AUTH.lstrip().startswith("from google.cloud import bigquery") and "SESSION_USER" in AUTH

intro = r'''# %% [markdown]
# # STEP 6 (v5) — Nhãn CheXpert từ báo cáo X-quang (gắn đúng từng ảnh): chia theo ảnh (P1) vs theo bệnh nhân (P2)
#
# **Mục đích.** Kiểm tra các kết luận của bài trên một nguồn nhãn thứ hai, chuẩn và gắn đúng từng ảnh: nhãn CheXpert
# (U-zeros) của MIMIC-CXR-JPG cho chính 29.083 ảnh của cohort. Câu hỏi:
# 1. Khi nhãn đúng từng ảnh, chia theo ảnh (P1) còn thổi phồng điểm bao nhiêu so với chia theo bệnh nhân (P2)?
# 2. Các chiến lược cân bằng dữ liệu có giúp gì trên nhãn chuẩn không?
# 3. Ảnh (11 điểm, RAD-DINO) so với EHR thế nào?
#
# Thêm nhóm thí nghiệm **EHR đúng thời điểm**: với nhãn sửa (A) và nhãn sạch (C) của STEP5, thay 11 biến EHR cũ bằng
# đặc trưng lấy đúng thời điểm chụp: tuổi tại chính lần nhập viện + 8 cờ "đã được chẩn đoán ở lần nhập viện TRƯỚC"
# (`P4_hist`, `P4C_hist`, dùng đúng fold P2), một mình và kết hợp với RAD-DINO.
#
# **Cách chạy** — tài khoản phụ (Colab Pro), Runtime **CPU, High-RAM** → **Run all**. Không cần sửa gì.
# - Notebook tự tìm thư mục `mimic-cxr-jpg-2.1.0.physionet.org` trên Drive (như STEP1) và đọc `mimic-cxr-2.0.0-chexpert.csv.gz`
#   + `mimic-cxr-2.0.0-metadata.csv.gz`. Nếu thiếu file chexpert, notebook mới hỏi BigQuery (link → chọn
#   <your PhysioNet-credentialed Google account> → dán mã), giống STEP4.
# - Cần thư mục kết quả STEP5 (`MMMDA_runs/v4_2026-10-19_admlabels`) và bảng BigQuery v3 (`MMMDA_runs/v3_2026-10-19_P4P5/v3_bq`) còn trên Drive.
# - Ước tính ~90–110 phút với 8 core. Rớt kết nối: Run all lại (tự bỏ qua phần đã xong).
# - Xong khi thấy `FLUSH XONG`. Tải **một file** `v5_for_claude.zip` về `D:\project_TTNT_Data_aug\analysis_v5\`.
#
# Mọi thư mục/file notebook tạo ra đều có tiền tố phiên bản `v5`.

'''
control = r'''# %%
# ============================== CELL 1 — CONTROL PANEL ==============================
RUN_MODE      = "full"                # "full" | "preflight" (1 fold, 2 nhãn, thư mục riêng)
VERSION       = "v5_2026-10-19"       # tag phiên bản: có trong tên mọi thư mục/file tạo ra
MIMIC_JPG_ROOT = None                 # None = tự tìm; hoặc vd "/content/drive/MyDrive/MIMIC_CXR/mimic-cxr-jpg-2.1.0.physionet.org"
GCP_PROJECT   = "YOUR_GCP_PROJECT_ID"   # chỉ dùng nếu trên Drive không có file chexpert
BQ_AUTH       = "other_account"

DRIVE_ROOT    = "/content/drive/MyDrive/data_aug"
CSV_PATH      = f"{DRIVE_ROOT}/x_ray_and_features_data.csv"
EXPECTED_HASH = "da85519958e4bccb"
RADDINO_DIR   = f"{DRIVE_ROOT}/embeddings/raddino"
V2_FOLDS      = f"{DRIVE_ROOT}/leakage_aware_v2/leakage_aware_v2/manifests/folds_subject_s42.csv"   # fold P2 (chỉ đọc)
V2_RECFOLDS   = f"{DRIVE_ROOT}/leakage_aware_v2/leakage_aware_v2/manifests/folds_record_s42.csv"    # fold P1 (chỉ đọc)
OUT_ROOT      = f"{DRIVE_ROOT}/MMMDA_runs"                               # -> MMMDA_runs/v5_2026-10-19_chexpert/
V4_ADMLABELS  = f"{DRIVE_ROOT}/MMMDA_runs/v4_2026-10-19_admlabels/v4_labels/v4_admission_labels.csv.gz"   # nhãn A/C + cờ tiền sử (STEP5)
V3_BQ_DIR     = f"{DRIVE_ROOT}/MMMDA_runs/v3_2026-10-19_P4P5/v3_bq"     # admissions/patients (tuổi tại lần nhập viện)
LOCAL_CACHE   = "/content/mmmda_v5_cache"
LOCAL_OUT     = "/content/mmmda_v5_out"

# 8 CheXpert observations (U-zeros: only 1.0 counts as positive; -1/blank -> 0)
CX_TARGETS = {"cx_pneumonia": "Pneumonia", "cx_lung_opacity": "Lung Opacity", "cx_effusion": "Pleural Effusion",
              "cx_cardiomegaly": "Cardiomegaly", "cx_edema": "Edema", "cx_pneumothorax": "Pneumothorax",
              "cx_fracture": "Fracture", "cx_lung_lesion": "Lung Lesion"}
N_PAR         = None                  # số process song song; None = tất cả core

'''
setup = cell("CELL 2 — INSTALL + MOUNT").replace(
    '''    V2_RECFOLDS = os.environ.get("V3_V2RECFOLDS", ""); PREV_BQ_DIRS = [x for x in os.environ.get("V3_PREVBQ", "").split(",") if x]''',
    '''    V2_RECFOLDS = os.environ.get("V3_V2RECFOLDS", ""); V4_ADMLABELS = os.environ["V5_ADMLABELS"]; V3_BQ_DIR = os.environ["V5_BQDIR"]''')
helpers = cell("CELL 3 — IMPORTS + HELPERS").replace('V = "v4"', 'V = "v5"').replace(
    'OUT = Path(OUT_ROOT) / (f"{VERSION}_admlabels" + ("_preflight" if RUN_MODE == "preflight" else ""))',
    'OUT = Path(OUT_ROOT) / (f"{VERSION}_chexpert" + ("_preflight" if RUN_MODE == "preflight" else ""))')
assert 'V = "v5"' in helpers and "_chexpert" in helpers
load = cell("CELL 4 — LOAD + LOCK DATA")

chex = r'''# %%
# ============================== CELL 5 — CHEXPERT LABELS (Drive first, BigQuery fallback) ==============================
LABF = DIRS["labels"] / f"{V}_chexpert_cohort_studies.csv.gz"     # chexpert rows of the cohort's studies
MAPF = DIRS["labels"] / f"{V}_dicom_study_map.csv.gz"             # dicom -> study_id for the cohort's radiographs
CX_SRC = DIRS["labels"] / f"{V}_chexpert_source.json"
def find_mimic_root():
    name = "mimic-cxr-jpg-2.1.0.physionet.org"
    pats = [f"/content/drive/MyDrive/{name}", f"/content/drive/MyDrive/*/{name}", f"/content/drive/MyDrive/*/*/{name}",
            f"/content/drive/.shortcut-targets-by-id/*/{name}", f"/content/drive/.shortcut-targets-by-id/*/*/{name}",
            f"/content/drive/Shareddrives/*/{name}", f"/content/drive/Shareddrives/*/*/{name}"]
    for p in pats:
        for c in sorted(glob.glob(p)):
            if os.path.exists(f"{c}/mimic-cxr-2.0.0-metadata.csv.gz"):
                return c
    return None
CX_COLS = sorted(set(CX_TARGETS.values()))
if not (LABF.exists() and MAPF.exists()):
    if LOCAL_TEST:
        cxs = pd.read_csv(os.environ["V5_CHEXPERT"]); meta = pd.read_csv(os.environ["V5_META"]); src = {"source": "LOCAL_TEST"}
    else:
        root = MIMIC_JPG_ROOT or find_mimic_root()
        cand = sorted(glob.glob(f"{root}/*chexpert*.csv*")) if root else []
        if root and cand:
            print("MIMIC_JPG_ROOT:", root, "| chexpert:", Path(cand[0]).name)
            cxs = pd.read_csv(cand[0])
            meta = pd.read_csv(f"{root}/mimic-cxr-2.0.0-metadata.csv.gz", usecols=["dicom_id", "subject_id", "study_id"])
            src = {"source": "drive", "root": root, "file": Path(cand[0]).name}
        else:
            print("Không thấy file chexpert trên Drive -> lấy từ BigQuery")
            SIDS_BQ = [int(s) for s in df.subject_id.unique()]
''' + "\n".join("        " + l if l.strip() else l for l in AUTH.splitlines()) + r'''
            P_S = [bigquery.ArrayQueryParameter("s", "INT64", SIDS_BQ)]
            tabs = {}
            for ds in ["physionet-data.mimic_cxr_jpg", "physionet-data.mimic_cxr"]:
                try:
                    for tb in client.list_tables(ds):
                        tabs.setdefault(tb.table_id, f"{ds}.{tb.table_id}")
                except Exception as e:
                    print(ds, "không liệt kê được bảng:", repr(e)[:120])
            print("bảng tìm thấy:", sorted(tabs))
            tcx = next((v for k, v in tabs.items() if "chexpert" in k.lower()), None)
            tmeta = next((v for k, v in tabs.items() if "metadata" in k.lower()), None)
            if not tcx or not tmeta:
                raise RuntimeError(f"Không tìm thấy bảng chexpert/metadata trên BigQuery: {sorted(tabs)}. Gửi thông báo này cho Claude.")
            cxs = bq(f"SELECT * FROM `{tcx}` WHERE subject_id IN UNNEST(@s)", P_S)
            meta = bq(f"SELECT dicom_id, subject_id, study_id FROM `{tmeta}` WHERE subject_id IN UNNEST(@s)", P_S)
            cxs.columns = [{c.lower().replace(" ", "_"): c for c in CX_COLS}.get(c.lower(), c) for c in cxs.columns]
            src = {"source": "bigquery", "chexpert_table": tcx, "metadata_table": tmeta}
    meta = meta.rename(columns={"dicom_id": "dicom"})
    mp = df[["dicom", "subject_id"]].merge(meta[["dicom", "study_id"]], on="dicom", how="left")
    assert mp.study_id.notna().all(), f"{int(mp.study_id.isna().sum())} ảnh không có study_id trong metadata"
    missing_cols = [c for c in CX_COLS if c not in cxs.columns]
    assert not missing_cols, f"file chexpert thiếu cột {missing_cols}"
    cxs = cxs[cxs.study_id.isin(set(mp.study_id))][["subject_id", "study_id"] + CX_COLS]
    atomic_to_csv(mp, MAPF); atomic_to_csv(cxs, LABF); atomic_json(src, CX_SRC)
mp = pd.read_csv(MAPF); cxs = pd.read_csv(LABF)
assert (mp.dicom.values == df.dicom.values).all()
lab_rows = mp[["study_id"]].merge(cxs.drop(columns="subject_id").drop_duplicates("study_id"), on="study_id", how="left")
no_report = int(lab_rows[CX_COLS].isna().all(1).sum())
img = df[["dataset_position", "subject_id", "hadm_id", "dicom", "StudyDate"]].copy()
cx_rows = []
for t, col in CX_TARGETS.items():
    v = lab_rows[col].values
    img[t] = (v == 1.0).astype(np.int8)                         # U-zeros
    cx_rows.append({"target": t, "chexpert": col, "pos_images": int((v == 1.0).sum()), "uncertain_images": int((v == -1.0).sum()),
                    "prev_%": round(100 * float((v == 1.0).mean()), 2)})
TARGETS_CX = list(CX_TARGETS)
# label varies within patient? (released labels never did)
multi = img.groupby("subject_id").size() > 1
var = img[img.subject_id.isin(multi[multi].index)].groupby("subject_id")[TARGETS_CX].nunique().gt(1)
for r in cx_rows:
    r["multi_img_patients_with_varying_label_%"] = round(100 * float(var[r["target"]].mean()), 1)
    r["pos_patients"] = int(img.groupby("subject_id")[r["target"]].max().sum())
# prior-study rule: positive if any EARLIER radiograph of the patient (by StudyDate) was positive
img = img.sort_values(["subject_id", "StudyDate", "dataset_position"])
for t in TARGETS_CX:
    prev_any = img.groupby("subject_id")[t].transform(lambda s: s.cummax().shift(1)).fillna(0)
    img[f"histcx_{t}"] = prev_any.astype(np.int8)
img = img.sort_values("dataset_position").reset_index(drop=True)
CXSUM = pd.DataFrame(cx_rows)
# ---- time-aligned EHR features for the repaired-label protocols (from STEP5 + v3 BigQuery tables) ----
AL = pd.read_csv(V4_ADMLABELS)
assert (AL.dataset_position.values == np.arange(len(df))).all() and (AL.dicom.values == df.dicom.values).all()
for t in TARGETS:
    for pref in ["adm", "cln", "hist", "histcln"]:
        img[f"{pref}_{t}"] = AL[f"{pref}_{t}"].values.astype(np.int8)
adm_bq = pd.read_csv(f"{V3_BQ_DIR}/v3_bq_admissions.csv.gz", parse_dates=["admittime"]).drop_duplicates("hadm_id").set_index("hadm_id")
pts_bq = pd.read_csv(f"{V3_BQ_DIR}/v3_bq_patients.csv.gz").drop_duplicates("subject_id").set_index("subject_id")
ay = adm_bq.admittime.reindex(img.hadm_id).dt.year.values.astype(float)
ay = np.where(np.isnan(ay), pd.to_datetime(df.admittime).dt.year.values, ay)
img["age_own"] = (pts_bq.anchor_age.reindex(img.subject_id).values + (ay - pts_bq.anchor_year.reindex(img.subject_id).values)).astype(float)
print("age at own admission differs from released age:", round(float((img.age_own != df.age).mean()), 4))
atomic_to_csv(CXSUM, DIRS["labels"] / f"{V}_chexpert_label_summary.csv")
print("radiographs without any CheXpert finding row (all labels -> 0):", no_report)
print(CXSUM.to_string(index=False))

'''
table = r'''# %%
# ============================== CELL 6 — TABLE, FEATURE SETS, SPLITS ==============================
E = np.load(f"{RADDINO_DIR}/emb.npy").astype(np.float32)
ids = pd.read_csv(f"{RADDINO_DIR}/ids.csv")
posm = df.dicom.map(dict(zip(ids.dicom, range(len(ids)))))
assert posm.notna().all(), f"{int(posm.isna().sum())} ảnh không có embedding RAD-DINO"
EMB = [f"raddino_{j}" for j in range(E.shape[1])]
HIST_A = [f"hist_{t}" for t in TARGETS]; HIST_C = [f"histcln_{t}" for t in TARGETS]
rec = pd.concat([df[["dataset_position", "subject_id"] + EHR + IMG].reset_index(drop=True),
                 img[["age_own"] + HIST_A + HIST_C].reset_index(drop=True),
                 pd.DataFrame(E[posm.astype(int).values], columns=EMB)], axis=1)
FEATSETS = {"ehr": dict(cont=EHR_CONT, emb=[], bin=EHR_BIN),
            "img_xrv": dict(cont=IMG, emb=[], bin=[]),
            "img_raddino": dict(cont=[], emb=EMB, bin=[]),
            "fusion_raddino": dict(cont=EHR_CONT, emb=EMB, bin=EHR_BIN),
            # time-aligned EHR: age at the radiograph's own admission + prior-diagnosis flags (earlier admissions only)
            "ehrhist_a": dict(cont=["age_own"], emb=[], bin=HIST_A),
            "fusionhist_a": dict(cont=["age_own"], emb=EMB, bin=HIST_A),
            "ehrhist_c": dict(cont=["age_own"], emb=[], bin=HIST_C),
            "fusionhist_c": dict(cont=["age_own"], emb=EMB, bin=HIST_C)}
EMB_PCA_DIM = 64
rec.to_pickle(CACHE / f"{V}_rec.pkl")
np.savez(CACHE / f"{V}_labels.npz", **{f"cx__{t}": img[t].values.astype(np.int8) for t in TARGETS_CX},
         **{f"histcx__{t}": img[f"histcx_{t}"].values.astype(np.int8) for t in TARGETS_CX},
         **{f"adm__{t}": img[f"adm_{t}"].values.astype(np.int8) for t in TARGETS},
         **{f"cln__{t}": img[f"cln_{t}"].values.astype(np.int8) for t in TARGETS},
         **{f"histadm__{t}": img[f"hist_{t}"].values.astype(np.int8) for t in TARGETS},
         **{f"histcln__{t}": img[f"histcln_{t}"].values.astype(np.int8) for t in TARGETS})
atomic_to_csv(img[["dataset_position", "subject_id", "dicom"] + TARGETS_CX], DIRS["labels"] / f"{V}_chexpert_image_labels.csv.gz")

if V2_FOLDS and os.path.exists(V2_FOLDS):
    m2 = pd.read_csv(V2_FOLDS); print("P2 folds: đọc từ", V2_FOLDS)
else:
    print("P2 folds: không thấy file v2 -> dựng lại (cùng seed 42, cùng thuật toán)")
    Ys = S[TARGETS].values; parts = []
    for k, (tr, te) in enumerate(MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(np.zeros(len(Ys)), Ys)):
        itr, iva = next(MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42 * 100 + k).split(np.zeros(len(tr)), Ys[tr]))
        role = np.empty(len(Ys), dtype=object); role[te] = "test"; role[tr[itr]] = "train"; role[tr[iva]] = "val"
        parts.append(pd.DataFrame({"subject_id": S.index.values, "fold": k, "role": role}))
    m2 = pd.concat(parts, ignore_index=True)
if V2_RECFOLDS and os.path.exists(V2_RECFOLDS):
    m1 = pd.read_csv(V2_RECFOLDS); print("P1 folds: đọc từ", V2_RECFOLDS)
else:
    print("P1 folds: không thấy file v2 -> dựng lại (cùng seed 42, cùng thuật toán)")
    Yr = df[TARGETS].values; parts = []
    for k, (tr, te) in enumerate(MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(np.zeros(len(Yr)), Yr)):
        itr, iva = next(MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42 * 100 + k).split(np.zeros(len(tr)), Yr[tr]))
        role = np.empty(len(Yr), dtype=object); role[te] = "test"; role[tr[itr]] = "train"; role[tr[iva]] = "val"
        parts.append(pd.DataFrame({"dataset_position": df.dataset_position.values, "fold": k, "role": role}))
    m1 = pd.concat(parts, ignore_index=True)
SPLITS = {}
for k in range(5):
    SPLITS[("P1", k)] = m1[m1.fold == k].set_index("dataset_position").role.reindex(rec.dataset_position).values
    SPLITS[("P2", k)] = rec.subject_id.map(m2[m2.fold == k].set_index("subject_id").role).values
for (sname, k), role in SPLITS.items():
    assert pd.notna(role).all()
    i_tr, i_va, i_te = [np.where(role == r)[0] for r in ("train", "val", "test")]
    s_tr, s_va, s_te = [set(rec.subject_id.values[i]) for i in (i_tr, i_va, i_te)]
    seen_te = np.isin(rec.subject_id.values[i_te], list(s_tr | s_va))
    if sname != "P1":
        assert not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te) and not seen_te.any()
    np.savez(CACHE / f"{V}_splits" / f"{sname}__f{k}.npz", idx_tr=i_tr, idx_va=i_va, idx_te=i_te, seen_te=seen_te)

CX_FEATS = ["ehr", "img_xrv", "img_raddino", "fusion_raddino"]
PROTO = {"P2_chexpert": dict(split="P2", folds=list(range(5)), label="cx", feats=CX_FEATS, targets=TARGETS_CX),
         "P1_chexpert": dict(split="P1", folds=list(range(5)), label="cx", feats=CX_FEATS, targets=TARGETS_CX),
         "P4_hist":     dict(split="P2", folds=list(range(5)), label="adm", feats=["ehrhist_a", "fusionhist_a"], targets=TARGETS),
         "P4C_hist":    dict(split="P2", folds=list(range(5)), label="cln", feats=["ehrhist_c", "fusionhist_c"], targets=TARGETS)}
CLFS = ["LR", "LGBM", "ET"]
G_FEATS = list(FEATSETS)
if RUN_MODE == "preflight":
    for _p in PROTO: PROTO[_p]["folds"] = [0]
    G_FEATS = ["ehr", "img_raddino", "ehrhist_a", "ehrhist_c"]; CLFS = ["LR", "LGBM"]
if LOCAL_TEST and os.environ.get("V3_TINY") == "1":       # offline smoke test only — ignored on Colab
    for _p in PROTO: PROTO[_p]["folds"] = [0]
    CLFS = ["LR", "LGBM"]; G_FEATS = ["ehr", "img_raddino", "ehrhist_a", "fusionhist_c"]

lab = np.load(CACHE / f"{V}_labels.npz")
TASKS, skipped = [], []
for p, cfg in PROTO.items():
    tgts = cfg["targets"]
    if RUN_MODE == "preflight" or (LOCAL_TEST and os.environ.get("V3_TINY") == "1"):
        tgts = [t for t in tgts if t in ("cx_pneumonia", "cx_effusion", "cx_fracture", "pneumonia", "fracture", "arthritis")][: 2 if RUN_MODE == "preflight" else 3]
    for t in tgts:
        y = lab[f"{cfg['label']}__{t}"]
        for k in cfg["folds"]:
            sp = np.load(CACHE / f"{V}_splits" / f"{cfg['split']}__f{k}.npz")
            ptr, pva, pte = int(y[sp["idx_tr"]].sum()), int(y[sp["idx_va"]].sum()), int(y[sp["idx_te"]].sum())
            if ptr < 10 or pva < 3 or pte < 3:
                skipped.append({"protocol": p, "fold": k, "target": t, "pos_tr": ptr, "pos_va": pva, "pos_te": pte}); continue
            for ft in [f for f in G_FEATS if f in cfg.get("feats", G_FEATS)]:
                TASKS.append((p, k, ft, t))
atomic_to_csv(pd.DataFrame(skipped, columns=["protocol", "fold", "target", "pos_tr", "pos_va", "pos_te"]), DIRS["manifests"] / f"{V}_skipped.csv")
print(f"tasks {len(TASKS)} | skipped (quá ít ca dương) {len(skipped)}")
if skipped:
    print(pd.DataFrame(skipped).to_string(index=False))

'''
funcs = cell("CELL 10 — MODEL / SAMPLER / METRIC FUNCTIONS")
stageA = cell("CELL 11 — STAGE A").replace('sec = {"ehr": 4, "img_xrv": 5, "img_raddino": 12, "fusion_raddino": 15}',
    'sec = {"ehr": 4, "img_xrv": 5, "img_raddino": 12, "fusion_raddino": 15, "ehrhist_a": 2, "ehrhist_c": 2, "fusionhist_a": 13, "fusionhist_c": 13}')
assert "ehrhist_a" in stageA
stageB = cell("CELL 12 — STAGE B")
base = cell("CELL 13 — BASELINES").replace(
    '''        y = lab[f"{cfg['label']}__{t}"]; hist = lab[f"{'histcln' if cfg['label'] == 'cln' else 'hist'}__{t}"]''',
    '''        y = lab[f"{cfg['label']}__{t}"]; hist = lab[f"hist{cfg['label']}__{t}"]''').replace(
    '''if cfg["label"] in ("adm", "cln") else np.nan''', '')
assert 'hist{cfg' in base
ana = cell("CELL 14 — MERGE + ANALYSIS")
ana = ana.replace('if "P1_admission" in set(R.protocol):', 'if "P1_chexpert" in set(R.protocol):').replace(
    'sel1 = best[best.protocol == "P1_admission"]', 'sel1 = best[best.protocol == "P1_chexpert"]').replace(
    'DIRS["analysis"] / f"{V}_tableC_P1adm_seen_unseen.csv"', 'DIRS["analysis"] / f"{V}_tableC_P1cx_seen_unseen.csv"').replace(
    'print("\\nP1_admission — seen vs unseen patients (macro)\\n"', 'print("\\nP1_chexpert — seen vs unseen patients (macro)\\n"')
a = ana.index("# --- patient-cluster bootstrap"); b = ana.index("numbers = {")
strat = r'''# --- imbalance strategies vs none (paired, same classifier/target/fold) ---
srows = []
for p in sorted(set(R.protocol)):
    Rp = R[R.protocol == p]; b0 = Rp[Rp.strategy == "none"].set_index(["feature_set", "target", "classifier", "fold"])
    for (fs, st), d in Rp[Rp.strategy != "none"].groupby(["feature_set", "strategy"]):
        dd = d.set_index(["feature_set", "target", "classifier", "fold"]); c = dd.index.intersection(b0.index)
        diff = dd.loc[c, "test_f2"] - b0.loc[c, "test_f2"]
        srows.append({"protocol": p, "feature_set": fs, "strategy": st, "dF2_mean": diff.mean(), "n": len(diff)})
STRAT = pd.DataFrame(srows)
atomic_to_csv(STRAT.round(5), DIRS["analysis"] / f"{V}_strategies_vs_none.csv")
print("\nΔF2 vs no augmentation\n", STRAT.pivot_table(index=["protocol", "feature_set"], columns="strategy", values="dF2_mean").round(4).to_string())

'''
ana = ana[:a] + strat + ana[b:]
ana = re.sub(r'numbers = \{.*?\}\}\natomic_json', '''numbers = {"version": VERSION, "data_hash": data_hash, "chexpert_source": json.load(open(CX_SRC)),
           "label_summary": CXSUM.to_dict("records"), "radiographs_without_report_labels": no_report,
           "tableA": TA.round(4).reset_index().to_dict("records"), "n_fits": len(R), "n_errors": len(ERR), "skipped": skipped,
           "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__}}
atomic_json''', ana, flags=re.S)
ana = ana.replace('''    z.write(BQ_SRC, f"{OUT.name}/{DIRS['bq'].name}/{BQ_SRC.name}")\n''', '')
ana = ana.replace('''    for d in ["labels", "manifests", "analysis"]:
        for f in sorted(DIRS[d].glob("*")):
            if f.is_file() and not f.name.endswith(".tmp"):''', '''    for d in ["labels", "manifests", "analysis"]:
        for f in sorted(DIRS[d].glob("*")):
            if f.is_file() and not f.name.endswith(".tmp") and "cohort_studies" not in f.name and "dicom_study_map" not in f.name:''')
assert "BQ_SRC" not in ana and "RECON" not in ana and "BOOT" not in ana, "leftover v4 references"
flush = cell("CELL 15 — FLUSH")

src = intro + control + setup + helpers + load + chex + table + funcs + stageA + stageB + base + ana + flush
# renumber cell headers sequentially
n = [0]
def renum(m):
    n[0] += 1
    return f"# ============================== CELL {n[0]} —"
src = re.sub(r"# ============================== CELL \d+ —", renum, src)
open(HERE + "/nb_v5_src.py", "w").write(src)
compile(src, "nb_v5", "exec")
convert(HERE + "/nb_v5_src.py", HERE + "/../notebooks/STEP6_v5_2026-10-19_chexpert_labels.ipynb")
print("ok")
