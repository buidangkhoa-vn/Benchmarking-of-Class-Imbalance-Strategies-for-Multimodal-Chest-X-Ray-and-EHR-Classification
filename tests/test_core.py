import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from cxr_ehr_imbalance.cohort import TARGETS
from cxr_ehr_imbalance.labels import AdmissionLabels
from cxr_ehr_imbalance.modeling import choose_threshold, f2_score
from cxr_ehr_imbalance.runner import subset_roles
from cxr_ehr_imbalance.splits import split_indices
from cxr_ehr_imbalance.stats import N_TEST_OVER_N_TRAIN, corrected_t, holm, select

# ICD-10 titles that exercise the released rule and the clean definition, one admission per title.
TITLES = {
    "S2231XA": "Fracture of one rib, right side, initial encounter for closed fracture",
    "M810": "Age-related osteoporosis without current pathological fracture",
    "Z8781": "Personal history of (healed) traumatic fracture",
    "J22": "Unspecified acute lower respiratory infection",
    "Z122": "Encounter for screening for malignant neoplasm of respiratory organs",
    "C3411": "Malignant neoplasm of upper lobe, right bronchus or lung",
}


@pytest.fixture(scope="module")
def admission_labels():
    codes = list(TITLES)
    titles = pd.DataFrame({"icd_code": codes, "icd_version": 10, "long_title": list(TITLES.values())})
    diagnoses = pd.DataFrame(
        {"subject_id": 1, "hadm_id": range(len(codes)), "seq_num": 1, "icd_code": codes, "icd_version": 10}
    )
    return AdmissionLabels(diagnoses, titles), {code: h for h, code in enumerate(codes)}


@pytest.mark.parametrize(
    "code, target, released, clean",
    [
        ("S2231XA", "fracture", 1, 1),  # ICD-10 injury title: "initial encounter for" is not a non-diagnostic title
        ("M810", "fracture", 1, 0),  # negated: "without ... fracture"
        ("Z8781", "fracture", 1, 0),  # personal history
        ("J22", "lung_infection", 1, 1),  # acute lower respiratory infection
        ("Z122", "lung_cancer", 1, 0),  # screening encounter
        ("C3411", "lung_cancer", 1, 1),  # primary lung malignancy
    ],
)
def test_label_rules(admission_labels, code, target, released, clean):
    labels, hadm = admission_labels
    j = TARGETS.index(target)
    assert labels.vec([hadm[code]])[0, j] == released
    assert labels.vec([hadm[code]], clean=True)[0, j] == clean


def test_admission_without_codes_is_negative(admission_labels):
    labels, _ = admission_labels
    assert labels.vec([999]).sum() == 0


def test_f2_score():
    y = np.array([1, 1, 0, 0])
    assert f2_score(y, np.array([1, 0, 1, 0])) == pytest.approx(5 / 10)  # tp = fn = fp = 1
    assert f2_score(y, np.ones(4)) == pytest.approx(10 / 12)  # all-positive: 5 p / (4 p + 1) with p = 0.5
    assert f2_score(np.zeros(3), np.zeros(3)) == 0.0


def test_threshold_ties_go_to_the_value_closest_to_one_half():
    # Every threshold in (0.3, 0.6] separates the two cases perfectly.
    assert choose_threshold(np.array([1, 0]), np.array([0.6, 0.3])) == 0.5


def test_threshold_ties_prefer_higher_recall():
    # Two positives, eight negatives. Thresholds up to 0.3 flag everything (tp = 2, fp = 8, F2 = 10/18); thresholds
    # above 0.5 flag one positive only (tp = 1, fn = 1, F2 = 5/9). The F2 values are equal; higher recall wins.
    y = np.array([1, 1] + [0] * 8)
    p = np.array([0.9, 0.3] + [0.5] * 8)
    assert choose_threshold(y, p) == 0.3


def test_corrected_t():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    m, lo, hi, p = corrected_t(x)
    se = np.sqrt((1 / 5 + N_TEST_OVER_N_TRAIN) * x.var(ddof=1))
    assert m == pytest.approx(3.0)
    assert hi - m == pytest.approx(sps.t.ppf(0.975, 4) * se)
    assert m - lo == pytest.approx(hi - m)
    assert p == pytest.approx(2 * sps.t.sf(3.0 / se, 4))


def test_holm():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def _grid(val_f2):
    rows = []
    for fold, scores in enumerate(val_f2):
        for (strategy, classifier), v in scores.items():
            rows.append(
                {
                    "protocol": "P2",
                    "feature_set": "ehr",
                    "target": "fracture",
                    "seed": 42,
                    "fold": fold,
                    "strategy": strategy,
                    "classifier": classifier,
                    "val_f2": v,
                    "test_f2": v,
                }
            )
    return pd.DataFrame(rows)


def test_selection_within_fold_and_tie_break():
    R = _grid(
        [
            {("smote", "LR"): 0.50, ("none", "LR"): 0.50, ("none", "RF"): 0.40},  # tie: grid order picks none
            {("smote", "LR"): 0.30, ("none", "LR"): 0.20, ("none", "RF"): 0.60},
        ]
    )
    S = select(R).sort_values("fold")
    assert list(zip(S.strategy, S.classifier, strict=True)) == [("none", "LR"), ("none", "RF")]


def test_selection_over_folds_keeps_one_configuration():
    R = _grid(
        [
            {("none", "LR"): 0.50, ("none", "RF"): 0.40},
            {("none", "LR"): 0.30, ("none", "RF"): 0.60},
        ]
    )
    S = select(R, within_fold=False)
    assert set(S.classifier) == {"RF"} and len(S) == 2


def test_patient_grouped_split_rejects_shared_patients():
    table = pd.DataFrame({"subject_id": [1, 1, 2, 3]})
    ok = split_indices(table, np.array(["train", "train", "val", "test"], dtype=object), grouped=True)
    assert list(ok["test"]) == [3] and not ok["seen_te"].any()
    with pytest.raises(ValueError):
        split_indices(table, np.array(["train", "test", "val", "test"], dtype=object), grouped=True)


def test_subsets_drop_rows_outside_the_protocol():
    table = pd.DataFrame({"aligned": [1, 0, 1, 1], "frontal": [True, True, False, True]})
    roles = np.array(["train", "val", "test", "test"], dtype=object)
    frontal = subset_roles(table, roles, {"name": "P2-F", "split": "patient", "views": "frontal"})
    assert list(frontal) == ["train", "val", None, "test"]
    both = subset_roles(table, roles, {"name": "x", "split": "patient_aligned", "views": "frontal"})
    assert list(both) == ["train", None, None, "test"]
    assert list(subset_roles(table, roles, {"name": "P2", "split": "patient"})) == list(roles)
    with pytest.raises(ValueError):
        subset_roles(table, roles, {"name": "x", "split": "patient", "views": "lateral"})
