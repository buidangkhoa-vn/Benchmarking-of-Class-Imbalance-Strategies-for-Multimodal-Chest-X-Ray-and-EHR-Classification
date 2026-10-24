import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cxr_ehr_imbalance.config import cli  # noqa: E402
from cxr_ehr_imbalance.runner import run_experiment  # noqa: E402

EXPERIMENTS = [
    "main",
    "admission",
    "chexpert_history",
    "tuning",
    "trees",
    "sensitivity",
    "frontal",
    "densenet_admission",
    "joint_tuning",
]


def extra(ap):
    ap.add_argument("--experiment", nargs="+", default=EXPERIMENTS, help="experiments to run (default: all)")
    ap.add_argument("--folds", nargs="+", type=int, help="run only these outer folds (quick check)")
    ap.add_argument("--targets", nargs="+", help="run only these targets (quick check)")
    ap.add_argument("--n-jobs", type=int, help="parallel worker processes (default: n_jobs of the config)")


args, cfg = cli("Model fits and baselines for the experiments of the config.", extra)
for name in args.experiment:
    run_experiment(cfg, name, folds=args.folds, targets=args.targets, n_jobs=args.n_jobs)
