"""Model fits for one or more experiments of the config (main, admission, chexpert_history, tuning, sensitivity).

Results go to outputs/runs/<experiment>/results.csv (one row per fit) and baselines.csv. A finished
(protocol, seed, fold, feature set) group is not refitted when the script is run again."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cxr_ehr_imbalance.config import cli  # noqa: E402
from cxr_ehr_imbalance.runner import run_experiment  # noqa: E402


def extra(ap):
    ap.add_argument("--experiment", nargs="+", default=["main", "admission", "chexpert_history", "tuning", "trees", "sensitivity"])
    ap.add_argument("--folds", nargs="+", type=int, help="run only these folds (quick check)")
    ap.add_argument("--targets", nargs="+", help="run only these targets (quick check)")
    ap.add_argument("--n-jobs", type=int)


args, cfg = cli(__doc__, extra)
for name in args.experiment:
    run_experiment(cfg, name, folds=args.folds, targets=args.targets, n_jobs=args.n_jobs)
