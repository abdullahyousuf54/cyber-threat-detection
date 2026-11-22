"""Run the complete project end-to-end.

    python run_all.py              # everything (incl. DNN + Step 7 tuning)
    python run_all.py --skip-dnn   # no PyTorch
    python run_all.py --skip-tuning  # skip the slow RF tuning (Step 7)

Order: dataset -> RF/SVM/IsolationForest -> Q-learning -> PyTorch DNN ->
ensemble evaluation -> Step 7 analysis -> export -> report.
Then start the dashboard with:  streamlit run dashboard/app.py
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
from src.asset_loader import ensure_assets


def run(script, optional=False):
    print(f"\n>>> python {script}", flush=True)
    try:
        subprocess.run([sys.executable, str(ROOT / script)], check=True, cwd=ROOT)
    except subprocess.CalledProcessError as exc:
        if not optional:
            raise
        print(f"[WARN] {script} failed ({exc}); continuing.")


def main():
    ensure_assets(ROOT)
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-dnn", action="store_true")
    ap.add_argument("--skip-tuning", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    # step5_download fetches the dataset (with mirror fallback) if missing.
    import step5_download  # noqa: F401  (downloads data if missing)

    run("step2_train.py")
    run("train_qlearning.py")
    if not args.skip_dnn:
        run("train_pytorch.py", optional=True)
    run("evaluate_step4.py")
    if not args.skip_tuning:
        run("step7_analysis.py")
    run("export_results.py")
    run("generate_report.py")

    print(f"\nAll done in {(time.time() - t0) / 60:.1f} min.")
    print("Start the dashboard:  streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
