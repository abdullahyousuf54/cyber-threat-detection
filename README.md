# Cyber Threat and Error Detection Using Machine Learning

B.Tech minor project — final compiled version (Steps 1–8 merged and bug-fixed).

## Pipeline

```text
NSL-KDD → preprocessing → Random Forest + Linear SVM + PyTorch DNN
        → weighted ensemble → Isolation Forest anomaly score → risk score
        → Q-learning response recommendation → human feedback → Q-table update
```

## Quick start

For hosted deployment with persistent feedback, see [DEPLOY.md](DEPLOY.md).

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

python run_all.py                    # download data, train, evaluate, tune, report
streamlit run dashboard/app.py       # interactive dashboard
```

Useful flags: `python run_all.py --skip-dnn` (no PyTorch), `--skip-tuning` (skips the slow Random-Forest search).
If your network blocks GitHub, put `KDDTrain+.txt` and `KDDTest+.txt` in `data/nsl_kdd/` yourself.

A **single-file** version of the entire project (CLI + dashboard) is provided separately as
`cyber_threat_detection.py`.

## Layout

| Path | Purpose |
|---|---|
| `run_all.py` | One-command full pipeline |
| `step5_download.py` | Dataset download with mirrors |
| `step2_train.py` | RF + Linear SVM + Isolation Forest |
| `train_qlearning.py` | Q-learning response policy |
| `train_pytorch.py` | PyTorch DNN |
| `evaluate_step4.py` | Ensemble evaluation, confusion matrices, ROC/AUC |
| `step7_analysis.py` | Class imbalance, RF tuning, permutation importance |
| `export_results.py`, `generate_report.py` | CSV export and `PROJECT_REPORT.md` |
| `src/` | Library code (models, ensemble, risk engine, Q-learning, feedback) |
| `dashboard/app.py` | Streamlit dashboard (8 pages) |
| `results/` | Measured outputs (metrics, comparison CSV, Step 7 artefacts) |
| `submission/` | Presentation, report template, viva Q&A, test cases, diagrams |
| `docs/history/` | Notes from the individual build steps |

## Measured results (KDDTest+, 22,544 records)

See `PROJECT_REPORT.md` and `results/model_comparison.csv` — they are generated from the actual run
and are the only numbers that should be quoted.

## Safety

The response layer only *recommends* MONITOR / ALERT / BLOCK / ISOLATE. It never executes firewall
rules, isolates devices, or changes any network.

## Fixes made while compiling the steps

1. **Ensemble class-order bug** – RF/SVM probability columns (alphabetical `classes_`) were mapped with
   `LABELS` order, so RF accuracy was reported as ~14 % in `evaluate_step4.py` and the ensemble was wrong.
2. **Anomaly score saturation** – `1/(1+x)` pushed every record to ≈0.9–1.0; replaced by a calibrated sigmoid.
3. **Q-learning under-training** – 5 000 episodes left LOW-risk states recommending ALERT; now 50 000 episodes
   (policy matches the intended mapping in 72/72 states) and training starts from a clean table.
4. **Dashboard** – duplicate `st.set_page_config` (crashed at start-up), Anomaly page chart crash,
   and a new *Explainability & Tuning* page for Step 7 results.
5. **`step7_analysis.py`** – imported names that did not exist and used wrong column names; rewritten.
   Also fixed a pandas 3 array-indexing error.
6. **`export_results.py`** – looked for metric keys that were never written (CSV was always empty).
7. Added macro-F1 to the metrics so the tuned-vs-baseline comparison is like-for-like.
