# Final Submission Guide — Cyber Threat Detection and Error Detection Using Machine Learning Algorithms

## Project flow

Data → Preprocessing → Supervised Detection (RF/SVM/DNN) → Ensemble → Anomaly Detection → Risk Score → Q-Learning Response → Human Feedback

## Final demo order

1. Open the Streamlit dashboard.
2. Show project overview and architecture.
3. Show model comparison.
4. Show ROC/AUC page after Step 5 results are generated.
5. Demonstrate threat detection and risk scoring.
6. Demonstrate recommended response.
7. Demonstrate anomaly detection.
8. Demonstrate Accept/Reject human feedback.
9. Run Step 7 and show class distribution, tuned RF, and feature importance.
10. Explain limitations and future work.

## Final commands

```powershell
python run_all.py            # data + training + evaluation + Step 7 + report (~7 min)
streamlit run dashboard/app.py
```

Run the commands after activating the virtual environment. Use `python run_all.py --skip-dnn`
if PyTorch is not installed, and `--skip-tuning` to skip the slow Random-Forest search.

## Academic honesty

Only report numerical results that are generated from the actual NSL-KDD files on the machine used for the experiment. The package intentionally does not fabricate metrics.
