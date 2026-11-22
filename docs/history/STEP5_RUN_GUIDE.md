# Step 5 Run Guide

## Windows PowerShell

```powershell
cd cyber-threat-detection-step5
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-pytorch.txt
python step5_experiment.py
python export_results.py
streamlit run dashboard/app.py
```

If PowerShell blocks activation:

```cmd
.venv\Scripts\activate.bat
```

## Linux/macOS

```bash
cd cyber-threat-detection-step5
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-pytorch.txt
python step5_experiment.py
python export_results.py
streamlit run dashboard/app.py
```

## What Step 5 produces

- RF/SVM/DNN training metrics
- Ensemble metrics
- Confusion matrices
- ROC/AUC evaluation
- Anomaly detection artifacts
- Updated project report
- CSV model comparison
- Dashboard-ready model files

If your network blocks GitHub, manually download KDDTrain+.txt and KDDTest+.txt and place them in `data/nsl_kdd/`.
