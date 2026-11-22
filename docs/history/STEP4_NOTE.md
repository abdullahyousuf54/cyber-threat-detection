# Step 4

Step 4 is the final implementation/presentation layer.

Added:
- PyTorch DNN evaluation and model persistence
- RF + SVM + optional DNN weighted ensemble
- Multiclass ROC/AUC evaluation
- Confusion-matrix dashboard
- Polished Streamlit dashboard
- End-to-end runner
- Automated project report generator

Run:

```bash
pip install -r requirements.txt
pip install -r requirements-pytorch.txt
python run_step4.py
python generate_report.py
streamlit run dashboard/app.py
```

If PyTorch is not installed, run:

```bash
python run_step4.py --skip-dnn
```

The project remains functional with RF + SVM + Isolation Forest + Q-learning.
