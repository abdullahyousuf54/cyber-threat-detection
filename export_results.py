"""Copy metrics to results/ and build results/model_comparison.csv."""
from pathlib import Path
import csv
import json

ROOT = Path(__file__).resolve().parent
M = ROOT / "models" / "nslkdd"
R = ROOT / "results"
R.mkdir(exist_ok=True)

for fn in ["metrics.json", "evaluation.json"]:
    p = M / fn
    if p.exists():
        (R / fn).write_text(
            json.dumps(json.loads(p.read_text(encoding="utf-8")), indent=2),
            encoding="utf-8",
        )

rows = []
p = M / "evaluation.json"
if p.exists():
    d = json.loads(p.read_text(encoding="utf-8"))
    for name, vals in d["models"].items():
        rows.append({
            "model": name,
            "accuracy": round(vals["accuracy"], 4),
            "precision_weighted": round(vals["precision"], 4),
            "recall_weighted": round(vals["recall"], 4),
            "f1_weighted": round(vals["f1"], 4),
            "f1_macro": round(vals.get("f1_macro", float("nan")), 4),
            "macro_auc": round(vals["roc"]["macro_auc"], 4),
        })
if rows:
    with open(R / "model_comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("Wrote results/model_comparison.csv")
print("Exported results to results/")
