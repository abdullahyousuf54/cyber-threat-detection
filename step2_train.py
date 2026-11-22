from pathlib import Path
from src.nslkdd import download_dataset, load_nsl_file, add_project_label
from src.nsl_models import train_models

def main():
    print("\n=== STEP 2: NSL-KDD PIPELINE ===\n")

    train_path, test_path = download_dataset()

    print("Loading NSL-KDD...")
    train_df = add_project_label(load_nsl_file(train_path))
    test_df = add_project_label(load_nsl_file(test_path))

    print(f"Training rows: {len(train_df):,}")
    print(f"Testing rows : {len(test_df):,}")

    print("\nProject-class distribution:")
    print(train_df["attack_family"].value_counts())

    print("\nTraining Random Forest + Linear SVM + Isolation Forest...")
    metrics = train_models(train_df, test_df)

    print("\n=== RESULTS ===")
    for name in ["random_forest", "linear_svm"]:
        m = metrics[name]
        print(
            f"{name}: "
            f"accuracy={m['accuracy']:.4f}, "
            f"precision={m['precision']:.4f}, "
            f"recall={m['recall']:.4f}, "
            f"f1={m['f1']:.4f}"
        )

    print("\nModels saved under models/nslkdd/")
    print("Start dashboard:")
    print("  streamlit run dashboard/app.py")

if __name__ == "__main__":
    main()
