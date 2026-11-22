from src.q_learning import QLearningResponseEngine, RISK_LEVELS, state_key
from src.nslkdd import download_dataset, load_nsl_file, add_project_label
from src.nsl_models import predict
from src.risk_engine import calculate_risk

def main():
    print("\n=== STEP 3: RISK + Q-LEARNING RESPONSE ENGINE ===\n")

    engine = QLearningResponseEngine(model_path="models/nslkdd/q_table.json")
    engine.q = {}  # always learn from a clean table
    engine.train_simulated(episodes=50000)
    ok, total = engine.policy_agreement()
    print(f"Policy matches intended risk->action mapping in {ok}/{total} states")

    print("Q-table saved to models/nslkdd/q_table.json")
    print("\nRecommended actions by risk level:")

    for risk in RISK_LEVELS:
        state = state_key(risk, "DoS", 0.80)
        print(f"  {risk:8s} -> {engine.choose_action(state)}")

    try:
        _, test_path = download_dataset()
        test_df = add_project_label(load_nsl_file(test_path)).sample(
            100, random_state=42
        )
        results = predict(test_df)
        counts = {}

        for _, row in results.iterrows():
            risk = calculate_risk(
                row["rf_prediction"],
                row["rf_confidence"],
                row["anomaly_score"],
            )
            state = state_key(
                risk.risk_level,
                risk.threat_family,
                risk.anomaly_score,
            )
            action = engine.choose_action(state)
            counts[action] = counts.get(action, 0) + 1

        print("\n100-record smoke test:")
        for action in ["MONITOR", "ALERT", "BLOCK", "ISOLATE"]:
            print(f"  {action:8s}: {counts.get(action, 0)}")
    except Exception as exc:
        print("\nReal-data smoke test skipped:")
        print(f"  {exc}")
        print("Run `python step2_train.py` first if Step 2 models are missing.")

    print("\nStart dashboard:")
    print("  streamlit run dashboard/app.py")

if __name__ == "__main__":
    main()
