from .nsl_models import predict
from .risk_engine import calculate_risk
from .q_learning import QLearningResponseEngine, state_key

class ThreatResponsePipeline:
    def __init__(self, model_dir="models/nslkdd"):
        self.model_dir = model_dir
        self.q_engine = QLearningResponseEngine(
            model_path=f"{model_dir}/q_table.json"
        )

    def analyze(self, df):
        predictions = predict(df, self.model_dir)
        rows = []

        for _, row in predictions.iterrows():
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
            item = row.to_dict()
            item.update(risk.as_dict())
            item["q_state"] = "|".join(state)
            item["recommended_action"] = self.q_engine.choose_action(state)
            rows.append(item)

        import pandas as pd
        return pd.DataFrame(rows)

    def apply_feedback(self, result_row, accepted):
        state = state_key(
            result_row["risk_level"],
            result_row["threat_family"],
            result_row["anomaly_score"],
        )
        return self.q_engine.learn_from_feedback(
            state,
            result_row["recommended_action"],
            accepted,
        )
