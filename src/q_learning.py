from pathlib import Path
import json
import random

ACTIONS = ["MONITOR", "ALERT", "BLOCK", "ISOLATE"]
RISK_LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
THREAT_FAMILIES = ["Normal", "Probe", "DoS", "R2L", "U2R", "Other"]
ANOMALY_LEVELS = ["LOW", "MEDIUM", "HIGH"]

IDEAL_ACTION = {
    "LOW": "MONITOR",
    "MEDIUM": "ALERT",
    "HIGH": "BLOCK",
    "CRITICAL": "ISOLATE",
}

ACTION_PENALTY = {
    "MONITOR": 0.0,
    "ALERT": 0.8,
    "BLOCK": 1.5,
    "ISOLATE": 2.0,
}

def anomaly_level(score):
    if float(score) < 0.35:
        return "LOW"
    if float(score) < 0.70:
        return "MEDIUM"
    return "HIGH"

def state_key(risk_level, threat_family, anomaly_score):
    return (
        str(risk_level),
        str(threat_family),
        anomaly_level(float(anomaly_score)),
    )

def reward_for(risk_level, action, human_feedback=None):
    ideal = IDEAL_ACTION.get(risk_level, "ALERT")
    if action == ideal:
        reward = 5.0
    else:
        distance = ACTIONS.index(action) - ACTIONS.index(ideal)
        if distance < 0:
            reward = -3.0 - abs(distance) * 1.5
        else:
            reward = 1.0 - abs(distance) * 0.8

    reward -= ACTION_PENALTY[action]

    if human_feedback == "accept":
        reward += 2.0
    elif human_feedback == "reject":
        reward -= 2.0
    return float(reward)

class QLearningResponseEngine:
    def __init__(
        self,
        alpha=0.10,
        gamma=0.90,
        epsilon=0.20,
        seed=42,
        model_path="models/nslkdd/q_table.json",
    ):
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.rng = random.Random(seed)
        self.model_path = Path(model_path)
        self.q = {}
        self._load()

    @staticmethod
    def _key(state):
        return "|".join(map(str, state))

    def _ensure_state(self, state):
        key = self._key(state)
        if key not in self.q:
            self.q[key] = {a: 0.0 for a in ACTIONS}
        return self.q[key]

    def choose_action(self, state, explore=False):
        values = self._ensure_state(state)
        if explore and self.rng.random() < self.epsilon:
            return self.rng.choice(ACTIONS)
        maximum = max(values.values())
        best = [a for a, v in values.items() if v == maximum]
        return self.rng.choice(best)

    def update(self, state, action, reward, next_state):
        current = self._ensure_state(state)
        future = self._ensure_state(next_state)
        target = float(reward) + self.gamma * max(future.values())
        current[action] += self.alpha * (target - current[action])

    def train_simulated(self, episodes=50000):
        states = [
            (risk, threat, anomaly)
            for risk in RISK_LEVELS
            for threat in THREAT_FAMILIES
            for anomaly in ANOMALY_LEVELS
        ]
        for _ in range(int(episodes)):
            state = self.rng.choice(states)
            risk_i = RISK_LEVELS.index(state[0])
            delta = self.rng.choice([-1, 0, 0, 0, 1])
            next_i = max(0, min(len(RISK_LEVELS) - 1, risk_i + delta))
            next_state = (
                RISK_LEVELS[next_i],
                self.rng.choice(THREAT_FAMILIES),
                self.rng.choice(ANOMALY_LEVELS),
            )
            action = self.choose_action(state, explore=True)
            self.update(
                state,
                action,
                reward_for(state[0], action),
                next_state,
            )
        self.save()
        return self

    def learn_from_feedback(self, state, action, accepted):
        feedback = "accept" if accepted else "reject"
        reward = reward_for(state[0], action, feedback)
        self.update(state, action, reward, state)
        self.save()
        return reward

    def policy_agreement(self):
        """Fraction of (risk, threat, anomaly) states whose greedy action
        equals the intended action for that risk level."""
        states = [
            (r, t, a)
            for r in RISK_LEVELS for t in THREAT_FAMILIES for a in ANOMALY_LEVELS
        ]
        ok = sum(self.choose_action(s) == IDEAL_ACTION[s[0]] for s in states)
        return ok, len(states)

    def save(self):
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "actions": ACTIONS,
            "q_table": self.q,
        }
        self.model_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load(self):
        if not self.model_path.exists():
            return
        try:
            payload = json.loads(self.model_path.read_text(encoding="utf-8"))
            self.q = payload.get("q_table", {})
        except (json.JSONDecodeError, OSError):
            self.q = {}

def recommend_response(risk_result, engine=None):
    engine = engine or QLearningResponseEngine()
    state = state_key(
        risk_result["risk_level"],
        risk_result["threat_family"],
        risk_result["anomaly_score"],
    )
    return state, engine.choose_action(state)
