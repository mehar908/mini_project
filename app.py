import csv
import io
import mimetypes
import os
import json
import platform
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from contextlib import contextmanager

import numpy as np
import pandas as pd
try:
    import importlib

    flask = importlib.import_module("flask")
    Flask = flask.Flask
    jsonify = flask.jsonify
    render_template = flask.render_template
    request = flask.request
    send_from_directory = flask.send_from_directory
    send_file = flask.send_file
except ModuleNotFoundError as exc:
    if exc.name == "flask":
        raise ModuleNotFoundError(
            "Flask is missing from the active Python environment. "
            "Install it with: python -m pip install Flask"
        ) from exc
    raise
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "creditcard.csv"
DATABASE_PATH = Path(os.environ.get("DATABASE_PATH", BASE_DIR / "transactions.sqlite3"))
mimetypes.add_type("text/css", ".css", strict=True)

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)

WORKSPACE_NAV_ITEMS = [
    ("overview", "Overview", "/"),
    ("live-stream", "Live Stream", "/live-stream"),
    ("rules", "Rules", "/rules"),
    ("sandbox", "Sandbox", "/sandbox"),
    ("disputes-sla", "Disputes SLA", "/disputes-sla"),
    ("ai-syndicatehunter", "AI SyndicateHunter", "/ai-syndicatehunter"),
]


@app.context_processor
def inject_workspace_navigation():
    return {"workspace_nav_items": WORKSPACE_NAV_ITEMS}


dataset_load_started = time.perf_counter()
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Put creditcard.csv next to app.py: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
dataset_load_seconds = time.perf_counter() - dataset_load_started
if "Class" not in df.columns:
    raise ValueError("Expected a 'Class' column in creditcard.csv.")

features = [column for column in df.columns if column != "Class"]
X = df[features].apply(pd.to_numeric, errors="coerce")
y = pd.to_numeric(df["Class"], errors="coerce")
if y.isna().any():
    raise ValueError("The 'Class' target column must contain numeric, non-missing labels.")
if not y.isin([0, 1]).all():
    raise ValueError("The 'Class' target column must contain only 0 and 1 labels.")
y = y.astype(int)

dataset_amount_average = (
    float(X["Amount"].mean())
    if "Amount" in X.columns and X["Amount"].notna().any()
    else None
)
dataset_proxy_metrics = {}
dataset_micro_window_counts = pd.Series(dtype=int)
dataset_micro_times = None
if {"Amount", "Time"}.issubset(X.columns):
    dataset_amounts = X["Amount"].fillna(0).to_numpy(dtype=float)
    dataset_times = X["Time"].fillna(0).to_numpy(dtype=float)
    micro_mask = dataset_amounts < 5
    sorted_micro_times = np.sort(dataset_times[micro_mask])
    window_starts = np.searchsorted(sorted_micro_times, dataset_times - 300, side="left")
    window_ends = np.searchsorted(sorted_micro_times, dataset_times, side="right")
    dataset_micro_window_counts = pd.Series(
        window_ends - window_starts,
        index=X.index,
    )
    dataset_micro_times = pd.Series(dataset_times, index=X.index)
    micro_candidates = micro_mask & ((window_ends - window_starts) > 3)
    legitimate_count = int((y == 0).sum())
    amount_proxy_candidates = (
        dataset_amounts > 2 * dataset_amount_average
        if dataset_amount_average is not None
        else np.zeros(len(df), dtype=bool)
    )
    amount_false_positives = int(np.sum(amount_proxy_candidates & (y.to_numpy() == 0)))
    micro_false_positives = int(np.sum(micro_candidates & (y.to_numpy() == 0)))
    dataset_proxy_metrics = {
        "micro_transaction_candidates": int(micro_candidates.sum()),
        "micro_transaction_fraud_labels": int(np.sum(micro_candidates & (y.to_numpy() == 1))),
        "micro_transaction_false_positive_rate_percent": round(
            micro_false_positives / legitimate_count * 100, 4
        ) if legitimate_count else None,
        "amount_over_twice_average_candidates": int(amount_proxy_candidates.sum()),
        "amount_over_twice_average_fraud_labels": int(
            np.sum(amount_proxy_candidates & (y.to_numpy() == 1))
        ),
        "amount_over_twice_average_false_positive_rate_percent": round(
            amount_false_positives / legitimate_count * 100, 4
        ) if legitimate_count else None,
    }


def evaluate_dataset_rules(amount, source_row_index, scenario_signals=None):
    row_amount = float(amount)
    if scenario_signals is not None:
        tor_match = scenario_signals["tor_or_vpn"]
        micro_match = scenario_signals["card_testing"]
        travel_match = scenario_signals["impossible_travel"]
        return [
            {
                "rule_id": "RULE_TOR_DATACENTER_BLOCK",
                "status": "simulated_match" if tor_match else "simulated_no_match",
                "matched_conditions": (
                    ["Synthetic source marked as Tor exit", "Synthetic proxy/VPN signal is active"]
                    if tor_match else ["Synthetic source not marked as Tor exit or proxy/VPN"]
                ),
                "detail": "Synthetic test signal only; not observed network or location data.",
            },
            {
                "rule_id": "RULE_CARDING_RAPID_MICRO_TX",
                "status": "simulated_match" if micro_match else "simulated_no_match",
                "matched_conditions": (
                    [
                        f"Amount ${row_amount:.2f} < $5.00",
                        f"{scenario_signals['transactions_in_five_minutes']} transactions in 5 minutes",
                        f"Simulated device/IP velocity {scenario_signals['device_ip_velocity_per_hour']} per hour",
                    ]
                    if micro_match
                    else ["Synthetic card/device velocity does not meet the rule"]
                ),
                "detail": "Synthetic scenario signal only; no real card, device, or IP identity is present.",
            },
            {
                "rule_id": "RULE_VELOCITY_IMPOSSIBLE_TRAVEL",
                "status": "simulated_match" if travel_match else "simulated_no_match",
                "matched_conditions": (
                    [
                        f"Amount ${row_amount:.2f} > 2× dataset average (${2 * dataset_amount_average:.2f})",
                        f"Synthetic card/IP country mismatch: {scenario_signals['card_country']} vs {scenario_signals['ip_country']}",
                        f"Synthetic travel speed {scenario_signals['travel_speed_kmh']} km/h > 800 km/h",
                    ]
                    if travel_match
                    else ["Synthetic locations or travel speed do not meet the rule"]
                ),
                "detail": "Synthetic scenario signal only; not observed geolocation or physical travel.",
            },
        ]

    if source_row_index is None:
        micro_window_count = None
    elif source_row_index in dataset_micro_window_counts.index:
        micro_window_count = int(dataset_micro_window_counts.loc[source_row_index])
    else:
        micro_window_count = None

    micro_candidate = row_amount < 5 and micro_window_count is not None and micro_window_count > 3
    amount_candidate = (
        dataset_amount_average is not None
        and row_amount > 2 * dataset_amount_average
    )
    return [
        {
            "rule_id": "RULE_TOR_DATACENTER_BLOCK",
            "status": "needs_data",
            "matched_conditions": [],
            "detail": "Not evaluated: source IP, Tor-exit, and proxy/VPN signals are absent.",
        },
        {
            "rule_id": "RULE_CARDING_RAPID_MICRO_TX",
            "status": "partial_candidate" if micro_candidate else "unverified",
            "matched_conditions": (
                [f"Amount ${row_amount:.2f} < $5.00", f"{micro_window_count} dataset-wide micro-transactions within 5 minutes"]
                if micro_candidate
                else (["Amount < $5.00"] if row_amount < 5 else [])
            ),
            "detail": (
                "Global dataset-time proxy only; cannot verify same-card/device/IP velocity."
                if source_row_index is not None
                else "Not evaluated: synthetic demo transaction."
            ),
        },
        {
            "rule_id": "RULE_VELOCITY_IMPOSSIBLE_TRAVEL",
            "status": "partial_candidate" if amount_candidate else "unverified",
            "matched_conditions": (
                [f"Amount ${row_amount:.2f} > 2× dataset average (${2 * dataset_amount_average:.2f})"]
                if amount_candidate
                else []
            ),
            "detail": (
                "Amount-only proxy; cannot verify countries, coordinates, travel speed, or shared card/session."
                if source_row_index is not None
                else "Not evaluated: synthetic demo transaction."
            ),
        },
    ]


sample_size = min(len(df), 20000)
sample_indices = df.sample(n=sample_size, random_state=42).index
X_sample = X.loc[sample_indices]
y_sample = y.loc[sample_indices]
X_train, X_replay, y_train, y_replay = train_test_split(
    X_sample,
    y_sample,
    test_size=0.2,
    random_state=42,
    stratify=y_sample,
)
replay_source_indices = X_replay.index.to_numpy()
training_medians = X_train.median()
X_train = X_train.fillna(training_medians).fillna(0)
X_replay = X_replay.fillna(training_medians).fillna(0)

demo_count = 15
scenario_count = 6
training_fraud = X_train.loc[y_train == 1]
if training_fraud.empty:
    raise ValueError("The training sample must contain fraud examples to create the demo feed.")

demo_rng = np.random.default_rng(42)


def make_demo_fraud(source, count):
    rows = source.iloc[demo_rng.integers(0, len(source), size=count)].copy().reset_index(drop=True)
    for column in features:
        scale = max(float(source[column].std()), 1e-9) * 0.002
        rows[column] += demo_rng.normal(0, scale, size=count)
    return rows


training_demos = make_demo_fraud(training_fraud, 150)
X_train = pd.concat([X_train.reset_index(drop=True), training_demos], ignore_index=True)
y_train = pd.concat([y_train.reset_index(drop=True), pd.Series(np.ones(len(training_demos), dtype=int))], ignore_index=True)

models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        ),
    ),
    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced",
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=15,
        max_depth=8,
        min_samples_leaf=6,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    ),
}

try:
    from xgboost import XGBClassifier
except ModuleNotFoundError as exc:
    if exc.name != "xgboost":
        raise
    xgboost_unavailable_reason = "XGBoost is not installed."
else:
    positive_count = int(y_train.sum())
    negative_count = len(y_train) - positive_count
    models["XGBoost"] = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=negative_count / max(positive_count, 1),
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
    )
    xgboost_unavailable_reason = None

model_training_seconds = {}
for name, classifier in models.items():
    training_started = time.perf_counter()
    classifier.fit(X_train, y_train)
    model_training_seconds[name] = time.perf_counter() - training_started

model = models["Random Forest"]

normal = X_train.loc[y_train == 0]
ranges = {column: (normal[column].quantile(0.01), normal[column].quantile(0.99))
          for column in features}

isolation_model = IsolationForest(
    n_estimators=100,
    contamination="auto",
    n_jobs=-1,
    random_state=42,
)
isolation_training_started = time.perf_counter()
isolation_model.fit(X_train.loc[y_train == 0])
isolation_training_seconds = time.perf_counter() - isolation_training_started
normal_anomaly_scores = np.sort(isolation_model.decision_function(X_train.loc[y_train == 0]))


def isolation_risk_scores(rows):
    anomaly_scores = isolation_model.decision_function(rows)
    anomaly_ranks = np.searchsorted(normal_anomaly_scores, anomaly_scores, side="left")
    return 1.0 - anomaly_ranks / len(normal_anomaly_scores)


def score_components(row):
    random_forest_risk = float(model.predict_proba(row.to_frame().T)[0][1])
    anomaly_score = float(isolation_model.decision_function(row.to_frame().T)[0])
    anomaly_rank = np.searchsorted(normal_anomaly_scores, anomaly_score, side="left")
    isolation_risk = 1.0 - float(anomaly_rank) / len(normal_anomaly_scores)
    return random_forest_risk, anomaly_score, 0.6 * random_forest_risk + 0.4 * isolation_risk


def evaluate_model(risk_scores):
    predictions = (risk_scores >= 0.5).astype(int)
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        y_replay,
        predictions,
        labels=[0, 1],
    ).ravel()
    return {
        "accuracy": round(float(accuracy_score(y_replay, predictions) * 100), 2),
        "precision": round(float(precision_score(y_replay, predictions, zero_division=0) * 100), 2),
        "recall": round(float(recall_score(y_replay, predictions, zero_division=0) * 100), 2),
        "f1_score": round(float(f1_score(y_replay, predictions, zero_division=0) * 100), 2),
        "roc_auc": round(float(roc_auc_score(y_replay, risk_scores) * 100), 2),
        "pr_auc": round(float(average_precision_score(y_replay, risk_scores) * 100), 2),
        "confusion_matrix": [
            [int(true_negative), int(false_positive)],
            [int(false_negative), int(true_positive)],
        ],
        "false_positive_rate": round(
            float(false_positive / (false_positive + true_negative) * 100), 2
        ) if false_positive + true_negative else 0.0,
        "false_negative_rate": round(
            float(false_negative / (false_negative + true_positive) * 100), 2
        ) if false_negative + true_positive else 0.0,
    }


model_comparison = []
model_risks = {}
model_inference_seconds = {}
model_evaluation_seconds = {}
for name, classifier in models.items():
    inference_started = time.perf_counter()
    risks = classifier.predict_proba(X_replay)[:, 1]
    inference_seconds = time.perf_counter() - inference_started
    model_risks[name] = risks
    evaluation_started = time.perf_counter()
    metrics = evaluate_model(risks)
    model_evaluation_seconds[name] = time.perf_counter() - evaluation_started
    model_inference_seconds[name] = inference_seconds
    model_comparison.append({
        "model": name,
        "metrics": metrics,
        "system_performance": {
            "training_seconds": round(model_training_seconds[name], 4),
            "inference_seconds": round(inference_seconds, 4),
            "test_transactions_per_second": round(
                len(X_replay) / max(inference_seconds, 1e-9), 2
            ),
        },
    })

if xgboost_unavailable_reason:
    model_comparison.append({
        "model": "XGBoost",
        "unavailable_reason": xgboost_unavailable_reason,
    })

isolation_inference_started = time.perf_counter()
isolation_risks = isolation_risk_scores(X_replay)
isolation_inference_seconds = time.perf_counter() - isolation_inference_started
model_risks["Isolation Forest"] = isolation_risks
isolation_evaluation_started = time.perf_counter()
model_comparison.append({
    "model": "Isolation Forest",
    "metrics": evaluate_model(isolation_risks),
    "system_performance": {
        "training_seconds": round(isolation_training_seconds, 4),
        "inference_seconds": round(isolation_inference_seconds, 4),
        "test_transactions_per_second": round(
            len(X_replay) / max(isolation_inference_seconds, 1e-9), 2
        ),
    },
})
model_evaluation_seconds["Isolation Forest"] = time.perf_counter() - isolation_evaluation_started

replay_risks = 0.6 * model_risks["Random Forest"] + 0.4 * isolation_risks
hybrid_evaluation_started = time.perf_counter()
hybrid_metrics = evaluate_model(replay_risks)
hybrid_evaluation_seconds = time.perf_counter() - hybrid_evaluation_started
hybrid_inference_seconds = (
    model_inference_seconds["Random Forest"] + isolation_inference_seconds
)
model_comparison.append({
    "model": "Hybrid RF + Isolation Forest",
    "metrics": hybrid_metrics,
    "system_performance": {
        "training_seconds": round(
            model_training_seconds["Random Forest"] + isolation_training_seconds, 4
        ),
        "inference_seconds": round(hybrid_inference_seconds, 4),
        "test_transactions_per_second": round(
            len(X_replay) / max(hybrid_inference_seconds, 1e-9), 2
        ),
    },
})
model_evaluation_seconds["Hybrid RF + Isolation Forest"] = hybrid_evaluation_seconds
model_metrics = {
    metric: hybrid_metrics[metric]
    for metric in ("accuracy", "precision", "recall", "f1_score")
}
demo_replay = make_demo_fraud(training_fraud, demo_count)
demo_start = len(X_replay)
X_replay = pd.concat([X_replay.reset_index(drop=True), demo_replay], ignore_index=True)
demo_indices = set(range(demo_start, len(X_replay)))
scenario_start = len(X_replay)
scenario_sources = X_train.loc[y_train == 0].reset_index(drop=True)
scenario_source_positions = [0, 1, 2, 3, 4, 5]
if len(scenario_sources) < len(scenario_source_positions):
    raise ValueError("At least six legitimate training rows are required for rule scenario examples.")
scenario_rows = scenario_sources.iloc[scenario_source_positions].copy().reset_index(drop=True)
dataset_amount_average = dataset_amount_average if dataset_amount_average is not None else 0.0
scenario_amounts = [
    24.50,
    2.49,
    max(2 * dataset_amount_average + 25.0, 250.0),
    36.00,
    3.25,
    max(2 * dataset_amount_average + 45.0, 300.0),
]
if "Amount" in scenario_rows.columns:
    scenario_rows["Amount"] = scenario_amounts
if "Time" in scenario_rows.columns:
    latest_dataset_time = float(X["Time"].max()) if "Time" in X.columns else 0.0
    scenario_rows["Time"] = latest_dataset_time + np.arange(1, scenario_count + 1) * 60
SCENARIO_FIXTURES = [
    {
        "id": "SCENARIO-TOR-001",
        "name": "Tor exit and active proxy",
        "signals": {"tor_or_vpn": True, "card_testing": False, "impossible_travel": False},
    },
    {
        "id": "SCENARIO-CARD-001",
        "name": "Four micro-transactions from one simulated device",
        "signals": {
            "tor_or_vpn": False,
            "card_testing": True,
            "transactions_in_five_minutes": 4,
            "device_ip_velocity_per_hour": 4,
            "impossible_travel": False,
        },
    },
    {
        "id": "SCENARIO-TRAVEL-001",
        "name": "High amount and impossible simulated travel",
        "signals": {
            "tor_or_vpn": False,
            "card_testing": False,
            "impossible_travel": True,
            "card_country": "US",
            "ip_country": "GB",
            "travel_speed_kmh": 920,
        },
    },
    {
        "id": "SCENARIO-CLEAN-001",
        "name": "Clean control transaction",
        "signals": {"tor_or_vpn": False, "card_testing": False, "impossible_travel": False},
    },
    {
        "id": "SCENARIO-CARD-CONTROL-001",
        "name": "Low amount without rapid velocity",
        "signals": {"tor_or_vpn": False, "card_testing": False, "impossible_travel": False},
    },
    {
        "id": "SCENARIO-TRAVEL-CONTROL-001",
        "name": "High amount without country mismatch",
        "signals": {"tor_or_vpn": False, "card_testing": False, "impossible_travel": False},
    },
]
X_replay = pd.concat([X_replay, scenario_rows], ignore_index=True)
scenario_indices = set(range(scenario_start, scenario_start + scenario_count))
scenario_by_index = {
    scenario_start + position: {
        **fixture,
        "scenario_signals": fixture["signals"],
    }
    for position, fixture in enumerate(SCENARIO_FIXTURES)
}
scenario_id_to_index = {
    fixture["id"]: scenario_start + position
    for position, fixture in enumerate(SCENARIO_FIXTURES)
}
order = np.concatenate([
    np.arange(scenario_start, scenario_start + scenario_count),
    np.arange(demo_start, scenario_start),
    demo_rng.permutation(demo_start),
])
cursor = 0
lock = Lock()
dataset_total = int(len(df))
dataset_fraud = int(df["Class"].sum())
dataset_legit = dataset_total - dataset_fraud
DEFAULT_DECISION_POLICY = {"approve_max": 30.0, "challenge_max": 70.0}
FRAUD_RULE_REFERENCES = {
    "RULE_TOR_DATACENTER_BLOCK": {
        "interceptions": 3842,
        "estimated_savings_usd": 489000,
        "reported_fpr_percent": 0.18,
    },
    "RULE_CARDING_RAPID_MICRO_TX": {
        "interceptions": 9140,
        "estimated_savings_usd": 1120000,
        "reported_fpr_percent": 0.05,
    },
    "RULE_VELOCITY_IMPOSSIBLE_TRAVEL": {
        "interceptions": 1420,
        "estimated_savings_usd": 311000,
        "reported_fpr_percent": 0.82,
    },
}

FRAUD_RULE_DEFINITIONS = [
    {
        "id": "RULE_TOR_DATACENTER_BLOCK",
        "name": "Block Anonymized Tor & VPN Exits",
        "severity": "CRITICAL",
        "action": "BLOCKED",
        "description": "Block transactions connecting through active Tor exits or high-risk commercial proxy infrastructure.",
        "conditions": ["Origin is Tor Network Exit", "Proxy or VPN active"],
        "required_fields": ["source_ip", "tor_exit_flag", "proxy_or_vpn_flag"],
        "dataset_proxy": None,
    },
    {
        "id": "RULE_CARDING_RAPID_MICRO_TX",
        "name": "Card Testing Botnet Velocity Breaker",
        "severity": "CRITICAL",
        "action": "BLOCKED",
        "description": "Detect rapid small-value transaction bursts that may indicate automated card testing.",
        "conditions": ["Order amount < $5.00", "More than 3 transactions in 5 minutes", "Device/IP velocity > 3 per hour"],
        "required_fields": ["Amount", "Time", "card_id_or_device_id_or_ip"],
        "dataset_proxy": "Amount and Time support a global burst proxy only; the CSV has no card, device, or IP identifier.",
    },
    {
        "id": "RULE_VELOCITY_IMPOSSIBLE_TRAVEL",
        "name": "Impossible Physical Travel Anomaly",
        "severity": "HIGH",
        "action": "IN_REVIEW",
        "description": "Review transactions with an unusually large amount and location changes impossible for physical travel.",
        "conditions": ["Amount > 200% of dataset average", "Card vs IP country differs", "Travel speed > 800 km/h"],
        "required_fields": ["Amount", "Time", "card_country", "ip_country", "geographic_coordinates", "card_or_session_id"],
        "dataset_proxy": "The amount threshold can be checked; the anonymized CSV has no country, coordinates, card, or session identity.",
    },
]


def decision_for_risk(risk_percent, policy):
    if risk_percent < policy["approve_max"]:
        return "Approved"
    if risk_percent < policy["challenge_max"]:
        return "Challenged 3DS"
    return "Blocked"


def get_decision_policy(connection):
    rows = connection.execute(
        "SELECT rule_key, numeric_value FROM decision_rules"
    ).fetchall()
    policy = DEFAULT_DECISION_POLICY.copy()
    policy.update({row["rule_key"]: row["numeric_value"] for row in rows})
    return policy


@contextmanager
def database_connection():
    connection = sqlite3.connect(DATABASE_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def initialize_database():
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with database_connection() as connection:
        existing = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'transactions'"
        ).fetchone()
        legacy_rows = []
        has_legacy_table = False
        if existing:
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(transactions)").fetchall()
            }
            if "transaction_json" in columns:
                connection.execute("ALTER TABLE transactions RENAME TO transactions_legacy")
                has_legacy_table = True
                legacy_rows = connection.execute(
                    "SELECT transaction_json FROM transactions_legacy ORDER BY id"
                ).fetchall()

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS transactions (
                transaction_id TEXT PRIMARY KEY,
                amount REAL NOT NULL,
                transaction_timestamp TEXT,
                last_replayed_at TEXT,
                original_dataset_label INTEGER,
                random_forest_risk REAL,
                isolation_forest_anomaly_score REAL,
                hybrid_risk_score REAL NOT NULL,
                predicted_class INTEGER NOT NULL,
                risk_category TEXT NOT NULL,
                investigation_status TEXT NOT NULL DEFAULT 'Pending',
                reasons_json TEXT NOT NULL DEFAULT '[]',
                explanation_note TEXT NOT NULL DEFAULT '',
                source_row_index INTEGER,
                dataset_time_seconds REAL,
                rule_results_json TEXT NOT NULL DEFAULT '[]',
                scenario_name TEXT,
                scenario_signals_json TEXT NOT NULL DEFAULT '{}'
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS investigations (
                investigation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT NOT NULL REFERENCES transactions(transaction_id),
                review_status TEXT NOT NULL,
                investigator_notes TEXT NOT NULL DEFAULT '',
                reviewed_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS decision_rules (
                rule_key TEXT PRIMARY KEY,
                numeric_value REAL NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        for rule_key, numeric_value in DEFAULT_DECISION_POLICY.items():
            connection.execute(
                """
                INSERT OR IGNORE INTO decision_rules (rule_key, numeric_value, updated_at)
                VALUES (?, ?, ?)
                """,
                (rule_key, numeric_value, datetime.now(timezone.utc).isoformat(timespec="seconds")),
            )
        transaction_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(transactions)").fetchall()
        }
        if "decision_status" not in transaction_columns:
            connection.execute(
                "ALTER TABLE transactions ADD COLUMN decision_status TEXT NOT NULL DEFAULT 'Approved'"
            )
        if "source_row_index" not in transaction_columns:
            connection.execute("ALTER TABLE transactions ADD COLUMN source_row_index INTEGER")
        if "dataset_time_seconds" not in transaction_columns:
            connection.execute("ALTER TABLE transactions ADD COLUMN dataset_time_seconds REAL")
        if "rule_results_json" not in transaction_columns:
            connection.execute(
                "ALTER TABLE transactions ADD COLUMN rule_results_json TEXT NOT NULL DEFAULT '[]'"
            )
        if "scenario_name" not in transaction_columns:
            connection.execute("ALTER TABLE transactions ADD COLUMN scenario_name TEXT")
        if "scenario_signals_json" not in transaction_columns:
            connection.execute(
                "ALTER TABLE transactions ADD COLUMN scenario_signals_json TEXT NOT NULL DEFAULT '{}'"
            )
        if "last_replayed_at" not in transaction_columns:
            connection.execute(
                "ALTER TABLE transactions ADD COLUMN last_replayed_at TEXT"
            )
        policy = get_decision_policy(connection)
        stored_transactions = connection.execute(
            "SELECT transaction_id, hybrid_risk_score FROM transactions"
        ).fetchall()
        connection.executemany(
            "UPDATE transactions SET decision_status = ? WHERE transaction_id = ?",
            [
                (decision_for_risk(row["hybrid_risk_score"] * 100, policy), row["transaction_id"])
                for row in stored_transactions
            ],
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_transactions_category ON transactions(risk_category)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_transactions_predicted_class ON transactions(predicted_class)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_transactions_last_replayed ON transactions(last_replayed_at)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_investigations_transaction ON investigations(transaction_id, investigation_id)"
        )

        for legacy_row in legacy_rows:
            transaction = json.loads(legacy_row["transaction_json"])
            transaction_id = transaction["id"]
            suffix = transaction_id.removeprefix("DEMO-TX-").removeprefix("TX-")
            source_index = int(suffix) - 1
            is_demo = transaction_id.startswith("DEMO-")
            original_label = (
                int(y_replay.iloc[source_index])
                if not is_demo and 0 <= source_index < demo_start
                else None
            )
            hybrid_risk = float(transaction["risk"]) / 100
            connection.execute(
                """
                INSERT OR IGNORE INTO transactions (
                    transaction_id, amount, original_dataset_label,
                    hybrid_risk_score, predicted_class, risk_category,
                    decision_status, reasons_json, explanation_note
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    transaction_id,
                    float(transaction["amount"]),
                    original_label,
                    hybrid_risk,
                    int(hybrid_risk >= 0.5),
                    transaction["category"],
                    decision_for_risk(float(transaction["risk"]), policy),
                    json.dumps(transaction.get("reasons", [])),
                    transaction.get("explanation_note", ""),
                ),
            )
        if has_legacy_table:
            connection.execute("DROP TABLE transactions_legacy")
        for stored_row in connection.execute(
            "SELECT transaction_id, amount, source_row_index, dataset_time_seconds "
            "FROM transactions WHERE rule_results_json = '[]'"
        ).fetchall():
            source_row_index = stored_row["source_row_index"]
            if source_row_index is None and not stored_row["transaction_id"].startswith("DEMO-"):
                try:
                    replay_position = int(stored_row["transaction_id"].rsplit("-", 1)[1]) - 1
                except (IndexError, ValueError):
                    replay_position = -1
                if 0 <= replay_position < len(replay_source_indices):
                    source_row_index = int(replay_source_indices[replay_position])
            dataset_time_seconds = stored_row["dataset_time_seconds"]
            if dataset_time_seconds is None and source_row_index is not None and dataset_micro_times is not None:
                dataset_time_seconds = float(dataset_micro_times.loc[source_row_index])
            connection.execute(
                """
                UPDATE transactions
                SET source_row_index = ?, dataset_time_seconds = ?, rule_results_json = ?
                WHERE transaction_id = ?
                """,
                (
                    source_row_index,
                    dataset_time_seconds,
                    json.dumps(evaluate_dataset_rules(stored_row["amount"], source_row_index)),
                    stored_row["transaction_id"],
                ),
            )


initialize_database()


def transaction_record(row):
    return {
        "id": row["transaction_id"],
        "amount": row["amount"],
        "timestamp": row["transaction_timestamp"],
        "original_dataset_label": row["original_dataset_label"],
        "random_forest_risk": row["random_forest_risk"],
        "isolation_forest_anomaly_score": row["isolation_forest_anomaly_score"],
        "hybrid_risk_score": row["hybrid_risk_score"],
        "risk": round(row["hybrid_risk_score"] * 100, 1),
        "predicted_class": row["predicted_class"],
        "category": row["risk_category"],
        "decision_status": row["decision_status"],
        "source_row_index": row["source_row_index"],
        "reasons": json.loads(row["reasons_json"]),
        "rule_evaluations": json.loads(row["rule_results_json"]),
        "scenario_name": row["scenario_name"],
        "scenario_signals": json.loads(row["scenario_signals_json"]),
        "dataset_time_seconds": row["dataset_time_seconds"],
        "explanation_note": row["explanation_note"],
        "investigation_status": row["investigation_status"],
        "investigator_notes": row["investigator_notes"] or "",
        "reviewed_at": row["reviewed_at"],
    }


def database_statistics(connection):
    total = connection.execute(
        "SELECT COUNT(*) FROM transactions"
    ).fetchone()[0]
    fraud_alerts = connection.execute(
        "SELECT COUNT(*) FROM transactions WHERE predicted_class = 1"
    ).fetchone()[0]
    reviewed = connection.execute(
        "SELECT COUNT(*) FROM transactions WHERE investigation_status = 'Reviewed'"
    ).fetchone()[0]
    decision_counts = {
        row["decision_status"]: row["count"]
        for row in connection.execute(
            "SELECT decision_status, COUNT(*) AS count FROM transactions GROUP BY decision_status"
        ).fetchall()
    }
    in_review = connection.execute(
        "SELECT COUNT(*) FROM transactions WHERE investigation_status = 'In Review'"
    ).fetchone()[0]
    low_risk = connection.execute(
        "SELECT COUNT(*) FROM transactions WHERE risk_category = 'Low'"
    ).fetchone()[0]
    size_mb = DATABASE_PATH.stat().st_size / (1024 * 1024) if DATABASE_PATH.exists() else 0
    return {
        "database_records": total,
        "decision_counts": {
            "Approved": decision_counts.get("Approved", 0),
            "In Review": in_review,
            "Blocked": decision_counts.get("Blocked", 0),
            "Challenged 3DS": decision_counts.get("Challenged 3DS", 0),
            "Min risk": low_risk,
        },
        "fraud_alerts": fraud_alerts,
        "reviewed_transactions": reviewed,
        "database_size_mb": round(size_mb, 3),
    }


MODEL_DESCRIPTIONS = {
    "Logistic Regression": "Supervised linear classifier with standardized features and class balancing.",
    "Decision Tree": "Supervised tree classifier with class balancing.",
    "Random Forest": "Supervised ensemble of decision trees; its fraud probability supplies 60% of the hybrid risk.",
    "XGBoost": "Optional gradient-boosted tree classifier; included only when the xgboost package is installed.",
    "Isolation Forest": "Unsupervised anomaly detector fitted only on normal training rows; anomaly ranking supplies 40% of hybrid risk.",
    "Hybrid RF + Isolation Forest": "Weighted score: 0.60 × Random Forest fraud probability + 0.40 × Isolation Forest normal-reference anomaly percentile.",
}

REPORT_METHODOLOGY = [
    ("Dataset collection", "The application loads the project-provided creditcard.csv file; Class is the required binary target. The app does not independently verify the dataset's source or collection process."),
    ("Data preprocessing", "Feature columns are converted to numeric values. Rows are sampled reproducibly (up to 20,000; random_state=42), then missing feature values are imputed using training-set medians only, with zero fallback for all-missing training columns."),
    ("Fraud rule coverage", "Supplementary Tor/VPN, card-testing, and impossible-travel rules are compared with the supplied dataset's available fields. Amount and dataset Time support limited aggregate proxies; the CSV has no IP, card/device identity, geographic, country, or account-age fields. Proxy matches are not blocks, travel detections, or verified fraud classifications."),
    ("Training and testing", "One stratified 80/20 split (random_state=42) is shared by the supervised and anomaly evaluations. Synthetic fraud examples are generated from training fraud rows for supervised model training and are excluded from holdout evaluation."),
    ("Random Forest", "A class-weighted random forest is trained on the training partition with synthetic training fraud examples. It estimates fraud probability for each holdout row."),
    ("Isolation Forest", "Isolation Forest is fit using only non-fraud training rows. Its decision score is ranked against those normal-training scores; a lower normal percentile maps to a higher anomaly risk."),
    ("Hybrid scoring", "Hybrid risk = 0.60 × Random Forest fraud probability + 0.40 × Isolation Forest anomaly percentile. The 0.50 threshold converts risk scores to predicted classes for classification metrics."),
    ("Database architecture", "SQLite stores typed transaction records in transactions and review events/notes in investigations. Stable transaction IDs and INSERT OR IGNORE prevent duplicate rows when the same replay is requested again."),
    ("Model evaluation", "All available models are evaluated on the same stratified holdout rows. Threshold metrics use 0.50; ROC-AUC and average precision use continuous scores. Confusion matrices are TN/FP/FN/TP."),
]

REPORT_LIMITATIONS = [
    "The project-provided dataset's provenance and representativeness are not verified by the application.",
    "The reproducible model comparison uses at most 20,000 sampled dataset rows; results may vary with another sample, split, or dataset.",
    "Approval, block, and 3-D Secure outcomes are policy simulations; the source data contains no payment processor outcomes.",
    "The illustrative review SLA and repeated-feature groups are not real disputes or evidence of linked people or merchants.",
    "Synthetic training fraud examples are generated from labeled training fraud rows and may affect supervised-model behavior.",
    "Isolation Forest anomaly percentiles are not calibrated fraud probabilities; the hybrid score is a heuristic, not a financial decision.",
    "System timings depend on the host machine, dependency versions, and current system load.",
    "SQLite history on ephemeral hosting storage may be lost on redeploy unless persistent storage is configured.",
]

REPORT_FUTURE_WORK = [
    "Validate on time-based and external datasets and monitor drift.",
    "Tune thresholds and model hyperparameters using a separate validation set.",
    "Calibrate probabilities and assess precision-recall trade-offs with domain reviewers.",
    "Add authenticated investigator accounts, audit controls, and a managed persistent database for production.",
]


def build_report_data():
    with database_connection() as connection:
        db_stats = database_statistics(connection)

    fraud_count = int((y == 1).sum())
    dataset_bytes = DATA_PATH.stat().st_size
    available_models = [
        {
            "name": item["model"],
            "description": MODEL_DESCRIPTIONS[item["model"]],
            "metrics": item.get("metrics"),
            "system_performance": item.get("system_performance"),
            "unavailable_reason": item.get("unavailable_reason"),
            "evaluation_seconds": round(
                model_evaluation_seconds.get(item["model"], 0.0), 4
            ),
        }
        for item in model_comparison
    ]
    supervised_training_seconds = sum(model_training_seconds.values())
    system_performance = {
        "dataset_load_seconds": round(dataset_load_seconds, 4),
        "supervised_training_seconds": round(supervised_training_seconds, 4),
        "isolation_forest_training_seconds": round(isolation_training_seconds, 4),
        "holdout_rows": len(X_replay) - demo_count,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
    }
    return {
        "project_name": "Banking Fraud Detection",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "file": DATA_PATH.name,
            "description": "Project-provided labeled transaction dataset; Class is the binary fraud label.",
            "rows": len(df),
            "feature_count": len(features),
            "feature_columns": ", ".join(features),
            "fraud_rows": fraud_count,
            "legitimate_rows": int(len(y) - fraud_count),
            "fraud_rate_percent": round(fraud_count / len(y) * 100, 4) if len(y) else 0,
            "missing_feature_cells": int(X.isna().sum().sum()),
            "missing_target_rows": int(y.isna().sum()),
            "file_size_mb": round(dataset_bytes / (1024 * 1024), 3),
            "sampled_rows": len(X_sample),
            "training_rows": len(X_train),
            "holdout_rows": len(y_replay),
        },
        "methodology": REPORT_METHODOLOGY,
        "models": available_models,
        "hybrid_results": hybrid_metrics,
        "system_performance": system_performance,
        "database": db_stats,
        "limitations": REPORT_LIMITATIONS,
        "future_improvements": REPORT_FUTURE_WORK,
    }


def report_csv(report):
    output = io.StringIO(newline="")
    columns = [
        "section", "item", "value", "details",
        "accuracy_percent", "precision_percent", "recall_percent", "f1_score_percent",
        "roc_auc_percent", "pr_auc_percent", "confusion_matrix_tn_fp_fn_tp",
        "false_positive_rate_percent", "false_negative_rate_percent",
        "training_seconds", "inference_seconds", "test_transactions_per_second",
        "evaluation_seconds",
    ]
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()

    def write(section, item, value="", details=""):
        writer.writerow({"section": section, "item": item, "value": value, "details": details})

    write("Project", "Project name", report["project_name"])
    write("Report", "Generated at UTC", report["generated_at"])
    dataset = report["dataset"]
    for key, value in dataset.items():
        write("Dataset", key, value)
    for item, description in report["methodology"]:
        write("Methodology", item, description)
    for item in report["models"]:
        metrics = item["metrics"]
        performance = item["system_performance"] or {}
        row = {
            "section": "Model comparison",
            "item": item["name"],
            "details": item["unavailable_reason"] or item["description"],
            "training_seconds": performance.get("training_seconds", ""),
            "inference_seconds": performance.get("inference_seconds", ""),
            "test_transactions_per_second": performance.get("test_transactions_per_second", ""),
            "evaluation_seconds": item["evaluation_seconds"],
        }
        if metrics:
            row.update({
                "accuracy_percent": metrics["accuracy"],
                "precision_percent": metrics["precision"],
                "recall_percent": metrics["recall"],
                "f1_score_percent": metrics["f1_score"],
                "roc_auc_percent": metrics["roc_auc"],
                "pr_auc_percent": metrics["pr_auc"],
                "confusion_matrix_tn_fp_fn_tp": "/".join(
                    str(value) for matrix_row in metrics["confusion_matrix"] for value in matrix_row
                ),
                "false_positive_rate_percent": metrics["false_positive_rate"],
                "false_negative_rate_percent": metrics["false_negative_rate"],
            })
        writer.writerow(row)
    for key, value in report["system_performance"].items():
        write("System performance", key, value)
    for key, value in report["database"].items():
        write("Database", key, value)
    for item in report["limitations"]:
        write("Limitations", item)
    for item in report["future_improvements"]:
        write("Future improvements", item)
    return output.getvalue()


def report_pdf(report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from textwrap import fill

    output = io.BytesIO()
    with PdfPages(output) as pdf:
        def text_page(title, lines):
            def create_page(continued=False):
                page = plt.figure(figsize=(8.27, 11.69))
                page_title = f"{title} (continued)" if continued else title
                page.text(0.08, 0.95, page_title, fontsize=18, weight="bold", va="top")
                return page

            figure = create_page()
            y_position = 0.90
            for heading, body in lines:
                wrapped = fill(str(body), width=95)
                required_space = 0.025 * (wrapped.count("\n") + 2) + 0.035
                if y_position - required_space < 0.06:
                    pdf.savefig(figure, bbox_inches="tight")
                    plt.close(figure)
                    figure = create_page(continued=True)
                    y_position = 0.90
                figure.text(0.08, y_position, str(heading), fontsize=11, weight="bold", va="top")
                y_position -= 0.025
                figure.text(0.08, y_position, wrapped, fontsize=9, va="top", linespacing=1.4)
                y_position -= 0.025 * (wrapped.count("\n") + 1) + 0.018
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)

        dataset = report["dataset"]
        text_page("Banking Fraud Detection — Model Report", [
            ("Generated (UTC)", report["generated_at"]),
            ("Project", report["project_name"]),
            ("Dataset", dataset["description"]),
            ("Dataset rows / features", f"{dataset['rows']} rows; {dataset['feature_count']} features; {dataset['file_size_mb']} MB."),
            ("Class balance", f"{dataset['fraud_rows']} fraud ({dataset['fraud_rate_percent']}%) and {dataset['legitimate_rows']} legitimate rows."),
            ("Missing values", f"{dataset['missing_feature_cells']} missing feature cells; {dataset['missing_target_rows']} missing target rows."),
            ("Training and holdout", f"{dataset['sampled_rows']} sampled rows; {dataset['training_rows']} training rows including generated training examples; {dataset['holdout_rows']} held-out rows."),
            ("Hybrid formula", report["models"][[model["name"] for model in report["models"]].index("Hybrid RF + Isolation Forest")]["description"]),
            ("Database", f"{report['database']['database_records']} transaction records; {report['database']['fraud_alerts']} fraud alerts; {report['database']['reviewed_transactions']} reviewed; {report['database']['database_size_mb']} MB."),
        ])
        text_page("Model Descriptions", [
            (item["name"], item["description"] if item["metrics"] else item["unavailable_reason"])
            for item in report["models"]
        ])

        figure, axis = plt.subplots(figsize=(11.69, 8.27))
        axis.axis("off")
        axis.set_title("Held-out Model Comparison", fontsize=15, weight="bold", pad=18)
        headers = ["Model", "Acc.", "Prec.", "Recall", "F1", "ROC-AUC", "PR-AUC", "TN/FP/FN/TP", "FPR", "FNR"]
        rows = []
        for item in report["models"]:
            metrics = item["metrics"]
            if not metrics:
                rows.append([item["name"], item["unavailable_reason"], "", "", "", "", "", "", "", ""])
                continue
            matrix = "/".join(str(value) for line in metrics["confusion_matrix"] for value in line)
            rows.append([
                item["name"], f"{metrics['accuracy']:.2f}", f"{metrics['precision']:.2f}",
                f"{metrics['recall']:.2f}", f"{metrics['f1_score']:.2f}",
                f"{metrics['roc_auc']:.2f}", f"{metrics['pr_auc']:.2f}", matrix,
                f"{metrics['false_positive_rate']:.2f}", f"{metrics['false_negative_rate']:.2f}",
            ])
        table = axis.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1, 1.8)
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)

        system_lines = [
            ("Dataset load seconds", report["system_performance"]["dataset_load_seconds"]),
            ("Supervised model training seconds", report["system_performance"]["supervised_training_seconds"]),
            ("Isolation Forest training seconds", report["system_performance"]["isolation_forest_training_seconds"]),
            ("Python / platform", f"{report['system_performance']['python_version']} / {report['system_performance']['platform']}"),
            ("CPU count", report["system_performance"]["cpu_count"]),
        ]
        system_lines.extend(
            (f"{item['name']} performance",
             f"Training: {item['system_performance']['training_seconds']} s; "
             f"holdout inference: {item['system_performance']['inference_seconds']} s; "
             f"throughput: {item['system_performance']['test_transactions_per_second']} rows/s; "
             f"metric evaluation: {item['evaluation_seconds']} s.")
            for item in report["models"] if item["system_performance"]
        )
        text_page("System Performance", system_lines)
        text_page("Methodology, Limitations and Future Work", [
            *report["methodology"],
            ("Limitations", "\n".join(f"- {item}" for item in report["limitations"])),
            ("Future improvements", "\n".join(f"- {item}" for item in report["future_improvements"])),
        ])

    output.seek(0)
    return output


def category(probability):
    if probability >= 0.70:
        return "High"
    if probability >= 0.30:
        return "Medium"
    return "Low"


def score(index):
    row = X_replay.iloc[index]
    scenario = scenario_by_index.get(index)
    source_row_index = (
        int(replay_source_indices[index]) if index < demo_start else None
    )
    random_forest_risk, anomaly_score, probability = score_components(row)
    level = category(probability)

    unusual_features = []
    for column in features:
        low, high = ranges[column]
        value = float(row[column])
        width = max(float(high - low), 1e-9)
        if value < low:
            deviation = (float(low) - value) / width
            direction = "below the 1st-percentile value"
        elif value > high:
            deviation = (value - float(high)) / width
            direction = "above the 99th-percentile value"
        else:
            continue

        if column == "Amount":
            description = f"Amount ${value:,.2f} is {direction} seen in normal training transactions."
        else:
            description = f"{column} is {direction} seen in normal training transactions."
        unusual_features.append((deviation, description))

    unusual_features.sort(key=lambda item: item[0], reverse=True)
    reasons = [description for _, description in unusual_features[:2]]
    if not reasons:
        reasons.append(
            "No individual feature is outside the normal training range; the score reflects the combined feature pattern."
            if probability >= 0.30
            else "No strong risk signal or individual feature outside the normal training range was detected."
        )

    return {
        "id": (
            scenario["id"] if scenario
            else f"{'DEMO-' if index in demo_indices else ''}TX-{index + 1:07d}"
        ),
        "scenario_name": scenario["name"] if scenario else None,
        "scenario_signals": scenario["scenario_signals"] if scenario else {},
        "amount": round(float(row.get("Amount", 0)), 2),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_row_index": source_row_index,
        "dataset_time_seconds": (
            float(row["Time"])
            if source_row_index is not None and "Time" in row.index
            else None
        ),
        "original_dataset_label": (
            int(y_replay.iloc[index]) if index < demo_start else None
        ),
        "random_forest_risk": random_forest_risk,
        "isolation_forest_anomaly_score": anomaly_score,
        "risk": round(probability * 100, 1),
        "predicted_class": int(probability >= 0.5),
        "category": level,
        "rule_evaluations": evaluate_dataset_rules(
            float(row.get("Amount", 0)),
            source_row_index,
            scenario["scenario_signals"] if scenario else None,
        ),
        "reasons": (["Synthetic demo fraud transaction based on labeled dataset patterns."] + reasons
                    if index in demo_indices else reasons),
        "explanation_note": (
            "Synthetic rule scenario built from a legitimate dataset feature row; external signals are fictional test inputs."
            if scenario
            else "These are unusual-value indicators, not proof of fraud or causal explanations."
        ),
    }


@app.get("/")
def dashboard():
    return render_template("index.html", active_page="overview")


@app.get("/live-stream")
def live_stream_page():
    return render_template("live_stream.html", active_page="live-stream")


@app.get("/transaction-history")
def transaction_history_page():
    return render_template("transaction_history.html", active_page="live-stream")


@app.get("/rules")
def rules_page():
    return render_template("rules.html", active_page="rules")


@app.get("/sandbox")
def sandbox_page():
    return render_template("sandbox.html", active_page="sandbox")


@app.get("/disputes-sla")
def disputes_sla_page():
    return render_template("disputes_sla.html", active_page="disputes-sla")


@app.get("/ai-syndicatehunter")
def syndicate_hunter_page():
    return render_template("syndicate_hunter.html", active_page="ai-syndicatehunter")


@app.get("/methodology")
def methodology():
    return render_template(
        "methodology.html",
        active_page="methodology",
        methodology=REPORT_METHODOLOGY,
        limitations=REPORT_LIMITATIONS,
        future_improvements=REPORT_FUTURE_WORK,
    )


@app.get("/reports/csv")
def download_csv_report():
    report = build_report_data()
    content = report_csv(report)
    return send_file(
        io.BytesIO(content.encode("utf-8-sig")),
        mimetype="text/csv; charset=utf-8",
        as_attachment=True,
        download_name="banking-fraud-detection-report.csv",
    )


@app.get("/reports/pdf")
def download_pdf_report():
    report = build_report_data()
    content = report_pdf(report)
    return send_file(
        content,
        mimetype="application/pdf",
        as_attachment=True,
        download_name="banking-fraud-detection-report.pdf",
    )


@app.get("/static/style.css")
def stylesheet():
    return send_from_directory(BASE_DIR / "static", "style.css", mimetype="text/css")


@app.get("/api/stream")
def stream():
    global cursor
    batch = []

    with lock:
        for _ in range(3):
            if cursor >= len(order):
                cursor = 0
            batch.append(int(order[cursor]))
            cursor += 1

    transactions = [score(index) for index in batch]

    with lock:
        with database_connection() as connection:
            policy = get_decision_policy(connection)
            connection.executemany(
                """
                INSERT INTO transactions (
                    transaction_id, amount, transaction_timestamp, last_replayed_at,
                    original_dataset_label, random_forest_risk,
                    isolation_forest_anomaly_score, hybrid_risk_score,
                    predicted_class, risk_category, decision_status, reasons_json,
                    explanation_note, source_row_index, dataset_time_seconds,
                    rule_results_json, scenario_name, scenario_signals_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(transaction_id) DO UPDATE SET
                    last_replayed_at = excluded.last_replayed_at
                """,
                [
                    (
                        transaction["id"],
                        transaction["amount"],
                        transaction["timestamp"],
                        transaction["timestamp"],
                        transaction["original_dataset_label"],
                        transaction["random_forest_risk"],
                        transaction["isolation_forest_anomaly_score"],
                        transaction["risk"] / 100,
                        transaction["predicted_class"],
                        transaction["category"],
                        decision_for_risk(transaction["risk"], policy),
                        json.dumps(transaction["reasons"]),
                        transaction["explanation_note"],
                        transaction["source_row_index"],
                        transaction["dataset_time_seconds"],
                        json.dumps(transaction["rule_evaluations"]),
                        transaction["scenario_name"],
                        json.dumps(transaction["scenario_signals"]),
                    )
                    for transaction in transactions
                ],
            )

    return jsonify({"transactions": transactions})


@app.get("/api/history")
def transaction_history():
    offset = max(request.args.get("offset", default=0, type=int), 0)
    limit = min(max(request.args.get("limit", default=25, type=int), 1), 100)
    risk_category = request.args.get("category", default="").strip()
    decision_status = request.args.get("decision", default="").strip()
    investigation_status = request.args.get("investigation_status", default="").strip()
    dataset_label = request.args.get("dataset_label", default="").strip()
    search = request.args.get("search", default="").strip()[:100]
    if risk_category and risk_category not in {"Low", "Medium", "High"}:
        return jsonify({"error": "category must be Low, Medium, or High."}), 400
    if decision_status and decision_status not in {"Approved", "Blocked", "Challenged 3DS"}:
        return jsonify({"error": "decision must be Approved, Blocked, or Challenged 3DS."}), 400
    if investigation_status and investigation_status not in {"Pending", "In Review", "Reviewed"}:
        return jsonify({"error": "investigation_status must be Pending, In Review, or Reviewed."}), 400
    if dataset_label and dataset_label not in {"0", "1"}:
        return jsonify({"error": "dataset_label must be 0 (not fraud) or 1 (fraud)."}), 400
    conditions = []
    parameters = []
    if risk_category:
        conditions.append("t.risk_category = ?")
        parameters.append(risk_category)
    if decision_status:
        conditions.append("t.decision_status = ?")
        parameters.append(decision_status)
    if investigation_status:
        conditions.append(
            "COALESCE((SELECT latest.review_status FROM investigations latest "
            "WHERE latest.transaction_id = t.transaction_id "
            "ORDER BY latest.investigation_id DESC LIMIT 1), t.investigation_status) = ?"
        )
        parameters.append(investigation_status)
    if dataset_label:
        conditions.append("t.original_dataset_label = ?")
        parameters.append(int(dataset_label))
    if search:
        conditions.append("(t.transaction_id LIKE ? OR CAST(t.amount AS TEXT) LIKE ?)")
        parameters.extend((f"%{search}%", f"%{search}%"))
    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    with database_connection() as connection:
        total = connection.execute(
            f"SELECT COUNT(*) FROM transactions t {where_clause}", parameters
        ).fetchone()[0]
        rows = connection.execute(
            f"""
            SELECT t.*,
                   i.review_status AS latest_review_status,
                   i.investigator_notes,
                   i.reviewed_at
            FROM transactions t
            LEFT JOIN investigations i
              ON i.investigation_id = (
                  SELECT MAX(latest.investigation_id)
                  FROM investigations latest
                  WHERE latest.transaction_id = t.transaction_id
              )
            {where_clause}
            ORDER BY COALESCE(t.last_replayed_at, t.transaction_timestamp) DESC,
                     t.rowid DESC
            LIMIT ? OFFSET ?
            """,
            (*parameters, limit, offset),
        ).fetchall()
    return jsonify({
        "transactions": [
            {
                **transaction_record(row),
                "investigation_status": row["latest_review_status"] or row["investigation_status"],
            }
            for row in rows
        ],
        "total": total,
        "offset": offset,
        "limit": limit,
    })


@app.get("/api/rules")
def decision_rules():
    with database_connection() as connection:
        policy = get_decision_policy(connection)
    return jsonify({
        "policy": policy,
        "rules": [
            {"outcome": "Approved", "condition": f"Hybrid risk < {policy['approve_max']:.1f}%"},
            {
                "outcome": "Challenged 3DS",
                "condition": f"{policy['approve_max']:.1f}% <= hybrid risk < {policy['challenge_max']:.1f}%",
            },
            {"outcome": "Blocked", "condition": f"Hybrid risk >= {policy['challenge_max']:.1f}%"},
            {"outcome": "In Review", "condition": "Manual investigation status; tracked separately"},
        ],
    })


@app.get("/api/fraud-rules")
def fraud_rules():
    dataset_fields = list(df.columns)
    rules = []
    for definition in FRAUD_RULE_DEFINITIONS:
        rule = {
            **definition,
            "missing_fields": [
                field for field in definition["required_fields"]
                if field not in dataset_fields
            ],
            "reference_metrics": FRAUD_RULE_REFERENCES[definition["id"]],
            "reference_metrics_note": (
                "Figures supplied in the request; not measured or verified by this dataset."
            ),
        }
        if definition["id"] == "RULE_CARDING_RAPID_MICRO_TX":
            rule["dataset_measurement"] = {
                **dataset_proxy_metrics,
                "measurement_scope": "Whole labeled CSV, global Time window; not grouped by card, device, or IP.",
            }
        elif definition["id"] == "RULE_VELOCITY_IMPOSSIBLE_TRAVEL":
            rule["dataset_measurement"] = {
                "amount_average": round(dataset_amount_average, 2)
                if dataset_amount_average is not None else None,
                "amount_threshold": round(2 * dataset_amount_average, 2)
                if dataset_amount_average is not None else None,
                "proxy_candidates": dataset_proxy_metrics.get(
                    "amount_over_twice_average_candidates", 0
                ),
                "proxy_fraud_labels": dataset_proxy_metrics.get(
                    "amount_over_twice_average_fraud_labels", 0
                ),
                "proxy_false_positive_rate_percent": dataset_proxy_metrics.get(
                    "amount_over_twice_average_false_positive_rate_percent"
                ),
                "measurement_scope": "Amount-only proxy across the whole CSV; no location or travel inference.",
            }
        else:
            rule["dataset_measurement"] = None
        rules.append(rule)
    return jsonify({
        "dataset": {
            "rows": int(len(df)),
            "columns": dataset_fields,
            "available_fields": [field for field in ("Amount", "Time", "Class") if field in dataset_fields],
            "missing_rule_signals": [
                field for field in ("source_ip", "tor_exit_flag", "proxy_or_vpn_flag",
                                    "card_id", "device_id", "ip_address",
                                    "card_country", "ip_country", "latitude",
                                    "longitude", "account_id")
                if field not in dataset_fields
            ],
        },
        "rules": rules,
        "classification_note": (
            "These rule signals are supplementary candidates. They do not overwrite the CSV's Class label "
            "or the model decision, and partial proxies do not trigger a block or review."
        ),
    })


@app.post("/api/rules")
def update_decision_rules():
    payload = request.get_json()
    if not isinstance(payload, dict):
        return jsonify({"error": "Expected a JSON object."}), 400
    try:
        approve_max = float(payload["approve_max"])
        challenge_max = float(payload["challenge_max"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Provide numeric approve_max and challenge_max values."}), 400
    if (
        not np.isfinite(approve_max)
        or not np.isfinite(challenge_max)
        or not 0 <= approve_max < challenge_max <= 100
    ):
        return jsonify({
            "error": "Thresholds must satisfy 0 <= approve_max < challenge_max <= 100."
        }), 400

    updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    policy = {"approve_max": approve_max, "challenge_max": challenge_max}
    with database_connection() as connection:
        connection.executemany(
            """
            INSERT INTO decision_rules (rule_key, numeric_value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(rule_key) DO UPDATE SET
                numeric_value = excluded.numeric_value,
                updated_at = excluded.updated_at
            """,
            [
                (rule_key, value, updated_at)
                for rule_key, value in policy.items()
            ],
        )
        rows = connection.execute(
            "SELECT transaction_id, hybrid_risk_score FROM transactions"
        ).fetchall()
        connection.executemany(
            "UPDATE transactions SET decision_status = ? WHERE transaction_id = ?",
            [
                (decision_for_risk(row["hybrid_risk_score"] * 100, policy), row["transaction_id"])
                for row in rows
            ],
        )
    return jsonify({"policy": policy, "updated_at": updated_at})


@app.post("/api/sandbox")
def sandbox_decision():
    payload = request.get_json()
    if not isinstance(payload, dict):
        return jsonify({"error": "Expected a JSON object."}), 400
    try:
        risk = float(payload["risk"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Provide a numeric risk score from 0 to 100."}), 400
    if not np.isfinite(risk) or not 0 <= risk <= 100:
        return jsonify({"error": "Risk score must be between 0 and 100."}), 400
    with database_connection() as connection:
        policy = get_decision_policy(connection)
    return jsonify({
        "risk": risk,
        "decision": decision_for_risk(risk, policy),
        "policy": policy,
        "simulation": True,
    })


@app.get("/api/disputes-sla")
def disputes_sla_data():
    with database_connection() as connection:
        total = connection.execute(
            "SELECT COUNT(*) FROM transactions WHERE original_dataset_label = 1"
        ).fetchone()[0]
        reviewed = connection.execute(
            """
            SELECT COUNT(*) FROM transactions
            WHERE original_dataset_label = 1 AND investigation_status = 'Reviewed'
            """
        ).fetchone()[0]
        rows = connection.execute(
            """
            SELECT transaction_id, amount, transaction_timestamp, hybrid_risk_score,
                   risk_category, decision_status, investigation_status
            FROM transactions
            WHERE original_dataset_label = 1
            ORDER BY rowid DESC
            LIMIT 100
            """
        ).fetchall()
    now = datetime.now(timezone.utc)
    queue = []
    for row in rows:
        created_at_value = row["transaction_timestamp"]
        created_at = datetime.fromisoformat(created_at_value) if created_at_value else now
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        age_hours = max((now - created_at).total_seconds() / 3600, 0)
        queue.append({
            "id": row["transaction_id"],
            "amount": row["amount"],
            "risk": round(row["hybrid_risk_score"] * 100, 1),
            "category": row["risk_category"],
            "decision": row["decision_status"],
            "investigation_status": row["investigation_status"],
            "age_hours": round(age_hours, 2),
            "sla_remaining_hours": round(max(4 - age_hours, 0), 2),
            "sla_breached": age_hours >= 4 and row["investigation_status"] != "Reviewed",
        })
    return jsonify({
        "total": total,
        "reviewed": reviewed,
        "open": total - reviewed,
        "sla_hours": 4,
        "transactions": queue,
        "note": "Illustrative 4-hour review SLA for replayed dataset-labeled fraud; not real chargeback or dispute data.",
    })


@app.get("/api/ai-syndicatehunter")
def syndicate_hunter_data():
    rows = syndicate_signal_rows()
    signal_groups = {}
    for row in rows:
        for reason in json.loads(row["reasons_json"]):
            signal = signal_from_reason(reason)
            if signal is None:
                continue
            group = signal_groups.setdefault(signal, {"signal": signal, "count": 0, "transactions": []})
            group["count"] += 1
            if len(group["transactions"]) < 5:
                group["transactions"].append({
                    "id": row["transaction_id"],
                    "amount": row["amount"],
                    "risk": round(row["hybrid_risk_score"] * 100, 1),
                })
    groups = sorted(signal_groups.values(), key=lambda item: (-item["count"], item["signal"]))[:20]
    return jsonify({
        "transactions_analyzed": len(rows),
        "groups": groups,
        "note": "Groups summarize repeated model-explanation features only. The dataset has no customer, card, or merchant identifiers, so these are not confirmed syndicates or linked people.",
    })


def syndicate_signal_rows():
    with database_connection() as connection:
        return connection.execute(
            """
            SELECT transaction_id, amount, hybrid_risk_score, reasons_json
            FROM transactions
            WHERE predicted_class = 1 OR original_dataset_label = 1
            ORDER BY rowid DESC
            LIMIT 2000
            """
        ).fetchall()


def signal_from_reason(reason):
    if reason.startswith("Synthetic demo") or reason.startswith("No "):
        return None
    return "Amount" if reason.startswith("Amount $") else reason.split(" ", 1)[0]


@app.post("/api/v1/syndicate/match")
def match_syndicate_signals():
    payload = request.get_json()
    if not isinstance(payload, dict):
        return jsonify({"error": "Expected a JSON object."}), 400

    transaction_id = payload.get("transaction_id")
    supplied_features = payload.get("features")
    if bool(transaction_id) == bool(supplied_features):
        return jsonify({
            "error": "Provide either transaction_id or a non-empty features array."
        }), 400

    feature_weights = {}
    if transaction_id:
        if not isinstance(transaction_id, str) or len(transaction_id) > 100:
            return jsonify({"error": "transaction_id must be text of at most 100 characters."}), 400
        with database_connection() as connection:
            transaction = connection.execute(
                "SELECT reasons_json FROM transactions WHERE transaction_id = ?",
                (transaction_id.strip(),),
            ).fetchone()
        if transaction is None:
            return jsonify({"error": "Transaction not found."}), 404
        for reason in json.loads(transaction["reasons_json"]):
            signal = signal_from_reason(reason)
            if signal:
                feature_weights[signal] = 1.0
        if not feature_weights:
            return jsonify({"error": "This transaction has no individual explanation features to match."}), 400
        equal_weight = 100.0 / len(feature_weights)
        feature_weights = {signal: equal_weight for signal in feature_weights}
    else:
        if not isinstance(supplied_features, list) or not supplied_features or len(supplied_features) > 30:
            return jsonify({"error": "features must be a non-empty array with at most 30 items."}), 400
        for item in supplied_features:
            if not isinstance(item, dict):
                return jsonify({"error": "Each feature must include feature and contribution."}), 400
            feature = item.get("feature")
            contribution = item.get("contribution")
            if not isinstance(feature, str) or not feature.strip() or len(feature.strip()) > 80:
                return jsonify({"error": "Each feature must be non-empty text of at most 80 characters."}), 400
            try:
                contribution = float(str(contribution).strip().rstrip("%"))
            except (TypeError, ValueError):
                return jsonify({"error": "Each contribution must be a number from 0 to 100."}), 400
            if not np.isfinite(contribution) or not 0 <= contribution <= 100:
                return jsonify({"error": "Each contribution must be a number from 0 to 100."}), 400
            signal = feature.strip().split()[0]
            if signal in feature_weights:
                return jsonify({"error": f"Feature {signal} was provided more than once."}), 400
            feature_weights[signal] = contribution
        if sum(feature_weights.values()) <= 0:
            return jsonify({"error": "At least one contribution must be greater than zero."}), 400

    signal_counts = {}
    signal_transactions = {}
    for row in syndicate_signal_rows():
        transaction_signals = {
            signal for reason in json.loads(row["reasons_json"])
            if (signal := signal_from_reason(reason)) is not None
        }
        for signal in transaction_signals:
            signal_counts[signal] = signal_counts.get(signal, 0) + 1
            signal_transactions.setdefault(signal, []).append({
                "id": row["transaction_id"],
                "amount": row["amount"],
                "risk": round(row["hybrid_risk_score"] * 100, 1),
            })

    repeated_signals = {
        group["signal"]
        for group in sorted(
            (
                {"signal": signal, "count": count}
                for signal, count in signal_counts.items()
            ),
            key=lambda group: (-group["count"], group["signal"]),
        )[:20]
    }
    matched_signals = set(feature_weights) & repeated_signals
    total_weight = sum(feature_weights.values())
    matched_weight = sum(feature_weights[signal] for signal in matched_signals)
    match_score = round(matched_weight / total_weight * 100) if total_weight else 0
    risk_level = "Low" if match_score <= 33 else "Medium" if match_score <= 66 else "High"

    matched_groups = [
        {
            "signal": signal,
            "count": signal_counts[signal],
            "contribution": round(feature_weights[signal], 2),
        }
        for signal in sorted(matched_signals)
    ]
    overlapping_by_id = {}
    for group in matched_groups:
        for transaction in signal_transactions[group["signal"]]:
            if transaction["id"] == transaction_id:
                continue
            overlap = overlapping_by_id.setdefault(
                transaction["id"],
                {**transaction, "signals": []},
            )
            overlap["signals"].append(group["signal"])
    overlapping_txs = sorted(
        overlapping_by_id.values(),
        key=lambda item: (-len(item["signals"]), -item["risk"], item["id"]),
    )[:100]

    if matched_groups:
        explanation = (
            f"{len(matched_groups)} of {len(feature_weights)} supplied signals matched "
            f"repeated explanation groups, covering {match_score}% of the supplied contribution weight. "
            f"{len(overlapping_txs)} other recorded transactions share at least one matched signal."
        )
    else:
        explanation = (
            "None of the supplied explanation features matched the repeated signal groups currently available."
        )
    explanation += " This is pattern matching on model explanations only, not confirmed linked accounts."

    return jsonify({
        "match_score": match_score,
        "risk_level": risk_level,
        "matched_groups": matched_groups,
        "overlapping_txs": overlapping_txs,
        "explanation": explanation,
    })


@app.post("/api/transactions/<transaction_id>/investigation")
def update_investigation(transaction_id):
    payload = request.get_json()
    if not isinstance(payload, dict):
        return jsonify({"error": "Expected a JSON object."}), 400
    review_status = payload.get("review_status")
    if not isinstance(review_status, str) or review_status not in {
        "Pending", "In Review", "Reviewed"
    }:
        return jsonify({
            "error": "review_status must be Pending, In Review, or Reviewed."
        }), 400
    notes = payload.get("investigator_notes", "")
    if not isinstance(notes, str) or len(notes) > 2000:
        return jsonify({"error": "investigator_notes must be text of at most 2000 characters."}), 400

    reviewed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with database_connection() as connection:
        exists = connection.execute(
            "SELECT 1 FROM transactions WHERE transaction_id = ?",
            (transaction_id,),
        ).fetchone()
        if exists is None:
            return jsonify({"error": "Transaction not found."}), 404
        connection.execute(
            "UPDATE transactions SET investigation_status = ? WHERE transaction_id = ?",
            (review_status, transaction_id),
        )
        connection.execute(
            """
            INSERT INTO investigations (
                transaction_id, review_status, investigator_notes, reviewed_at
            ) VALUES (?, ?, ?, ?)
            """,
            (transaction_id, review_status, notes.strip(), reviewed_at),
        )
    return jsonify({
        "transaction_id": transaction_id,
        "review_status": review_status,
        "investigator_notes": notes.strip(),
        "reviewed_at": reviewed_at,
    })


@app.get("/api/summary")
def summary():
    with database_connection() as connection:
        rows = connection.execute(
            """
            SELECT t.*,
                   i.review_status AS latest_review_status,
                   i.investigator_notes,
                   i.reviewed_at
            FROM transactions t
            LEFT JOIN investigations i
              ON i.investigation_id = (
                  SELECT MAX(latest.investigation_id)
                  FROM investigations latest
                  WHERE latest.transaction_id = t.transaction_id
              )
            ORDER BY t.rowid DESC
            LIMIT 20
            """
        ).fetchall()
        grouped_counts = connection.execute(
            "SELECT risk_category, COUNT(*) AS count FROM transactions GROUP BY risk_category"
        ).fetchall()
        current_counts = {"Low": 0, "Medium": 0, "High": 0}
        current_counts.update({row["risk_category"]: row["count"] for row in grouped_counts})
        db_stats = database_statistics(connection)
        policy = get_decision_policy(connection)
    return jsonify({
        "counts": current_counts,
        "transactions": [
            {
                **transaction_record(row),
                "investigation_status": row["latest_review_status"] or row["investigation_status"],
            }
            for row in rows
        ],
        "stats": {
            "total_transactions": db_stats["database_records"],
            "dataset_transactions": dataset_total,
            "fraud_transactions": dataset_fraud,
            "legit_transactions": dataset_legit,
            "model_metrics": model_metrics,
            "model_comparison": model_comparison,
            "decision_policy": policy,
            **db_stats,
        },
    })


if __name__ == "__main__":
    app.run(
        debug=False,
        use_reloader=False,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
    )