"""
MLflow Tracking & Autolog Setup
Based on professor's Task 10 in 7_Generator.py

Install:
    pip install mlflow openai dspy-ai
Run MLflow server first:
    mlflow server --host 0.0.0.0 --port 5000
Then visit: http://localhost:5000
"""

import mlflow
import dspy
from datetime import datetime


# ── Configuration ──────────────────────────────────────────────────────────────

MLFLOW_URI       = "http://localhost:5000"
EXPERIMENT_NAME  = f"generation_{datetime.now().strftime('%d%m%Y')}"   # e.g. generation_08062026


def setup_mlflow(framework: str = "dspy") -> None:
    """
    Configure MLflow and enable autologging.

    Parameters
    ----------
    framework : str
        One of "dspy" | "langchain" | "openai".
        Matches the professor's Task 10 options exactly.
    """
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    if framework == "dspy":
        mlflow.dspy.autolog()           # professor's exact instruction
    elif framework == "langchain":
        mlflow.langchain.autolog()
    elif framework == "openai":
        mlflow.openai.autolog()
    else:
        raise ValueError(f"Unknown framework: {framework}. Choose dspy | langchain | openai.")

    print(f"[MLflow] Tracking URI : {MLFLOW_URI}")
    print(f"[MLflow] Experiment   : {EXPERIMENT_NAME}")
    print(f"[MLflow] Autolog      : {framework}")


# ── Manual run helpers ─────────────────────────────────────────────────────────

def log_retrieval_metrics(run_name: str, metrics: dict, params: dict | None = None) -> str:
    """
    Open a new MLflow run and log retrieval evaluation metrics.

    Parameters
    ----------
    run_name : str
        Human-readable label for this run (e.g. "faiss_mmr_eval").
    metrics : dict
        Flat dict of metric_name → float, e.g.
        {"hit_at_1": 0.72, "mrr": 0.81, "ndcg_at_5": 0.78}.
    params : dict | None
        Optional hyper-parameters or config values to tag the run.

    Returns
    -------
    str
        The MLflow run_id for later reference.
    """
    with mlflow.start_run(run_name=run_name) as run:
        if params:
            mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        print(f"[MLflow] Logged retrieval run '{run_name}' → {run.info.run_id}")
        return run.info.run_id


def log_generation_metrics(run_name: str, metrics: dict, artifacts: dict | None = None) -> str:
    """
    Log generation quality metrics (ROUGE-L, BERTScore, faithfulness).

    Parameters
    ----------
    artifacts : dict | None
        Optional {label: local_file_path} pairs; files are uploaded to
        MLflow artifact store.
    """
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_metrics(metrics)
        if artifacts:
            for label, path in artifacts.items():
                mlflow.log_artifact(path, artifact_path=label)
        print(f"[MLflow] Logged generation run '{run_name}' → {run.info.run_id}")
        return run.info.run_id


# ── Quick smoke-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    setup_mlflow("dspy")

    # Example: log a dummy retrieval run
    log_retrieval_metrics(
        run_name="smoke_test_retrieval",
        metrics={"hit_at_1": 1.0, "mrr": 1.0, "ndcg_at_5": 1.0},
        params={"vector_store": "faiss", "search_type": "mmr", "k": 5},
    )
    print("Smoke test passed. Open http://localhost:5000 to inspect.")
