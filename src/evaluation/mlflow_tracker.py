"""
MLflow Tracking & Autolog Setup
Based on professor's Task 10 in 7_Generator.py

Install:
    pip install mlflow openai dspy-ai
Run MLflow server first:
    mlflow server --host 0.0.0.0 --port 5000
Then visit: http://localhost:5000
"""

import os
from pathlib import Path

# Allow legacy file store if user explicitly sets a file:// URI via env
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
import dspy
from datetime import datetime


# ── Configuration ──────────────────────────────────────────────────────────────
# Default: local SQLite (no server, works on Windows, no file-store deprecation).
# Override with MLFLOW_TRACKING_URI=http://localhost:5000 for the MLflow UI server.

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DB   = _PROJECT_ROOT / "mlflow.db"
_RAW_URI      = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
EXPERIMENT_NAME = f"generation_{datetime.now().strftime('%d%m%Y')}"


def _sqlite_uri(path: Path) -> str:
    return f"sqlite:///{path.resolve().as_posix()}"


def _resolve_mlflow_uri(raw: str) -> str:
    """
    Normalize tracking URIs. Bare Windows paths become file:/// URIs;
    empty env uses a project-local SQLite database.
    """
    if not raw:
        return _sqlite_uri(_DEFAULT_DB)

    if raw.startswith(("http://", "https://", "file:", "sqlite:",
                       "postgresql:", "mysql:", "mssql:", "databricks")):
        return raw

    path = Path(raw)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve().as_uri()


MLFLOW_URI = _resolve_mlflow_uri(_RAW_URI)


def _upgrade_mlflow_schema(uri: str) -> None:
    """Auto-migrate SQLite MLflow DB when package version outpaces schema."""
    if not uri.startswith("sqlite:"):
        return
    import subprocess
    import sys

    try:
        proc = subprocess.run(
            [sys.executable, "-m", "mlflow", "db", "upgrade", uri],
            capture_output=True,
            text=True,
            timeout=120,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode == 0:
            print("[MLflow] Database schema upgraded/verified")
            return
        print(f"[MLflow] db upgrade returned {proc.returncode}: {out.strip()}")
    except Exception as exc:
        print(f"[MLflow] db upgrade failed ({exc})")

    # Last resort: back up stale DB and start fresh (metrics only, no model artifacts)
    db_path = uri.replace("sqlite:///", "")
    if os.path.isfile(db_path):
        backup = f"{db_path}.bak"
        try:
            if os.path.isfile(backup):
                os.remove(backup)
            os.replace(db_path, backup)
            print(f"[MLflow] Reset stale database -> {backup}")
        except Exception as exc:
            print(f"[MLflow] could not reset database ({exc})")


def setup_mlflow(framework: str = "dspy") -> None:
    """
    Configure MLflow and enable autologging.

    Parameters
    ----------
    framework : str
        One of "dspy" | "langchain" | "openai".
        Matches the professor's Task 10 options exactly.
    """
    _upgrade_mlflow_schema(MLFLOW_URI)
    mlflow.set_tracking_uri(MLFLOW_URI)
    if MLFLOW_URI.startswith("sqlite:"):
        # SQLite tracking — no separate model registry needed for metric logging
        pass
    else:
        try:
            mlflow.set_registry_uri(MLFLOW_URI)
        except Exception:
            pass
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Skip dspy autolog — it calls dspy.configure() and breaks Streamlit threads.
    # Manual metric logging via log_retrieval_metrics / log_generation_metrics is enough.
    if framework == "dspy":
        print("[MLflow] dspy autolog disabled (Streamlit-safe mode)")
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
