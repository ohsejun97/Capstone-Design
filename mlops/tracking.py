"""
tracking.py — MLflow 실험 추적 어댑터

기존 학습 스크립트는 `results/{run_name}/result.json`에 dict 하나를 저장한다.
이 모듈은 그 dict를 그대로 받아 MLflow run으로 변환한다. 학습 코드는
`log_result(result, out_dir)` 한 줄만 추가하면 되고, MLflow가 없거나
서버가 죽어 있어도 학습은 절대 실패하지 않는다(로그만 남기고 통과).

Tracking URI 결정 순서:
    1. MLFLOW_TRACKING_URI 환경변수 (docker-compose가 Postgres를 주입)
    2. sqlite:///mlflow.db  (로컬 폴백)

MLflow 3.x부터 파일 백엔드(./mlruns)는 유지보수 모드라 신규 기록이 막혀 있다.
로컬에서도 SQLite를 쓰며, 운영 구성에서는 그대로 Postgres로 갈아끼운다.

사용:
    from mlops.tracking import log_result
    log_result(result, out_dir)                 # 학습 스크립트 말미
    log_result(result, out_dir, start_time=ts)  # 백필(과거 타임스탬프 복원)
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_EXPERIMENT = "FusionDTI-DTI"
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"

# ── result.json 키 분류 ───────────────────────────────────────────────────────
# MLflow는 param(문자열, 불변 설정)과 metric(수치, 시계열)을 구분한다.
# result.json은 둘을 한 dict에 섞어 두므로 여기서 갈라준다.

_PARAM_KEYS = {
    "run_name", "dataset", "eval_dataset", "target_dataset", "source_model",
    "encoder", "quant", "lora", "lora_r", "use_3di", "drug_encoder", "split",
    "prot_dim", "warm_start", "lr", "model_dir", "unfreeze", "batch_size",
    "max_length", "seed", "n_train", "n_val", "n_test", "n_pairs",
}

_METRIC_KEYS = {
    "test_pearson_r", "test_spearman_r", "test_rmse", "test_mae", "test_ci",
    "test_p_value", "best_val_r", "epochs_trained", "train_time_sec",
    "infer_ms_mean", "infer_ms_std", "peak_vram_mb",
    "cross_pearson_r", "cross_rmse", "cross_mae", "cross_ci", "cross_p_value",
    "kiba_label_mean", "kiba_label_std",
}

# results/ 하위에서 MLflow 아티팩트로 함께 올릴 파일.
# 모델 가중치(.pt)는 수백 MB라 기본 제외 — --with-weights로만 올린다.
_ARTIFACT_NAMES = ("result.json", "report.md", "training_history.csv",
                   "test_predictions.csv")
_WEIGHT_NAMES = ("dti_head.pt", "chemberta_ft.pt")


def split_fields(result: dict[str, Any]) -> tuple[dict, dict, dict]:
    """result.json dict를 (params, metrics, extra)로 분리한다.

    분류표에 없는 키는 extra로 빠지며, 값 타입을 보고 param/metric에 재배치한다.
    새 실험 스크립트가 필드를 추가해도 조용히 유실되지 않게 하기 위함.
    """
    params: dict[str, Any] = {}
    metrics: dict[str, float] = {}
    extra: dict[str, Any] = {}

    for key, value in result.items():
        if value is None:
            # MLflow는 None을 못 받는다. 설정값이 '없음'인 것도 정보이므로 문자열로.
            params[key] = "none"
        elif key in _PARAM_KEYS:
            params[key] = value
        elif key in _METRIC_KEYS:
            metrics[key] = float(value)
        elif key == "timestamp":
            extra[key] = value
        elif isinstance(value, bool):
            params[key] = value          # bool은 int 하위타입 → 먼저 걸러야 한다
        elif isinstance(value, (int, float)):
            metrics[key] = float(value)
        else:
            params[key] = value

    return params, metrics, extra


def phase_of(result: dict[str, Any]) -> str:
    """result.json 내용으로 README의 Phase 라벨(1a~1i)을 역추론한다.

    results/ 디렉토리에는 phase 필드가 없다. 하지만 어떤 실험이 어느 단계에
    속하는지는 (dataset, use_3di, drug_encoder, 전이 여부)로 완전히 결정된다.
    MLflow UI에서 phase 태그로 필터링하기 위해 복원한다.
    """
    dataset = result.get("dataset", "")
    drug_encoder = result.get("drug_encoder", "morgan")
    use_3di = result.get("use_3di", False)

    if "cross_pearson_r" in result:
        return "1f-zeroshot"          # BindingDB 모델 → DAVIS/KIBA 무학습 평가
    if result.get("source_model") or result.get("warm_start") is not None:
        # Head만 재학습하는 전이. ft ChemBERTa 여부로 1g/1i를 가른다.
        return "1i" if "ft" in str(result.get("model_dir", "")) else "1g"
    if result.get("unfreeze"):
        return "1h"                   # ChemBERTa 상위 레이어 unfreeze
    if dataset == "bindingdb":
        return "1f"
    if drug_encoder == "gnn":
        return "1d"
    if drug_encoder == "chemberta":
        return "1e"
    if use_3di:
        return "1c"
    return "1b" if dataset == "kiba" else "1a"


def _tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)


def _artifact_root() -> str | None:
    """실험을 새로 만들 때 쓸 아티팩트 저장 위치.

    주의: MLflow는 artifact_location을 지정하지 않으면 **현재 작업 디렉토리**
    기준 ./mlruns 에 파일을 쌓는다. 추적 DB를 다른 곳에 두더라도 마찬가지라,
    프로젝트 루트에서 스크립트를 돌리면 저장소 안에 아티팩트 사본이 생긴다.
    그래서 항상 명시한다.
    """
    root = os.environ.get("MLFLOW_ARTIFACT_ROOT")
    if root:
        return root
    uri = _tracking_uri()
    if uri.startswith(("http://", "https://")):
        return None          # 서버가 자기 정책대로 정한다
    return str(Path("mlartifacts").resolve())


def log_result(
    result: dict[str, Any],
    out_dir: str | Path | None = None,
    *,
    experiment: str | None = None,
    start_time: datetime | None = None,
    with_weights: bool = False,
    extra_tags: dict[str, str] | None = None,
) -> str | None:
    """result.json dict 하나를 MLflow run 하나로 기록한다.

    Args:
        result:       학습 스크립트가 만든 결과 dict.
        out_dir:      results/{run_name}/ 경로. 있으면 아티팩트도 함께 올린다.
        experiment:   MLflow experiment 이름. 기본 FusionDTI-DTI.
        start_time:   과거 실험 백필용. None이면 현재 시각.
        with_weights: True면 .pt 가중치도 아티팩트로 업로드(수백 MB 주의).
        extra_tags:   추가 태그.

    Returns:
        생성된 run_id, 실패 시 None. **예외를 밖으로 내보내지 않는다** —
        추적 실패가 학습 결과 손실로 이어져서는 안 된다.
    """
    try:
        from mlflow.tracking import MlflowClient
    except ImportError:
        log.warning("mlflow 미설치 — 실험 추적을 건너뜁니다 (pip install mlflow)")
        return None

    try:
        client = MlflowClient(tracking_uri=_tracking_uri())
        exp_name = experiment or os.environ.get("MLFLOW_EXPERIMENT_NAME",
                                                DEFAULT_EXPERIMENT)
        exp = client.get_experiment_by_name(exp_name)
        exp_id = (exp.experiment_id if exp
                  else client.create_experiment(exp_name,
                                                artifact_location=_artifact_root()))

        params, metrics, extra = split_fields(result)
        run_name = result.get("run_name", "unnamed")

        # 타임스탬프 복원: 과거 실험을 백필할 때 MLflow UI의 시간축이
        # 실제 실험 순서와 일치해야 Phase 진행이 읽힌다.
        if start_time is None and extra.get("timestamp"):
            try:
                start_time = datetime.fromisoformat(extra["timestamp"])
            except ValueError:
                start_time = None
        start_ms = int(start_time.timestamp() * 1000) if start_time else None

        tags = {
            "phase": phase_of(result),
            "mlflow.runName": run_name,
            "source": "result.json",
        }
        if extra.get("timestamp"):
            tags["original_timestamp"] = extra["timestamp"]
        if extra_tags:
            tags.update(extra_tags)

        run = client.create_run(exp_id, start_time=start_ms, tags=tags)
        run_id = run.info.run_id

        for key, value in params.items():
            client.log_param(run_id, key, value)
        for key, value in metrics.items():
            client.log_metric(run_id, key, value)

        if out_dir is not None:
            _log_artifacts(client, run_id, Path(out_dir), with_weights)

        # 학습 시간을 알면 종료 시각도 실제에 맞춘다.
        end_ms = None
        if start_ms is not None:
            end_ms = start_ms + int(metrics.get("train_time_sec", 0) * 1000)
        client.set_terminated(run_id, status="FINISHED", end_time=end_ms)

        log.info("MLflow run 기록: %s (%s)", run_name, run_id)
        return run_id

    except Exception as exc:
        log.warning("MLflow 기록 실패 (%s) — 학습 결과는 %s 에 그대로 있습니다",
                    exc, out_dir)
        return None


def _log_artifacts(client, run_id: str, out_dir: Path, with_weights: bool) -> None:
    names = _ARTIFACT_NAMES + (_WEIGHT_NAMES if with_weights else ())
    for name in names:
        path = out_dir / name
        if path.exists():
            client.log_artifact(run_id, str(path))

    # cross_eval_{dataset}.json 은 개수가 가변이라 글롭으로 처리
    for path in sorted(out_dir.glob("cross_eval_*.json")):
        client.log_artifact(run_id, str(path))
