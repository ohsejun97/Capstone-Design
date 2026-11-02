"""MLOps 레이어 — 실험 추적(MLflow), 아티팩트 관리.

기존 results/{run_name}/result.json 파일 기반 워크플로를 유지한 채,
MLflow Tracking Server로 동일 내용을 이중 기록한다.
"""

from mlops.tracking import log_result, phase_of, split_fields

__all__ = ["log_result", "phase_of", "split_fields"]
