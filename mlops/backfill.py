"""
backfill.py — results/ 디렉토리의 과거 실험을 MLflow로 일괄 이관

Phase 1a~1i 동안 쌓인 실험은 results/{run_name}/result.json 파일로만 남아 있다.
비교하려면 디렉토리를 일일이 열어야 하고, "3Di가 몇 점 올렸나" 같은 질문에
답하려면 매번 표를 손으로 만들어야 했다. 이 스크립트는 그 파일들을
MLflow Tracking으로 옮겨 UI에서 정렬·필터·비교가 되게 한다.

멱등성: 같은 run_name + original_timestamp 조합이 이미 있으면 건너뛴다.
       따라서 몇 번을 돌려도 중복 run이 생기지 않는다.

사용:
    python -m mlops.backfill                      # file:./mlruns 로 백필
    python -m mlops.backfill --dry-run            # 무엇이 올라갈지만 출력
    MLFLOW_TRACKING_URI=http://localhost:5000 python -m mlops.backfill
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mlops.tracking import (
    DEFAULT_EXPERIMENT,
    _tracking_uri,
    log_result,
    phase_of,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("backfill")


def _existing_keys(experiment: str) -> set[tuple[str, str]]:
    """이미 기록된 (run_name, original_timestamp) 집합. 멱등 재실행용."""
    try:
        from mlflow.tracking import MlflowClient
    except ImportError:
        return set()

    client = MlflowClient(tracking_uri=_tracking_uri())
    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        return set()

    keys: set[tuple[str, str]] = set()
    page_token = None
    while True:
        page = client.search_runs([exp.experiment_id], max_results=1000,
                                  page_token=page_token)
        for run in page:
            tags = run.data.tags
            keys.add((tags.get("mlflow.runName", ""),
                      tags.get("original_timestamp", "")))
        page_token = getattr(page, "token", None)
        if not page_token:
            break
    return keys


def discover(results_dir: Path) -> list[tuple[Path, dict, Path | None]]:
    """백필 대상을 (json 경로, dict, 아티팩트 디렉토리) 목록으로 수집한다.

    result.json은 아티팩트 디렉토리를 함께 올리지만,
    cross_eval_*.json은 같은 폴더의 학습 아티팩트와 무관하므로 붙이지 않는다.
    """
    found = []
    for run_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        main = run_dir / "result.json"
        if main.exists():
            found.append((main, json.loads(main.read_text()), run_dir))
        for cross in sorted(run_dir.glob("cross_eval_*.json")):
            data = json.loads(cross.read_text())
            # cross_eval 결과에는 run_name이 없다 — 평가 대상으로 이름을 만든다.
            data.setdefault(
                "run_name",
                f"{run_dir.name}--zeroshot-{data.get('eval_dataset', '?')}",
            )
            found.append((cross, data, None))
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--dry-run", action="store_true",
                        help="MLflow에 쓰지 않고 대상만 출력")
    parser.add_argument("--with-weights", action="store_true",
                        help=".pt 가중치도 아티팩트로 업로드 (수백 MB)")
    parser.add_argument("--force", action="store_true",
                        help="이미 기록된 run도 다시 올림")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        log.error("results 디렉토리를 찾을 수 없습니다: %s", results_dir)
        return 1

    targets = discover(results_dir)
    log.info("발견: %d개 실험 (%s)", len(targets), results_dir)

    seen = set() if (args.dry_run or args.force) else _existing_keys(args.experiment)
    if seen:
        log.info("이미 기록됨: %d개 run — 건너뜁니다", len(seen))

    logged = skipped = failed = 0
    for json_path, data, artifact_dir in targets:
        key = (data.get("run_name", ""), data.get("timestamp", ""))
        phase = phase_of(data)
        label = f"[{phase:>11}] {data.get('run_name')}"

        if key in seen:
            skipped += 1
            continue

        if args.dry_run:
            r = data.get("test_pearson_r", data.get("cross_pearson_r"))
            log.info("%s  r=%s  ← %s", label, r, json_path.relative_to(ROOT))
            logged += 1
            continue

        run_id = log_result(
            data,
            artifact_dir,
            experiment=args.experiment,
            with_weights=args.with_weights,
            extra_tags={"backfilled": "true"},
        )
        if run_id:
            logged += 1
            log.info("%s  → %s", label, run_id[:8])
        else:
            failed += 1
            log.warning("%s  → 실패", label)

    verb = "대상" if args.dry_run else "기록"
    log.info("\n%s %d개 | 건너뜀 %d개 | 실패 %d개", verb, logged, skipped, failed)
    if not args.dry_run:
        log.info("Tracking URI: %s", _tracking_uri())
        log.info("UI: mlflow ui  (또는 docker compose up mlflow → localhost:5000)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
