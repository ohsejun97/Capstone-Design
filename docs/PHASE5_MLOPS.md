# Phase 5 — MLOps: 실험 추적 · 에이전트 그래프 · 컨테이너화

Phase 1~4는 "모델이 되는가"를 물었다. Phase 5는 그 위에 **운영 계층**을 얹는다:
실험을 다시 찾을 수 있는가, 에이전트가 실패해도 복구되는가, 남이 이 시스템을
재현해서 띄울 수 있는가.

---

## 1. 무엇이 문제였나

| 문제 | 증상 | 해결 |
|---|---|---|
| 실험 결과가 파일로만 존재 | `results/` 28개 디렉토리를 일일이 열어야 비교됨. "3Di가 몇 점 올렸나"에 답하려면 매번 손으로 표를 만듦 | MLflow Tracking (`mlops/`) |
| Phase 3 오케스트레이션 미구현 | 도구 5개는 있는데 이어 붙이는 계층이 없음 | LangGraph 상태 그래프 (`agent/`) |
| UI가 `tools/`를 직접 import | UI와 모델 서버를 분리할 수 없음 → GPU 프로세스를 UI가 물고 있음 | 백엔드 추상화 (`web_pipeline/backend.py`) |
| 재현 수단이 `setup_env.sh` 뿐 | "내 노트북에선 되는데" | Docker Compose + k3s 매니페스트 |
| 검증이 수동 | 리팩터링하면 뭐가 깨지는지 모름 | pytest 26개 + GitHub Actions |

---

## 2. 실험 추적 — `mlops/`

### 설계 원칙: result.json을 버리지 않는다

기존 학습 스크립트는 `results/{run_name}/result.json`에 dict 하나를 저장한다.
이걸 MLflow API 호출로 **대체**하면 추적 서버가 죽었을 때 실험 결과가 통째로
날아간다. 그래서 파일 저장은 그대로 두고, 같은 dict를 MLflow에도 **이중 기록**한다.

```python
# scripts/train_chemberta_unfreeze.py 말미 — 추가된 것은 이게 전부다
if not args.no_mlflow:
    from mlops.tracking import log_result
    log_result(result, out_dir, extra_tags={"hparams": f"unfreeze={args.unfreeze},..."})
```

`log_result`는 **예외를 밖으로 내보내지 않는다.** 추적 실패가 GPU 몇 시간을
날리는 일은 없어야 한다 — 경고만 찍고 통과한다.

### result.json → MLflow 매핑

MLflow는 param(불변 설정)과 metric(수치)을 구분하지만 `result.json`은 둘을
한 dict에 섞어 둔다. `split_fields()`가 분류표로 가르고, **분류표에 없는 키는
타입을 보고 재배치**한다 — 새 실험이 필드를 추가해도 조용히 유실되지 않는다.

`phase_of()`는 `result.json`에 없는 Phase 라벨(1a~1i)을 내용으로 역추론한다:

```
cross_pearson_r 있음        → 1f-zeroshot
source_model/warm_start 있음 → 1g (frozen) 또는 1i (ft)
unfreeze 있음               → 1h
dataset == bindingdb        → 1f
drug_encoder == gnn         → 1d
drug_encoder == chemberta   → 1e
use_3di                     → 1c
dataset == kiba             → 1b, 그 외 1a
```

### 과거 실험 이관

```bash
python -m mlops.backfill --dry-run     # 무엇이 올라갈지 확인
python -m mlops.backfill               # 실제 이관
```

멱등하다 — `(run_name, original_timestamp)`가 이미 있으면 건너뛴다.
`timestamp` 필드로 MLflow의 run 시작 시각을 복원하므로 UI의 시간축이
실제 실험 순서와 일치한다.

**실측 결과 (2026-09-30):**

```
발견: 28개 실험 (results/)
기록 28개 | 건너뜀 0개 | 실패 0개
재실행 → 기록 0개 | 건너뜀 28개 | 실패 0개     ← 멱등성 확인
```

이관 후 가능해진 질의 (기존에는 손으로 표를 만들어야 했던 것):

```python
client.search_runs([exp_id], filter_string="tags.phase='1c' and metrics.test_pearson_r > 0.80")
# → SaProt-650M-davis-3di 0.8082 / SaProt-35M-kiba-3di 0.8035 / ...
```

### 함정 — 아티팩트가 저장소를 오염시킨다

MLflow는 `artifact_location`을 지정하지 않으면 추적 DB 위치와 **무관하게**
현재 작업 디렉토리 기준 `./mlruns`에 파일을 쌓는다. 추적 DB를 `/tmp`에 두고
프로젝트 루트에서 백필을 돌렸더니 저장소 안에 아티팩트 사본 85개가 생겼다.
`_artifact_root()`가 항상 명시하도록 고쳤고, `.gitignore`에도 추가했다.

> 또 하나: MLflow 3.x부터 파일 백엔드(`file:./mlruns`)는 유지보수 모드라
> 신규 기록이 거부된다. 로컬 기본값을 `sqlite:///mlflow.db`로 잡은 이유다.

---

## 3. 에이전트 오케스트레이션 — `agent/`

### 왜 ReAct(smolagents)가 아니라 상태 그래프인가

DTI 질의의 **정상 경로**는 정해져 있다: 이름 해석 → 범위 점검 → 구조 보강 →
예측 → 답변. LLM이 매 스텝 도구를 고르게 두면 비용과 비결정성만 늘고, 틀린
순서로 도구를 부르거나 루프에 빠질 수 있다.

반면 **실패 경로**는 실제로 분기가 필요하다:

| 상황 | 그래프의 대응 |
|---|---|
| 브랜드명이라 PubChem이 못 찾음 (`Gleevec`) | `refine_drug_name`이 성분명으로 치환 → 1회 재시도 |
| 융합 유전자라 UniProt 항목 없음 (`BCR-ABL`) | `refine_protein_name`이 `ABL1`로 치환 → 재시도 |
| 키나아제가 아닌 표적 (`HMGCR`) | **차단하지 않고** OOD 경고를 달아 계속 진행 |
| AlphaFold 구조 없음 | 3Di 없이 예측하고 "Pearson r 약 0.023 하락" 고지 |
| 계속 실패 | 카운터 상한에서 포기 → 도구를 더 부르지 않고 실패 답변 |

그래서 **LLM은 맨 앞 엔티티 추출에만** 쓰고 나머지는 명시적 그래프로 둔다.

### 그래프 구조

```
START → parse ──(파싱 실패)──────────────────────────┐
          ▼                                          │
     resolve_drug ──retry──→ refine_drug_name ──┐    │
          │ ok        ▲                         │    │
          │           └─────────────────────────┘    │
          │ give_up → mark_drug_failure ─────────────┤
          ▼                                          │
   resolve_protein ──retry──→ refine_protein_name ───┤
          │ ok                                       │
          │ give_up → mark_protein_failure ──────────┤
          ▼                                          │
     scope_check                                     │
          │                                          │
   ┌──────┴──────┐  병렬 fan-out                     │
   ▼             ▼                                   │
fetch_structure  generate_ligand                     │
   └──────┬──────┘  fan-in                           │
          ▼                                          │
       predict ──────────────────────────────────→ compose → END
```

`agent/graph.py --draw`로 Mermaid를 뽑을 수 있다 (CI가 매번 검증한다).

### 설계상 결정 세 가지

**① 두 이름 해석은 순차, 두 보강 도구는 병렬.**
이름 해석은 각자 재시도 사이클을 가지는데, 이걸 병렬 fan-in과 섞으면
LangGraph의 배리어가 재시도 분기를 기다리다 교착될 수 있다. 반면
`fetch_structure`(네트워크)와 `generate_ligand`(CPU)는 서로 독립이고
사이클이 없어 안전하게 겹쳐 돌린다.

**② 정규화는 추출이 아니라 재시도 노드에서.**
처음에는 `_extract_with_rules`가 `Gleevec → Imatinib`까지 해버렸다. 그랬더니
PubChem이 늘 성공해서 **재시도 분기가 영영 실행되지 않는 죽은 코드**가 됐다.
추출은 표면형만 돌려주고 치환은 실패 후에 하도록 뒤집었다.

**③ 경고 필드에 reducer가 필요하다.**
`fetch_structure`와 `generate_ligand`가 같은 슈퍼스텝에서 `warnings`에 쓴다.
`Annotated[list[str], operator.add]`가 없으면 한쪽이 다른 쪽을 덮어쓴다.
`test_parallel_branches_accumulate_warnings`가 이걸 지킨다.

### 실행

```bash
python -m agent.graph "Does Imatinib bind to BCR-ABL?" --verbose
python -m agent.graph --draw                              # Mermaid 출력
python -m agent.graph "..." --checkpoint sqlite:///agent_checkpoints.db --thread-id user-42
```

체크포인터를 켜면 AlphaFold 다운로드처럼 수십 초 걸리는 단계에서 끊겨도
같은 `thread_id`로 이어서 재개된다.

### LLM은 선택이다

`ANTHROPIC_API_KEY`가 있으면 Claude로 엔티티를 추출하고, 없으면 규칙 경로로
폴백한다. 이 폴백이 있어야 CI에서 네트워크·과금 없이 그래프 전체를 테스트할 수
있다. LLM 호출이 실패해도 조용히 규칙 경로로 떨어진다.

---

## 4. UI ↔ API 분리 — `web_pipeline/backend.py`

`app.py`는 원래 `tools/`를 직접 import했다. 한 프로세스에서는 잘 돌지만
컨테이너로 쪼개면 UI가 SaProt-650M과 torch를 통째로 지고 다니게 된다.

호출부를 `backend.py` 뒤로 숨기고 두 구현을 둔다:

```
FUSIONDTI_BACKEND=local   tools/를 직접 import (기본값 — 기존 사용법 그대로)
FUSIONDTI_BACKEND=http    web_pipeline/api.py를 HTTP로 호출 (컨테이너 구성)
```

두 구현은 **같은 dict 모양**을 돌려준다. `/predict`가 응답을
`{drug, protein, prediction, ...}`로 감싸는 것도 `backend.predict()`가
벗겨서 `tools.dti_tool.predict_binding`과 동일한 모양으로 맞춘다.
FastAPI의 `{"detail": ...}` 404도 `{"error": ...}`로 정규화한다.

기본값이 `local`이라 **기존 실행 방법은 아무것도 바뀌지 않는다.**

---

## 5. 컨테이너 — `docker/`, `docker-compose.yml`

```
postgres  MLflow 백엔드 스토어
mlflow    실험 추적 UI/REST     → localhost:5000
api       FastAPI 추론 (GPU)    → localhost:8000/docs
ui        Streamlit (CPU)       → localhost:8501
```

```bash
docker compose up -d postgres mlflow       # 추적만 (GPU 불필요)
docker compose --profile serving up -d     # 추론 스택까지
```

### 결정 사항

- **이미지 두 개로 분리.** UI는 torch도 SaProt 가중치도 필요 없다 —
  추론은 HTTP로 API에 맡긴다. 같은 이미지를 쓰면 CPU 컨테이너가 3GB 넘는
  GPU 스택을 지고 다닌다. (rdkit은 UI에도 필요하다 — 분자 그림을 직접 그린다.)
- **모델 가중치는 이미지에 굽지 않는다.** `results/**/*.pt`는 수백 MB라
  재빌드마다 레이어가 통째로 무효화된다. HF 캐시(SaProt 2.6GB)는 더 크다.
  둘 다 볼륨으로 마운트한다.
- **Postgres를 쓰는 이유는 동시 쓰기.** 학습 스크립트와 백필이 같이 돌면
  SQLite는 `database is locked`로 죽는다.
- **healthcheck의 `start-period`가 180초.** 모델 로딩에 그만큼 걸리고,
  UI는 `condition: service_healthy`로 기다린다.

---

## 6. Kubernetes — `k8s/`

**범위를 정확히 표기한다:** 단일 GPU 노드 k3s를 대상으로 **작성**한 매니페스트이고,
검증한 것은 스키마와 참조 무결성뿐이다. 클러스터 적용은 아직 하지 않았다.
자세한 것은 [k8s/README.md](../k8s/README.md).

compose에서는 드러나지 않는 제약을 명시적으로 다룬다:

| 제약 | k8s에서의 표현 |
|---|---|
| GPU를 한 파드만 점유 | `nvidia.com/gpu: 1` — replicas를 2로 올리면 두 번째는 `Pending` |
| 롤링 업데이트가 GPU에서 교착 | `strategy: Recreate` |
| 모델 로딩 3분 동안 kubelet이 파드를 죽임 | `startupProbe` (failureThreshold 30) |
| 준비 전 트래픽 유입 | `readinessProbe`가 Service 엔드포인트에서 제외 |
| UI만 수평 확장 | `HorizontalPodAutoscaler`를 UI Deployment에만 |

---

## 7. CI — `.github/workflows/ci.yml`

GPU도 네트워크도 쓰지 않는 검증만 돌린다. 학습·추론은 CI 대상이 아니다
(SaProt 로딩만 수 분). 대신 그 위 계층을 전부 검증한다:

| 잡 | 내용 |
|---|---|
| `test` | pytest 26개 (Python 3.10 / 3.12), 그래프 컴파일, MLflow 이관 dry-run |
| `lint` | ruff (신규 코드만), 전체 구문 검사 |
| `containers` | `docker compose config`, hadolint, kubeconform |
| `build` | UI 이미지 빌드 (API는 torch CUDA wheel이 러너 디스크를 넘김) |

### 테스트가 실제로 잡는 것

모델 성능이 아니라 **제어 흐름**이다:

- 정상 경로가 도구 5개를 전부 부르는가
- 브랜드명 실패 → 보정 → 재시도가 실제로 일어나는가 (`pubchem` 호출 2회)
- 계속 실패해도 상한에서 멈추는가, 그리고 DTI를 안 부르는가
- OOD 표적이 차단이 아니라 경고인가
- AlphaFold 실패가 예측을 막지 않는가
- 병렬 분기 경고 3건이 서로 덮어쓰지 않는가
- `/predict` 래퍼가 벗겨져 `tools/`와 같은 모양이 되는가
- API가 죽어 있을 때 예외가 아니라 error dict가 나오는가

도구는 전부 목킹한다 (`tests/conftest.py`가 `sys.modules`를 교체). GPU도
외부 API도 필요 없어 2.8초에 끝난다.

---

## 8. 재현 절차

```bash
# 1. MLOps 의존성 (학습 스택과 분리 — torch/rdkit 불필요)
pip install -r requirements-mlops.txt

# 2. 테스트
pytest tests/ -v                    # 26 passed
ruff check agent/ mlops/ tests/ web_pipeline/backend.py

# 3. 에이전트 그래프
python -m agent.graph --draw
python -m agent.graph "Does Imatinib bind to BCR-ABL?" --verbose   # 실제 API 호출

# 4. 실험 추적
python -m mlops.backfill --dry-run
python -m mlops.backfill
mlflow ui --backend-store-uri sqlite:///mlflow.db    # → localhost:5000

# 5. 컨테이너 (Docker 필요)
docker compose config --quiet
docker compose up -d postgres mlflow
MLFLOW_TRACKING_URI=http://localhost:5000 python -m mlops.backfill
docker compose --profile serving up -d
```

### 검증 로그 (2026-09-30)

이 문서의 주장 중 **이 환경에서 실제로 실행해 확인한 것**과 **정적 검증만 한 것**을
구분해 둔다.

| 항목 | 상태 | 근거 |
|---|---|---|
| pytest 26개 | ✅ 실행 확인 | `26 passed in 2.84s` |
| ruff | ✅ 실행 확인 | `All checks passed!` |
| 그래프 컴파일 + Mermaid | ✅ 실행 확인 | 13개 노드, 조건부 엣지 6개 렌더링 |
| MLflow 백필 28개 | ✅ 실행 확인 | 기록 28 / 실패 0, 재실행 시 건너뜀 28 |
| MLflow 필터 쿼리·아티팩트 | ✅ 실행 확인 | `phase=1c and r>0.80` → 4건, 아티팩트 4종 업로드 |
| 학습 스크립트 훅 | ⚠️ 구문만 확인 | GPU/`bioinfo` 환경이 이 머신에 없어 실학습 미실행 |
| `docker compose` | ⚠️ 정적 검증만 | 이 머신에 Docker 미설치 — YAML 파싱 + 볼륨/의존성 참조 무결성만 확인 |
| Dockerfile 빌드 | ❌ 미실행 | 동일 사유. CI의 `containers`/`build` 잡이 첫 실검증이 된다 |
| k8s 매니페스트 | ⚠️ 정적 검증만 | YAML 파싱 + 리소스 간 참조 무결성(11개 리소스, Secret 예시 포함) 확인. 클러스터 적용 미실행 |

검증에 쓴 버전 (프로젝트 밖 임시 venv, Python 3.14.7):

```
mlflow==3.16.1   langgraph==1.2.12   langgraph-checkpoint-sqlite==3.1.1
pytest==9.1.1    ruff (latest)       pyyaml
```

> 실행하지 못한 항목은 Docker가 설치된 환경에서 `docker compose config` →
> `docker compose --profile serving up` 순으로 확인해야 한다.

---

## 9. 남은 작업

- [ ] Dockerfile 실제 빌드 검증 (GPU 머신 필요)
- [ ] k3s 클러스터 적용 및 HPA 동작 확인
- [ ] 예측 이력을 Postgres에 적재 (현재는 MLflow에 실험만)
- [ ] 단백질 임베딩 캐시(`cache/*.pt`)를 pgvector로 이관 → 유사 단백질 검색
- [ ] BindingDB 전처리를 Prefect/Airflow DAG로 정식화
- [ ] Phase 1h/1i 학습을 새 훅으로 재실행해 MLflow에 기록
