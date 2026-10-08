# Phase 6 — 자동화 층: SelfTest · n8n · Ralph Loop

Phase 5가 "시스템을 재현 가능하게" 만들었다면, Phase 6은 **사람이 보지 않는
동안에도 시스템이 자기를 점검하고 실험을 이어 가게** 만든다.

---

## 1. 층을 나눈 이유

오케스트레이션 도구가 둘이다. 경쟁이 아니라 담당 시간 단위가 다르다.

```
┌─ n8n (자동화 층) ── 초 ~ 분 ─────────────────────────────────┐
│  스케줄 · 배치 · 외부 이벤트 · 알림 · 사람에게 보고           │
│                                                              │
│   ① 야간 SelfTest  ② 배치 스크리닝                           │
│   ③ BindingDB 감시 ④ 웹훅 질의                               │
└──────────────────────────┬───────────────────────────────────┘
                           │ HTTP
┌──────────────────────────▼───────────────────────────────────┐
│  FastAPI  ──►  LangGraph (질의 1건의 제어 흐름) ── 밀리초     │
│                재시도 · OOD 분기 · 병렬 보강 · 체크포인팅     │
└──────────────────────────────────────────────────────────────┘
```

**왜 질의 경로에 n8n을 쓰지 않았나.** 기록된 추론 시간이
`infer_ms_mean ≈ 0.945ms`다. 질의 하나를 여러 n8n 노드로 쪼개면 노드 간 왕복이
추론보다 비싸진다. 게다가 워크플로 JSON은 단위 테스트가 어려워, 재시도나
OOD 판단 같은 로직을 넣는 순간 검증 수단이 사라진다.

그래서 원칙 하나를 지킨다:

> **n8n에는 로직을 두지 않는다. 배선만 한다.**

이 경계를 설명할 수 있다는 것 자체가 도구를 이해하고 쓴다는 증거다.

### 에이전트인가 서비스인가

처음 구상은 Query / DTI / ChemBERTa / Result / SelfTest 다섯 에이전트였다.
실제로 판단을 하는 것만 에이전트로 부르고 나머지는 이름을 바로잡았다.

| 구상 | 실제 | 왜 |
|---|---|---|
| Query Agent | ✅ Query Agent (`agent/llm.py`) | 엔티티 추출·모호성 해소 — 판단한다 |
| DTI Agent | 🔧 DTI **추론 서비스** (`tools/dti_tool.py`) | 입력→출력 함수. 판단하지 않는다 |
| ChemBERTa Agent | ❌ **분리하지 않음** | 아래 참조 |
| Result Agent | 🔧 `compose_answer` 노드 | 현재는 템플릿. LLM 리포트로 바꾸면 에이전트가 된다 |
| SelfTest Agent | ✅ SelfTest Agent (`selftest/`) | 무엇을 검사할지·통과인지·무엇을 알릴지 판단한다 |

**ChemBERTa를 떼지 않은 이유.** ft-ChemBERTa와 MLP Head는 Phase 1h에서 **함께
학습된 한 모델**이다. 떼면 요청마다 768차원 배열을 HTTP로 넘겨야 하는데,
추론 자체가 1ms 수준이라 순손해다. 다만 **배치 스크리닝**에서는 임베딩을
미리 계산해 캐싱하는 게 정석이고, 그때는 분리할 이유가 생긴다 — 온라인이
아니라 배치 경로에서.

---

## 2. SelfTest Agent — `selftest/`

### 골든셋: 정답을 주장하지 않는다

`data/davis_test.csv`의 `Label`은 0~1 정규화값이고, 현재 학습 스크립트는
DeepPurpose 로더로 pKd 5.00~10.54를 직접 받는다. 스케일이 다르다. 그래서
그 라벨을 정답으로 쓰지 않는다.

대신 **모델 자신의 출력을 기준선으로 스냅샷**하고 거기서의 이탈을 본다.
서빙 회귀 테스트의 정석이기도 하다 — "문헌값과 맞는가"가 아니라
"어제와 같은 답을 내는가"를 묻는 것이다.

| 계층 | 개수 | 입력 | 무엇을 잡나 |
|---|---|---|---|
| `model_level` | 20쌍 | SMILES + 아미노산 서열 | 가중치 교체, 양자화 변경, 3Di 캐시 유실 |
| `agent_level` | 10쌍 | 약물명 + 유전자명 | 이름 해석 체인, 외부 API 계약, OOD 경고 |
| `ordering` | 3건 | 위 id 참조 | 저해제가 음성 대조보다 높은가 |

`model_level`은 `davis_test.csv`에서 결정론적으로 뽑는다(단백질 20종·약물
20종이 겹치지 않게). `agent_level`은 손으로 골랐고 **어려운 경우를 일부러
섞었다**:

- `Gleevec` + `BCR-ABL` — 브랜드명과 융합 유전자 표기를 동시에 태워
  재시도 분기 두 개가 매일 실제로 실행되게 한다
- `Atorvastatin` + `HMGCR` — 비키나아제 표적. OOD 경고가 계속 붙는지 본다
- `Aspirin` + `EGFR`, `Caffeine` + `ABL1` — 음성 대조

### 검사 6종

| 검사 | 잡는 것 |
|---|---|
| `drift` | 쌍별 pKd가 기준선에서 0.05 초과 이탈 |
| `ordering` | 알려진 저해제 < 음성 대조. **기준선과 독립** — 모델을 새로 학습해 값이 통째로 바뀌어도 살아남는다 |
| `range` | NaN / inf / 불가능한 pKd. FP16 오버플로는 예외를 던지지 않고 조용히 NaN을 흘린다 |
| `latency` | 추론 p95 악화 |
| `ood_warning` | `expect: ood` 쌍에 경고가 실제로 붙었는가 |
| `contract` | PubChem/UniProt/AlphaFold 응답 스키마 변경 |

`contract`가 특히 중요하다. 외부 API가 필드 이름을 바꾸면 예측은 **계속
나오는데 값이 틀린다.** 예외도 안 나고 상태 코드도 200이다.

### 실행

```bash
python -m selftest.build_golden_set            # 골든셋 생성 (커밋돼 있음)
python -m selftest.runner --record-baseline    # 최초 1회 — 기준선 스냅샷
python -m selftest.runner                      # 이후 — 드리프트 점검
```

종료 코드: `0` 정상 / `1` 회귀 / `2` 실행 실패.
API로도 노출된다: `POST /selftest`.

### SelfTest 자신을 어떻게 믿나

감시 장치의 가장 위험한 실패는 **고장 난 채로 "이상 없음"을 보고하는 것**이다.
그래서 테스트는 고장을 일부러 주입하고 각 검사가 빨간불을 켜는지 본다:

```
가중치 이동(+0.3)      → drift가 잡는가        ✅
FP16 잡음(±0.01)       → 거짓 경보를 안 내는가  ✅   ← 이것도 중요하다
NaN 유출               → range가 잡는가         ✅
저해제/대조 순서 역전  → ordering이 잡는가      ✅
OOD 경고 소실          → ood_warning이 잡는가   ✅
PubChem 필드명 변경    → contract가 잡는가      ✅
기준선과 겹치는 쌍 0개 → 조용히 통과하지 않는가 ✅
```

18개 테스트, GPU·네트워크 없이 0.1초.

---

## 3. Ralph Loop — `ralph/`

"같은 지시를 완료될 때까지 반복"은 그 자체로는 무한 루프와 구별되지 않는다.
쓸 수 있게 만들려면 **정지 조건**이 전부다.

| 정지 조건 | 설정 |
|---|---|
| 목표 달성 | `--target 0.90` |
| 반복 예산 | `--max-iters` |
| 시간 예산 | `--max-seconds` |
| 개선 정체 | `--patience` (기본 3회) |
| 제안 고갈 | Proposer가 `None` 반환 |

이 프로젝트에서의 쓰임: **하이퍼파라미터 탐색**. Phase 1h에서 unfreeze 층수·
학습률·배치 크기를 손으로 바꿔 가며 돌린 반복을 자동화한다. MLflow에 이미
이력이 쌓여 있어 루프가 읽을 상태 저장소가 준비돼 있다.

갈아 끼울 수 있는 세 부분:

```
Proposer   GridProposer(결정론적) | LLMProposer(이력을 읽고 다음을 제안)
Executor   TrainingExecutor(subprocess) | DryRunExecutor | 테스트용 가짜
Loop       정지 조건 + 상태 영속화
```

**서브프로세스로 띄우는 이유:** 학습이 OOM이나 CUDA 오류로 죽어도 루프는
살아남아야 하고, 반복 사이에 GPU 메모리가 확실히 반환되어야 한다.

**상태를 디스크에 남기는 이유:** 학습 한 번이 수십 분이라 루프가 몇 시간
돈다. 끊기면 `--resume`으로 이어 간다. 이미 끝낸 조합은 다시 돌지 않는다.

**LLM 제안자의 함정:** LLM은 "중복하지 마라"고 지시해도 같은 설정을 다시
낸다. 그러면 루프가 제자리를 돈다. 중복 제안은 거부하고 격자로 폴백한다.
LLM이 "수렴했다"고 선언하면 `None`을 받아 정상 종료한다.

```bash
python -m ralph.loop --dry-run                  # 무엇을 돌릴지만 출력
python -m ralph.loop --max-iters 8 --target 0.90
python -m ralph.loop --proposer llm --resume
```

16개 테스트가 정지 조건 5가지를 각각 독립 검증한다 — 하나라도 새면 GPU
시간이 조용히 증발하기 때문이다.

---

## 4. n8n 워크플로 — `n8n/`

### executeCommand를 버린 이유

처음에는 n8n이 `python -m selftest.runner`를 직접 부르게 설계했다.
**동작하지 않는다** — `n8nio/n8n` 이미지는 Node 기반이라 Python도 torch도
없다. 프로젝트를 마운트해도 인터프리터가 없다.

해결: 실행이 필요한 것을 API가 노출하고 n8n은 HTTP로만 부른다. 그래서
`api.py`에 두 엔드포인트를 추가했다.

| 엔드포인트 | 하는 일 |
|---|---|
| `POST /agent/ask` | 자연어 질의 → LangGraph 전 경로 → 답변 + `trace` |
| `POST /selftest` | 골든셋 회귀 점검 → 리포트 JSON |

부수 효과로 n8n 이미지를 커스터마이즈할 필요가 없어졌고, 모델은 GPU 컨테이너
하나에만 남았다.

### 워크플로 4개

| 파일 | 트리거 | 하는 일 |
|---|---|---|
| `selftest_nightly` | 매일 02:00 | 회귀 점검 → **실패 시에만** Slack |
| `batch_screening` | 웹훅 | 약물 N개 × 표적 1개 → pKd 순위 → CSV + 요약 |
| `bindingdb_watch` | 매주 월 03:00 | 릴리스 변경 감지 → 알림 (**재학습은 안 건다**) |
| `webhook_query` | 웹훅 | 자연어 질의 → 에이전트 → JSON |

### 설계상 결정

**① 회귀는 HTTP 200으로 보고한다.** `/selftest`는 회귀를 찾아도 200에
`status="FAIL"`을 담는다. 5xx를 주면 n8n이 노드 실패로 처리해 알림 분기까지
가지 못한다. 회귀는 서버 오류가 아니라 보고할 사실이다.

**② 리포트가 안 나오는 것도 장애로 친다.** `Parse Report` 노드는 응답이
비었거나 형식이 다르면 `status: FAIL`을 만든다.

**③ 배치는 한 건씩.** GPU 1장에 동시 요청을 보내면 VRAM이 터진다.
`SplitInBatches(batchSize=1)` + `neverError`로 개별 실패를 흡수한다.

**④ BindingDB 감시는 알리기만 한다.** 재학습은 수 시간짜리 GPU 작업이다.
사람 확인 없이 트리거하지 않는다 — n8n을 쓸 때 가장 흔한 실수가 이 경계를
흐리는 것이다.

### 워크플로도 코드로 관리한다

UI에서 만들면 diff를 읽을 수 없다. `n8n/build_workflows.py`가 JSON을
생성하고, 그 JSON을 커밋한다. CI가 `--check`로 최신 여부를 확인한다.

정적 검증 58개가 인스턴스 없이 잡는 것:

- 없는 노드를 가리키는 연결 (import는 되는데 실행이 조용히 끊긴다)
- 고아 노드 / 트리거에서 도달 불가한 노드
- IF의 false 분기, SplitInBatches의 done 분기 누락
- Code 노드의 JavaScript 구문 오류 (`node --check`로 실제 파싱)
- 하드코딩된 시크릿
- 워크플로가 부르는 API 경로가 `api.py`에 실재하는지
- URL 호스트가 compose 서비스명과 맞는지 (`localhost`면 자기 자신을 가리킨다)
- 느린 호출에 충분한 타임아웃과 `neverError`가 걸렸는지

---

## 5. 검증 로그 (2026-09-30)

| 항목 | 상태 | 근거 |
|---|---|---|
| 전체 테스트 118개 | ✅ 실행 확인 | `118 passed in 3.22s` |
| SelfTest 고장 주입 18개 | ✅ 실행 확인 | 6종 검사가 모두 빨간불을 켬 |
| Ralph 정지 조건 16개 | ✅ 실행 확인 | 5가지 조건 각각 독립 검증 |
| n8n 워크플로 58개 | ✅ 실행 확인 | Code 노드는 `node --check`로 실제 파싱 |
| 골든셋 생성 | ✅ 실행 확인 | 20+10쌍, 서열이 유효 아미노산인지 확인 |
| Ralph dry-run | ✅ 실행 확인 | 격자 6개 명령 조립 |
| ruff | ✅ 실행 확인 | `All checks passed!` |
| `/agent/ask`, `/selftest` | ⚠️ 구문만 확인 | 모델 로딩이 필요해 이 머신에서 미실행 |
| SelfTest 실제 실행 | ❌ 미실행 | GPU·`bioinfo` 환경 없음. 기준선도 아직 없다 |
| n8n import 후 실행 | ❌ 미실행 | Docker 미설치. 노드 typeVersion 호환성 미확인 |

> 야간 SelfTest를 켜기 전에 반드시 `--record-baseline`을 한 번 돌려야 한다.
> 기준선이 없으면 drift 검사가 매번 `SKIP`이라 아무것도 잡지 못한다.

---

## 6. 남은 작업

- [ ] GPU 머신에서 `selftest.runner --record-baseline` 실행 → `baseline.json` 커밋
- [ ] n8n 인스턴스에 워크플로 import → 노드 버전 호환성 확인
- [ ] `Result Agent`를 템플릿에서 LLM 리포트 생성으로 승격
- [ ] 배치 경로용 ChemBERTa 임베딩 서비스 + 캐시 (온라인 경로는 그대로)
- [ ] Ralph 루프를 실제 학습에 한 번 돌려 MLflow `FusionDTI-Ralph` 채우기
