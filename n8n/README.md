# n8n 워크플로 — 자동화 층

## 왜 n8n을 질의 경로에 넣지 않았나

이 프로젝트에는 오케스트레이션 도구가 둘 있다. 경쟁이 아니라 **층이 다르다.**

| | LangGraph (`agent/`) | n8n (`n8n/`) |
|---|---|---|
| 담당 | 질의 1건의 제어 흐름 | 스케줄·배치·알림·외부 연동 |
| 시간 단위 | 밀리초 | 초 ~ 분 |
| 상태 | 타입 있는 `DTIState` + 체크포인터 | 워크플로 스태틱 데이터 |
| 테스트 | pytest 19개 | 정적 검증 58개 (인스턴스 없이) |
| 버전 관리 | Python diff | JSON diff (읽기 어려움) |

기록된 추론 시간이 `infer_ms_mean ≈ 0.945ms`다. 질의 하나를 5개 n8n 노드로
쪼개면 노드 간 HTTP 왕복이 추론 자체보다 훨씬 비싸진다. 게다가 워크플로 JSON은
단위 테스트가 어려워서 재시도·OOD 분기 같은 **판단 로직**을 넣으면 검증할 방법이
사라진다.

그래서 원칙은 하나다:

> **n8n에는 로직을 두지 않는다. 배선만 한다.**
> 언제 부를지(크론/웹훅), 어디로 알릴지(Slack), 실패하면 어디로 갈지.
> 판단은 전부 Python(`selftest/`, `ralph/`, FastAPI)에 있다.

## 왜 executeCommand를 쓰지 않았나

처음에는 n8n이 `python -m selftest.runner`를 직접 부르게 설계했다. **동작하지
않는다** — `n8nio/n8n` 이미지는 Node 기반이라 Python도 torch도 없다. 프로젝트를
마운트해도 인터프리터가 없다.

해결: 실행이 필요한 것을 API가 노출하고 n8n은 HTTP로만 부른다.

| 워크플로가 부르는 곳 | api.py |
|---|---|
| `POST /selftest` | 골든셋 회귀 점검 → 리포트 JSON |
| `POST /agent/ask` | 자연어 질의 → LangGraph 전 경로 |
| `POST /predict` | SMILES/이름 → pKd |

부수 효과로 n8n 이미지를 커스터마이즈할 필요가 없어졌고, 모델은 GPU 컨테이너
하나에만 남았다.

## 워크플로 4개

| 파일 | 트리거 | 하는 일 |
|---|---|---|
| `selftest_nightly.json` | 매일 02:00 | 골든셋 회귀·드리프트 점검 → **실패 시에만** Slack |
| `batch_screening.json` | 웹훅 `POST /webhook/fusiondti/screen` | 약물 N개 × 표적 1개 → pKd 순위 → CSV + Slack 요약 |
| `bindingdb_watch.json` | 매주 월 03:00 | BindingDB 릴리스 변경 감지 → 알림 (**재학습은 안 건다**) |
| `webhook_query.json` | 웹훅 `POST /webhook/fusiondti/ask` | 자연어 질의 → 에이전트 → JSON 응답 |

### 설계상 결정

**① SelfTest는 회귀를 200으로 보고한다.**
`/selftest`는 회귀를 찾아도 HTTP 200에 `status="FAIL"`을 담는다. 5xx를 주면
n8n이 노드 실패로 처리해 알림 분기까지 가지 못한다. 회귀는 서버 오류가 아니라
**보고할 사실**이다.

**② 리포트가 안 나오는 것도 장애로 친다.**
`Parse Report` 노드는 응답이 비었거나 형식이 다르면 `status: FAIL`을 만들어
낸다. 감시 장치가 죽은 채로 "이상 없음"을 보고하는 게 가장 나쁜 실패다.

**③ 배치는 한 건씩 보낸다.**
`SplitInBatches(batchSize=1)` — GPU가 1장뿐이라 동시 요청을 보내면 VRAM이
터진다. 개별 요청 실패는 `neverError`로 흡수해 배치 전체를 멈추지 않는다.

**④ BindingDB 감시는 알리기만 한다.**
재학습은 수 시간짜리 GPU 작업이다. 사람 확인 없이 트리거하지 않는다.
n8n을 쓸 때 가장 흔한 실수가 이 경계를 흐리는 것이다.

**⑤ Slack은 자격증명 노드 대신 Incoming Webhook URL.**
`$env.SLACK_WEBHOOK_URL`을 쓰므로 워크플로가 이 저장소 밖 설정에 묶이지 않는다.

## 사용

```bash
# 1. 추론 스택 + n8n 기동
export SLACK_WEBHOOK_URL='https://hooks.slack.com/services/...'   # 선택
docker compose --profile serving --profile automation up -d

# 2. http://localhost:5678 접속 → 계정 생성
# 3. 각 워크플로를 Import from File (컨테이너 안 /workflows/*.json)
# 4. 켜기 전에 SelfTest 기준선을 먼저 기록한다 (아래)
```

### 기준선 기록이 먼저다

기준선 없이 야간 SelfTest를 켜면 drift 검사가 매번 `SKIP`이라 아무것도 잡지
못한다. 모델을 배포한 직후 한 번 실행한다:

```bash
python -m selftest.build_golden_set        # 골든셋 생성 (이미 커밋돼 있음)
python -m selftest.runner --record-baseline
```

`selftest/baseline.json`은 **모델 자신의 출력 스냅샷**이다. 문헌값이 아니다.
모델을 의도적으로 바꿨을 때만 다시 기록한다.

## 워크플로를 고치는 법

UI에서 고치면 diff를 읽을 수 없다. **`build_workflows.py`를 고치고 재생성한다.**

```bash
python -m n8n.build_workflows        # JSON 재생성
python -m n8n.build_workflows --check  # CI가 쓰는 최신 여부 확인
pytest tests/test_n8n_workflows.py   # 58개 정적 검증
```

## 검증 수준 — 정직하게

| 항목 | 상태 |
|---|---|
| JSON 구조·연결 무결성·고아 노드·분기 출력 | ✅ 자동 검증 (pytest 58개) |
| Code 노드 JavaScript 구문 | ✅ `node --check`로 실제 파싱 |
| API 경로가 `api.py`에 실재하는지 | ✅ 자동 검증 |
| compose 서비스명·타임아웃·`neverError` | ✅ 자동 검증 |
| **n8n 인스턴스에 import 후 실행** | ❌ **미실행** |
| 노드 typeVersion이 n8n 1.70.1과 호환되는지 | ❌ 미확인 |

import 후 노드에 버전 경고가 뜨면 UI에서 해당 노드를 다시 선택해 저장한 뒤,
export한 JSON을 `build_workflows.py`에 반영하면 된다.
