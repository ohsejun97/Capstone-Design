"""
build_workflows.py — n8n 워크플로 JSON 생성

n8n 워크플로는 손으로 쓰기엔 반복이 많고(노드마다 id·position·typeVersion),
UI에서 만들면 diff를 읽을 수 없다. 그래서 파이썬으로 생성하고 JSON을 커밋한다.
구조가 바뀌면 이 파일의 diff로 의도가 드러난다.

설계 원칙 — **n8n에는 로직을 두지 않는다.**
    워크플로 JSON은 단위 테스트도 코드 리뷰도 어렵다. 그래서 판단은 전부
    파이썬(selftest/, ralph/, FastAPI)에 두고, n8n은 **배선만** 한다:
    언제 부를지(크론/웹훅), 어디로 보낼지(Slack/시트), 실패하면 무엇을 할지.

Slack은 자격증명 노드 대신 Incoming Webhook URL로 보낸다. 자격증명을 요구하면
워크플로가 이 저장소 밖 설정에 묶여 재현이 어려워진다.

사용:
    python -m n8n.build_workflows          # n8n/*.json 생성
    python -m n8n.build_workflows --check  # 생성물이 최신인지만 확인 (CI)
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import pairwise
from pathlib import Path

HERE = Path(__file__).resolve().parent

# n8n 컨테이너에는 Python도 torch도 없다. 그래서 스크립트를 executeCommand로
# 돌리지 않고, 모델과 코드를 가진 API 컨테이너가 노출한 엔드포인트를 호출한다.
# 부수 효과로 n8n 이미지를 커스터마이즈할 필요가 없어졌다.
API_BASE = "http://api:8000"


def node(name: str, type_: str, type_version: float, position: tuple[int, int],
         parameters: dict, **extra) -> dict:
    """n8n 노드 하나. id는 이름에서 결정론적으로 만든다 — 재생성해도 diff가 안 생기게."""
    return {
        "parameters": parameters,
        "id": name.lower().replace(" ", "-").replace("/", "-"),
        "name": name,
        "type": type_,
        "typeVersion": type_version,
        "position": list(position),
        **extra,
    }


def connect(*chain: str) -> dict:
    """a → b → c 선형 연결."""
    links: dict = {}
    for src, dst in pairwise(chain):
        links[src] = {"main": [[{"node": dst, "type": "main", "index": 0}]]}
    return links


def workflow(name: str, nodes: list[dict], connections: dict,
             description: str) -> dict:
    return {
        "name": name,
        "nodes": nodes,
        "connections": connections,
        "settings": {"executionOrder": "v1"},
        "meta": {"description": description},
        "tags": [{"name": "fusiondti"}],
    }


def slack_node(name: str, position: tuple[int, int], text_expr: str) -> dict:
    """Slack Incoming Webhook으로 메시지 전송.

    $env.SLACK_WEBHOOK_URL 를 쓰므로 n8n 컨테이너에 그 환경변수만 주면 된다.
    """
    return node(name, "n8n-nodes-base.httpRequest", 4.2, position, {
        "method": "POST",
        "url": "={{ $env.SLACK_WEBHOOK_URL }}",
        "sendBody": True,
        "specifyBody": "json",
        "jsonBody": f"={{{{ JSON.stringify({{ text: {text_expr} }}) }}}}",
        "options": {},
    })


# ══════════════════════════════════════════════════════════════════════════════
# 1. 야간 SelfTest
# ══════════════════════════════════════════════════════════════════════════════

def selftest_nightly() -> dict:
    """매일 02:00 — 골든셋 회귀 점검 → 실패 시에만 Slack 알림.

    API의 POST /selftest 를 부른다. 그 엔드포인트는 회귀를 발견해도 HTTP 200에
    status="FAIL"을 담아 돌려준다 — 회귀는 서버 오류가 아니라 보고할 사실이고,
    5xx로 주면 n8n이 노드 실패로 처리해 알림 분기까지 못 간다.
    """
    nodes = [
        node("Nightly 02:00", "n8n-nodes-base.scheduleTrigger", 1.2, (0, 300), {
            "rule": {"interval": [{"field": "cronExpression",
                                   "expression": "0 2 * * *"}]},
        }),
        node("Run SelfTest", "n8n-nodes-base.httpRequest", 4.2, (220, 300), {
            "method": "POST",
            "url": f"{API_BASE}/selftest",
            "sendQuery": True,
            "queryParameters": {"parameters": [
                {"name": "skip_contracts", "value": "false"},
            ]},
            "options": {
                # 골든셋 30쌍 예측 + 외부 API 계약 검사 — 몇 분 걸릴 수 있다.
                "timeout": 900000,
                # 500이 나도 노드를 실패시키지 않고 본문을 다음 노드로 넘긴다.
                "response": {"response": {"neverError": True,
                                          "fullResponse": True}},
            },
        }),
        node("Parse Report", "n8n-nodes-base.code", 2, (440, 300), {
            "jsCode": """
const res = $input.first().json;
const report = res.body || {};

// API가 리포트를 못 낸 것도 장애다 — 조용히 통과시키면 감시가 죽은 줄 모른다.
if (res.statusCode !== 200 || !Array.isArray(report.checks)) {
  return [{ json: {
    status: 'FAIL',
    n_failed_checks: -1,
    failed_checks: ['selftest_endpoint_error'],
    summary: `SelfTest 엔드포인트 오류 (HTTP ${res.statusCode}): `
             + String(report.detail || JSON.stringify(report)).slice(0, 400),
  }}];
}

const failed = report.checks.filter(c => c.status === 'FAIL');
report.summary = failed.map(c => `• ${c.name}: ${c.summary}`).join('\\n')
                 || '모든 검사 통과';
return [{ json: report }];
""".strip(),
        }),
        node("Regression?", "n8n-nodes-base.if", 2, (660, 300), {
            "conditions": {
                "options": {"caseSensitive": True, "version": 2},
                "conditions": [{
                    "leftValue": "={{ $json.status }}",
                    "rightValue": "FAIL",
                    "operator": {"type": "string", "operation": "equals"},
                }],
                "combinator": "and",
            },
        }),
        slack_node("Alert Slack", (900, 200),
                   "'🚨 FusionDTI SelfTest 회귀 감지\\n\\n' + $json.summary "
                   "+ '\\n\\n리포트: results/selftest/latest.json'"),
        node("No Action", "n8n-nodes-base.noOp", 1, (900, 400), {}),
    ]

    connections = connect("Nightly 02:00", "Run SelfTest", "Parse Report",
                          "Regression?")
    # IF 노드는 출력이 둘이다 — [0]=true(회귀), [1]=false(정상)
    connections["Regression?"] = {"main": [
        [{"node": "Alert Slack", "type": "main", "index": 0}],
        [{"node": "No Action", "type": "main", "index": 0}],
    ]}

    return workflow(
        "FusionDTI — Nightly SelfTest", nodes, connections,
        "매일 02:00 골든셋 회귀·드리프트 점검. 실패 시에만 Slack 알림.",
    )


# ══════════════════════════════════════════════════════════════════════════════
# 2. 배치 스크리닝
# ══════════════════════════════════════════════════════════════════════════════

def batch_screening() -> dict:
    """약물 목록 × 표적 1개 → 예측 → pKd 순위 → CSV + Slack 요약.

    배치에서만 의미 있는 경로다. 온라인 질의와 달리 수백~수천 건이므로
    한 건씩 API를 때리고 결과를 모은다. SplitInBatches로 배치당 1건씩 보내
    GPU 서버가 동시 요청에 압사하지 않게 한다.
    """
    nodes = [
        node("Manual / Webhook", "n8n-nodes-base.webhook", 2, (0, 300), {
            "path": "fusiondti/screen",
            "httpMethod": "POST",
            "responseMode": "lastNode",
            "options": {},
        }),
        node("Read Drug List", "n8n-nodes-base.code", 2, (220, 300), {
            "jsCode": """
// 입력: { "target": "ABL1", "drugs": ["Imatinib", "Dasatinib", ...] }
// 또는  { "target": "ABL1", "drugs_csv": "Imatinib,Dasatinib" }
const body = $input.first().json.body || $input.first().json;
const target = body.target;
if (!target) throw new Error('target이 없습니다.');

let drugs = body.drugs;
if (!drugs && body.drugs_csv) {
  drugs = String(body.drugs_csv).split(',').map(s => s.trim()).filter(Boolean);
}
if (!Array.isArray(drugs) || drugs.length === 0) {
  throw new Error('drugs 배열이 비어 있습니다.');
}

// 한 건씩 흐르도록 item으로 펼친다.
return drugs.map(d => ({ json: { drug_name: d, protein_name: target } }));
""".strip(),
        }),
        node("One At A Time", "n8n-nodes-base.splitInBatches", 3, (440, 300), {
            "batchSize": 1,
            "options": {},
        }),
        node("Predict", "n8n-nodes-base.httpRequest", 4.2, (660, 400), {
            "method": "POST",
            "url": f"{API_BASE}/predict",
            "sendBody": True,
            "specifyBody": "json",
            "jsonBody": ("={{ JSON.stringify({ drug_name: $json.drug_name, "
                         "protein_name: $json.protein_name }) }}"),
            "options": {
                # 모델 로딩/추론이 느릴 수 있다. 그리고 한 약물이 실패해도
                # 배치 전체를 멈추면 안 된다.
                "timeout": 120000,
                "response": {"response": {"neverError": True,
                                          "fullResponse": True}},
            },
        }),
        node("Rank & Summarize", "n8n-nodes-base.code", 2, (880, 300), {
            "jsCode": """
// 모든 배치의 응답을 모아 pKd 내림차순으로 정렬한다.
const rows = [];
for (const item of $input.all()) {
  const status = item.json.statusCode;
  const body = item.json.body || {};
  if (status !== 200) {
    rows.push({ drug: body.query_name || '?', pKd: null,
                error: body.detail || `HTTP ${status}` });
    continue;
  }
  rows.push({
    drug: (body.drug && body.drug.name) || '?',
    target: (body.protein && body.protein.gene) || '?',
    pKd: body.prediction ? body.prediction.pKd : null,
    interpretation: body.prediction ? body.prediction.interpretation : '',
    used_3di: body.prediction ? body.prediction.used_3di : null,
  });
}

const ok = rows.filter(r => r.pKd !== null).sort((a, b) => b.pKd - a.pKd);
const failed = rows.filter(r => r.pKd === null);

const top = ok.slice(0, 10)
  .map((r, i) => `${i + 1}. ${r.drug}  pKd ${r.pKd.toFixed(2)}  ${r.interpretation}`)
  .join('\\n');

return [{ json: {
  target: ok.length ? ok[0].target : '?',
  n_total: rows.length,
  n_ok: ok.length,
  n_failed: failed.length,
  top10: top,
  results: ok,
  failures: failed,
}}];
""".strip(),
        }),
        node("Write CSV", "n8n-nodes-base.convertToFile", 1.1, (1100, 200), {
            "operation": "csv",
            "options": {"fileName": "={{ 'screening_' + $json.target + '.csv' }}"},
        }),
        slack_node("Post Summary", (1100, 400),
                   "'🧪 배치 스크리닝 완료 — 표적 ' + $json.target + '\\n' "
                   "+ $json.n_ok + '/' + $json.n_total + '건 성공\\n\\n' + $json.top10"),
    ]

    connections = connect("Manual / Webhook", "Read Drug List", "One At A Time")
    # SplitInBatches: 출력 0 = 완료(done), 출력 1 = 배치(loop)
    connections["One At A Time"] = {"main": [
        [{"node": "Rank & Summarize", "type": "main", "index": 0}],
        [{"node": "Predict", "type": "main", "index": 0}],
    ]}
    connections["Predict"] = {"main": [
        [{"node": "One At A Time", "type": "main", "index": 0}]]}
    connections["Rank & Summarize"] = {"main": [[
        {"node": "Write CSV", "type": "main", "index": 0},
        {"node": "Post Summary", "type": "main", "index": 0},
    ]]}

    return workflow(
        "FusionDTI — Batch Screening", nodes, connections,
        "약물 목록 × 표적 1개를 일괄 예측해 pKd 순위와 CSV를 낸다.",
    )


# ══════════════════════════════════════════════════════════════════════════════
# 3. BindingDB 갱신 감시
# ══════════════════════════════════════════════════════════════════════════════

def bindingdb_watch() -> dict:
    """주 1회 BindingDB 릴리스를 확인하고 바뀌었으면 알린다.

    **자동 재학습은 걸지 않는다.** 학습은 수 시간짜리 GPU 작업이라
    사람 확인 없이 트리거하면 안 된다. 알림만 보내고 결정은 사람이 한다 —
    n8n을 쓸 때 가장 자주 하는 실수가 이 경계를 흐리는 것이다.
    """
    nodes = [
        node("Weekly Mon 03:00", "n8n-nodes-base.scheduleTrigger", 1.2, (0, 300), {
            "rule": {"interval": [{"field": "cronExpression",
                                   "expression": "0 3 * * 1"}]},
        }),
        node("Fetch Release Page", "n8n-nodes-base.httpRequest", 4.2, (220, 300), {
            "method": "GET",
            "url": "https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp",
            "options": {"timeout": 30000,
                        "response": {"response": {"neverError": True}}},
        }),
        node("Extract Version", "n8n-nodes-base.code", 2, (440, 300), {
            "jsCode": """
// 페이지에서 BindingDB_All_<YYYYmM>.tsv 형태의 릴리스 표기를 찾는다.
// 파싱이 깨지면 조용히 통과시키지 않고 그 사실을 알린다 — 감시가 죽은 채로
// '변화 없음'을 보고하는 게 가장 나쁜 실패다.
const html = String($input.first().json.data || $input.first().json.body || '');
const m = html.match(/BindingDB_All[_-]?(\\d{6,8}|\\d{4}m\\d{1,2})/i);

if (!m) {
  return [{ json: { parsed: false, version: null,
                    note: '릴리스 표기를 찾지 못했습니다 — 페이지 구조가 바뀌었을 수 있습니다.' }}];
}
return [{ json: { parsed: true, version: m[1], note: '' } }];
""".strip(),
        }),
        node("Compare To Known", "n8n-nodes-base.code", 2, (660, 300), {
            "jsCode": """
// 마지막으로 본 버전을 워크플로 스태틱 데이터에 보관한다
// (외부 저장소 없이 n8n 안에서 상태를 유지하는 표준 방법).
const staticData = $getWorkflowStaticData('global');
const current = $json.version;

if (!$json.parsed) {
  return [{ json: { changed: true, reason: 'parse_failed',
                    message: $json.note, version: null } }];
}

const previous = staticData.bindingdb_version || null;
staticData.bindingdb_version = current;

if (previous === null) {
  return [{ json: { changed: false, reason: 'first_run',
                    message: `최초 실행 — 현재 버전 ${current} 기록`, version: current } }];
}
if (previous === current) {
  return [{ json: { changed: false, reason: 'unchanged',
                    message: `변화 없음 (${current})`, version: current } }];
}
return [{ json: { changed: true, reason: 'new_release',
                  message: `새 릴리스: ${previous} → ${current}`, version: current } }];
""".strip(),
        }),
        node("Changed?", "n8n-nodes-base.if", 2, (880, 300), {
            "conditions": {
                "options": {"caseSensitive": True, "version": 2},
                "conditions": [{
                    "leftValue": "={{ $json.changed }}",
                    "rightValue": True,
                    "operator": {"type": "boolean", "operation": "true",
                                 "singleValue": True},
                }],
                "combinator": "and",
            },
        }),
        slack_node("Notify", (1120, 200),
                   "'📦 BindingDB 감시\\n' + $json.message + "
                   "'\\n\\n재학습은 자동으로 돌지 않습니다. 필요하면 GPU 머신에서 "
                   "`python -m ralph.loop` 를 직접 실행하세요.'"),
        node("Quiet", "n8n-nodes-base.noOp", 1, (1120, 400), {}),
    ]

    connections = connect("Weekly Mon 03:00", "Fetch Release Page",
                          "Extract Version", "Compare To Known", "Changed?")
    connections["Changed?"] = {"main": [
        [{"node": "Notify", "type": "main", "index": 0}],
        [{"node": "Quiet", "type": "main", "index": 0}],
    ]}

    return workflow(
        "FusionDTI — BindingDB Watch", nodes, connections,
        "주 1회 BindingDB 릴리스 변경 감지. 알림만 보내고 재학습은 사람이 결정.",
    )


# ══════════════════════════════════════════════════════════════════════════════
# 4. 웹훅 질의
# ══════════════════════════════════════════════════════════════════════════════

def webhook_query() -> dict:
    """외부 폼/봇이 자연어 질의를 던지면 포맷된 답을 돌려준다.

    LangGraph 에이전트를 그대로 태운다 — n8n에서 도구를 다시 배선하지 않는다.
    """
    nodes = [
        node("Webhook", "n8n-nodes-base.webhook", 2, (0, 300), {
            "path": "fusiondti/ask",
            "httpMethod": "POST",
            "responseMode": "responseNode",
            "options": {},
        }),
        node("Run Agent", "n8n-nodes-base.httpRequest", 4.2, (220, 300), {
            "method": "POST",
            "url": f"{API_BASE}/agent/ask",
            "sendBody": True,
            "specifyBody": "json",
            "jsonBody": "={{ JSON.stringify({ query: $json.body.query || '' }) }}",
            "options": {
                "timeout": 180000,
                "response": {"response": {"neverError": True,
                                          "fullResponse": True}},
            },
        }),
        node("Format Answer", "n8n-nodes-base.code", 2, (440, 300), {
            "jsCode": """
const res = $input.first().json;
const body = res.body || {};

if (res.statusCode !== 200) {
  return [{ json: {
    ok: false,
    answer: `요청을 처리하지 못했습니다 (HTTP ${res.statusCode}).`,
    detail: String(body.detail || '').slice(0, 300),
  }}];
}

return [{ json: {
  ok: Boolean(body.ok),
  answer: body.answer || '에이전트가 응답을 내지 못했습니다.',
  pKd: body.prediction ? body.prediction.pKd : null,
  warnings: body.warnings || [],
}}];
""".strip(),
        }),
        node("Respond", "n8n-nodes-base.respondToWebhook", 1.1, (660, 300), {
            "respondWith": "json",
            "responseBody": "={{ JSON.stringify($json) }}",
            "options": {},
        }),
    ]

    return workflow(
        "FusionDTI — Webhook Query",
        nodes,
        connect("Webhook", "Run Agent", "Format Answer", "Respond"),
        "POST /webhook/fusiondti/ask 로 자연어 질의 → API /agent/ask → JSON 응답.",
    )


# ══════════════════════════════════════════════════════════════════════════════

BUILDERS = {
    "selftest_nightly": selftest_nightly,
    "batch_screening": batch_screening,
    "bindingdb_watch": bindingdb_watch,
    "webhook_query": webhook_query,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true",
                        help="파일을 쓰지 않고 현재 커밋된 JSON과 같은지 확인")
    args = parser.parse_args()

    stale = []
    for name, build in BUILDERS.items():
        path = HERE / f"{name}.json"
        content = json.dumps(build(), indent=2, ensure_ascii=False) + "\n"

        if args.check:
            if not path.exists() or path.read_text() != content:
                stale.append(name)
            continue

        path.write_text(content)
        data = json.loads(content)
        print(f"{path.name:<28} 노드 {len(data['nodes'])}개")

    if args.check:
        if stale:
            print("생성물이 최신이 아닙니다: " + ", ".join(stale), file=sys.stderr)
            print("  python -m n8n.build_workflows 를 실행하고 커밋하세요.",
                  file=sys.stderr)
            return 1
        print("n8n 워크플로 JSON이 최신입니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
