"""
test_n8n_workflows.py — n8n 워크플로 JSON 정적 검증

n8n 워크플로는 단위 테스트가 어렵다. 실제로 돌려 보려면 n8n 인스턴스와
외부 서비스가 필요하고, 그러다 보니 보통 아무도 검증하지 않은 채 커밋된다.

여기서는 **인스턴스 없이 잡을 수 있는 것**을 전부 잡는다:

  · 없는 노드를 가리키는 연결 (import는 되지만 실행이 조용히 끊긴다)
  · 어디에도 연결되지 않은 고아 노드
  · 트리거 노드 부재 (수동으로만 돌아가는 워크플로가 되어 버린다)
  · 분기 노드(IF/SplitInBatches)의 출력 개수 불일치
  · Code 노드의 JavaScript 구문 오류
  · 하드코딩된 시크릿
  · API/경로가 docker-compose 구성과 어긋남
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

N8N_DIR = Path(__file__).resolve().parent.parent / "n8n"
WORKFLOW_PATHS = sorted(N8N_DIR.glob("*.json"))

TRIGGER_TYPES = {
    "n8n-nodes-base.scheduleTrigger",
    "n8n-nodes-base.webhook",
    "n8n-nodes-base.cron",
    "n8n-nodes-base.manualTrigger",
}

# 출력이 2개인 노드. 연결이 하나만 달려 있으면 한쪽 분기가 죽은 것이다.
TWO_OUTPUT_TYPES = {
    "n8n-nodes-base.if": ("true", "false"),
    "n8n-nodes-base.splitInBatches": ("done", "loop"),
}


def load(path: Path) -> dict:
    return json.loads(path.read_text())


@pytest.fixture(params=WORKFLOW_PATHS, ids=lambda p: p.stem)
def wf(request) -> dict:
    return load(request.param)


def test_workflows_exist():
    assert WORKFLOW_PATHS, "n8n/*.json 이 하나도 없다"
    assert len(WORKFLOW_PATHS) == 4


def test_generated_files_are_up_to_date():
    """build_workflows.py를 고치고 JSON 재생성을 잊는 경우를 잡는다."""
    from n8n.build_workflows import BUILDERS

    for name, build in BUILDERS.items():
        path = N8N_DIR / f"{name}.json"
        expected = json.dumps(build(), indent=2, ensure_ascii=False) + "\n"
        assert path.read_text() == expected, (
            f"{name}.json이 오래됐다 — python -m n8n.build_workflows 실행 필요")


# ── 구조 ──────────────────────────────────────────────────────────────────────

def test_required_top_level_keys(wf):
    for key in ("name", "nodes", "connections", "settings"):
        assert key in wf, f"최상위 키 '{key}' 누락"
    assert wf["nodes"], "노드가 비어 있다"


def test_node_ids_and_names_are_unique(wf):
    names = [n["name"] for n in wf["nodes"]]
    ids = [n["id"] for n in wf["nodes"]]
    assert len(names) == len(set(names)), f"노드 이름 중복: {names}"
    assert len(ids) == len(set(ids)), f"노드 id 중복: {ids}"


def test_nodes_have_required_fields(wf):
    for n in wf["nodes"]:
        for key in ("parameters", "id", "name", "type", "typeVersion", "position"):
            assert key in n, f"{n.get('name', '?')}: '{key}' 누락"
        assert n["type"].startswith("n8n-nodes-base."), \
            f"{n['name']}: 예상 밖 노드 타입 {n['type']}"
        assert len(n["position"]) == 2


def test_connections_reference_existing_nodes(wf):
    """가장 흔하고 가장 조용한 오류 — import는 되는데 실행이 안 이어진다."""
    names = {n["name"] for n in wf["nodes"]}

    for source, outputs in wf["connections"].items():
        assert source in names, f"연결 출발점 '{source}'가 노드에 없다"
        for branch in outputs.get("main", []):
            for link in branch:
                assert link["node"] in names, \
                    f"'{source}' → 없는 노드 '{link['node']}'"


def test_no_orphan_nodes(wf):
    """트리거를 제외한 모든 노드는 무언가로부터 입력을 받아야 한다."""
    targets = {
        link["node"]
        for outputs in wf["connections"].values()
        for branch in outputs.get("main", [])
        for link in branch
    }
    for n in wf["nodes"]:
        if n["type"] in TRIGGER_TYPES:
            continue
        assert n["name"] in targets, f"고아 노드: {n['name']}"


def test_has_exactly_one_trigger(wf):
    triggers = [n for n in wf["nodes"] if n["type"] in TRIGGER_TYPES]
    assert len(triggers) == 1, \
        f"트리거가 {len(triggers)}개 — 정확히 1개여야 한다"


def test_branching_nodes_wire_both_outputs(wf):
    """IF의 false 분기나 SplitInBatches의 done 분기를 빠뜨리는 실수를 잡는다."""
    for n in wf["nodes"]:
        labels = TWO_OUTPUT_TYPES.get(n["type"])
        if labels is None:
            continue
        outputs = wf["connections"].get(n["name"], {}).get("main", [])
        assert len(outputs) == 2, (
            f"{n['name']}({n['type']}): 출력 {len(outputs)}개 — "
            f"{labels[0]}/{labels[1]} 둘 다 연결해야 한다")
        for index, label in enumerate(labels):
            assert outputs[index], f"{n['name']}: '{label}' 분기가 비어 있다"


def test_graph_is_reachable_from_trigger(wf):
    """트리거에서 출발해 모든 노드에 도달할 수 있는가."""
    trigger = next(n["name"] for n in wf["nodes"] if n["type"] in TRIGGER_TYPES)

    seen, stack = {trigger}, [trigger]
    while stack:
        current = stack.pop()
        for branch in wf["connections"].get(current, {}).get("main", []):
            for link in branch:
                if link["node"] not in seen:
                    seen.add(link["node"])
                    stack.append(link["node"])

    unreachable = {n["name"] for n in wf["nodes"]} - seen
    assert not unreachable, f"트리거에서 도달 불가: {unreachable}"


# ── Code 노드 ────────────────────────────────────────────────────────────────

def test_code_nodes_are_syntactically_valid(wf):
    """Code 노드의 JS 구문 오류는 실행 시점에야 터진다 — 미리 잡는다.

    node가 있으면 진짜 파서로 검사하고, 없으면 괄호 균형만 본다.
    """
    import shutil
    import subprocess

    code_nodes = [n for n in wf["nodes"] if n["type"] == "n8n-nodes-base.code"]
    if not code_nodes:
        pytest.skip("Code 노드 없음")

    node_bin = shutil.which("node")

    for n in code_nodes:
        source = n["parameters"]["jsCode"]
        assert source.strip(), f"{n['name']}: jsCode가 비어 있다"

        if node_bin:
            # --check 는 실행하지 않고 파싱만 한다.
            result = subprocess.run([node_bin, "--check", "-"],
                                    input=source, text=True,
                                    capture_output=True, check=False)
            assert result.returncode == 0, \
                f"{n['name']} JS 구문 오류:\n{result.stderr}"
        else:
            for opener, closer in (("{", "}"), ("(", ")"), ("[", "]")):
                assert source.count(opener) == source.count(closer), \
                    f"{n['name']}: '{opener}{closer}' 괄호 불균형"


def test_code_nodes_return_items(wf):
    """n8n Code 노드는 항목 배열을 반환해야 한다. 빠뜨리면 흐름이 끊긴다."""
    for n in wf["nodes"]:
        if n["type"] != "n8n-nodes-base.code":
            continue
        assert "return" in n["parameters"]["jsCode"], \
            f"{n['name']}: return 문이 없다"


# ── 보안 · 구성 정합성 ────────────────────────────────────────────────────────

SECRET_PATTERNS = [
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),        # Slack 토큰
    re.compile(r"sk-ant-[A-Za-z0-9\-_]{20,}"),           # Anthropic 키
    re.compile(r"hooks\.slack\.com/services/T[A-Z0-9]+"),  # 실제 웹훅 URL
]


def test_no_hardcoded_secrets(wf):
    blob = json.dumps(wf)
    for pattern in SECRET_PATTERNS:
        assert not pattern.search(blob), \
            f"시크릿으로 보이는 문자열 발견: {pattern.pattern}"


def test_slack_uses_env_var(wf):
    """Slack URL은 환경변수로 주입해야 재현 가능하다."""
    for n in wf["nodes"]:
        url = n["parameters"].get("url", "")
        if "slack" in url.lower():
            assert "$env.SLACK_WEBHOOK_URL" in url, \
                f"{n['name']}: Slack URL이 하드코딩됐다"


def test_api_calls_use_compose_service_name():
    """워크플로의 API 주소가 docker-compose 서비스 이름과 맞는가.

    localhost로 적으면 n8n 컨테이너 자기 자신을 가리켜 조용히 실패한다.
    """
    import yaml

    root = N8N_DIR.parent
    compose = yaml.safe_load((root / "docker-compose.yml").read_text())
    services = set(compose["services"])

    for path in WORKFLOW_PATHS:
        for n in load(path)["nodes"]:
            url = n["parameters"].get("url", "")
            if not url.startswith("http://"):
                continue
            host = url.split("//", 1)[1].split("/", 1)[0].split(":")[0]
            assert host in services, (
                f"{path.stem}/{n['name']}: 호스트 '{host}'가 compose 서비스에 없다 "
                f"(있는 것: {sorted(services)})")


def test_no_execute_command_nodes(wf):
    """n8n 이미지에는 Python도 torch도 없다 — 스크립트를 직접 돌리면 실패한다.

    실행이 필요한 것은 전부 API 엔드포인트로 노출하고 HTTP로 부른다.
    """
    offenders = [n["name"] for n in wf["nodes"]
                 if n["type"] == "n8n-nodes-base.executeCommand"]
    assert not offenders, (
        f"executeCommand 노드 {offenders} — n8n 컨테이너에는 Python이 없다. "
        f"api.py에 엔드포인트를 만들고 httpRequest로 부를 것")


def test_api_endpoints_exist_in_api_module():
    """워크플로가 부르는 API 경로가 실제로 api.py에 정의돼 있는가.

    엔드포인트 이름을 바꾸면 워크플로가 조용히 404를 받는다 — 여기서 잡는다.
    """
    source = (N8N_DIR.parent / "web_pipeline" / "api.py").read_text()

    called = set()
    for path in WORKFLOW_PATHS:
        for n in load(path)["nodes"]:
            url = n["parameters"].get("url", "")
            if url.startswith("http://api:8000"):
                called.add(url.split("http://api:8000", 1)[1].split("?")[0])

    assert called, "API를 부르는 노드가 하나도 없다"
    for route in sorted(called):
        assert f'"{route}"' in source, f"api.py에 {route} 라우트가 없다"


def test_long_running_calls_have_generous_timeout():
    """모델 로딩·골든셋 예측은 기본 타임아웃(5분 미만)으로는 모자란다."""
    slow_routes = {"/selftest", "/agent/ask", "/predict"}

    for path in WORKFLOW_PATHS:
        for n in load(path)["nodes"]:
            url = n["parameters"].get("url", "")
            route = (url.split("http://api:8000", 1)[1].split("?")[0]
                     if url.startswith("http://api:8000") else None)
            if route not in slow_routes:
                continue
            timeout = n["parameters"].get("options", {}).get("timeout")
            assert timeout and timeout >= 60000, (
                f"{path.stem}/{n['name']}: {route} 타임아웃이 {timeout}ms — "
                f"모델 추론에는 부족하다")


def test_api_failures_do_not_kill_the_workflow():
    """API가 5xx를 줘도 알림 분기까지는 가야 한다 (neverError)."""
    for path in WORKFLOW_PATHS:
        for n in load(path)["nodes"]:
            url = n["parameters"].get("url", "")
            if not url.startswith("http://api:8000"):
                continue
            response = n["parameters"].get("options", {}).get("response", {})
            assert response.get("response", {}).get("neverError") is True, (
                f"{path.stem}/{n['name']}: neverError가 없어 "
                f"API 오류 시 워크플로가 멈춘다")
