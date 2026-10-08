"""
test_backend.py — UI ↔ API 백엔드 추상화 검증

컨테이너를 쪼갠 뒤 UI가 실제로 API와 말이 통하는지 확인한다. 모델을 띄우지
않고 api.py의 응답 **모양**만 흉내 내는 스텁 서버를 세워서 검증한다:

  · /predict는 {drug, protein, prediction, elapsed_seconds}로 감싸서 준다
    → backend.predict()가 prediction만 꺼내 tools/와 같은 모양으로 맞추는지
  · FastAPI의 404는 {"detail": "..."} 형태다
    → {"error": ...}로 정규화되는지
  · 서버가 죽어 있어도 예외가 아니라 error dict가 나오는지
"""

from __future__ import annotations

import importlib
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

import pytest

PREDICT_BODY = {
    "drug": {"name": "Imatinib", "smiles": "CC1"},
    "protein": {"gene": "ABL1", "uniprot_id": "P00519"},
    "prediction": {
        "pKd": 8.74, "kd_nM": 1.8, "interpretation": "Strong binding",
        "smiles": "CC1", "seq_length": 1130, "used_3di": True,
    },
    "elapsed_seconds": 1.23,
}

DRUG_BODY = {"query_name": "Imatinib", "name": "Imatinib", "cid": "5291",
             "smiles": "CC1", "cached": True}


class _Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: dict):
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/health":
            self._send(200, {"status": "ok", "models_loaded": True})
        elif path.startswith("/resolve/drug/"):
            self._send(200, DRUG_BODY)
        elif path.startswith("/resolve/protein/"):
            # api.py는 찾지 못하면 404 + detail 로 답한다.
            self._send(404, {"detail": "Protein 'ZZZ' not found in UniProt."})
        else:
            self._send(404, {"detail": "no route"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        self._send(200, PREDICT_BODY)

    def log_message(self, *args):                       # 테스트 출력 조용히
        pass


@pytest.fixture
def stub_api(monkeypatch):
    """스텁 API를 띄우고 backend를 http 모드로 다시 로드한다."""
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address

    monkeypatch.setenv("FUSIONDTI_BACKEND", "http")
    monkeypatch.setenv("FUSIONDTI_API_URL", f"http://{host}:{port}")

    # 모듈 상수가 import 시점에 환경변수를 읽으므로 다시 로드해야 한다.
    from web_pipeline import backend
    importlib.reload(backend)
    yield backend

    server.shutdown()
    server.server_close()


def test_http_mode_is_reported(stub_api):
    assert stub_api.is_remote()
    assert "HTTP" in stub_api.describe()


def test_warmup_checks_server_health(stub_api):
    assert stub_api.warmup()["models_loaded"] is True


def test_resolve_drug_returns_tool_shaped_dict(stub_api):
    result = stub_api.resolve_drug("Imatinib")
    assert result["smiles"] == "CC1"
    assert result["cid"] == "5291"
    assert "error" not in result


def test_predict_unwraps_prediction_envelope(stub_api):
    """/predict의 래퍼를 벗겨 tools.dti_tool과 같은 모양으로 돌려줘야 한다."""
    result = stub_api.predict("CC1", "MAAA")
    assert result["pKd"] == 8.74
    assert result["used_3di"] is True
    assert "drug" not in result, "래퍼가 벗겨지지 않았다"


def test_fastapi_detail_becomes_error(stub_api):
    result = stub_api.resolve_protein("ZZZ")
    assert "error" in result
    assert "not found in UniProt" in result["error"]


def test_connection_failure_is_not_an_exception(monkeypatch):
    monkeypatch.setenv("FUSIONDTI_BACKEND", "http")
    monkeypatch.setenv("FUSIONDTI_API_URL", "http://127.0.0.1:9")
    from web_pipeline import backend
    importlib.reload(backend)

    result = backend.predict("CC1", "MAAA")
    assert "error" in result
    assert "API 연결 실패" in result["error"]


def test_local_mode_is_default(monkeypatch):
    monkeypatch.delenv("FUSIONDTI_BACKEND", raising=False)
    from web_pipeline import backend
    importlib.reload(backend)

    # 기본값이 local이어야 기존 단일 프로세스 사용법이 그대로 동작한다.
    assert not backend.is_remote()
    assert "in-process" in backend.describe()
