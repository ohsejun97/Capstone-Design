"""
backend.py — Streamlit UI가 쓰는 백엔드 추상화

app.py는 원래 tools/를 직접 import했다. 한 프로세스에서는 잘 돌지만,
컨테이너로 쪼개면 문제가 된다: UI 컨테이너가 SaProt-650M과 torch를 통째로
지고 다니게 되고, UI를 수평 확장하면 GPU 한 장을 여러 프로세스가 물게 된다.

그래서 호출부를 이 모듈 뒤로 숨기고 두 가지 구현을 둔다:

    local  — tools/를 직접 import (기존 동작 그대로. 노트북/단일 프로세스용)
    http   — services/agent/app/api.py 를 HTTP로 호출 (컨테이너 분리 구성)

선택은 환경변수로 한다:
    FUSIONDTI_BACKEND=local            (기본값 — 기존 사용법이 바뀌지 않는다)
    FUSIONDTI_BACKEND=http
    FUSIONDTI_API_URL=http://api:8000  (http 모드일 때 대상)

두 구현은 **같은 dict 모양**을 돌려준다 — 성공 시 tools/가 주는 dict,
실패 시 {"error": ...}. app.py는 어느 쪽이 돌고 있는지 알 필요가 없다.
"""

from __future__ import annotations

import os
from typing import Any

BACKEND = os.environ.get("FUSIONDTI_BACKEND", "local").lower()
API_URL = os.environ.get("FUSIONDTI_API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = float(os.environ.get("FUSIONDTI_API_TIMEOUT", "120"))


def is_remote() -> bool:
    """UI가 원격 API를 쓰는 중인지. 사이드바 표시에 쓴다."""
    return BACKEND == "http"


def describe() -> str:
    return f"HTTP → {API_URL}" if is_remote() else "in-process (tools/)"


# ══════════════════════════════════════════════════════════════════════════════
# 공개 API — app.py는 아래 4개만 쓴다
# ══════════════════════════════════════════════════════════════════════════════

def warmup() -> dict[str, Any]:
    """모델을 미리 로드한다(local) 또는 서버가 준비됐는지 확인한다(http)."""
    if is_remote():
        return _get("/health")
    from tools.dti_tool import _load_models
    _load_models()
    return {"status": "ok", "models_loaded": True}


def resolve_drug(name: str) -> dict[str, Any]:
    if is_remote():
        return _get(f"/resolve/drug/{_quote(name)}")
    from tools.pubchem_tool import resolve_drug_name
    return resolve_drug_name(name)


def resolve_protein(name: str, organism_id: int = 9606) -> dict[str, Any]:
    if is_remote():
        return _get(f"/resolve/protein/{_quote(name)}",
                    params={"organism_id": organism_id})
    from tools.uniprot_tool import resolve_protein_name
    return resolve_protein_name(name, organism_id=organism_id)


def predict(smiles: str, aa_seq: str) -> dict[str, Any]:
    """SMILES + 서열 → pKd. 반환 dict는 tools.dti_tool.predict_binding과 동일."""
    if is_remote():
        response = _post("/predict", {"smiles": smiles, "sequence": aa_seq})
        if "error" in response:
            return response
        # /predict는 drug/protein/prediction을 감싸서 준다 — prediction만 꺼낸다.
        return response.get("prediction", response)
    from tools.dti_tool import predict_binding
    return predict_binding(smiles, aa_seq)


# ══════════════════════════════════════════════════════════════════════════════
# HTTP 헬퍼 — 어떤 실패든 {"error": ...} 로 정규화한다
# ══════════════════════════════════════════════════════════════════════════════

def _quote(value: str) -> str:
    from urllib.parse import quote
    return quote(value, safe="")


def _get(path: str, params: dict | None = None) -> dict[str, Any]:
    return _request("GET", path, params=params)


def _post(path: str, payload: dict) -> dict[str, Any]:
    return _request("POST", path, json=payload)


def _request(method: str, path: str, **kwargs) -> dict[str, Any]:
    import requests

    try:
        response = requests.request(method, f"{API_URL}{path}",
                                    timeout=TIMEOUT, **kwargs)
    except requests.exceptions.RequestException as exc:
        return {"error": f"API 연결 실패 ({API_URL}): {exc}"}

    if response.status_code >= 400:
        # FastAPI의 HTTPException은 {"detail": "..."} 형태로 온다.
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        return {"error": f"{detail}" if detail else f"HTTP {response.status_code}"}

    try:
        return response.json()
    except ValueError:
        return {"error": f"API가 JSON이 아닌 응답을 보냈습니다: {response.text[:200]}"}
