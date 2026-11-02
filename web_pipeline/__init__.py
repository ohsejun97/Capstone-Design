"""FusionDTI 웹 계층 — FastAPI 백엔드(services/agent/app/api.py) + Streamlit UI(app.py).

UI는 tools/를 직접 부르지 않고 backend.py를 거친다. 그래야 같은 app.py가
단일 프로세스(local)에서도, API 컨테이너와 분리된 구성(http)에서도 돈다.
"""
