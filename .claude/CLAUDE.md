# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## 프로젝트 개요

**Agentic FusionDTI** — 저사양 하드웨어(GTX 1650, 4GB VRAM)에서 실시간으로 약물-표적 상호작용(DTI)을 예측하는 지능형 신약 재창출 플랫폼.

**핵심 연구 질문:** 4-bit 양자화 + 경량 백본(SaProt-35M)으로 SaProt-650M 대비 Pearson r ≥ 0.8을 유지하면서 VRAM 사용량을 80% 이상 줄일 수 있는가?

**검증 벤치마크:** DAVIS 테스트셋 (`davis_test.csv`, 6,011개 약물-단백질 쌍)

---

## 실험 히스토리 (중요)

### V1 — 실패
- 모델: `westlake-repl/SaProt_650M_AF2` (Pre-trained, DTI 파인튜닝 없음)
- 결과: Pearson R = **0.03** (완전히 실패)
- 원인: SaProt은 단백질 언어 모델로 사전학습된 것이지, DTI 태스크로 파인튜닝된 게 아님. 단독으로는 결합 친화도를 예측 불가.

### V2 — 미완성 (현재 과제)
- 목표: DTI에 특화된 파이프라인으로 전환
- Error 1: HuggingFace 401 Unauthorized (Gated 모델 접근 권한 문제)
- Error 2: 토크나이저가 단백질+SMILES 동시 처리 시 입력 정합성 오류

**결론:** SaProt-650M 단독으로는 DTI를 못 함. 반드시 DTI 태스크용 아키텍처로 감싸야 함.

---

## 참조 레포지토리 4개

| 레포 | 역할 |
|------|------|
| [panspecies-dti (abhinadduri)](https://github.com/abhinadduri/panspecies-dti) | `davis_test.csv` 출처. SaProt을 DTI에 적용한 SPRINT 모델 구현체 |
| [SaProt (westlake-repl)](https://github.com/westlake-repl/SaProt) | 메인 단백질 인코더. SA 토큰 포맷 및 가중치 |
| [FusionDTI (ZhaohanM)](https://github.com/ZhaohanM/FusionDTI) | 토큰 수준 Cross-Attention(CAN) 융합 아키텍처 |
| [DeepPurpose (kexinhuang12345)](https://github.com/kexinhuang12345/DeepPurpose) | DTI 표준 프레임워크. DAVIS 파인튜닝 가중치 제공 (`MPNN_CNN_DAVIS` 등) |

---

## SA 토큰 포맷 (핵심)

SaProt이 사용하는 SA(Structural Aware) 토큰은 **아미노산 문자(대문자) + FoldSeek 3Di 구조 문자(소문자)**를 쌍으로 묶은 것.

```python
# SA 토큰 생성 공식
combined_seq = "".join([aa.upper() + di.lower() for aa, di in zip(aa_seq, foldseek_3di_seq)])
# 예: "MaEvKcIp..."  →  토큰: ["Ma", "Ev", "Kc", "Ip", ...]

# pLDDT < 70인 잔기는 3Di 문자를 "#"으로 대체
# 예: "M#L#V#K#..."  (davis_test.csv의 Target Sequence 포맷이 이것)
```

- 어휘: 21 AA × 21 3Di = 441 토큰 + 5 특수 토큰 = 446 총 어휘
- 토크나이저: `EsmTokenizer.from_pretrained("westlake-repl/SaProt_650M_AF2")`
- `davis_test.csv`의 `Target Sequence` 컬럼이 이 포맷 (`P#F#W#K#...` 형태)

---

## 올바른 DTI 파이프라인 설계

### 핵심 깨달음
SaProt은 **단백질만 인코딩**하는 모델임. SMILES(약물)를 함께 넣으면 안 됨.
DTI 예측을 위해서는 **단백질 인코더 + 약물 인코더 + 융합 레이어**가 별도로 필요.

### SPRINT 방식 (panspecies-dti 기반, 권장)
```
단백질: SA 시퀀스 → SaProt-650M → 어텐션 풀링 → MLP → latent vector (1024-dim)
약물:   SMILES   → Morgan Fingerprint (2048-bit) → MLP → latent vector (1024-dim)
융합:   코사인 유사도 → 시그모이드 → 결합 친화도 점수
```

### FusionDTI 방식 (고성능, 복잡)
```
단백질: SA 시퀀스 → SaProt-650M → 토큰별 임베딩 (seq_len, 1280) → Linear (768)
약물:   SELFIES  → SELFormer   → 토큰별 임베딩 (seq_len, 768)
융합:   CAN(Cross Attention Network) 8-head cross-attention → MLP → 점수
```
- FusionDTI는 SMILES 대신 **SELFIES** 포맷 필요. DAVIS 벤치마크 없음.

### 사용 가능한 DTI 파인튜닝 가중치
| 출처 | 모델 | 데이터셋 | 사용법 |
|------|------|---------|--------|
| DeepPurpose | `MPNN_CNN_DAVIS`, `CNN_CNN_DAVIS`, `Morgan_CNN_DAVIS` 등 | DAVIS | `models.model_pretrained(model='MPNN_CNN_DAVIS')` |
| panspecies-dti | SPRINT (Google Drive) | MERGED | SPRINT 아키텍처 위에 로드 |
| SaProt/FusionDTI | **없음** | — | 직접 파인튜닝 필요 |

---

## 실행 방법

```bash
# 기준 예측값 생성 (SaProt-650M, CPU, 약 17시간 소요)
python run_reference.py
```

출력: `reference_scores_osj.csv` (컬럼: `smiles`, `reference_score`, `label`)

---

## 전체 시스템 아키텍처

```
사용자 자연어 쿼리
        ↓
smolagents + Gemini 1.5 Flash API (도구 호출 조정)
        ↓
    ┌──────────────────────────────┐
    │ 경량 경로 (GTX 1650, 4GB)    │  실시간 결과
    │ SaProt-35M + 4-bit 양자화    │
    └──────────────────────────────┘
        +
    ┌──────────────────────────────┐
    │ 풀 경로 (연구 서버, 32코어)   │  검증용
    │ SaProt-650M on CPU           │
    └──────────────────────────────┘
        ↓
  R 통계 분석 (Pearson r)
        ↓
  Streamlit 프론트엔드 / FastAPI 백엔드
```

---

## 기술 스택

- **모델:** SaProt (35M / 650M AF2), FusionDTI, SPRINT
- **최적화:** bitsandbytes (4-bit/8-bit 양자화)
- **에이전트:** smolagents (Hugging Face), Gemini 1.5 Flash API
- **ML:** PyTorch, Hugging Face transformers
- **프론트/백엔드:** Streamlit, FastAPI
- **인프라:** Docker, WSL2, Linux 서버 (32코어)
- **통계:** R (Pearson r 상관분석)

---

## 환경

Conda 환경: `bioinfo` (Python 3.10)

```bash
bash setup_env.sh          # 자동 설치
python run_reference.py    # 추론 실행
```

**주의사항:**
- `torch>=2.6.0` 필수 (CVE-2025-32434 대응, transformers가 2.6 미만 거부)
- `torchvision`, `torchaudio` 설치 금지 → `transformers`와 버전 충돌 발생
- GPU: GTX 1650 SUPER (4GB, CUDA 12.6), SPRINT 추론 약 1시간 소요
- SPRINT 체크포인트: `sprint_weights.ckpt` (192MB, Google Drive에서 다운로드)

## 실험 결과 히스토리

| 버전 | 모델 | Pearson r | 원인 |
|------|------|----------|------|
| V1 | SaProt_650M_AF2 (분류 헤드 직접) | 0.030 | 랜덤 헤드 + SMILES 오입력 |
| V2 | SaProt_650M + SPRINT 가중치 (MERGED) | 0.141 | MERGED→DAVIS OOD, 이진분류→회귀 불일치 |
| V3 (예정) | SaProt_650M_DTI_Davis 또는 DAVIS 파인튜닝 | 0.8+ 목표 | — |
