# Lightweight Bio-AI Agent System for Drug–Target Interaction Analysis

> **캡스톤 디자인 프로젝트**
> "경량화된 DTI 모델과 구조 정보 도구를 Agent AI로 통합하여, 자연어 기반 약물-표적 상호작용 분석 시스템 구축"

---

## 1. 프로젝트 개요

기존 DTI(Drug–Target Interaction) 분석 도구는 **standalone 모델**로, 단순 수치 예측에 그친다.

본 프로젝트는 이를 **Agent AI 기반 시스템**으로 재구성하여:

- 자연어 질문 → 자동 도구 선택 → 결과 + 구조 설명까지 End-to-End 제공
- 경량화 / 양자화된 SaProt을 **실시간 호출 가능한 DTI Tool**로 통합
- AlphaFold DB(단백질 3D 구조) + RDKit(리간드 3D 구조)까지 연동

---

## 2. 핵심 연구 질문

> **"SaProt-650M full-precision 대비, 경량 백본(35M)과 양자화(4-bit / 8-bit)가**
> **동등한 DTI 예측 성능을 유지하면서 Agent tool로 실시간 호출 가능한가?"**

- 검증 데이터: DAVIS (연속 pKd 회귀, 30,056 쌍)
- 평가 지표: Pearson r, VRAM 사용량, 추론 속도(latency)

---

## 3. 시스템 구조

```
사용자 자연어 쿼리
        ↓
  Agent AI (LLM Orchestrator)
        ↓
  ┌─────────────────────────────────────┐
  │  Tool 1: DTI Prediction             │  ← SaProt (양자화) + MLP 헤드
  │  Tool 2: Protein Structure          │  ← AlphaFold DB 조회
  │  Tool 3: Ligand Structure           │  ← RDKit 3D conformer 생성
  │  Tool 4: Explanation / Summary      │  ← LLM 기반 결과 해석
  └─────────────────────────────────────┘
        ↓
  최종 응답 (예측 수치 + 구조 설명)
```

### Tool 1: DTI Prediction Tool 내부 구조

```
SMILES  → Morgan Fingerprint (2048-bit)      ─┐
                                               ├→ DTI MLP 헤드 → pKd
AA서열  → SA 토큰 → SaProt (frozen, 양자화)  ─┘
              ↑
     AA문자 + FoldSeek 3Di 구조 토큰 쌍
```

---

## 4. Phase 1 완료 — DTI Tool 경량화 실험 결과

> SaProt frozen + MLP 헤드를 DAVIS로 학습 후 경량 백본 / 양자화 수준별 성능 비교

| 모델 | 파라미터 | 양자화 | Test Pearson r | Val Pearson r | 학습 시간 |
|------|---------|--------|---------------|--------------|---------|
| SaProt-650M | 652M | none (FP16) | 0.7855 | 0.7990 | 59초 |
| SaProt-35M | 34M | none (FP16) | 0.7832 | 0.7872 | 55초 |
| SaProt-650M-8bit | 652M | INT8 | 0.7812 | 0.7951 | 62초 |
| **SaProt-650M-4bit** | 652M | **NF4 4-bit** | **0.7914** | **0.8016** | 198초 |

### 핵심 발견

- **경량 백본:** SaProt-35M (파라미터 19배↓)이 650M 대비 성능 차이 **0.0023** — 사실상 동일
- **양자화:** 4-bit NF4가 FP16 대비 성능 **소폭 향상** (val r: 0.7990 → 0.8016)
- **결론:** 경량 백본 또는 4-bit 양자화 모델이 Agent Tool로 실시간 사용 가능한 수준 확인

---

## 5. Phase 2 계획 — Agent 시스템 구현

| 단계 | 내용 | 상태 |
|------|------|------|
| 2-1 | DTI Tool 모듈화 (양자화 모델 wrapping) | 예정 |
| 2-2 | Protein Tool — AlphaFold DB API 연동 | 예정 |
| 2-3 | Ligand Tool — RDKit 3D conformer 생성 | 예정 |
| 2-4 | Agent 구현 (smolagents / LangChain) | 예정 |
| 2-5 | End-to-End 데모 (자연어 쿼리 → 결과) | 예정 |

---

## 6. 실행 방법

### 환경 설치

```bash
bash setup_env.sh
# Conda 환경: bioinfo (Python 3.10)
```

### DTI Tool 학습 (Phase 1)

```bash
python train_dti_saprot.py --encoder 650M                    # FP16 기준 (r=0.7855)
python train_dti_saprot.py --encoder 35M                     # 경량 백본 (r=0.7832)
python train_dti_saprot.py --encoder 650M --quant 8bit       # INT8 양자화 (r=0.7812)
python train_dti_saprot.py --encoder 650M --quant 4bit       # NF4 4-bit (r=0.7914)
```

### 결과 시각화

```bash
python experiments/visualize_results.py
# 출력: outputs/figures/ (학습 곡선, 산점도, 모델 비교 바 차트)
```

---

## 7. 기술 스택

| 분류 | 스택 |
|------|------|
| 단백질 인코더 | SaProt (35M / 650M AF2), SA 토큰 (AA + FoldSeek 3Di) |
| 약물 인코딩 | RDKit Morgan Fingerprint (radius=2, nBits=2048) |
| 경량화 / 양자화 | bitsandbytes (NF4 4-bit, INT8) |
| ML 프레임워크 | PyTorch 2.6, Hugging Face Transformers |
| Agent | smolagents (Hugging Face) |
| 구조 DB | AlphaFold DB (단백질 3D), RDKit (리간드 3D) |
| 프론트/백엔드 | Streamlit, FastAPI (예정) |
| 인프라 | Docker, WSL2, Linux 32코어 |

---

## 8. 실험 히스토리 (전체)

| 버전 | 방법 | Pearson r | 상태 |
|------|------|----------|------|
| V1 | SaProt-650M 랜덤 헤드 (CPU) | 0.030 | ❌ 입력 오류 |
| V2 | SPRINT + MERGED 가중치 | 0.141 | ❌ OOD 문제 |
| V3-A | SaProt-650M frozen + MLP | 0.7855 | ✅ |
| V3-B | SaProt-35M frozen + MLP | 0.7832 | ✅ |
| V3-C | SaProt-650M-8bit frozen + MLP | 0.7812 | ✅ |
| V3-D | SaProt-650M-4bit frozen + MLP | **0.7914** | ✅ |

---

## 9. 환경 주의사항

- `torch >= 2.6.0` 필수 (CVE-2025-32434 대응)
- `torchvision`, `torchaudio` 설치 금지 (버전 충돌)
- GPU: GTX 1650 SUPER (4GB VRAM, CUDA 12.6)
- Conda 환경: `bioinfo` (Python 3.10)

---

## 10. 실험 상세 기록

- [Phase 1 실험 일지](docs/PHASE1_EXPERIMENT_LOG.md)
- [Phase 1 파이프라인 설계](docs/PHASE1_REFERENCE_PIPELINE.md)
- [Phase 1 학습 실험 보고서](docs/PHASE1_TRAINING_EXPERIMENTS.md)
