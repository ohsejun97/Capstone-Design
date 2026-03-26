# Phase 1: Reference Score 생성 실험 보고서

> **목표:** SaProt-650M을 이용한 DAVIS 데이터셋 DTI Reference Score 산출 (Pearson r ≥ 0.8)
>
> **상태:** 🔧 V2 파이프라인 구현 완료 — DTI 파인튜닝 가중치 확보 필요

---

## 목차

1. [실험 배경](#1-실험-배경)
2. [V1 실패 원인 분석](#2-v1-실패-원인-분석)
3. [V2 아키텍처 설계](#3-v2-아키텍처-설계)
4. [SA 토큰 포맷 이해](#4-sa-토큰-포맷-이해)
5. [실행 방법](#5-실행-방법)
6. [Pearson r ≥ 0.8 달성 전략](#6-pearson-r--08-달성-전략)
7. [다음 단계: 경량화 실험](#7-다음-단계-경량화-실험)

---

## 1. 실험 배경

### 사용 데이터셋

| 항목 | 내용 |
|------|------|
| 파일 | `davis_test.csv` |
| 출처 | [panspecies-dti (abhinadduri)](https://github.com/abhinadduri/panspecies-dti) |
| 샘플 수 | 6,011개 약물-단백질 쌍 |
| 포맷 | FoldSeek 3Di 구조 시퀀스 포함 |
| 레이블 | pKd 기반 이진 분류 (Label: 0/1) |

### 참조 레포지토리

| 레포 | 역할 |
|------|------|
| [SaProt](https://github.com/westlake-repl/SaProt) | 단백질 인코더 (SA 토큰 포맷) |
| [panspecies-dti (SPRINT)](https://github.com/abhinadduri/panspecies-dti) | 이중-타워 DTI 아키텍처 |
| [FusionDTI](https://github.com/ZhaohanM/FusionDTI) | 토큰 수준 Cross-Attention 융합 |
| [DeepPurpose](https://github.com/kexinhuang12345/DeepPurpose) | DAVIS 파인튜닝 가중치 제공 |

---

## 2. V1 실패 원인 분석

### 실험 결과
- **Pearson R = 0.03** (기대치: ≥ 0.8)
- 사실상 랜덤 예측 수준

### 원인 1: 잘못된 모델 클래스 사용

```python
# ❌ V1 — 랜덤 분류 헤드 추가 (DTI 학습 없음)
model = AutoModelForSequenceClassification.from_pretrained(
    "westlake-repl/SaProt_650M_AF2", num_labels=1
)
```

`AutoModelForSequenceClassification`은 SaProt 인코더 위에 **새로운 랜덤 선형 레이어**를 추가한다.
이 헤드는 DTI 데이터로 전혀 학습되지 않았으므로 출력이 완전한 노이즈다.

### 원인 2: SaProt에 SMILES 입력

```python
# ❌ V1 — SaProt 토크나이저에 SMILES 동시 입력
inputs = tokenizer(row['Target Sequence'], row['SMILES'], ...)
```

SaProt은 **단백질 언어 모델**이다. 어휘집(vocabulary)은 441개의 SA 토큰(아미노산 + FoldSeek 3Di)으로 구성되며, SMILES 문자를 알지 못한다.
SMILES를 두 번째 시퀀스로 넣으면 토크나이저가 각 문자를 미지의 토큰으로 처리하여 입력이 완전히 깨진다.

### 원인 3: DTI 파인튜닝 없는 PLM 단독 사용

`SaProt_650M_AF2`는 Masked Language Modeling(MLM)으로 사전학습된 **단백질 언어 모델**이다.
이것만으로는 약물-단백질 결합 친화도를 예측할 수 없다.
DTI 예측을 위해서는 반드시 **별도의 약물 인코더 + 융합 레이어**가 필요하다.

```
단순 PLM (사전학습만) → DTI 예측 불가 ❌
단백질 인코더 + 약물 인코더 + 융합 레이어 → DTI 예측 가능 ✅
```

---

## 3. V2 아키텍처 설계

SPRINT (panspecies-dti) 아키텍처를 기반으로 설계했다.

### 전체 파이프라인

```
davis_test.csv
    │
    ├── Target Sequence (SA 토큰)
    │       │
    │       ▼
    │   EsmTokenizer
    │       │
    │       ▼
    │   SaProt-650M (EsmModel, frozen)
    │       │  last_hidden_state [seq_len, 1280]
    │       ▼
    │   Mean Pooling (CLS/EOS 제외)
    │       │  [1280]
    │       ▼
    │   ProjectionHead (MLP + GELU + LayerNorm)
    │       │  [1024]
    │       ▼
    │   L2 Normalize ──────────────────────────┐
    │                                          │
    └── SMILES                                 │  코사인 유사도
            │                                  │
            ▼                                  │
        RDKit MolFromSmiles                    │
            │                                  │
            ▼                                  │
        Morgan FP (radius=2, nBits=2048)       │
            │  [2048]                          │
            ▼                                  │
        ProjectionHead (MLP + GELU + LayerNorm)│
            │  [1024]                          │
            ▼                                  │
        L2 Normalize ──────────────────────────┘
                                               │
                                               ▼
                                       binding score (float)
```

### 모델 클래스 변경

| 항목 | V1 (잘못됨) | V2 (수정됨) |
|------|------------|------------|
| 단백질 인코더 | `AutoModelForSequenceClassification` | `EsmModel` |
| 약물 인코더 | 없음 (SMILES를 토크나이저에 직접 입력) | Morgan Fingerprint (RDKit) |
| 융합 방식 | SaProt logits (랜덤 헤드) | 코사인 유사도 (ProjectionHead 투영 후) |
| 입력 방식 | `tokenizer(protein, SMILES)` | 단백질/약물 분리 인코딩 |

### ProjectionHead 구조

```python
class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, out_dim),
        )
    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)
```

---

## 4. SA 토큰 포맷 이해

### 포맷 정의

SaProt의 SA(Structural Aware) 토큰은 **아미노산 1문자(대문자) + FoldSeek 3Di 구조 1문자(소문자)**를 쌍으로 묶은 2문자 단위다.

```
아미노산: ACDEFGHIKLMNPQRSTVWY  (20종)
3Di 구조: pynwrqhgdlvtmfsaeikc   (20종)
특수 마스크: #  (pLDDT < 70, 구조 신뢰도 낮음)
```

### 예시

```
고신뢰도 잔기:  "Ma" → Met(M) + 3Di='a'
                "Ev" → Glu(E) + 3Di='v'

저신뢰도 잔기:  "P#" → Pro(P) + 3Di='#' (pLDDT < 70)
                "F#" → Phe(F) + 3Di='#'
```

### davis_test.csv 포맷

```
Target Sequence 컬럼 예시:
P#F#W#K#I#L#N#P#L#L#E#R#DdPqNqLkFkVfAfLlYaDfFd...
│ │ │ │                  │ │ │ │ │ │
P# F# W# K# ...          Dd Pq Nq Lk Fk ...
(저신뢰도 구간)            (고신뢰도 구간)
```

데이터가 **이미 올바른 SA 포맷**으로 저장되어 있으므로 별도 전처리 없이 `EsmTokenizer`에 직접 입력 가능하다.

### 어휘집 구성

| 구분 | 수량 |
|------|------|
| SA 토큰 (21 AA × 21 3Di) | 441개 |
| 특수 토큰 (CLS, EOS, PAD 등) | 5개 |
| **합계** | **446개** |

---

## 5. 실행 방법

### 환경 설정

```bash
# 의존성 설치
pip install -r requirements.txt

# (선택) Hugging Face 토큰 설정 — gated model 접근 시 필요
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx

# (선택) SPRINT 가중치 경로 설정 — Pearson r ≥ 0.8 달성 시 필요
export SPRINT_CHECKPOINT=/path/to/sprint_weights.ckpt
```

### 실행

```bash
python run_reference.py
```

### 출력 파일

| 파일 | 내용 |
|------|------|
| `reference_scores_osj.csv` | `smiles`, `reference_score`, `label` |

### 실행 예상 시간

| 환경 | 예상 시간 |
|------|----------|
| 연구 서버 (32코어 CPU) | ~3~5시간 |
| GTX 1650 (GPU) | ~30분 |
| M1 Mac / 일반 노트북 | ~8시간 |

---

## 6. Pearson r ≥ 0.8 달성 전략

### 현재 상태 분석

| 방법 | 예상 Pearson r | 가중치 상태 |
|------|--------------|------------|
| V1 (SaProt 단독 + 랜덤 헤드) | 0.03 | ❌ 의미 없음 |
| V2 zero-shot (가중치 없음) | ~0.1~0.3 | ⚠️ 파인튜닝 없음 |
| **V2 + SPRINT 가중치** | **~0.8+** | ✅ DTI 파인튜닝 완료 |
| V2 + SaProt_DTI_Davis (gated) | **~0.85+** | ✅ DAVIS 전용 파인튜닝 |

### 옵션 A: SPRINT 가중치 (권장)

panspecies-dti에서 공개한 SPRINT 체크포인트를 사용한다. MERGED 데이터셋으로 파인튜닝된 가중치이며 DAVIS에서도 좋은 성능을 보인다.

```bash
# 1. Google Drive에서 다운로드
#    https://drive.google.com/file/d/1uojdSn1otFKi-DBJyTKoOA6OZbwOQX6U

# 2. 환경변수 설정
export SPRINT_CHECKPOINT=./sprint_weights.ckpt

# 3. 실행
python run_reference.py
```

### 옵션 B: SaProt DTI Davis (Gated)

Westlake 연구팀의 DAVIS 전용 파인튜닝 모델. HuggingFace 접근 승인 필요.

```bash
# 1. https://huggingface.co/westlake-repl/SaProt_650M_AF2_DTI_Davis 에서 접근 요청
# 2. 승인 후 토큰 설정
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
# 3. run_reference.py의 SAPROT_MODEL을 변경
#    SAPROT_MODEL = "westlake-repl/SaProt_650M_AF2_DTI_Davis"
python run_reference.py
```

### 옵션 C: DeepPurpose 즉시 검증 (빠른 기준값 확보)

DAVIS 전용 파인튜닝 가중치가 공개된 DeepPurpose를 통해 즉시 r ≥ 0.8 기준값을 확보할 수 있다.
SaProt 기반은 아니지만 경량화 비교의 상한선으로 활용 가능하다.

```python
from DeepPurpose import DTI as models
net = models.model_pretrained(model='MPNN_CNN_DAVIS')
# DAVIS에서 Pearson r ≈ 0.88 달성
```

---

## 7. 다음 단계: 경량화 실험

Reference Score 확보 후 아래 단계로 진행한다.

### 실험 계획

```
Phase 1 (현재): Reference Score 산출
    └── SaProt-650M (full precision, CPU)
        → reference_scores_osj.csv

Phase 2: 경량화 모델 구현
    ├── SaProt-35M (backbone 교체)
    ├── SaProt-650M + 8-bit 양자화 (bitsandbytes)
    └── SaProt-650M + 4-bit 양자화 (bitsandbytes NF4)

Phase 3: 성능 비교 분석 (R 언어)
    └── Pearson r 비교: Full vs Quantized
        목표: r_lightweight ≥ 0.8 × r_reference

Phase 4: 에이전트 통합
    └── smolagents + Gemini 1.5 Flash API
        → 자연어 쿼리 → DTI 예측 파이프라인
```

### 경량화 코드 스니펫 (Phase 2 미리보기)

```python
# 4-bit 양자화 적용 예시
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

saprot_4bit = EsmModel.from_pretrained(
    "westlake-repl/SaProt_650M_AF2",
    quantization_config=bnb_config,
    device_map="auto",
)
# 예상 VRAM: ~1.5GB (원본 2.5GB 대비 40% 감소)
```

### 목표 지표

| 모델 | VRAM | 예상 Pearson r | 목표 |
|------|------|--------------|------|
| SaProt-650M (FP32) | ~2.5 GB | 0.85+ | 기준값 |
| SaProt-650M (8-bit) | ~1.3 GB | 0.83+ | ✅ |
| SaProt-650M (4-bit) | ~0.8 GB | 0.80+ | ✅ |
| SaProt-35M (FP32) | ~0.15 GB | 0.75+ | △ |
| SaProt-35M (4-bit) | ~0.05 GB | 0.70+ | △ |

---

## 부록: 파일 구조

```
Capstone_Design/
├── run_reference.py          # V2 DTI 추론 파이프라인 (본 보고서 대상)
├── davis_test.csv            # DAVIS 테스트셋 (6,011 샘플)
├── reference_scores_osj.csv  # 생성된 Reference Score
├── requirements.txt          # 의존성 목록
└── docs/
    └── PHASE1_REFERENCE_PIPELINE.md   # 본 보고서
```

---

*최종 업데이트: 2026-03-25*
