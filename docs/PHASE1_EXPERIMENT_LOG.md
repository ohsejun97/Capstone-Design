# Phase 1 실험 일지 — Reference Score 생성

> **작성일시:** 2026년 3월 25일 (수) KST
> **목표:** SaProt-650M 기반 DAVIS DTI Reference Score 산출 (Pearson r ≥ 0.8)
> **현재 상태:** 🔄 진행 중 — V3 완료 (r = 0.7855), 목표까지 0.015 차이

---

## 실험 환경

| 항목 | 값 |
|------|-----|
| OS | Ubuntu 22.04 (WSL2, Linux 6.6.87.2) |
| GPU | NVIDIA GeForce GTX 1650 SUPER (4GB VRAM, CUDA 12.6) |
| CPU | 32-thread 서버 |
| Python | 3.10.20 (Conda: `bioinfo`) |
| PyTorch | 2.6.0+cu124 |
| Transformers | 5.3.0 |
| 데이터셋 | DAVIS — DeepPurpose (연속 pKd, 30,056 쌍) |

---

## 실험 V1 — SaProt 단독 추론

**일시:** 2026년 3월 25일 (오전)
**소요 시간:** ~17시간 (CPU)

### 설정

| 항목 | 값 |
|------|-----|
| 모델 | `westlake-repl/SaProt_650M_AF2` |
| 모델 클래스 | `AutoModelForSequenceClassification(num_labels=1)` |
| 약물 입력 | `tokenizer(protein, SMILES)` — 시퀀스 쌍 직접 입력 |
| 데이터 | davis_test.csv (panspecies-dti, 이진 레이블) |

### 결과

| 지표 | 값 |
|------|-----|
| **Pearson R** | **0.030** |
| 처리 샘플 | 6,011개 |

### 실패 원인

1. **잘못된 모델 클래스:** 랜덤 초기화된 분류 헤드 추가 → 완전한 노이즈 출력
2. **잘못된 입력 포맷:** SMILES를 단백질 토크나이저에 입력 → 어휘 미인식
3. **DTI 파인튜닝 없음:** SaProt은 Masked LM 사전학습 모델, 결합 친화도 예측 불가

---

## 실험 V2 — SPRINT 아키텍처 + panspecies-dti 가중치

**일시:** 2026년 3월 25일 20:04 ~ 21:09 KST
**소요 시간:** 1시간 5분 44초 (GPU)

### 아키텍처

```
단백질: SA 시퀀스 → SaProt-650M (frozen) → 4-head MultiHeadAttentionPool
        → Sequential(LN+LeakyReLU+Linear×3) → [1024] → L2 normalize

약물:   SMILES → RDKit Morgan FP (2048-bit)
        → Linear(2048→1260)+BN+LeakyReLU × 3 → Linear(1024) → L2 normalize

융합:   cosine similarity (dot product)
```

### 가중치

| 항목 | 값 |
|------|-----|
| 출처 | panspecies-dti (SPRINT) |
| 학습 데이터 | MERGED (BIOSNAP + BindingDB + Human) |
| 학습 태스크 | 이진 분류 (CE loss) |

### 결과

| 지표 | 값 |
|------|-----|
| **Pearson R** | **+0.1412** |
| p-value | 3.73e-28 |
| 처리 샘플 | 6,011개 |

### 실패 원인

1. **OOD 가중치:** SPRINT은 MERGED 데이터로 학습, DAVIS는 포함 안 됨
2. **태스크 불일치:** 이진 분류 모델 → 연속 pKd 회귀에 부적합
3. **평가 데이터 문제:** panspecies-dti DAVIS는 이진 레이블(0/1), 연속 pKd가 아님

---

## 실험 V3 — SaProt + DTI MLP 헤드 (DAVIS 직접 학습)

**일시:** 2026년 3월 25일 21:55 ~ 22:19 KST

### 방법론 전환 배경

V2 실패 분석 결과, 핵심 문제는 **"DTI 태스크용으로 DAVIS에서 직접 학습된 가중치가 없다"**는 것.

해결책: SaProt을 frozen 인코더로 사용하고, 소형 MLP 헤드만 DAVIS 연속 pKd 데이터로 학습.

- **데이터:** DeepPurpose DAVIS (연속 pKd, 30,056 쌍, `binary=False, convert_to_log=True`)
- **분할:** Train 70% / Val 10% / Test 20% (seed=42)

### 아키텍처

```
단백질: AA 시퀀스 → SA 포맷("P#F#...") → SaProt (frozen)
        → mean pool → [prot_dim]  ← pre-computed cache

약물:   SMILES → Morgan FP (radius=2, nBits=2048) → [2048]

DTI 헤드 (훈련 대상):
  prot_enc: Linear(prot_dim→512) + LayerNorm + GELU + Linear(512→256) + GELU
  drug_enc: Linear(2048→512) + BatchNorm + GELU + Linear(512→256) + GELU
  regressor: Linear(512→256) + GELU + Dropout(0.1) + Linear(256→64) + GELU + Linear(64→1)

Loss: HuberLoss(delta=1.0)  |  Optimizer: Adam(lr=1e-3)  |  Scheduler: CosineAnnealingLR
```

### 실험 V3-A: SaProt-650M

**시작:** 2026-03-25 21:55 | **완료:** 22:16

| 항목 | 값 |
|------|-----|
| 인코더 | SaProt-650M-AF2 (1,280-dim, 652M params, frozen) |
| 단백질 임베딩 계산 | 379개 unique proteins, ~20분 (GPU) |
| 학습 시간 | **58.7초** (DTI 헤드만) |
| Epochs | 50 (Early stopping 미작동) |
| **Test Pearson r** | **0.7855** |
| Val Best r | 0.7990 |
| p-value | 0.0 (매우 유의) |
| Test 샘플 수 | 6,012개 |

### 실험 V3-B: SaProt-35M

**시작:** 2026-03-25 22:16 | **완료:** 22:18

| 항목 | 값 |
|------|-----|
| 인코더 | SaProt-35M-AF2 (480-dim, 327M params, frozen) |
| 학습 시간 | **54.8초** |
| Epochs | 50 |
| **Test Pearson r** | **0.7832** |
| Val Best r | 0.7872 |
| p-value | 0.0 |

### 실험 V3-C: SaProt-650M 4-bit

**시작:** 2026-03-25 22:18 | **완료:** 22:19 (즉시 실패)

**오류:** `AssertionError` — bitsandbytes가 EsmModel pooler 레이어를 양자화하면서 shape 불일치

**해결책:** `EsmModel.from_pretrained(..., add_pooling_layer=False)` 추가 → 재실행 예정

---

## V1 ~ V3 종합 비교

| 버전 | 방법 | Pearson r | 비고 |
|------|------|----------|------|
| V1 | SaProt-650M (랜덤 헤드) | 0.030 | ❌ 입력 오류 + 랜덤 헤드 |
| V2 | SPRINT + MERGED 가중치 | 0.141 | ❌ OOD + 태스크 불일치 |
| **V3-650M** | **SaProt-650M + DTI 헤드** | **0.7855** | ✅ 목표 0.015 차이 |
| **V3-35M** | **SaProt-35M + DTI 헤드** | **0.7832** | ✅ 650M 대비 -0.0023 |
| V3-4bit | SaProt-650M 4-bit | — | ❌ 재실행 예정 |

### 핵심 발견
- **650M vs 35M 성능 차이: 0.0023** — 파라미터 약 18배 차이임에도 사실상 동일한 성능
- Val에서는 650M이 0.799까지 도달 → 모델 개선 여지 있음
- 목표(r ≥ 0.8)에 근접, 아직 Phase 1 미완료

---

## 다음 단계

1. **V3-C 4bit 재실행** — `add_pooling_layer=False` 패치 후 실행
2. **r ≥ 0.8 달성 시도** — 하이퍼파라미터 튜닝 또는 사전학습 활용
3. **Phase 2 전환** — Reference Score 확보 후 경량화 트레이드오프 분석 시작

---

## 파일 목록

```
Capstone_Design/
├── train_dti_saprot.py          # V3 메인 학습 스크립트
├── run_all_experiments.sh       # 3개 모델 자동 실행
├── experiments/
│   ├── run_reference.py         # V2 SPRINT 파이프라인
│   └── run_baseline_deeppurpose.py
├── results/
│   ├── SaProt-650M/             # V3-A 결과 (r=0.7855)
│   └── SaProt-35M/              # V3-B 결과 (r=0.7832)
├── outputs/
│   └── reference_scores_osj.csv # V2 예측값
├── data/
│   └── davis_test.csv
└── cache/
    ├── prot_embs_650M_none.pt   # 379개 단백질 임베딩 캐시
    └── prot_embs_35M_none.pt
```
