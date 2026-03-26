# Phase 2 실험 보고서 — SaProt DTI 헤드 학습

> **작성일시:** 2026년 3월 25일 (수) KST
> **목표:** SaProt (650M / 35M / 650M-4bit) + DTI MLP 헤드를 DAVIS 연속 pKd 데이터로 학습하여 세 모델의 Pearson r 비교
> **현재 상태:** 🔄 학습 진행 중

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
| 데이터셋 | DAVIS (DeepPurpose, 연속 pKd) — 30,056 쌍 |

---

## 실험 설계

### 연구 가설

> **SaProt-650M 대비 SaProt-35M(경량) + 4bit 양자화 모델이 Pearson r 기준 90% 이상을 유지하면서 VRAM 사용량을 80% 이상 줄일 수 있는가?**

### 데이터 분할

| 세트 | 샘플 수 | 비율 |
|------|---------|------|
| Train | 21,039 | 70% |
| Val   | 3,005  | 10% |
| Test  | 6,012  | 20% |

- 시드: 42 (재현 가능)

### 아키텍처

```
단백질 경로:
  AA 시퀀스 → SA 포맷("P#F#W#...") → SaProt (frozen)
  → mean pool → [prot_dim] → pre-computed cache

약물 경로:
  SMILES → RDKit Morgan FP (radius=2, nBits=2048) → [2048]

DTI 헤드 (훈련 대상):
  prot_enc: Linear(prot_dim→512) + LayerNorm + GELU + Linear(512→256) + GELU
  drug_enc: Linear(2048→512) + BatchNorm + GELU + Linear(512→256) + GELU
  regressor: Linear(512→256) + GELU + Dropout(0.1) + Linear(256→64) + GELU + Linear(64→1)
```

### 학습 설정

| 파라미터 | 값 |
|---------|-----|
| Loss | HuberLoss (delta=1.0) |
| Optimizer | Adam (lr=1e-3, weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR (T_max=50, eta_min=1e-5) |
| Batch size | 128 |
| Max epochs | 50 |
| Early stopping | patience=10 |

---

## 실험 결과

### 실험 A — SaProt-650M (기준 모델)

| 항목 | 값 |
|------|-----|
| 인코더 | SaProt-650M-AF2 (1,280-dim, ~650M params) |
| 양자화 | 없음 (FP16) |
| VRAM 사용 | TBD |
| 학습 시간 | TBD |
| **Test Pearson r** | **TBD** |
| Val Best r | TBD |
| Epochs | TBD |

**상태:** 🔄 학습 진행 중

---

### 실험 B — SaProt-35M (경량 모델)

| 항목 | 값 |
|------|-----|
| 인코더 | SaProt-35M-AF2 (480-dim, ~35M params) |
| 양자화 | 없음 (FP16) |
| VRAM 사용 | TBD |
| 학습 시간 | TBD |
| **Test Pearson r** | **TBD** |
| Val Best r | TBD |
| Epochs | TBD |

**상태:** ⏳ 대기 중

---

### 실험 C — SaProt-650M 4-bit (양자화 모델)

| 항목 | 값 |
|------|-----|
| 인코더 | SaProt-650M-AF2 (NF4, double quant) |
| 양자화 | 4-bit (bitsandbytes NF4) |
| VRAM 사용 | TBD |
| 학습 시간 | TBD |
| **Test Pearson r** | **TBD** |
| Val Best r | TBD |
| Epochs | TBD |

**상태:** ⏳ 대기 중

---

## 비교 분석

| 모델 | Pearson r | VRAM | 속도 | 목표 달성 |
|------|-----------|------|------|----------|
| SaProt-650M (기준) | TBD | TBD | TBD | — |
| SaProt-35M | TBD | TBD | TBD | r ≥ 기준×0.9 |
| SaProt-650M-4bit | TBD | TBD | TBD | r ≥ 기준×0.9 |

### 목표 기준

- **Pearson r:** 기준 모델(650M) 대비 90% 이상 유지
- **VRAM:** 기준 대비 80% 이상 절감 (또는 4GB 이하로 동작)
- **핵심 가설 검증:** 경량/양자화 모델로 실시간 DTI 예측 가능성 확인

---

## V1~V3 전체 실험 히스토리

| 버전 | 일시 | 방법 | Pearson r | 상태 |
|------|------|------|----------|------|
| V1 | 2026-03-25 오전 | SaProt_650M (랜덤 헤드, CPU, 17h) | 0.030 | ❌ 실패 |
| V2 | 2026-03-25 20:04~21:09 | SPRINT + panspecies-dti 가중치 (GPU) | 0.141 | ⚠️ OOD |
| V3-A | 2026-03-25 (진행 중) | SaProt-650M + DTI MLP (DAVIS pKd) | TBD | 🔄 진행 |
| V3-B | (예정) | SaProt-35M + DTI MLP | TBD | ⏳ |
| V3-C | (예정) | SaProt-650M-4bit + DTI MLP | TBD | ⏳ |

---

## V2 대비 V3 방법론 개선

| 항목 | V2 (SPRINT) | V3 (DTI 헤드 학습) |
|------|-------------|-------------------|
| 데이터 | DAVIS test (binarized, 0/1) | DAVIS 전체 (연속 pKd 5.0~10.8) |
| 목표 | 이진 분류 cosine sim → r 측정 | 연속 pKd 직접 회귀 |
| 가중치 | MERGED 사전학습 (OOD) | DAVIS로 직접 학습 |
| 평가 지표 | Pearson r (이진 레이블에 부적합) | Pearson r (연속값, 적합) |
| 기대 r | ≈ 0.14 | ≈ 0.7~0.9+ |

---

## 결론 및 다음 단계

> (학습 완료 후 업데이트 예정)

---

*이 보고서는 학습 완료 후 실제 결과로 업데이트됩니다.*
