# Phase 1 학습 실험 보고서 — SaProt DTI 헤드 학습 결과

> **작성일시:** 2026년 3월 27일 KST (최종 업데이트)
> **목표:** SaProt frozen / LoRA 방식으로 DAVIS 연속 pKd 학습 후 경량화 트레이드오프 비교
> **현재 상태:** ⏳ LoRA 실험 — Colab GPU 한도 초과로 사용 불가 → Kaggle(P100, 주 30h) 또는 로컬 35M LoRA로 대체 예정

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
| PEFT | LoRA (rank=16, alpha=32, target=query/key/value) |
| 데이터셋 | DAVIS (DeepPurpose, 연속 pKd) — 30,056 쌍 |

---

## 실험 설계 요약

### 두 가지 학습 모드 비교

| 모드 | SaProt | 학습 대상 | 속도 | 목표 |
|------|--------|---------|------|------|
| **Frozen** | 완전 고정 | DTI MLP 헤드만 | 빠름 (~1분) | 베이스라인 확인 |
| **LoRA** | 어텐션 레이어 일부 적응 | LoRA 어댑터 + 헤드 | 느림 (~수 시간) | r ≥ 0.9 달성 |

### LoRA 설정

```
LoraConfig(
    r=16,                                   # rank
    lora_alpha=32,                          # scaling = alpha/r = 2.0
    target_modules=["query", "key", "value"],  # 어텐션 레이어
    lora_dropout=0.05,
    task_type=TaskType.FEATURE_EXTRACTION,
)
```

| 모델 | 전체 파라미터 | LoRA 학습 파라미터 | 비율 |
|------|------------|-----------------|------|
| SaProt-35M | 34.3M | 0.55M | 1.6% |
| SaProt-650M | 652M | ~10M | ~1.5% |

### 학습 설정

| 파라미터 | Frozen 모드 | LoRA 모드 |
|---------|-----------|---------|
| Optimizer | Adam | AdamW |
| LR (헤드) | 1e-3 | 5e-4 (lr×10) |
| LR (LoRA) | — | 5e-5 |
| Batch size | 128 | 35M:32 / 650M:8 |
| Max epochs | 50 | 50 |
| Early stopping | patience=10 | patience=10 |
| Loss | HuberLoss | HuberLoss |
| Scheduler | CosineAnnealingLR | CosineAnnealingLR |
| Gradient clipping | 1.0 | 1.0 |

---

## Frozen 실험 결과 (완료)

| 모델 | Test Pearson r | Val Best r | 학습 시간 | 상태 |
|------|---------------|-----------|---------|------|
| SaProt-650M frozen | **0.7855** | 0.7990 | 58.7초 | ✅ |
| SaProt-35M frozen | **0.7832** | 0.7872 | 54.8초 | ✅ |
| SaProt-650M-4bit frozen | **0.7914** | 0.8016 | 197.6초 | ✅ |

**핵심 발견:**
- 35M이 650M 대비 파라미터 약 19배 적음에도 성능 차이 단 0.0023
- 4bit 양자화 적용 시 성능 소폭 향상 (val r 0.7990 → **0.8016**, 목표치 0.8 달성)
- 4bit frozen 만으로도 val r ≥ 0.8 달성 — Q3의 충분 조건 부분 충족

---

## LoRA 실험 결과 (진행 중)

### 실험 D — SaProt-35M + LoRA

| 항목 | 값 |
|------|-----|
| 로컬 시도 | 2026-03-26, 1 에폭 완료 후 중단 |
| Train Loss (Epoch 1) | 0.3283 |
| Val Pearson r (Epoch 1) | 0.5410 |
| 에폭당 소요 시간 (GTX 1650) | ~2.5시간 (Tensor Core 없음) |
| **재실행 환경** | **Kaggle (P100 16GB, 주 30h 무료) 또는 로컬 장시간 실행** |
| **Test Pearson r** | **TBD** |

**상태:** ⏳ 재실행 필요

> **로컬 중단 이유:** GTX 1650 SUPER에는 Tensor Core가 없어 FP16 행렬 연산이 매우 느림.
> Epoch 1 완료까지 ~2.5시간 소요 → 50에폭 완료까지 ~125시간 필요.
> **Colab 사용 불가 (2026-03-27 GPU 한도 초과)** → Kaggle P100으로 대체 검토 중.

---

### 실험 E — SaProt-650M + LoRA

| 항목 | 값 |
|------|-----|
| 학습 파라미터 | ~10M (전체 652M의 1.5%) |
| batch_size | 8 |
| Gradient checkpointing | ✅ (VRAM 절약) |
| Colab 실행 결과 | **Epoch 36에서 GPU 한도 초과로 강제 중단** |
| Val Pearson r (Epoch 36 부근) | **~0.8x (목표치 확인, 정확한 수치 미기록)** |
| **Test Pearson r** | **TBD (체크포인트 없어 가중치 소실)** |

**상태:** ❌ Colab GPU 한도 초과 중단 — 재실행 시 체크포인트 저장 필수

> **교훈:** 학습 루프에 best 갱신 시 즉시 디스크 저장 로직 추가 필요.
> 현재 코드는 학습 완료 후에만 저장 → 중단 시 가중치 전부 소실.

---

### 실험 F — SaProt-650M-4bit + LoRA

| 항목 | 값 |
|------|-----|
| 양자화 | NF4 4-bit (bitsandbytes) |
| **Test Pearson r** | **TBD** |

**상태:** ⏳ 대기 중

---

## 전체 실험 히스토리

| 버전 | 일시 | 방법 | Pearson r | 상태 |
|------|------|------|----------|------|
| V1 | 2026-03-25 오전 | SaProt-650M 랜덤 헤드 (CPU, 17h) | 0.030 | ❌ |
| V2 | 2026-03-25 20:04 | SPRINT + MERGED 가중치 | 0.141 | ❌ OOD |
| V3-A | 2026-03-25 21:55 | SaProt-650M frozen + MLP | 0.7855 | ✅ |
| V3-B | 2026-03-25 22:16 | SaProt-35M frozen + MLP | 0.7832 | ✅ |
| V3-C | 2026-03-26 | SaProt-650M-4bit frozen | **0.7914** (val 0.8016) | ✅ |
| V4-D | 2026-03-26 (Epoch 1 only) | SaProt-35M + LoRA (Val r=0.5410 @ ep1) | TBD | ⏳ Kaggle 재실행 예정 |
| V4-E | 2026-03-26~27 (Epoch 36에서 중단) | SaProt-650M + LoRA (Val r ~0.8x @ ep36) | TBD | ❌ Colab GPU 한도 초과 |
| V4-F | 예정 | SaProt-650M-4bit + LoRA | TBD | ⏳ |

---

## 핵심 연구 질문

```
[Q1] LoRA가 frozen 대비 얼마나 성능을 올리는가?
     frozen 35M: r=0.7832  →  LoRA 35M: r=???  (목표: +0.1 이상)

[Q2] 35M + LoRA가 frozen 650M을 초과할 수 있는가?
     frozen 650M: r=0.7855  →  LoRA 35M: r=???  (초과 시 핵심 기여)

[Q3] 4-bit + LoRA가 4GB에서 작동하면서 r ≥ 0.8을 유지하는가?
     최소 VRAM으로 최대 성능 → GTX 1650 실용화 기준
```

---

## Frozen vs LoRA 비교 예측 (학습 완료 후 업데이트)

| 모델 | Frozen r | LoRA r | 향상폭 | VRAM |
|------|---------|--------|--------|------|
| SaProt-650M | 0.7855 | TBD | TBD | ~2.0 → ~2.2 GB |
| SaProt-35M | 0.7832 | TBD | TBD | ~0.8 → ~1.0 GB |
| SaProt-650M-4bit | **0.7914** (val 0.8016) | TBD | TBD | ~0.8 → ~1.0 GB |

---

*결과는 학습 완료 후 업데이트됩니다.*
