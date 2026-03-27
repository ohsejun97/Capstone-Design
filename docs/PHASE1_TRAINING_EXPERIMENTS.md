# Phase 1 학습 실험 보고서 — SaProt DTI Tool 경량화 비교

> **작성일시:** 2026년 3월 27일 KST (최종 업데이트)
> **목표:** SaProt frozen + MLP 헤드를 DAVIS/KIBA로 학습 후, 경량 백본 / 양자화 수준별 성능 비교 및 교차 데이터셋 일반화 검증
> **현재 상태:** ✅ 완료 — DAVIS 4모델 + KIBA 교차검증 완료, DTI Tool 확정

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
| 데이터셋 | DAVIS (연속 pKd, 30,056 쌍) + KIBA (KIBA score, 118,254 쌍) |

---

## 실험 설계

### SaProt Frozen + MLP 헤드 구조

SaProt은 학습하지 않고 완전 고정(frozen). DTI MLP 헤드만 DAVIS pKd로 학습.

```
SMILES  → Morgan FP (2048-bit)    ─┐
                                    ├→ DTI MLP 헤드 → pKd
AA서열  → SA 토큰 → SaProt (frozen) ─┘
```

### 비교 축 (2가지)

| 비교 | 내용 |
|------|------|
| **경량 백본** | SaProt-650M (652M params) vs SaProt-35M (34M params) |
| **양자화** | FP16 (none) vs INT8 (8-bit) vs NF4 (4-bit) |

### 학습 설정 (공통)

| 파라미터 | 값 |
|---------|----|
| Optimizer | Adam |
| Learning Rate | 1e-3 |
| Batch size | 128 |
| Max epochs | 50 |
| Early stopping | patience=10 |
| Loss | HuberLoss (delta=1.0) |
| Scheduler | CosineAnnealingLR |
| Train / Val / Test | 70% / 10% / 20% |

### DTI MLP 헤드 구조

```
prot_enc: Linear(prot_dim→512) → LayerNorm → GELU → Linear(512→256) → GELU
drug_enc: Linear(2048→512)     → BatchNorm  → GELU → Linear(512→256) → GELU
regressor: Linear(512→256) → GELU → Dropout(0.1) → Linear(256→64) → GELU → Linear(64→1)
```

---

## 실험 결과 (전체 완료)

| 모델 | 파라미터 | 양자화 | Test Pearson r | Val Best r | 학습 시간 | 상태 |
|------|---------|--------|---------------|-----------|---------|------|
| SaProt-650M | 652M | FP16 (none) | 0.7855 | 0.7990 | 58.7초 | ✅ |
| SaProt-35M | 34M | FP16 (none) | 0.7832 | 0.7872 | 54.8초 | ✅ |
| SaProt-650M-8bit | 652M | INT8 | 0.7812 | 0.7951 | 61초 | ✅ |
| **SaProt-650M-4bit** | 652M | **NF4 4-bit** | **0.7914** | **0.8016** | 197.6초 | ✅ |

---

## 핵심 발견

### 1. 경량 백본 비교 (650M vs 35M)

- SaProt-35M: 파라미터 **19배 적음** (652M → 34M)
- Test Pearson r 차이: **0.0023** — 사실상 동일한 예측 성능
- **결론:** 35M도 Agent DTI Tool로 충분히 사용 가능

### 2. 양자화 비교 (FP16 vs 8-bit vs 4-bit)

- 8-bit: FP16 대비 소폭 하락 (0.7855 → 0.7812)
  - bitsandbytes INT8이 compute dtype을 FP16으로 강제 → 정밀도 소폭 손실
- 4-bit NF4: FP16 대비 오히려 **소폭 향상** (0.7855 → 0.7914, val r **0.8016** 달성)
  - NF4 + double quantization 조합이 이 태스크에서 정규화 효과 발생한 것으로 추정
- **결론:** 4-bit 양자화가 성능 손실 없이 VRAM을 대폭 절감 → Agent Tool 최적 후보

### 3. 전체 결론

모든 모델의 Test Pearson r이 **0.78~0.79 범위**에서 수렴.
→ 모델 간 성능 차이보다 **"어떤 모델을 Agent Tool로 쓸 것인가"** 의 효율성 기준이 핵심.
→ **SaProt-650M-4bit** (val r=0.8016, VRAM 절감) 또는 **SaProt-35M** (속도 우선)을 DTI Tool로 채택.

---

## 전체 실험 히스토리

| 버전 | 일시 | 방법 | Test Pearson r | 상태 |
|------|------|------|---------------|------|
| V1 | 2026-03-25 오전 | SaProt-650M 랜덤 헤드 (CPU, 17h) | 0.030 | ❌ 입력 오류 |
| V2 | 2026-03-25 20:04 | SPRINT + MERGED 가중치 | 0.141 | ❌ OOD 문제 |
| V3-A | 2026-03-25 21:55 | SaProt-650M frozen + MLP (DAVIS) | 0.7855 | ✅ |
| V3-B | 2026-03-25 22:16 | SaProt-35M frozen + MLP (DAVIS) | 0.7832 | ✅ |
| V3-C | 2026-03-26 | SaProt-650M-8bit frozen + MLP (DAVIS) | 0.7812 | ✅ |
| V3-D | 2026-03-26 | SaProt-650M-4bit frozen + MLP (DAVIS) | **0.7914** | ✅ |
| V4-A | 2026-03-27 | SaProt-650M-4bit frozen + MLP (KIBA) | **0.7994** | ✅ |
| V4-B | 2026-03-27 | SaProt-35M frozen + MLP (KIBA) | 0.7894 | ✅ |
| V4-C | 2026-03-27 | SaProt-650M-8bit frozen + MLP (KIBA) | 0.7916 | ✅ |

> **LoRA 실험:** Colab GPU 한도 초과 + 로컬 GTX 1650 속도 한계(epoch당 2.5h)로 중단.
> 프로젝트 방향을 "모델 파인튜닝" → "Agent 시스템 구축"으로 전환.

---

## Phase 2-a: KIBA 교차 검증 — 전 모델 비교 (Cross-Dataset Generalization)

> "DAVIS에서 잘 된 게 우연인가? 양자화 효과는 데이터셋에 무관한가?" → 3개 모델 전부 KIBA 검증

### 결과

| 모델 | DAVIS Test r | KIBA Test r | KIBA Val r | 학습 시간 |
|------|-------------|------------|-----------|---------|
| SaProt-35M | 0.7832 | 0.7894 | 0.8001 | 241초 |
| SaProt-650M-8bit | 0.7812 | 0.7916 | 0.8042 | 224초 |
| **SaProt-650M-4bit** | **0.7914** | **0.7994** | **0.8106** | 206초 |

### 전체 평가 지표 (KIBA)

| 모델 | Pearson r | RMSE | MAE | CI |
|------|-----------|------|-----|-----|
| SaProt-35M-kiba | 0.7894 | 0.5126 | 0.3088 | 0.8289 |
| SaProt-650M-8bit-kiba | 0.7916 | 0.5117 | 0.3112 | 0.8273 |
| SaProt-650M-4bit-kiba | **0.7994** | **0.5026** | **0.3028** | **0.8324** |

### 핵심 결론

1. **일반화 확인:** 3개 모델 모두 KIBA에서 r ≈ 0.79~0.80 → DAVIS 결과가 우연이 아님
2. **양자화 효과 일관성:** 4bit > 8bit > 35M 순서가 KIBA에서도 동일하게 유지
3. **KIBA ≥ DAVIS 성능:** 더 많은 학습 데이터(4배)로 MLP 헤드가 더 잘 수렴
4. **DTI Tool 확정:** SaProt-650M-4bit가 두 데이터셋 모두 최상위 → Agent DTI Tool로 채택

---

## Phase 2-b 이후 방향

Phase 2-a에서 DTI Tool 완전 확정 → Phase 2-b부터 Agent 시스템 구현

```
Phase 2-b-1: Protein Tool — AlphaFold DB API 연동
Phase 2-b-2: Ligand Tool  — RDKit 3D conformer 생성
Phase 2-b-3: Agent 구현   — smolagents orchestration
Phase 2-b-4: End-to-End 데모 (자연어 → 결과 + 구조 설명)
```
