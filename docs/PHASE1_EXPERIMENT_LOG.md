# Phase 1 실험 일지 — DTI 파이프라인 구축 과정

> **작성일시:** 2026년 3월 25일 (초안) / 2026년 3월 30일 (최종 업데이트)
> **목표:** SaProt-650M 기반 DAVIS DTI 파이프라인 구축
> **최종 상태:** ✅ 완료 — Phase 3까지 종료, 최종 r=0.8082 (SaProt-650M FP16 + 3Di)

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

### 결과

| 지표 | 값 |
|------|-----|
| **Pearson R** | **0.030** |

### 실패 원인

1. **잘못된 모델 클래스:** 랜덤 초기화된 분류 헤드 추가 → 완전한 노이즈 출력
2. **잘못된 입력 포맷:** SMILES를 단백질 토크나이저에 입력 → 어휘 미인식
3. **DTI 학습 없음:** SaProt은 Masked LM 사전학습 모델, pKd 예측 불가

---

## 실험 V2 — SPRINT 아키텍처 + panspecies-dti 가중치

**일시:** 2026년 3월 25일 20:04 ~ 21:09 KST

### 아키텍처

```
단백질: SA 시퀀스 → SaProt-650M (frozen) → MultiHeadAttentionPool → [1024] → L2 normalize
약물:   SMILES → Morgan FP (2048-bit) → Linear × 3 → [1024] → L2 normalize
융합:   cosine similarity
```

### 결과

| 지표 | 값 |
|------|-----|
| **Pearson R** | **+0.1412** |

### 실패 원인

1. **OOD 가중치:** SPRINT은 MERGED 데이터로 학습, DAVIS는 포함 안 됨
2. **태스크 불일치:** 이진 분류 모델 → 연속 pKd 회귀에 부적합

---

## 실험 V3 — SaProt + DTI MLP 헤드 (DAVIS 직접 학습)

**일시:** 2026년 3월 25일 21:55 ~

### 방법론 전환 배경

V2 실패 분석 결과, 핵심 문제는 **"DTI 태스크용으로 DAVIS에서 직접 학습된 가중치가 없다"**는 것.

해결책: SaProt을 frozen 인코더로 사용하고, **소형 MLP 헤드만 DAVIS 연속 pKd 데이터로 학습**.

- **데이터:** DeepPurpose DAVIS (연속 pKd, 30,056 쌍)
- **분할:** Train 70% / Val 10% / Test 20% (seed=42)

### 파이프라인

```
SMILES  → Morgan FP (radius=2, nBits=2048)              ─┐
                                                          ├→ DTI MLP 헤드 → pKd
AA서열  → SA 토큰("M#E#T#...") → SaProt (frozen) → mean pool ─┘
```

### V3-A: SaProt-650M FP16

| 항목 | 값 |
|------|-----|
| 단백질 임베딩 계산 | 379개 unique proteins, ~20분 (GPU) |
| 학습 시간 | **58.7초** (MLP 헤드만) |
| **Test Pearson r** | **0.7855** |
| Val Best r | 0.7990 |

### V3-B: SaProt-35M FP16

| 항목 | 값 |
|------|-----|
| 학습 시간 | **54.8초** |
| **Test Pearson r** | **0.7832** |

**핵심 발견:** 35M이 650M 대비 파라미터 19배 적지만 성능 차이 **0.0023**

### V3-C/D: 양자화 실험

| 모델 | Test r | 비고 |
|------|--------|------|
| SaProt-650M-8bit | 0.7812 | INT8 소폭 하락 |
| SaProt-650M-4bit | **0.7914** | NF4 오히려 향상 |

---

## LoRA 파인튜닝 시도 및 포기

| 항목 | 내용 |
|------|------|
| 시도 이유 | SOTA 대비 성능 갭(r=0.79 vs 0.89) 줄이기 위해 |
| 문제 | GTX 1650 SUPER는 Tensor Core 없음 → epoch당 ~2.5시간 |
| 결론 | 50 epoch × 2.5h = 5일 이상 → 프로젝트 기간 내 불가 |

**방향 전환:** "모델 파인튜닝으로 SOTA 달성" → "frozen 인코더 기반 Agent 시스템 구축"

이 전환은 연구 목적을 재정의한다:
> SaProt의 DTI 예측 능력 자체를 극대화하는 것이 아니라,
> **SaProt의 3Di 구조 토큰이 DTI 예측에 기여하는지 검증하고, 이를 Agent Tool로 활용**하는 것.

---

## Phase 3 — FoldSeek 3Di 통합

3Di 토큰 적용 후 전 모델에서 성능 향상 확인.
FP16이 3Di 구조 신호를 가장 잘 활용함.

**최종 결과 (DAVIS):**
```
650M FP16 + 3Di:  r = 0.8082  (+0.0227 vs placeholder)
35M FP16  + 3Di:  r = 0.7996  (+0.0165)
650M-8bit + 3Di:  r = 0.8027  (+0.0215)
650M-4bit + 3Di:  r = 0.7977  (+0.0063)
```

**최종 모델 확정: SaProt-650M FP16 + 3Di**

자세한 내용: [Phase 1 Training Report](PHASE1_TRAINING_EXPERIMENTS.md)
