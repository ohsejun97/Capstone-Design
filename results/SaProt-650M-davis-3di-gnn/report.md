# 실험 보고서 — SaProt-650M-davis-3di-gnn

**생성일시:** 2026-03-31T18:21:21.068214

---

## 실험 설정

| 항목 | 값 |
|---|---|
| 데이터셋 | DAVIS |
| Protein Encoder | SaProt-650M (FP16) |
| Drug Encoder | Morgan FP (2048-bit) + GNN/MPNN (256-dim) concat → 2304-dim |
| FoldSeek 3Di | ✅ 사용 |
| LoRA | ❌ Frozen |
| Split | Random 70 / 10 / 20 |
| Train / Val / Test | 21,039 / 3,005 / 6,012 |

---

## 성능 지표

| 지표 | 값 | 설명 |
|---|---|---|
| **Pearson r** | **0.5795** | 예측-실측 선형 상관계수 (주 지표) |
| RMSE | 0.7618 | 평균 예측 오차 (pKd 단위) |
| MAE | 0.4032 | 평균 절대 오차 (pKd 단위) |
| CI | 0.7907 | 결합력 순위 일치도 (0.5=랜덤, 1.0=완벽) |
| Val best r | 0.6125 | 검증셋 최고 Pearson r |

**판정:** ❌  성능 미달

### SOTA 비교

| 모델 | DAVIS Pearson r |
|---|---|
| 본 실험 | **0.5795** |
| DeepPurpose MPNN_CNN | ~0.89 (SOTA) |
| DeepPurpose CNN | ~0.86 |
| 서열 기반 baseline | ~0.78~0.80 |

---

## 학습 정보

| 항목 | 값 |
|---|---|
| 학습 에포크 | 20 |
| 총 학습 시간 | 774.3초 (12.9분) |
| Early stopping | patience=10 |

---

## 추론 속도 / 하드웨어

| 항목 | 값 |
|---|---|
| 단일 샘플 추론 시간 | **0.707 ms** (± 0.038 ms) |
| 추론 속도 | **1414 samples/sec** |
| 학습 중 최대 VRAM | 2062.9 MB |

---

## 결과 파일

| 파일 | 내용 |
|---|---|
| `result.json` | 전체 지표 요약 |
| `test_predictions.csv` | 예측값 vs 실측값 |
| `training_history.csv` | 에포크별 loss / val_r |
| `dti_head.pt` | 최적 모델 가중치 |
