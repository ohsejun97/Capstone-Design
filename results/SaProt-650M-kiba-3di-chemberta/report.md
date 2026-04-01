# 실험 보고서 — SaProt-650M-kiba-3di-chemberta

**생성일시:** 2026-04-01T16:25:04.877998

---

## 실험 설정

| 항목 | 값 |
|---|---|
| 데이터셋 | KIBA |
| Protein Encoder | SaProt-650M (FP16) |
| Drug Encoder | ChemBERTa (seyonec/ChemBERTa-zinc-base-v1, frozen, 768-dim) |
| FoldSeek 3Di | ✅ 사용 |
| LoRA | ❌ Frozen |
| Split | Random 70 / 10 / 20 |
| Train / Val / Test | 82,777 / 11,825 / 23,652 |

---

## 성능 지표

| 지표 | 값 | 설명 |
|---|---|---|
| **Pearson r** | **0.7667** | 예측-실측 선형 상관계수 (주 지표) |
| RMSE | 0.5351 | 평균 예측 오차 (pKd 단위) |
| MAE | 0.3293 | 평균 절대 오차 (pKd 단위) |
| CI | 0.8148 | 결합력 순위 일치도 (0.5=랜덤, 1.0=완벽) |
| Val best r | 0.7762 | 검증셋 최고 Pearson r |

**판정:** △   양호 (r ≥ 0.6)

### SOTA 비교

| 모델 | DAVIS Pearson r |
|---|---|
| 본 실험 | **0.7667** |
| DeepPurpose MPNN_CNN | ~0.89 (SOTA) |
| DeepPurpose CNN | ~0.86 |
| 서열 기반 baseline | ~0.78~0.80 |

---

## 학습 정보

| 항목 | 값 |
|---|---|
| 학습 에포크 | 50 |
| 총 학습 시간 | 218.6초 (3.6분) |
| Early stopping | patience=10 |

---

## 추론 속도 / 하드웨어

| 항목 | 값 |
|---|---|
| 단일 샘플 추론 시간 | **1.634 ms** (± 1.367 ms) |
| 추론 속도 | **612 samples/sec** |
| 학습 중 최대 VRAM | 2491.5 MB |

---

## 결과 파일

| 파일 | 내용 |
|---|---|
| `result.json` | 전체 지표 요약 |
| `test_predictions.csv` | 예측값 vs 실측값 |
| `training_history.csv` | 에포크별 loss / val_r |
| `dti_head.pt` | 최적 모델 가중치 |
