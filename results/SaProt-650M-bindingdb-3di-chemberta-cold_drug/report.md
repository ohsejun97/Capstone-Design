# 실험 보고서 — SaProt-650M-bindingdb-3di-chemberta-cold_drug

**생성일시:** 2026-04-09T12:41:02.982945

---

## 실험 설정

| 항목 | 값 |
|---|---|
| 데이터셋 | BINDINGDB |
| Protein Encoder | SaProt-650M (FP16) |
| Drug Encoder | ChemBERTa (seyonec/ChemBERTa-zinc-base-v1, frozen, 768-dim) |
| FoldSeek 3Di | ✅ 사용 |
| LoRA | ❌ Frozen |
| Split | Random 70 / 10 / 20 |
| Train / Val / Test | 57,436 / 7,913 / 15,446 |

---

## 성능 지표

| 지표 | 값 | 설명 |
|---|---|---|
| **Pearson r** | **0.7083** | 예측-실측 선형 상관계수 (주 지표) |
| RMSE | 1.2543 | 평균 예측 오차 (pKd 단위) |
| MAE | 0.9893 | 평균 절대 오차 (pKd 단위) |
| CI | 0.7473 | 결합력 순위 일치도 (0.5=랜덤, 1.0=완벽) |
| Val best r | 0.6866 | 검증셋 최고 Pearson r |

**판정:** △   양호 (r ≥ 0.6)

### SOTA 비교

| 모델 | DAVIS Pearson r |
|---|---|
| 본 실험 | **0.7083** |
| DeepPurpose MPNN_CNN | ~0.89 (SOTA) |
| DeepPurpose CNN | ~0.86 |
| 서열 기반 baseline | ~0.78~0.80 |

---

## 학습 정보

| 항목 | 값 |
|---|---|
| 학습 에포크 | 18 |
| 총 학습 시간 | 54.8초 (0.9분) |
| Early stopping | patience=10 |

---

## 추론 속도 / 하드웨어

| 항목 | 값 |
|---|---|
| 단일 샘플 추론 시간 | **1.343 ms** (± 1.545 ms) |
| 추론 속도 | **745 samples/sec** |
| 학습 중 최대 VRAM | 1294.6 MB |

---

## 결과 파일

| 파일 | 내용 |
|---|---|
| `result.json` | 전체 지표 요약 |
| `test_predictions.csv` | 예측값 vs 실측값 |
| `training_history.csv` | 에포크별 loss / val_r |
| `dti_head.pt` | 최적 모델 가중치 |
