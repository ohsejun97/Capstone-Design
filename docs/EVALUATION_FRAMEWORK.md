# 평가 프레임워크 — DTI 예측 성능 측정 기준

> **DTI 예측은 회귀(Regression) 태스크다.**
> 입력(Drug + Protein) → 출력(pKd, 연속 실수값)이므로, 분류의 "정확도(accuracy)"가 아닌 아래의 지표를 사용한다.

---

## 1. 사용 지표 및 해석

### 1.1 Pearson r — 주 지표 (예측 경향성)

예측값과 실측 pKd 값의 **선형 상관관계**. DTI 논문에서 가장 보편적으로 사용되는 1차 지표.

```
r = 1.0  완벽한 예측
r = 0.0  무관계 (랜덤과 동일)
r < 0.0  역상관 (잘못된 예측)
```

| 수준 | Pearson r | 의미 |
|---|---|---|
| 랜덤 헤드 | ~0.03 | 예측력 없음 (본 연구 V1 초기 실패) |
| 서열 기반 baseline | ~0.75~0.80 | 기존 서열 모델 수준 |
| **본 연구 현재 (3Di)** | **0.81** | 구조 토큰 통합 효과 |
| **본 연구 목표 (GNN)** | **≥ 0.85** | GNN drug encoder 추가 후 |
| SOTA | ~0.89 | end-to-end fine-tuning + GNN |

**"정확하다"는 기준:** DTI 분야에서 절대적 컷오프는 없다. **기존 연구 대비 comparable한가**가 판단 기준이며, 본 연구는 DAVIS r ≥ 0.85를 목표로 한다.

**p-value에 대하여:** DAVIS 테스트셋(~6,000쌍), KIBA 테스트셋(~23,000쌍) 규모에서 r ≥ 0.7이면 p-value는 사실상 항상 p < 0.001 이하로 통계적 유의성 자체는 의미가 없다. 대신 **동일 경향이 DAVIS와 KIBA 두 데이터셋에서 재현되는지**가 신뢰도의 근거가 된다.

---

### 1.2 RMSE — 절대 오차 (예측 정밀도)

예측값이 실측 pKd에서 평균적으로 얼마나 벗어나는지. **pKd 단위(log scale)**로 해석한다.

```
RMSE = 0.5  →  예측이 실제보다 평균 0.5 pKd 단위 빗나감
RMSE = 1.0  →  결합력 10배 차이를 구분 못할 수 있는 수준
```

Pearson r의 보조 지표로 사용. r이 높아도 RMSE가 크면 절대적 pKd 수치 예측은 부정확하다.

---

### 1.3 CI (Concordance Index) — 순위 일치도 (실용적 지표)

임의의 두 drug-protein 쌍 중 **"결합력이 더 강한 쪽을 맞히는 확률"**.

```
CI = 0.5  랜덤 (동전 던지기와 동일)
CI = 1.0  항상 순위를 정확히 맞힘
CI = 0.86 현재 본 연구 수준
```

신약 개발 실무에서는 절대적 pKd 값보다 "어떤 약이 이 단백질에 더 잘 붙는가"의 **순위**가 중요하다. 따라서 Pearson r이 주 지표이지만 CI가 실용적 의미를 보완한다.

---

## 2. 비교 분석 프레임워크

### 비교 ① 구조 토큰의 효과 — "3Di가 기여하는가?" ✅ 완료

**비교 대상:** Placeholder('#') vs FoldSeek 3Di 토큰
**통제 변수:** 동일 모델, 동일 데이터셋, 동일 학습 설정
**결론 조건:** 모든 모델에서 일관된 향상이 있으면 "3Di 토큰이 DTI 예측에 기여한다"

| 모델 | Placeholder r | 3Di r | Delta |
|---|---|---|---|
| SaProt-650M FP16 | 0.7855 | **0.8082** | +0.023 |
| SaProt-35M FP16 | 0.7832 | 0.7996 | +0.016 |
| SaProt-650M-8bit | 0.7812 | 0.8027 | +0.022 |
| SaProt-650M-4bit | 0.7914 | 0.7977 | +0.006 |

→ 전 모델에서 일관된 향상 확인. "3Di 구조 토큰이 DTI 예측 성능에 유의미하게 기여한다" ✅

---

### 비교 ② Drug Encoder 교체 효과 — "GNN이 Morgan FP보다 낫는가?" 🔄 Phase 1d

**비교 대상:** Morgan FP(현재) vs GNN — AttentiveFP/MPNN
**통제 변수:** 동일 Protein Encoder(SaProt-650M FP16 + 3Di), 동일 데이터셋
**결론 조건:** DAVIS와 KIBA 모두에서 향상이 나타나면 "GNN drug encoder가 유효하다"

| Drug Encoder | DAVIS r | KIBA r |
|---|---|---|
| Morgan FP (baseline) | 0.8082 | 0.8032 |
| GNN (목표) | ≥ 0.85 | ≥ 0.82 |

---

### 비교 ③ SOTA 대비 상대적 위치 — "저사양에서도 경쟁력이 있는가?"

**비교 대상:** 본 연구 vs DeepPurpose(2020) vs ConPLex(2023) vs SOTA
**결론 조건:** 유사하거나 그 이상의 성능을 4GB VRAM 환경에서 달성하면 "저사양 환경에서도 실용적 성능을 낼 수 있다"

| 모델 | 연도 | Drug Encoder | Protein Encoder | DAVIS r | VRAM |
|---|---|---|---|---|---|
| DeepPurpose CNN | 2020 | Morgan FP | CNN (서열만, 학습) | ~0.86 | ~2GB |
| DeepPurpose MPNN | 2020 | MPNN (GNN, 학습) | CNN (서열만, 학습) | ~0.89 | ~2GB |
| ConPLex | 2023 | GNN (학습) | ESM-2 (frozen) | ~0.90 | >8GB |
| **본 연구 (현재)** | **2026** | **Morgan FP** | **SaProt+3Di (frozen)** | **0.81** | **~2GB** |
| **본 연구 (목표)** | **2026** | **GNN (학습)** | **SaProt+3Di (frozen)** | **≥ 0.85** | **~2GB** |

**ConPLex와의 차별점:** ConPLex는 frozen PLM(ESM-2) + GNN 조합으로 r≈0.90을 달성했으나, 구조 정보 없는 ESM-2를 사용했다. 본 연구는 ESM-2 대신 FoldSeek 3Di 구조 토큰이 내장된 SaProt을 사용하며, Phase 1c에서 3Di 토큰이 실질적 성능 기여(+0.023)를 실험적으로 확인했다.

---

### 비교 ④ 일반화 능력 — "우연이 아닌가?" 🔄 Phase 1d

**Cross-dataset 검증 (이미 수행):**
- DAVIS(442개 키나아제)와 KIBA(229개 키나아제) 양쪽에서 r ≈ 0.80 재현
- → "결과가 특정 데이터셋에 과적합된 게 아니다"

**Cold-target split (예정):**

| Split 방식 | 설명 | 본 연구 계획 |
|---|---|---|
| Random (현재) | 쌍 단위 무작위 분할. 동일 단백질이 train·test에 모두 등장 | 기준선 |
| Cold-target | 테스트 단백질이 학습에 미등장. 실제 신규 타겟 스크리닝 시나리오 | Phase 1d 추가 |

→ Random split r과 Cold-target split r을 나란히 보고하면 "성능이 data leakage 없이 진짜 일반화된 결과"임을 확인 가능.

---

## 3. 최종 보고 형식

성능 결과를 보고할 때 아래 형식으로 기술한다.

**정확도 서술 예시:**
> "제안 모델은 DAVIS 테스트셋(random split)에서 Pearson r = 0.85, RMSE = 0.62, CI = 0.87을 달성하였다. 이는 full fine-tuning 기반 SOTA(DeepPurpose MPNN_CNN, r = 0.89) 대비 VRAM 4GB 환경에서 4% 이내의 성능 차이로, 저사양 환경에서의 실용적 DTI 예측이 가능함을 보여준다."

**경향성 서술 예시:**
> "FoldSeek 3Di 구조 토큰 적용 시 전 모델에서 Pearson r이 일관되게 향상되었으며(+0.006 ~ +0.023), GNN drug encoder 교체 후 Morgan FP 대비 추가 향상이 DAVIS/KIBA 양 데이터셋에서 동일하게 관찰되었다. 이는 protein encoder(3Di)와 drug encoder(GNN)를 독립적으로 개선하는 전략이 DTI 성능 향상에 유효함을 지지한다."
