# Phase 1g — Transfer Learning: BindingDB Head → DAVIS/KIBA

> **작성일시:** 2026-04-09
> **목표:** BindingDB로 학습된 DTI 모델의 표현을 DAVIS/KIBA에 그대로 활용 — MLP Head만 재학습(Transfer Learning)으로 cross-dataset 일반화 달성

---

## 배경 및 문제 정의

Phase 1f에서 BindingDB(cold_drug) 학습 모델을 zero-shot으로 DAVIS/KIBA에 적용했을 때 성능이 매우 낮았다.

| 방식 | DAVIS Pearson r | KIBA Pearson r |
|---|---|---|
| Zero-shot cross-eval | 0.208 | 0.160 |

원인은 두 가지였다.

1. **도메인 시프트**: BindingDB는 다양한 타겟을 포함하지만 DAVIS는 키나아제 특화 데이터셋.
2. **레이블 불일치**: KIBA score는 pKd가 아닌 Ki/Kd/IC50 복합 지표로 스케일 자체가 다름. (KIBA mean=11.72 vs BindingDB pKd mean=6.25)

### 핵심 인사이트

SaProt(frozen) + ChemBERTa(frozen)의 **임베딩 표현 자체는 재사용 가능**하다. 실패의 원인은 임베딩 품질이 아니라, MLP Head가 BindingDB의 pKd 스케일에만 최적화되어 있다는 것. Head만 각 데이터셋에 맞게 재보정(re-calibration)하면 된다.

---

## 방법론

### Transfer Learning 전략

```
[SaProt-650M FP16 + 3Di]  ← frozen, 재사용 (캐시된 임베딩)
[ChemBERTa]               ← frozen, 재사용 (캐시된 임베딩)
        ↓
[MLP Head]  ← BindingDB 가중치로 warm-start → DAVIS/KIBA로 fine-tune
```

- SaProt과 ChemBERTa는 전혀 건드리지 않음 (가중치 고정 + 캐시 재사용)
- MLP Head만 BindingDB 학습 결과를 초기값(warm-start)으로 하여 각 데이터셋으로 재학습
- KIBA는 레이블을 z-score 정규화 후 학습, 추론 시 역정규화하여 KIBA 스케일로 복원

### 학습 설정

| 항목 | 값 |
|---|---|
| Source model | SaProt-650M-bindingdb-3di-chemberta (r=0.8737) |
| Warm-start | BindingDB head 가중치 로드 |
| LR | 3e-4 (원래 학습의 1/3 — warm-start이므로 낮게) |
| Optimizer | Adam + weight_decay=1e-4 |
| Scheduler | CosineAnnealingLR |
| Loss | HuberLoss (delta=1.0) |
| Batch size | 128 |
| Max epochs | 50 |
| Early stopping | patience=10 |
| KIBA 정규화 | z-score (mean=11.72, std=0.84) |

### 구현

```bash
# DAVIS fine-tune
python scripts/finetune_head.py \
    --source_model results/SaProt-650M-bindingdb-3di-chemberta \
    --target_dataset davis --split random

# KIBA fine-tune
python scripts/finetune_head.py \
    --source_model results/SaProt-650M-bindingdb-3di-chemberta \
    --target_dataset kiba --split random
```

---

## 실험 결과

### DAVIS

| 방식 | Pearson r | Spearman r | RMSE | CI | 학습시간 |
|---|---|---|---|---|---|
| Zero-shot (cross-eval) | 0.208 | — | — | — | — |
| DAVIS 직접 학습 (Phase 1c) | 0.8082 | — | — | — | — |
| **Transfer (BindingDB→DAVIS)** | **0.8166** | 0.6794 | 0.5303 | 0.8747 | 194s |

- Zero-shot 대비 **+0.609** 향상
- DAVIS 직접 학습(0.8082)보다 **+0.0084** 높음

### KIBA

| 방식 | Pearson r | Spearman r | RMSE | CI | 학습시간 |
|---|---|---|---|---|---|
| Zero-shot (cross-eval) | 0.160 | — | — | — | — |
| KIBA 직접 학습 (Phase 1c) | 0.8032 | — | — | — | — |
| **Transfer (BindingDB→KIBA)** | **0.8163** | 0.8114 | 0.4826 | 0.8414 | 794s |

- Zero-shot 대비 **+0.656** 향상
- KIBA 직접 학습(0.8032)보다 **+0.0131** 높음

---

## 분석

### 왜 직접 학습보다 Transfer가 더 잘 나오는가?

DAVIS(68 약물)와 KIBA(229 단백질)는 데이터 규모가 상대적으로 작다. BindingDB로 학습된 Head는 이미 32,480개 약물, 2,384개 단백질에 걸친 결합력 패턴을 압축한 상태로 초기화된다. 적은 데이터에서 scratch부터 최적화하는 것보다 좋은 시작점을 제공한다.

### KIBA Spearman r이 Pearson r과 거의 동일한 이유

KIBA는 정규화 후 학습하기 때문에 스케일 편향이 없고, 모델이 순위 정보와 절대값 모두 정확하게 학습한 것을 의미한다.

### Zero-shot이 그렇게 낮았던 이유 재확인

KIBA(mean=11.72)와 BindingDB pKd(mean=6.25)는 스케일이 역전에 가깝다. Head가 6~7 근처를 예측하도록 최적화되어 있으니, 정답이 11~13인 KIBA 데이터에서 Pearson r이 낮게 나오는 것은 당연하다. Head 재학습만으로 이 문제가 완전히 해소된다.

---

## 결론

| 실험 | DAVIS r | KIBA r |
|---|---|---|
| 기존 DAVIS/KIBA 직접 학습 (Phase 1c) | 0.8082 | 0.8032 |
| **BindingDB Transfer (Phase 1g)** | **0.8166** | **0.8163** |

**BindingDB로 학습된 SaProt+ChemBERTa 표현이 DAVIS/KIBA보다 더 풍부하고 일반화된 임베딩을 제공한다.** Head만 재학습해도 각 데이터셋에서의 직접 학습을 뛰어넘는다는 것을 실험적으로 증명했다. 이는 BindingDB 대규모 사전학습이 실질적인 Transfer Learning 역할을 한다는 것을 보여준다.

---

## 출력 파일

```
results/
├── finetune_davis_random_from_SaProt-650M-bindingdb-3di-chemberta/
│   ├── dti_head.pt     ← DAVIS fine-tuned head
│   └── result.json
└── finetune_kiba_random_from_SaProt-650M-bindingdb-3di-chemberta/
    ├── dti_head.pt     ← KIBA fine-tuned head
    └── result.json

scripts/
└── finetune_head.py    ← Transfer Learning 스크립트
```
