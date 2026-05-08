# Phase 1i — Transfer Learning (ft ChemBERTa): BindingDB → DAVIS/KIBA

> **작성일시:** 2026-05-08
> **목표:** Phase 1h에서 fine-tune된 ChemBERTa로 DAVIS/KIBA drug embedding을 재계산, MLP Head만 재학습하여 Phase 1g(frozen) 대비 성능 향상 확인

---

## 배경 및 동기

### Phase 1g의 한계

Phase 1g에서 frozen ChemBERTa drug embedding + head 재학습으로 DAVIS r=0.8166, KIBA r=0.8163을 달성했다. 이는 직접 학습(Phase 1c, r=0.8082/0.8032)을 초과한 결과였지만, drug embedding이 여전히 범용 PubChem 분자 성질 예측 기반이었다.

### Phase 1h의 발견

Phase 1h에서 ChemBERTa layers 4~5를 BindingDB DTI pKd 예측 태스크로 fine-tune하자 BindingDB random split에서 r=0.8923(frozen 0.8737 대비 +0.0186)을 달성했다. Fine-tuned ChemBERTa는 pKd와 더 직접적으로 연관된 화학 표현(극성기, 소수성 부위, 결합 형태)을 인코딩한다는 것이 검증됐다.

### 핵심 가설

fine-tuned ChemBERTa로 재계산한 DAVIS/KIBA drug embedding은 frozen 버전보다 DTI 친화도와 더 강하게 연관될 것이며, 이로 인해 head fine-tuning 성능이 Phase 1g를 초과할 것이다.

---

## 방법론

### 파이프라인

```
[Phase 1h] ChemBERTa fine-tune on BindingDB
  → chemberta_ft.pt (layers 4~5 업데이트 가중치 저장)

[Phase 1i]
  Step 1: vanilla ChemBERTa + chemberta_ft.pt 병합 → freeze
  Step 2: DAVIS/KIBA unique 약물 SMILES → ft ChemBERTa → drug embedding 캐싱
  Step 3: SaProt protein embedding (기존 캐시 재사용)
  Step 4: Phase 1h head warm-start → DAVIS/KIBA fine-tune
```

### Phase 1g와의 차이점

| 항목 | Phase 1g | Phase 1i |
|------|---------|---------|
| Drug embedding 소스 | frozen ChemBERTa (PubChem 사전학습) | **ft ChemBERTa (DTI fine-tuned)** |
| Head warm-start | BindingDB frozen head | **BindingDB ft head** |
| Drug embedding 캐시 | `drug_embs_{dataset}_chemberta.pt` | `drug_embs_{dataset}_chemberta_ft.pt` |

### 데이터셋 전처리 (Phase 1g와 동일)

| 데이터셋 | 로더 | 레이블 | 전처리 |
|---------|------|--------|--------|
| DAVIS | DeepPurpose `load_process_DAVIS` | pKd | `convert_to_log=True` (Kd nM → pKd) |
| KIBA | DeepPurpose `load_process_KIBA` | KIBA score | z-score 정규화 (mean=11.72, std=0.84) → 추론 시 역정규화 |

### 학습 설정

| 항목 | 값 |
|------|-----|
| Head warm-start | Phase 1h head (`dti_head.pt`) |
| LR | 3e-4 |
| Optimizer | Adam (weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR |
| Loss | HuberLoss (delta=1.0) |
| Batch size | 128 |
| Max epochs | 50 |
| Early stopping | patience=10 |
| Data split | random (70/10/20) |

### 실행 명령

```bash
# DAVIS
python scripts/finetune_head_ft.py --target_dataset davis --split random

# KIBA
python scripts/finetune_head_ft.py --target_dataset kiba --split random
```

---

## 실험 결과

### DAVIS

| 방식 | Pearson r | Spearman r | RMSE | MAE | CI | R² | 학습 시간 |
|------|-----------|-----------|------|-----|----|----|---------|
| 직접 학습 (Phase 1c) | 0.8082 | — | — | — | — | — | — |
| Transfer frozen (Phase 1g) | 0.8166 | 0.6794 | 0.5303 | 0.2911 | 0.8747 | — | 194s |
| **Transfer ft (Phase 1i)** | **0.8677** | **0.7021** | **0.4572** | **0.2514** | **0.8925** | **0.7507** | **204s** |
| **Phase 1g 대비 개선** | **+0.0511** | **+0.0227** | **-0.0731** | **-0.0397** | **+0.0178** | — | — |

### KIBA

| 방식 | Pearson r | Spearman r | RMSE | MAE | CI | R² | 학습 시간 |
|------|-----------|-----------|------|-----|----|----|---------|
| 직접 학습 (Phase 1c) | 0.8032 | — | — | — | — | — | — |
| Transfer frozen (Phase 1g) | 0.8163 | 0.8114 | 0.4826 | 0.2873 | 0.8414 | — | 794s |
| **Transfer ft (Phase 1i)** | **0.8594** | **0.8464** | **0.4268** | **0.2578** | **0.8610** | **0.7370** | **814s** |
| **Phase 1g 대비 개선** | **+0.0431** | **+0.0350** | **-0.0558** | **-0.0295** | **+0.0196** | — | — |

---

## 분석

### 1. Fine-tuned ChemBERTa 표현의 우수성

DAVIS와 KIBA 모두에서 frozen 대비 유의미한 향상이 확인됐다. DAVIS +0.051, KIBA +0.043으로 두 데이터셋에서 일관된 개선폭은 ChemBERTa fine-tuning의 효과가 특정 데이터셋에 편향되지 않음을 보여준다.

**RMSE 개선폭이 r보다 크다:**
- DAVIS RMSE: -13.8% (0.5303 → 0.4572)
- KIBA RMSE: -11.6% (0.4826 → 0.4268)

이는 ft ChemBERTa 표현이 이상값(outlier) 예측을 개선했음을 의미한다. frozen 표현은 극단적인 결합력 값(매우 강하거나 약한 결합)에서 오차가 컸는데, fine-tuning 후 이 부분이 교정됐다.

### 2. DAVIS Spearman r 개선 한계 (0.7021)

DAVIS Pearson r은 크게 올랐지만 Spearman r은 0.7021로 KIBA(0.8464)보다 낮다.

원인: DAVIS는 68개 약물의 전수(complete) 행렬 구조이며, 동일 약물이 442개 단백질 모두와 쌍을 이룬다. 결합력의 **절대값** 예측은 잘 되지만(r=0.8677), 같은 약물 내 단백질 간 **순위** 예측은 단백질 표현(SaProt)의 한계가 더 크게 작용한다. Spearman r을 높이려면 SaProt fine-tuning이 필요하지만 4GB VRAM에서는 불가능.

### 3. KIBA Spearman r과 Pearson r의 수렴

KIBA: Pearson 0.8594 vs Spearman 0.8464 — 두 지표가 비교적 가깝다. KIBA 데이터셋은 다양한 단백질 계열에 걸쳐 있어 순위 예측과 절대값 예측 난이도가 비슷하기 때문이다.

### 4. 학습 시간이 Phase 1g와 유사한 이유

Phase 1i는 fine-tuned ChemBERTa로 drug embedding을 한 번 캐싱한 뒤, 이후 head 학습은 Phase 1g와 동일하게 캐시된 임베딩만 사용한다. 따라서 학습 시간이 Phase 1g(194s/794s)와 거의 동일(204s/814s)하다.

### 5. Phase 1 전체 진화 경로

```
Phase 1c:  SaProt+3Di + Morgan FP (직접 학습)    DAVIS r=0.8082, KIBA r=0.8032
Phase 1g:  BindingDB Transfer (frozen ChemBERTa) DAVIS r=0.8166, KIBA r=0.8163
Phase 1h:  ChemBERTa fine-tune (BindingDB)       BindingDB r=0.8923
Phase 1i:  BindingDB Transfer (ft ChemBERTa)     DAVIS r=0.8677, KIBA r=0.8594 ← 최고
```

각 단계가 이전 단계의 한계를 명확히 해결하는 연속적 개선이다.

---

## 결론

**Fine-tuned ChemBERTa Transfer Learning이 Phase 1 최종 최고 성능을 달성했다.**

| | DAVIS r | KIBA r | 의미 |
|--|---------|--------|------|
| Phase 1c 직접 학습 | 0.8082 | 0.8032 | 베이스라인 |
| Phase 1g frozen transfer | 0.8166 | 0.8163 | +1% |
| **Phase 1i ft transfer** | **0.8677** | **0.8594** | **+7.3% / +6.9%** |

SOTA(DeepPurpose r≈0.89, ConPLex r≈0.90)와의 격차가 0.02~0.04 수준으로 좁혀졌으며, **GTX 1650 SUPER (4GB VRAM)** 환경 제약을 고려하면 실질적으로 경쟁력 있는 성능이다.

**Bio-AI Agent Tool 1** 에는 Phase 1i 모델을 사용한다:
- BindingDB 범위 약물/단백질: ft ChemBERTa BindingDB 모델 직접 사용 (r=0.8923)
- DAVIS 유사 타겟(키나아제): ft ChemBERTa → DAVIS transfer 모델 (r=0.8677)
- KIBA 측정 기반 타겟: ft ChemBERTa → KIBA transfer 모델 (r=0.8594)

---

## 출력 파일

```
results/
├── finetune_davis_random_from_SaProt-650M-bindingdb-3di-chemberta-unfreeze2-random_ft/
│   ├── dti_head.pt     ← DAVIS fine-tuned head
│   └── result.json
└── finetune_kiba_random_from_SaProt-650M-bindingdb-3di-chemberta-unfreeze2-random_ft/
    ├── dti_head.pt     ← KIBA fine-tuned head
    └── result.json

cache/
├── drug_embs_davis_chemberta_ft.pt  ← ft ChemBERTa DAVIS drug embeddings
└── drug_embs_kiba_chemberta_ft.pt   ← ft ChemBERTa KIBA drug embeddings

scripts/
└── finetune_head_ft.py  ← Phase 1i 스크립트
```
