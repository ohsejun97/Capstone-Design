# Phase 1h — ChemBERTa Fine-tuning: BindingDB Random Split

> **작성일시:** 2026-05-06
> **목표:** ChemBERTa 마지막 2개 레이어를 DTI 태스크에 맞게 fine-tune하여 frozen 기준선(r=0.8737)을 초과하는지 확인

---

## 배경 및 동기

### Phase 1f 한계 — Frozen Encoder 표현력 상한

Phase 1f에서 ChemBERTa를 완전 frozen으로 사용했을 때 BindingDB random split r=0.8737을 달성했다. 이는 PubChem에서 분자 성질 예측으로 사전학습된 범용 표현을 그대로 활용한 결과다.

그러나 **DTI 결합 친화도(pKd) 예측**은 사전학습 목적과 다소 다른 태스크다. Frozen encoder는 pKd 예측에 최적화된 화학 표현을 생성하지 못할 가능성이 있다.

### 가설

ChemBERTa 상위 레이어를 DTI 태스크에 맞게 fine-tune하면:
- 약물의 결합 친화도와 연관된 화학적 특징(극성기, 소수성 부위, 입체 구조)에 attention이 집중됨
- Mean pooling된 표현이 pKd와 더 강하게 연관된 방향으로 이동
- Frozen 기준선(r=0.8737)을 초과할 것으로 기대

### 왜 BindingDB Random Split인가?

Bio-AI Agent의 실제 사용 시나리오는 알려진 약물-단백질 쌍에 대한 질의다. 새로운 미지 약물 스크리닝(cold split 시나리오)보다 넓은 화학 공간에서의 일반적인 결합 친화도 예측이 주된 목적이다. 따라서 random split r이 에이전트의 실질 성능에 더 가까운 지표다.

---

## 방법론

### 아키텍처

```
SMILES → [ChemBERTa layers 0~3: frozen]
              → [ChemBERTa layers 4~5: fine-tune, lr=1e-5]  ← 6레이어 모델, 마지막 2개
              → Mean Pooling → [768-dim]
                                             ─┐
                                              ├→ MLP Head → pKd
Sequence → 3Di tokens → SaProt-650M (frozen) ─┘
              ↑ 임베딩 캐시 재사용 (prot_embs_bindingdb_650M_none_3di.pt)
```

**핵심 설계 원칙:**
- SaProt은 완전 frozen + 사전 계산된 임베딩 캐시 재사용 (VRAM 비사용)
- ChemBERTa는 on-the-fly 인코딩 (캐싱 불가 — 가중치가 변하므로)
- 차등 학습률: MLP Head(5e-4) vs ChemBERTa fine-tune layers(1e-5, 50배 작게)

### Fine-tuning 설정

| 항목 | 값 |
|------|-----|
| Base model | seyonec/ChemBERTa-zinc-base-v1 |
| Frozen layers | 0~3 (4개) |
| Trainable layers | 4~5 + pooler (ChemBERTa-zinc-base = 6 layers) |
| Trainable params | 14.77M (전체 44.1M 중) |
| DTI Head params | 1.46M |
| Data split | BindingDB random (70/10/20) |
| Batch size | 32 (GTX 1650 SUPER 4GB VRAM) |
| LR (Head) | 5e-4 |
| LR (ChemBERTa) | 1e-5 |
| Optimizer | AdamW (weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR (T_max=50) |
| Loss | HuberLoss (delta=1.0) |
| Early stopping | patience=10 |
| Max epochs | 50 |
| Grad clip | 1.0 |

### DTI Head 구조 (Phase 1f와 동일)

```
prot_enc:  Linear(1280→512) → LayerNorm → GELU → Linear(512→256) → GELU
drug_enc:  Linear(768→512)  → BatchNorm → GELU → Linear(512→256) → GELU
regressor: Linear(512→256)  → GELU → Dropout(0.1) → Linear(256→64) → GELU → Linear(64→1)
```

### 실행 환경

| 항목 | 값 |
|------|-----|
| GPU | NVIDIA GeForce GTX 1650 SUPER (4GB VRAM) |
| VRAM 사용 (실측) | 884.8 MB (peak) |
| 학습 데이터 | BindingDB 80,795 쌍 (train: 56,556) |

### 실행 명령

```bash
python scripts/train_chemberta_unfreeze.py \
    --unfreeze 2 \
    --split random \
    --batch_size 32 \
    --epochs 50 \
    --lr_head 5e-4 \
    --lr_cb 1e-5 \
    --patience 10
```

---

## 실험 결과

> **실험 일시:** 2026-05-06 완료 (23:07 KST)

### 기준선 비교

| 방식 | Pearson r | RMSE | CI | 학습 시간 |
|------|-----------|------|----|---------|
| ChemBERTa frozen (Phase 1f) | 0.8737 | 0.7933 | 0.8633 | 141s |
| **ChemBERTa fine-tune (Phase 1h)** | **0.8923** | **0.7387** | **0.8770** | **24,824s (413.7분)** |
| **개선량** | **+0.0186** | **-0.0546** | **+0.0137** | — |

### 세부 지표

| 지표 | 값 |
|------|-----|
| Pearson r | **0.8923** |
| p-value | 0.00e+00 (통계적으로 완전 유의) |
| Spearman r | 0.8722 |
| R² | 0.7924 |
| RMSE | **0.7387** |
| MAE | 0.4617 |
| CI | 0.8770 |
| Best val r | 0.8893 |
| Epochs trained | 50 (max epoch 도달, patience 미소진) |
| 학습 시간 | 24,824s (413.7분, ~6.9시간) |
| Peak VRAM | 884.8 MB |

---

## 분석

### 1. Fine-tuning 효과 — 가설 검증

| 지표 | Frozen (1f) | Fine-tune (1h) | 개선 |
|------|------------|----------------|------|
| Pearson r | 0.8737 | **0.8923** | **+0.0186** |
| Spearman r | — | 0.8722 | — |
| RMSE | 0.7933 | **0.7387** | **-0.0546** |
| MAE | 0.5130 | **0.4617** | **-0.0513** |
| CI | 0.8633 | **0.8770** | **+0.0137** |

ChemBERTa 상위 2개 레이어를 DTI pKd 예측 태스크에 맞게 fine-tune하자 모든 지표에서 개선이 확인됐다. Pearson r +0.0186, RMSE -0.0546으로 가설이 실험적으로 검증됐다.

### 2. 수렴 패턴 분석

- **epoch 1~15**: 빠른 상승 (r: 0.745 → 0.865, +0.12)
- **epoch 15~25**: 중간 속도 상승 (r: 0.865 → 0.878, +0.013)
- **epoch 25~45**: 느린 상승 (r: 0.878 → 0.889, +0.011)
- **epoch 45~50**: 수렴 (r: ~0.889 유지)

50 epoch을 모두 소진하여 early stopping이 발동하지 않았다. patience(10)를 채우기 전에 다시 갱신되는 패턴이 반복됐다.

### 3. 과적합 여부

Best val r=0.8893 vs test r=0.8923 — test가 val보다 오히려 높다. 과적합 징후 없음. BindingDB 80K 쌍 규모에서 patience=10 설정이 충분히 작동했다.

### 4. VRAM 효율

Peak VRAM 884.8 MB — GTX 1650 SUPER (4GB) 기준 22% 사용. max_length=128 설정(초기 512에서 변경)이 VRAM을 절반 이하로 줄이는 데 결정적 역할을 했다 (초기 시도 시 3,804 MB → 884 MB).

### 5. 학습 시간

413.7분(~7시간)은 frozen(141s) 대비 177배 긴 시간이다. ChemBERTa 레이어 unfreeze로 인해 drug embedding을 매 배치마다 재계산해야 하기 때문이다. 성능 향상(+0.0186) 대비 시간 비용을 감안해야 한다.

---

## 결론

**ChemBERTa fine-tuning은 효과적이다.** Frozen 기준선(r=0.8737) 대비 +0.0186 향상으로 **r=0.8923, RMSE=0.7387**을 달성했다. 모든 지표에서 개선이 확인됐고 과적합 없이 수렴했다.

**Bio-AI Agent 관점에서의 의미:**
- r=0.8923은 알려진 약물-단백질 쌍에 대한 높은 예측 정확도
- CI=0.877은 두 약물의 결합력 순위를 87.7% 정확도로 비교 가능
- SOTA(DeepPurpose r≈0.89) 수준 달성

**다음 단계:**
1. fine-tuned ChemBERTa 가중치로 DAVIS/KIBA drug embedding 재계산
2. MLP Head만 각 데이터셋에 fine-tune (Phase 1g 방식 반복)
3. frozen ChemBERTa Transfer Learning(r=0.8166/0.8163)을 초과하는지 확인

---

## 출력 파일

```
results/SaProt-650M-bindingdb-3di-chemberta-unfreeze2-random/
├── dti_head.pt          ← 최적 DTI Head 가중치
├── chemberta_ft.pt      ← fine-tuned ChemBERTa layers 10~11 가중치
├── training_history.csv ← epoch별 loss/val_r/RMSE/CI
└── result.json          ← 전체 지표 JSON

logs/
└── chemberta_unfreeze2_random.log
```

## 스크립트

```
scripts/train_chemberta_unfreeze.py
```

- `--unfreeze N`: 마지막 N개 레이어 fine-tune (default=2)
- `--split`: random / cold_drug / cold_protein
- `--lr_head`: MLP Head 학습률
- `--lr_cb`: ChemBERTa fine-tune 학습률 (헤드보다 50배 작게 유지)
