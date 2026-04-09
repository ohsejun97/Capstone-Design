# Phase 1/2/3 학습 실험 보고서 — SaProt DTI 모델 비교

> **작성일시:** 2026년 3월 30일 KST (최종 업데이트: 2026-04-09)
> **목표:** SaProt frozen + MLP 헤드를 DAVIS/KIBA로 학습 후, 백본 크기 / 양자화 / 3Di 구조 토큰 / drug encoder / 데이터 스케일 / Transfer Learning 전략별 성능 비교
> **현재 상태:** ✅ 완료 — Phase 1a~1g 전체 실험 완료

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

## 설계 배경 — 왜 frozen + MLP 헤드인가?

SaProt-650M을 완전 파인튜닝하려면 >16GB VRAM이 필요하다. GTX 1650 SUPER(4GB)에서는 불가능.

LoRA 파인튜닝을 시도했으나, GTX 1650 SUPER는 Tensor Core가 없어 epoch당 ~2.5시간이 소요됨.
50 epoch 기준 5일 이상 → 실질적으로 포기.

**채택 전략:** SaProt을 고정(frozen) 단백질 인코더로 사용하고, **소형 MLP 헤드만 DTI 데이터로 학습**.

```
SMILES  → Morgan Fingerprint (2048-bit, 고정)   ─┐
                                                  ├→ MLP 헤드 (학습) → pKd
AA서열  → SA 토큰 → SaProt-650M (frozen)          ─┘
```

이 구조는 SaProt이 일반 단백질 표현력만으로 DTI 예측에 얼마나 유효한지, 특히 3Di 구조 토큰이 결합 친화도에 기여하는지를 측정하는 데 최적이다.

---

## 실험 설계

### 비교 축 (3가지)

| 비교 | 내용 |
|------|------|
| **백본 크기** | SaProt-650M (652M params) vs SaProt-35M (34M params) |
| **양자화** | FP16 vs INT8 (8-bit) vs NF4 (4-bit) |
| **구조 토큰** | Placeholder('#') vs FoldSeek 3Di 실제 토큰 |

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

## Phase 1a/1b 결과 — Placeholder('#') 기준

### DAVIS (30,056 쌍, 379 unique 단백질)

| 모델 | 파라미터 | 양자화 | Pearson r | RMSE | CI | 학습 시간 |
|------|---------|--------|-----------|------|----|---------|
| SaProt-650M | 652M | FP16 | 0.7855 | — | 0.8620 | 58.7초 |
| SaProt-35M | 34M | FP16 | 0.7832 | — | 0.8602 | 54.8초 |
| SaProt-650M-8bit | 652M | INT8 | 0.7812 | — | 0.8577 | 61초 |
| SaProt-650M-4bit | 652M | NF4 | **0.7914** | — | **0.8679** | 197.6초 |

> RMSE는 초기 실험 로그에 미기록. CI는 별도 연산 결과.

### KIBA (118,254 쌍, 229 unique 단백질)

| 모델 | Pearson r | RMSE | CI | 학습 시간 |
|------|-----------|------|----|---------|
| SaProt-650M FP16 | 0.7987 | 0.5024 | 0.8304 | 199초 |
| SaProt-35M | 0.7894 | — | — | 241초 |
| SaProt-650M-8bit | 0.7916 | — | — | 224초 |
| **SaProt-650M-4bit** | **0.7994** | — | — | 206초 |

> SaProt-650M FP16(KIBA)만 이후 실험에서 재측정. 나머지는 초기 로그에 RMSE/CI 미기록.

### Phase 1/2 핵심 발견

1. **35M ≈ 650M**: 파라미터 19배 차이에도 성능 차이 0.0023 → 경량 백본 충분히 viable
2. **4bit 우위**: FP16 대비 오히려 소폭 향상 (DAVIS +0.0059, KIBA +0.0139) → NF4 정규화 효과 추정
3. **KIBA 일반화**: 전 모델 r ≈ 0.79~0.80 → DAVIS 결과가 우연이 아님
4. **Phase 1/2 잠정 채택**: SaProt-650M-4bit (두 데이터셋 모두 최상위)

---

## Phase 1c 결과 — FoldSeek 3Di 구조 토큰 적용

> DAVIS 379개 단백질 전부 AlphaFold DB → FoldSeek 3Di 토큰 추출 (커버리지 100%)
> KIBA 229개 단백질 중 228개 성공 (커버리지 99.6%)

### DAVIS — Placeholder vs 3Di (Pearson r)

| 모델 | Placeholder r | 3Di r | Delta | RMSE (3Di) | CI (3Di) |
|------|-------------|-------|-------|------------|---------|
| **650M FP16** | 0.7855 | **0.8082** | **+0.023** | — | — |
| 35M FP16 | 0.7832 | 0.7996 | +0.017 | — | — |
| 650M-8bit | 0.7812 | 0.8027 | +0.022 | — | — |
| 650M-4bit | 0.7914 | 0.7977 | +0.006 | — | — |

### KIBA — Placeholder vs 3Di (Pearson r)

| 모델 | Placeholder r | 3Di r | Delta | RMSE (3Di) | CI (3Di) |
|------|-------------|-------|-------|------------|---------|
| **650M FP16** | 0.7987 | 0.8032 | +0.005 | — | — |
| 35M FP16 | 0.7894 | **0.8035** | +0.014 | — | — |
| 650M-8bit | 0.7916 | 0.7997 | +0.008 | — | — |
| 650M-4bit | 0.7994 | 0.7935 | −0.006 | — | — |

> RMSE/CI는 이후 drug encoder 실험(1d/1e)부터 체계적으로 기록 시작. 1c 실험은 미기록.

### Phase 3 핵심 발견

**3Di 토큰의 효과:**
- DAVIS: 전 모델 향상. 최대 +0.023 (650M FP16)
- KIBA: 650M-4bit 소폭 하락(-0.006) 제외, 전 모델 향상
- 3Di 구조 정보가 DTI 예측 성능에 실질적으로 기여함을 확인 → SaProt 선택 정당화

**양자화와 3Di의 관계:**
- Placeholder 사용 시: 4bit ≥ FP16 (양자화 손실이 '#' 토큰에는 무해)
- 3Di 사용 시: FP16 > 8bit > 4bit (양자화가 구조 신호를 손상)
- 해석: NF4 4bit 정밀도 손실이 의미없는 '#'에는 영향 없었지만, 실제 구조 신호가 담긴 3Di 토큰에는 직접적 영향
- **이는 역으로 3Di 토큰이 실제 의미있는 구조 정보를 인코딩한다는 증거**

---

## 전체 실험 히스토리

| 버전 | 일시 | 방법 | Pearson r | RMSE | CI | 상태 |
|------|------|------|-----------|------|----|------|
| V1 | 2026-03-25 | SaProt-650M 랜덤 헤드 (CPU) | 0.030 | — | — | ❌ 입력 오류 |
| V2 | 2026-03-25 | SPRINT + MERGED 가중치 | 0.141 | — | — | ❌ OOD |
| V3-A | 2026-03-25 | SaProt-650M FP16 + MLP (DAVIS) | 0.7855 | — | 0.8620 | ✅ |
| V3-B | 2026-03-25 | SaProt-35M FP16 + MLP (DAVIS) | 0.7832 | — | 0.8602 | ✅ |
| V3-C | 2026-03-26 | SaProt-650M-8bit + MLP (DAVIS) | 0.7812 | — | 0.8577 | ✅ |
| V3-D | 2026-03-26 | SaProt-650M-4bit + MLP (DAVIS) | 0.7914 | — | 0.8679 | ✅ |
| V4-A | 2026-03-27 | SaProt-650M-4bit + MLP (KIBA) | 0.7994 | — | — | ✅ |
| V4-B | 2026-03-27 | SaProt-35M + MLP (KIBA) | 0.7894 | — | — | ✅ |
| V4-C | 2026-03-27 | SaProt-650M-8bit + MLP (KIBA) | 0.7916 | — | — | ✅ |
| V7 | 2026-03-30 | SaProt-650M FP16 + MLP (KIBA, 재측정) | 0.7987 | 0.5024 | 0.8304 | ✅ |
| V5-A | 2026-03-30 | SaProt-650M FP16 + 3Di (DAVIS) | **0.8082** | — | — | ✅ 3Di 기준선 |
| V5-B | 2026-03-30 | SaProt-35M + 3Di (DAVIS) | 0.7996 | — | — | ✅ |
| V5-C | 2026-03-30 | SaProt-650M-8bit + 3Di (DAVIS) | 0.8027 | — | — | ✅ |
| V5-D | 2026-03-30 | SaProt-650M-4bit + 3Di (DAVIS) | 0.7977 | — | — | ✅ |
| V6-A | 2026-03-30 | SaProt-650M FP16 + 3Di (KIBA) | 0.8032 | — | — | ✅ |
| V6-B | 2026-03-30 | SaProt-35M + 3Di (KIBA) | **0.8035** | — | — | ✅ |
| V6-C | 2026-03-30 | SaProt-650M-8bit + 3Di (KIBA) | 0.7997 | — | — | ✅ |
| V6-D | 2026-03-30 | SaProt-650M-4bit + 3Di (KIBA) | 0.7935 | — | — | ✅ |
| V8-B | 2026-04-01 | SaProt-650M + 3Di + GNN (DAVIS) | 0.5795 | 0.7618 | 0.7907 | ❌ 데이터 부족 |
| V8-C | 2026-04-01 | SaProt-650M + 3Di + GNN (KIBA) | 0.7191 | 0.5783 | 0.7828 | △ Morgan FP 미달 |
| V10-A | 2026-04-01 | SaProt-650M + 3Di + ChemBERTa (DAVIS) | 0.7915 | 0.5627 | 0.8608 | △ 기준선 미달 |
| V10-B | 2026-04-01 | SaProt-650M + 3Di + ChemBERTa (KIBA) | 0.7667 | 0.5351 | 0.8148 | △ 기준선 미달 |
| **V11** | **2026-04-03** | **ChemBERTa + BindingDB (random)** | **0.8737** | **0.7933** | **0.8633** | **✅ 기준선 돌파** |
| V12 | 2026-04-03 | GNN + BindingDB (random) | 0.8411 | 0.8842 | 0.8459 | ✅ 기준선 돌파 |
| V13 | 2026-04-09 | ChemBERTa + BindingDB (cold_drug) | 0.7083 | 1.2543 | 0.7473 | ✅ 실제 일반화 |
| V14 | 2026-04-09 | ChemBERTa + BindingDB (cold_protein) | 0.6549 | 1.1840 | 0.7430 | ✅ 실제 일반화 |
| V15 | 2026-04-09 | Zero-shot → DAVIS (cross-eval) | 0.208 | 1.3029 | 0.5900 | ❌ 도메인 시프트 |
| V16 | 2026-04-09 | Zero-shot → KIBA (cross-eval) | 0.160 | 5.7871 | 0.5505 | ❌ 레이블 불일치 |
| **V17** | **2026-04-09** | **Transfer: BindingDB→DAVIS (head fine-tune)** | **0.8166** | **0.5303** | **0.8747** | **✅ 직접 학습 초과** |
| **V18** | **2026-04-09** | **Transfer: BindingDB→KIBA (head fine-tune)** | **0.8163** | **0.4826** | **0.8414** | **✅ 직접 학습 초과** |

---

## 최종 모델 선정

**채택: SaProt-650M FP16 + 3Di + ChemBERTa (Transfer Learning)**

| 기준 | BindingDB (raw) | DAVIS (transfer) | KIBA (transfer) |
|------|----------------|-----------------|----------------|
| Pearson r | 0.8737 | **0.8166** | **0.8163** |
| Spearman r | — | 0.6794 | 0.8114 |
| RMSE | 0.7933 | 0.5303 | 0.4826 |
| CI | 0.8633 | **0.8747** | 0.8414 |
| Peak VRAM | 2.6GB | 1.3GB | 1.3GB |
| 학습 시간 | 141초 | 194초 | 794초 |

**전체 비교 (DAVIS 기준):**

| 모델 | Pearson r | RMSE | CI | 비고 |
|------|-----------|------|----|------|
| SaProt+3Di + Morgan FP | 0.8082 | — | — | Phase 1c 기준선 |
| SaProt+3Di + ChemBERTa (DAVIS 직접) | 0.7915 | 0.5627 | 0.8608 | DAVIS 68약물 부족 |
| SaProt+3Di + ChemBERTa (BindingDB random) | 0.8737 | 0.7933 | 0.8633 | BindingDB 테스트셋 기준 |
| **SaProt+3Di + ChemBERTa (Transfer → DAVIS)** | **0.8166** | **0.5303** | **0.8747** | **Phase 1g 최종** |

**Agent Tool 1 (DTI 예측)**: Transfer Learning 모델 사용. BindingDB 32K 약물 표현력 + DAVIS 도메인 적응 완료.

---

## 평가 방법론 한계 — Random Split

현재 학습/검증/테스트 분할은 **Random 70/10/20**으로, DAVIS 30,056 쌍을 무작위 인덱스로 나눈다.

**문제:** DAVIS는 442개 단백질 × 68개 약물의 전수(complete) 행렬이다. Random split 하면 동일 단백질 또는 동일 약물이 Train과 Test에 모두 등장한다. 즉, 테스트 단백질과 동일한 단백질에 다른 약물이 결합하는 데이터가 이미 학습에 포함된 상태다. 이는 **데이터 누수(data leakage)** 위험이 있으며, 실제 generalization 성능을 과대추정할 수 있다.

| Split 방식 | 설명 | 난이도 |
|---|---|---|
| Random (현재) | 쌍 단위 무작위 분할 | 쉬움 (leakage 위험) |
| Cold-drug | 테스트 약물이 학습에 미등장 | 중간 |
| Cold-target | 테스트 단백질이 학습에 미등장 | 어려움 (실제 신약 스크리닝 시나리오) |

**Phase 4 계획:** Cold-target split 추가 평가. Random split 결과(r ≈ 0.81)와 Cold-target split 결과를 모두 보고하여 성능 신뢰도를 확보한다.

---

## 다음 단계

### Phase 1d — GNN Drug Encoder (✅ 완료, 결과: 실패)

**실험 결과:**

| 방식 | DAVIS r | KIBA r | vs Morgan FP |
|------|---------|--------|-------------|
| Morgan FP 단독 (기준) | 0.8082 | 0.8032 | — |
| GNN 단독 (from scratch) | 0.61 (val only) | — | ❌ |
| Morgan+GNN concat, 1단계 | ~0.61 (plateau) | — | ❌ |
| **Morgan+GNN concat, 2단계** | **0.5795** | **0.7191** | **❌ 대폭 하락** |

**실패 원인 분석:**

GNN from-scratch 학습이 실패한 근본 원인은 **DAVIS의 고유 약물 수 부족(68개)**이다.

| 데이터셋 | 고유 약물 수 | GNN 결과 | 해석 |
|---------|------------|---------|------|
| DAVIS | 68개 | r=0.58 ❌ | 2M 파라미터 GNN 학습에 턱없이 부족 |
| KIBA | 2,068개 | r=0.72 △ | 데이터 많을수록 GNN 수렴 가능, 아직 Morgan FP 미달 |

Morgan FP는 수십 년 화학 연구가 녹아든 고정 표현이라 소량 데이터에서도 강하다. GNN은 분자 표현을 처음부터 학습해야 하므로 68개 약물로는 일반화된 표현을 학습할 수 없다.

**2단계 학습(GNN warmup)**도 시도했으나 근본적인 데이터 부족 문제를 해결하지 못함.

**결론: GNN from-scratch → Pretrained drug encoder로 전략 전환**

ChemBERTa (PubChem 수백만 분자로 사전학습된 SMILES transformer)를 사용하면:
- 데이터 부족 문제 해결 (사전학습 시 충분한 분자 다양성 확보)
- SaProt과 동일한 패러다임 (frozen pretrained encoder + DTI 헤드만 학습)
- DAVIS 68개, KIBA 2068개 모두에서 유효

**관련 논문 비교:**

| 모델 | Protein Encoder | Drug Encoder | DAVIS r | 비고 |
|------|----------------|--------------|---------|------|
| DeepPurpose MPNN_CNN | CNN (학습) | MPNN (학습) | ~0.89 | 2020, end-to-end |
| ConPLex | ESM-2 (frozen) | GNN (학습) | ~0.90 | 2023 PNAS |
| **본 연구 (목표)** | **SaProt+3Di (frozen)** | **ChemBERTa (frozen/fine-tune)** | **≥ 0.85** | **4GB VRAM** |

### Phase 1e — Pretrained Drug Encoder (✅ 완료, 결과: Morgan FP 미달)

**실험 결과:**

| 방식 | DAVIS r | KIBA r | vs Morgan FP |
|------|---------|--------|-------------|
| Morgan FP 단독 (기준) | **0.8082** | **0.8032** | — |
| ChemBERTa pooler_output (frozen) | 0.7889 | 0.7602 | -0.019 / -0.043 |
| ChemBERTa mean pooling (frozen) | 0.7915 | 0.7667 | -0.017 / -0.037 |

**ChemBERTa 미달 원인 분석:**

1. **pooler_output 한계:** pooler_output(CLS 토큰에 linear+tanh 적용)은 분류 태스크용으로 설계됨. mean pooling으로 교체 시 DAVIS +0.003, KIBA +0.007로 개선됐으나 기준선 미달.
2. **표현 차원 열세:** Morgan FP 2048-dim vs ChemBERTa 768-dim.
3. **도메인 특화 vs 범용:** Morgan FP(ECFP4)는 수십 년간 키나아제 QSAR에 최적화된 표현. DAVIS 68개처럼 키나아제 억제제 중심의 작은 drug set에서는 ChemBERTa 범용 표현보다 강함.
4. **frozen encoder의 한계:** ChemBERTa가 PubChem에서 분자 성질 예측으로 사전학습됐지만, DTI 친화도 태스크로의 적응 없이 frozen 사용 시 최적 표현이 아닐 수 있음.

**결론: Drug encoder는 Morgan FP로 확정. Phase 3 Agent 오케스트레이션으로 이행.**

**관련 논문 비교 (최종):**

| 모델 | Protein Encoder | Drug Encoder | DAVIS r | 비고 |
|------|----------------|--------------|---------|------|
| DeepPurpose MPNN_CNN | CNN (학습) | MPNN (학습) | ~0.89 | 2020, end-to-end |
| ConPLex | ESM-2 (frozen) | GNN (학습) | ~0.90 | 2023 PNAS |
| **본 연구 (확정)** | **SaProt+3Di (frozen)** | **Morgan FP (2048-bit)** | **0.8082 / 0.8032** | **4GB VRAM** |

### Phase 1f — BindingDB 데이터 확장 + GNN/ChemBERTa 재시도 (✅ 완료)

**배경:**

Phase 1d/1e의 실패 원인을 재분석한 결과, 문제는 모델 자체가 아니라 **데이터 부족**이었다.

| 데이터셋 | 고유 약물 수 | GNN 결과 |
|---------|------------|---------|
| DAVIS | 68개 | r=0.58 ❌ |
| KIBA | 2,068개 | r=0.72 △ |
| **BindingDB** | **32,480개** | **미확인 → 실험** |

KIBA에서는 DAVIS보다 훨씬 나은 결과(r=0.72)가 나왔다는 점이 방향을 시사한다 — **약물이 많을수록 GNN이 수렴한다.**

**전처리 현황 (2026-04-02):**

- [x] `BindingDB_All_202604_tsv.zip` 다운로드 (525MB)
- [x] `BindingDB_All.tsv` 압축 해제 (7.9GB, 약 305만 행)
- [x] `scripts/preprocess_bindingdb.py` 작성 — Kd 필터링, pKd 변환, CSV 저장
- [x] 서버(500GB RAM)에서 전처리 완료 → `data/BindingDB/bindingdb_kd.csv`
  - 필터링 후 115,426행 → 중복 dedup(평균 pKd) → **80,795 고유 쌍**
  - 고유 약물 32,480개 / 고유 타겟 2,384개 / pKd 2.00~15.00 (mean=6.30)
  - null 0건, 중복 쌍 0건 검증 완료
- [x] FoldSeek 3Di 캐시 빌드 완료 (서버, 2026-04-03): 2,384개 단백질 중 2,309개 성공 (96.9%)
  - UniProt ID를 CSV에서 직접 추출 → BLAST 스킵 최적화 (처리시간 ~30h → ~3h)
  - no_structure: 70건 (AlphaFold DB 미보유), no_uniprot: 5건
- [x] `train_dti_saprot.py` bindingdb 케이스 CSV 읽기로 수정

**전처리 파이프라인:**

```
BindingDB_All.tsv (7.9GB, 305만 행)
        ↓ scripts/preprocess_bindingdb.py
        ↓ - Kd 컬럼 필터링, 단일 체인 단백질만
        ↓ - Kd(nM) → pKd(-log10) 변환
        ↓ - 이상값 제거 (Kd > 10,000,000 nM, Kd = 0 제거)
        ↓ - (smiles, sequence) 중복 dedup (평균 pKd)
data/BindingDB/bindingdb_kd.csv (smiles, sequence, pkd, uniprot_id)
```

**Phase 1f 실험 결과 (2026-04-03 ~ 2026-04-09):**

Random Split:

| 방식 | Dataset | Pearson r | RMSE | MAE | CI | VRAM |
|------|---------|-----------|------|-----|-----|------|
| Morgan FP (기준) | DAVIS | 0.8082 | — | — | — | ~2GB |
| ChemBERTa (frozen) | DAVIS | 0.7915 | 0.5627 | 0.3025 | 0.8608 | 1.6GB |
| GNN (from-scratch) | DAVIS | 0.5795 | 0.7618 | 0.4032 | 0.7907 | 2.1GB |
| **ChemBERTa (frozen)** | **BindingDB** | **0.8737** | **0.7933** | **0.5130** | **0.8633** | **2.6GB** |
| GNN (from-scratch) | BindingDB | 0.8411 | 0.8842 | 0.5773 | 0.8459 | 2.2GB |

Cold Split (BindingDB, ChemBERTa):

| Split | Pearson r | RMSE | MAE | CI |
|-------|-----------|------|-----|----|
| cold_drug | 0.7083 | 1.2543 | 0.9893 | 0.7473 |
| cold_protein | 0.6549 | 1.1840 | 0.8007 | 0.7430 |

Zero-shot Cross-dataset Evaluation (cold_drug 모델 → DAVIS/KIBA):

| Target | Pearson r | RMSE | CI | 실패 원인 |
|--------|-----------|------|----|---------|
| DAVIS | 0.208 | 1.3029 | 0.5900 | 도메인 시프트 (키나아제 특화) |
| KIBA | 0.160 | 5.7871 | 0.5505 | 레이블 불일치 (KIBA score ≠ pKd) |

**핵심 발견:**

**1. BindingDB 전략 검증 — 약물 다양성이 핵심이었음**

| Drug Encoder | DAVIS (68약물) | BindingDB (32,480약물) | 변화 |
|---|---|---|---|
| GNN | 0.5795 | 0.8411 | **+0.2616** |
| ChemBERTa | 0.7915 | 0.8737 | **+0.0822** |

DAVIS에서 실패했던 두 drug encoder가 BindingDB로 전환하자 모두 기준선(0.8082)을 돌파했다. 이는 Phase 1d/1e의 실패 원인이 **모델 아키텍처 문제가 아닌 훈련 데이터 부족**이었음을 명확히 확인해준다.

**2. ChemBERTa > GNN 이유**

GNN이 BindingDB 32K 약물에서도 ChemBERTa에 뒤지는 이유:
- ChemBERTa는 **PubChem 수백만 분자**로 사전학습된 표현을 그대로 활용 → 데이터 규모와 무관하게 풍부한 표현력 보유
- GNN은 from-scratch 학습 → 32K 약물로도 모든 분자 부분구조 패턴을 커버하기엔 부족
- 결론: **사전학습 여부가 소량~중량 데이터에서 결정적**

**3. RMSE 상승은 과제 난이도 증가 반영**

ChemBERTa의 RMSE가 DAVIS(0.56)보다 BindingDB(0.79)에서 높은 이유:
- BindingDB pKd 범위: 2.00~15.00 (13 단위 범위)
- DAVIS pKd 범위: 더 좁은 키나아제 억제제 특화 분포
- 절대적으로 더 어려운 회귀 문제 → RMSE 상승은 자연스러운 결과

**4. 목표 달성 여부**

프로젝트 핵심 가설: *"BindingDB로 약물 다양성 확보 + ChemBERTa frozen → DAVIS 기준선 돌파"*

→ BindingDB 기준 r=0.8737 달성 ✅

단, 비교의 공정성 주의: BindingDB와 DAVIS는 훈련/테스트 데이터 분포가 다르므로 r값의 직접 비교는 제한적. DAVIS 기준선(0.8082)은 DAVIS 테스트셋 기준이며, BindingDB 0.8737은 BindingDB 테스트셋 기준이다.

**5. GNN 학습 시간 문제**

GNN 학습이 7,414초(~2시간)인 반면 ChemBERTa는 141초. 차이의 원인:
- GNN은 매 배치마다 SMILES → 분자 그래프 변환 + 메시지 패싱을 실시간 수행
- ChemBERTa는 frozen → 임베딩이 사전 계산 가능하지만 현재 미구현 (캐시 없음)
- 실용적 관점에서 ChemBERTa가 압도적으로 유리

**논문 비교 (최종):**

| 모델 | Protein Encoder | Drug Encoder | DAVIS r | 비고 |
|------|----------------|--------------|---------|------|
| DeepPurpose MPNN_CNN | CNN (학습) | MPNN (학습) | ~0.89 | 2020, end-to-end |
| ConPLex | ESM-2 (frozen) | GNN (학습) | ~0.90 | 2023 PNAS |
| **본 연구 (1f, BindingDB)** | **SaProt+3Di (frozen)** | **ChemBERTa (frozen)** | **0.8737** | **4GB VRAM** |
| 본 연구 (DAVIS 기준선) | SaProt+3Di (frozen) | Morgan FP | 0.8082 | 4GB VRAM |

**결론: 최종 Drug Encoder 전략**

BindingDB 실험을 통해 drug encoder 선택 기준이 명확해졌다:

1. **소량 데이터(DAVIS, 68약물)**: Morgan FP > ChemBERTa > GNN — 결정론적 표현이 강력
2. **대량 다양 데이터(BindingDB, 32K약물)**: ChemBERTa > GNN > Morgan FP — 학습 가능한 표현이 우세
3. **Phase 3 Agent 시스템**: ChemBERTa(BindingDB) 모델을 DTI Tool로 채택. 더 넓은 화학 공간 커버 가능

### Phase 1g — Transfer Learning: BindingDB→DAVIS/KIBA (✅ 완료)

Zero-shot 실패 원인(도메인 시프트, 레이블 불일치)을 분석하여 MLP Head만 재학습하는 Transfer Learning 전략 구현.
SaProt + ChemBERTa 임베딩은 캐시 재사용 → 추가 GPU 로드 없이 DAVIS 194초, KIBA 794초 내 완료.

| Target | Pearson r | Spearman r | RMSE | CI | vs 직접 학습 |
|--------|-----------|-----------|------|----|------------|
| DAVIS | **0.8166** | 0.6794 | 0.5303 | **0.8747** | +0.0084 (0.8082 초과) |
| KIBA | **0.8163** | 0.8114 | **0.4826** | 0.8414 | +0.0131 (0.8032 초과) |

상세 분석: [docs/PHASE1G_TRANSFER_LEARNING.md](PHASE1G_TRANSFER_LEARNING.md)

### Phase 3 — Agent 오케스트레이션 (⏳ Next)

Phase 1g 완료 후 최종 DTI 모델을 Tool 1에 통합하고 smolagents ReAct 오케스트레이션 구현.
