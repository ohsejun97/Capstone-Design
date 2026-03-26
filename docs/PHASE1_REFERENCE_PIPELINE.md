# Phase 1 파이프라인 설계 문서

> **목표:** SaProt 기반 DTI 파이프라인으로 DAVIS Reference Score 확보 (Pearson r ≥ 0.8)
> **상태:** 🔄 V3 완료 (r = 0.7855) — LoRA 파인튜닝으로 r ≥ 0.9 도전 예정

---

## 1. 왜 DeepPurpose를 그냥 쓰지 않나?

DeepPurpose는 `MPNN_CNN_DAVIS` 사전학습 모델로 즉시 r ≈ 0.88을 달성한다.
그런데 본 프로젝트가 DeepPurpose를 직접 쓰지 않는 이유:

| 항목 | DeepPurpose | 본 연구 (SaProt 기반) |
|------|------------|---------------------|
| 단백질 인코더 | CNN / AAC (서열만 사용) | SaProt (서열 + **3D 구조** 정보) |
| 약물 인코더 | MPNN / Morgan / CNN | Morgan FP (동일) |
| 학습 방식 | end-to-end (전체 재학습) | frozen 인코더 + 소형 헤드 학습 |
| VRAM 요구 | ~2GB (CNN 기반) | ~2GB frozen, 4bit 시 ~0.8GB |
| 경량화 실험 | 불가 (구조 고정) | ✅ 35M / 4bit 변형 비교 가능 |
| 연구 기여 | 없음 (기존 baseline) | 구조 인식 PLM의 경량화 가능성 검증 |

**핵심:** DeepPurpose는 데이터 로더로만 활용. 단백질 표현 방식 자체를 SaProt으로 교체하여 연구 기여도를 만든다.

---

## 2. 전체 파이프라인

### 2.1 데이터 흐름

```
[DeepPurpose DAVIS]
  load_process_DAVIS(binary=False, convert_to_log=True)
  → 30,056 쌍 (SMILES, AA서열, pKd)
  → Train 70% / Val 10% / Test 20%

[단백질 경로]
  AA서열
  → SA 포맷 변환: "MEVK..." → "M#E#V#K#..."
  → SaProt 토크나이저 (446 SA 어휘)
  → SaProt Transformer (frozen, 33레이어)
  → last_hidden_state[:, 1:-1, :] (CLS/EOS 제외)
  → mean pool → [prot_dim]
  → cache 저장 (379개 unique 단백질, 재사용)

[약물 경로]
  SMILES
  → RDKit MorganFP (radius=2, nBits=2048)
  → [2048]

[DTI 헤드 (학습 대상)]
  prot_enc: [prot_dim → 512 → 256]
  drug_enc: [2048 → 512 → 256]
  regressor: [512 → 256 → 64 → 1]  ← pKd 예측값

  Loss: HuberLoss(delta=1.0)
  Optimizer: Adam(lr=1e-3, wd=1e-4)
  Scheduler: CosineAnnealingLR
  Early stopping: patience=10
```

### 2.2 SA 토큰 포맷

SaProt의 입력은 일반 아미노산 서열이 아닌 **SA(Structural Aware) 토큰**:

```python
# 구조 정보 없을 때 (DeepPurpose DAVIS 원시 서열)
sa_seq = "".join(aa + "#" for aa in aa_seq)
# "MEVK" → "M#E#V#K#"

# 구조 정보 있을 때 (AlphaFold + FoldSeek 3Di)
sa_seq = "".join(aa.upper() + di.lower() for aa, di in zip(aa_seq, foldseek_seq))
# "MEVK" + "adcp" → "MaEdVcKp"
```

현재는 DeepPurpose DAVIS의 원시 AA 서열을 사용하므로 `#` 대체 방식 적용.
AlphaFold 구조를 추가하면 SaProt 성능이 더 올라갈 여지 있음.

---

## 3. 실험 결과 (V3)

| 모델 | Test r | Val Best r | 학습 시간 | 파라미터 |
|------|--------|-----------|---------|---------|
| SaProt-650M frozen | 0.7855 | 0.7990 | 58.7초 | 652M frozen + 2.4M head |
| SaProt-35M frozen | 0.7832 | 0.7872 | 54.8초 | 327M frozen + 1.1M head |
| SaProt-650M 4bit | — | — | — | 재실행 예정 |

**핵심 발견:** 35M이 650M 대비 파라미터 18배 적지만 성능 차이 **0.0023** → 단백질 인코더 크기가 결정적이지 않음을 시사

---

## 4. 현재 한계와 r ≥ 0.9 달성 전략

### 현재 한계

`frozen SaProt + MLP 헤드` 방식의 근본적 문제:
- SaProt은 단백질 Masked LM으로 사전학습됨 → **약물-단백질 상호작용 정보 없음**
- frozen 상태라 DAVIS 데이터셋 특성에 적응 불가
- 헤드가 고정된 임베딩에서 DTI 정보를 찾아야 하는 구조적 제약

### 전략별 비교

| 전략 | 예상 r | VRAM 추가 | 복잡도 | 추천 |
|------|--------|----------|--------|------|
| 에폭 확대 + LR 튜닝 | 0.80~0.82 | 없음 | 낮음 | 빠른 확인용 |
| 마지막 레이어 unfreeze | 0.83~0.88 | ~500MB | 중간 | — |
| **LoRA 파인튜닝** | **0.87~0.92** | **~200MB** | **중간** | **✅ 권장** |
| 전체 파인튜닝 | 0.90+ | >4GB | 높음 | 4GB에서 불가 |

### LoRA 전략 (권장)

```
기존: SaProt 가중치 W (완전 고정)

LoRA: W' = W + ΔW = W + B·A
  - A: [d × rank], B: [rank × d]  (rank=16)
  - 추가 파라미터: ~2M개 (전체의 0.3%)
  - 학습: A, B + DTI 헤드만
  - VRAM: +~200MB

기대 효과:
  SaProt이 DAVIS의 약물-단백질 상호작용 패턴에 적응
  → frozen 대비 Pearson r +0.05~0.10 기대
  → 35M + LoRA가 frozen 650M을 초과할 수 있음 (핵심 연구 포인트)
```

---

## 5. 단계별 로드맵

```
[현재] Phase 1 진행 중
  ├── V3 frozen 실험 완료 (650M: 0.7855, 35M: 0.7832)
  ├── 4bit 재실행 (add_pooling_layer=False 패치)
  └── LoRA 실험 → r ≥ 0.8 달성 후 Phase 1 완료

[Phase 2] 경량화 트레이드오프 분석
  ├── LoRA-650M vs LoRA-35M vs LoRA-650M-4bit 비교
  ├── VRAM 사용량, 추론 속도, Pearson r 종합 분석
  └── "35M + LoRA ≥ 650M frozen" 가설 검증

[Phase 3] 에이전트 연동
  ├── smolagents + Gemini API 툴 패키징
  └── AlphaFold DB 연동 → 실제 SA 토큰 생성

[Phase 4~5] 배포
  └── Streamlit + FastAPI + Docker
```

---

## 6. 참조

| 레포 | 역할 |
|------|------|
| [westlake-repl/SaProt](https://github.com/westlake-repl/SaProt) | SaProt 모델, SA 토큰 포맷 |
| [abhinadduri/panspecies-dti](https://github.com/abhinadduri/panspecies-dti) | SPRINT 아키텍처, davis_test.csv |
| [kexinhuang12345/DeepPurpose](https://github.com/kexinhuang12345/DeepPurpose) | DAVIS 데이터 로더, 기준선 비교 |
| [ZhaohanM/FusionDTI](https://github.com/ZhaohanM/FusionDTI) | Cross-Attention 융합 참고 |
