# Phase 1 실험 일지 — Reference Score 생성

> **작성일시:** 2026년 3월 25일 (수) 21:10 KST
> **목표:** SaProt-650M 기반 DAVIS DTI Reference Score 산출 (Pearson r ≥ 0.8)
> **현재 상태:** 🔄 진행 중 — V2 완료, r = 0.141, 가중치 전략 재검토 필요

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
| 데이터셋 | DAVIS test set — 6,011 샘플 |

---

## 실험 V1 — SaProt 단독 추론

**일시:** 2026년 3월 25일 (수) (정확한 시각 미기록)

### 설정

| 항목 | 값 |
|------|-----|
| 모델 | `westlake-repl/SaProt_650M_AF2` |
| 모델 클래스 | `AutoModelForSequenceClassification(num_labels=1)` |
| 약물 입력 | `tokenizer(protein, SMILES)` — 시퀀스 쌍 직접 입력 |
| 실행 환경 | CPU (32 threads) |
| 소요 시간 | ~17시간 |

### 결과

| 지표 | 값 |
|------|-----|
| **Pearson R** | **0.030** |
| 처리 샘플 | 6,011개 |
| 오류 | 0개 |

### 실패 원인 분석

1. **잘못된 모델 클래스:** `AutoModelForSequenceClassification`은 SaProt 위에 **랜덤 초기화된 선형 레이어**를 추가한다. 이 헤드는 DTI 데이터로 전혀 학습되지 않아 출력이 완전한 노이즈다.

2. **잘못된 입력 포맷:** `tokenizer(protein, SMILES)`로 SMILES를 두 번째 시퀀스로 입력했다. SaProt 어휘집(446 SA 토큰)은 SMILES 문자를 알지 못하므로 입력이 깨진다.

3. **DTI 파인튜닝 없는 PLM 단독 사용:** `SaProt_650M_AF2`는 Masked Language Modeling으로 사전학습된 단백질 언어 모델이다. 결합 친화도 예측을 위해서는 별도의 약물 인코더 + 융합 레이어가 필수다.

---

## 실험 V2 — SPRINT 아키텍처 + panspecies-dti 가중치

**일시:** 2026년 3월 25일 (수) 20:04 ~ 21:09 KST
**소요 시간:** 1시간 5분 44초

### 아키텍처 (panspecies-dti SPRINT, 체크포인트 역설계)

```
단백질 경로:
  SA 시퀀스 (davis_test.csv Target Sequence)
    → EsmTokenizer (446-token SA 어휘)
    → SaProt-650M (EsmModel, frozen, 33-layer Transformer)
    → last_hidden_state[:, 1:-1, :] — CLS/EOS 제외  → [seq_len, 1280]
    → MultiHeadAttentionPool (4-head, learnable CLS token)  → [1280]
    → Sequential: LN + LeakyReLU + Linear(1280→1024) × 3 + LN  → [1024]
    → L2 normalize

약물 경로:
  SMILES
    → RDKit MorganFP (radius=2, nBits=2048)  → [2048]
    → Linear(2048→1260) + BN + LeakyReLU
    → Linear(1260→1024) + BN + LeakyReLU
    → Linear(1024→1024) + BN + LeakyReLU
    → Linear(1024→1024)  → [1024]
    → L2 normalize

융합:
  코사인 유사도 (NoSigmoid) = dot product (L2 정규화 완료)
```

### 가중치

| 항목 | 값 |
|------|-----|
| 출처 | [panspecies-dti (SPRINT)](https://github.com/abhinadduri/panspecies-dti) |
| 파일 | `sprint_weights.ckpt` (192MB) |
| 학습 데이터 | MERGED (BIOSNAP + BindingDB + Human) |
| 학습 태스크 | 이진 분류 (CE loss) |
| Epoch | 12 / 15 |
| Global step | 693,979 |

### 가중치 로딩 결과

```
target_proj (단백질 헤드): 20/20 키 완전 매칭
drug_proj   (약물 헤드):   23/23 키 완전 매칭
```

### 설치 이슈 및 해결 과정

| 이슈 | 원인 | 해결 |
|------|------|------|
| `ModuleNotFoundError: No module named 'omegaconf'` | SPRINT 체크포인트 로드 시 omegaconf 필요 | `pip install omegaconf pytorch-lightning` |
| `Due to a serious vulnerability issue in torch.load` (CVE-2025-32434) | transformers 5.x가 torch < 2.6 거부 | `pip install torch==2.6.0 --index-url ...cu124` |
| `RuntimeError: operator torchvision::nms does not exist` | torch 2.6 + torchvision 2.5 버전 불일치 | `pip uninstall torchvision torchaudio` (DTI에 불필요) |

### 결과

| 지표 | 값 |
|------|-----|
| **Pearson R** | **+0.1412** |
| p-value | 3.73e-28 (통계적으로 유의미) |
| 처리 샘플 | 6,011개 |
| 오류 샘플 | 0개 |
| 소요 시간 | 65분 44초 (GPU, GTX 1650) |

### 예측 스코어 분포

| 통계 | 값 |
|------|-----|
| 평균 | 0.336 |
| 표준편차 | 0.197 |
| 최솟값 | -0.247 |
| 중앙값 | 0.368 |
| 최댓값 | 0.677 |

### 레이블 분포

| Label | 샘플 수 |
|-------|--------|
| 0 (비결합) | 5,708 (94.96%) |
| 1 (결합)   | 303 (5.04%) |

### V1 → V2 비교

| 지표 | V1 | V2 | 개선 |
|------|----|----|------|
| Pearson R | 0.030 | 0.141 | +0.111 (+370%) |
| 오류 | 0개 | 0개 | — |
| 아키텍처 | 랜덤 헤드 | SPRINT (DTI 파인튜닝) | ✅ |

---

## V2 실패 원인 분석 (Pearson r = 0.141, 목표 미달)

### 원인 1: Out-of-Distribution 가중치

SPRINT 체크포인트는 **MERGED 데이터셋** (BIOSNAP + BindingDB + Human)으로 학습됐다. DAVIS는 포함되지 않았으며 도메인 특성이 다르다.

| 데이터셋 | 특성 | SPRINT 학습 여부 |
|---------|------|----------------|
| MERGED (BIOSNAP+BindingDB+Human) | 이진 분류, 다양한 단백질 패밀리 | ✅ |
| DAVIS | pKd 연속값 회귀, 키나아제 중심 | ❌ |

### 원인 2: 학습 태스크 불일치

- SPRINT: **이진 분류** (CE loss, classify=True)
- DAVIS 목표: **연속값 회귀** (Pearson r 기준)
- 이진 분류로 학습된 모델의 cosine similarity가 연속 pKd 값과 상관관계가 낮음

### 원인 3: 클래스 불균형

DAVIS 데이터의 레이블 분포가 **극도로 불균형** (결합: 5%, 비결합: 95%).
이진 분류 모델은 비결합 샘플에 편향되어 학습됨.

---

## 다음 단계 전략

### 전략 A: `SaProt_650M_AF2_DTI_Davis` 접근 (최우선)

Westlake 연구팀의 DAVIS 전용 파인튜닝 모델. 현재 **비공개 저장소**로 HuggingFace 공개 목록에 없음.

```
접근 방법:
1. westlake-repl GitHub: https://github.com/westlake-repl/SaProt
   → Issues 또는 Discussion에 데이터 접근 요청
2. 논문 저자 이메일 문의
3. HuggingFace: https://huggingface.co/westlake-repl
   → 계정 로그인 후 모델 페이지에서 접근 요청 가능 여부 확인
```

예상 Pearson r: **~0.85+**

### 전략 B: SPRINT을 DAVIS로 파인튜닝

panspecies-dti의 훈련 파이프라인을 활용해 DAVIS train 데이터로 직접 파인튜닝.

```bash
# davis_train.csv 필요 (panspecies-dti에서 wget)
# panspecies-dti configs/saprot_agg_config.yaml 수정 후:
python ultrafast/train.py --config configs/saprot_davis_config.yaml
```

예상 Pearson r: **~0.80+**
필요 자원: GTX 1650 (4GB) — 배치 크기 조정 필요

### 전략 C: DeepPurpose DAVIS 즉시 기준값 확보

가장 빠른 방법. SaProt 기반은 아니지만 r ≥ 0.8 상한선 검증에 활용 가능.

```python
from DeepPurpose import DTI as models
net = models.model_pretrained(model='MPNN_CNN_DAVIS')
# DAVIS Pearson r ≈ 0.88
```

예상 Pearson r: **~0.88**

---

## 종합 실험 히스토리

| 버전 | 일시 | 모델 | Pearson R | 상태 |
|------|------|------|----------|------|
| V1 | 2026-03-25 (오전) | SaProt_650M (랜덤 헤드, CPU) | 0.030 | ❌ 실패 |
| V2 | 2026-03-25 20:04~21:09 | SaProt_650M + SPRINT 가중치 (GPU) | 0.141 | ⚠️ 부분 성공 |
| V3 | 예정 | SaProt_650M_DTI_Davis 또는 DAVIS 파인튜닝 | 0.8+ | 🔄 계획 중 |

---

## 파일 목록

```
Capstone_Design/
├── run_reference.py          # V2 DTI 추론 파이프라인
├── davis_test.csv            # DAVIS 테스트셋 (6,011 샘플)
├── reference_scores_osj.csv  # V2 예측 결과 (Pearson r=0.141)
├── sprint_weights.ckpt       # panspecies-dti SPRINT 가중치 (192MB)
├── requirements.txt          # 의존성 (torch>=2.6, torchvision 제외)
├── setup_env.sh              # 환경 설치 스크립트
└── docs/
    ├── PHASE1_REFERENCE_PIPELINE.md   # 파이프라인 설계 문서
    └── PHASE1_EXPERIMENT_LOG.md       # 본 실험 일지
```
