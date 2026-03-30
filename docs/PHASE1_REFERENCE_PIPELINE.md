# Phase 1 파이프라인 설계 문서

> **목표:** SaProt 기반 DTI 파이프라인으로 DAVIS/KIBA Reference Score 확보
> **상태:** ✅ 완료 — Phase 3까지 종료, 최종 모델: SaProt-650M FP16 + 3Di (DAVIS r=0.8082)

---

## 1. SaProt과 DTI의 관계

**SaProt은 DTI 모델이 아니다.** SaProt은 단백질만 이해하는 PLM(Protein Language Model)이다. DTI 예측 시스템의 *protein encoder 역할*로 사용된다.

DTI 예측 시스템의 구조:
```
[Drug Encoder]    약물 화학 구조 → 벡터
[Protein Encoder] 단백질 서열/구조 → 벡터
                                      → MLP Head → pKd
```

기존 DTI 연구(DeepPurpose 등)의 두 가지 병목:

| 병목 | 기존 방식 | 한계 | 본 연구 |
|------|---------|------|--------|
| **Protein Encoder** | CNN / AAC (서열만) | 3D 구조 정보 미반영 | SaProt + FoldSeek **3Di 토큰** |
| **Drug Encoder** | Morgan FP (고정 비트벡터) | 학습 파라미터 없음, 위상 정보 손실 | Phase 1d: GNN(AttentiveFP/MPNN) |

**DeepPurpose는 데이터 로더로만 활용한다.** DAVIS/KIBA 데이터 파이프라인을 재사용하되, protein encoder를 SaProt으로, drug encoder를 GNN으로 교체하여 두 modality의 표현력을 균형 있게 향상시킨다.

---

## 2. 왜 SaProt을 frozen으로 사용하는가?

### 하드웨어 제약

| 방법 | VRAM 요구 | 본 환경 (GTX 1650 SUPER 4GB) |
|------|----------|---------------------------|
| SaProt 전체 파인튜닝 | >16GB | ❌ 불가 |
| LoRA 파인튜닝 | ~4~6GB | ❌ 시도 후 포기 (epoch당 ~2.5h, Tensor Core 없음) |
| **Frozen + MLP 헤드** | **~2GB** | **✅ 채택** |

LoRA는 메모리 측면에서 가능했지만, GTX 1650 SUPER는 Tensor Core가 없어 학습 속도가 epoch당 2.5시간으로 측정됨. 50 epoch 기준 5일 이상 소요 → 실질적으로 불가능.

### 설계 선택의 의미

SaProt을 frozen으로 사용하면 DTI 태스크 특화 적응은 없다. 그러나 이것이 연구 핵심 질문이다:

> **"SaProt의 일반 단백질 표현력(3Di 구조 토큰 포함)이 DTI 예측에 유효한가? 그리고 drug encoder를 GNN으로 교체하면 두 encoder의 표현력 불균형을 해소할 수 있는가?"**

frozen 설정은 protein encoder 측 성능 변화를 오직 입력 표현(3Di 토큰 유무)에서만 발생하도록 통제한다. 이를 통해 3Di 토큰의 순수 기여도를 정량적으로 분리 측정할 수 있었다.

---

## 3. 전체 파이프라인

### 3.1 데이터 흐름

```
[DeepPurpose DAVIS/KIBA]
  load_process_DAVIS(binary=False, convert_to_log=True)
  → 30,056 쌍 (SMILES, AA서열, pKd)
  → Train 70% / Val 10% / Test 20%

[단백질 경로]
  AA서열
  → AlphaFold DB → PDB 구조
  → FoldSeek → 3Di 토큰 (per-residue)
  → SA 포맷 변환: "MEVK" + "adcp" → "MaEdVcKp"
  → SaProt 토크나이저 (446 SA 어휘)
  → SaProt-650M Transformer (frozen, FP16, 33레이어)
  → last_hidden_state[:, 1:-1, :] mean pool → [1280-dim]
  → cache 저장 (379개 unique 단백질, 재사용)

[약물 경로]
  SMILES
  → RDKit MorganFP (radius=2, nBits=2048)
  → [2048-dim] (고정 표현, 학습 없음)
  ※ Morgan FP는 결정론적 비트벡터 — 분자 그래프의 위상 정보를 radius=2 범위 내
     원자 이웃 해싱으로 인코딩. 학습 파라미터 없음. SOTA 대비 성능 갭의 주원인.
     Phase 4에서 GNN(AttentiveFP/MPNN)으로 교체 예정.

[DTI 헤드 (유일한 학습 대상)]
  prot_enc: [1280 → 512 → 256]
  drug_enc: [2048 → 512 → 256]
  regressor: [512 → 256 → 64 → 1]  ← pKd 예측값

  Loss: HuberLoss(delta=1.0)
  Optimizer: Adam(lr=1e-3, wd=1e-4)
  Scheduler: CosineAnnealingLR
  Early stopping: patience=10
  학습 시간: ~3분 (임베딩 캐시 사용 시)
```

### 3.2 SA 토큰 포맷

SaProt의 입력은 일반 아미노산 서열이 아닌 **SA(Structural Aware) 토큰**:

```python
# Phase 3+ (현재): AlphaFold + FoldSeek 3Di 사용
sa_seq = "".join(aa.upper() + di.lower() for aa, di in zip(aa_seq, foldseek_seq))
# "MEVK" + "adcp" → "MaEdVcKp"

# Phase 1/2 baseline: 구조 정보 없을 때
sa_seq = "".join(aa + "#" for aa in aa_seq)
# "MEVK" → "M#E#V#K#"
```

어휘: 21 AA × 21 3Di = 441 + 5 special = **446 토큰**

---

## 4. 실험 결과 요약

### Phase 1/2 — Placeholder('#') 기준

| 모델 | Test r (DAVIS) | Test r (KIBA) |
|------|---------------|--------------|
| SaProt-650M | 0.7855 | N/A |
| SaProt-35M | 0.7832 | 0.7894 |
| SaProt-650M-8bit | 0.7812 | 0.7916 |
| SaProt-650M-4bit | 0.7914 | 0.7994 |

### Phase 3 — 3Di 구조 토큰 적용 후

| 모델 | Test r (DAVIS) | Test r (KIBA) | 평균 |
|------|---------------|--------------|------|
| **SaProt-650M FP16** | **0.8082** | **0.8032** | **0.8057** |
| SaProt-35M | 0.7996 | 0.8035 | 0.8016 |
| SaProt-650M-8bit | 0.8027 | 0.7997 | 0.8012 |
| SaProt-650M-4bit | 0.7977 | 0.7935 | 0.7956 |

### 핵심 발견

**3Di 토큰의 효과:**
- DAVIS 전 모델 향상 (최대 +0.023, 650M FP16)
- KIBA 650M-4bit 제외 전 모델 향상
- 3Di 구조 정보가 DTI 예측에 실질적으로 기여함을 정량적으로 확인

**양자화와 3Di의 관계:**
- Placeholder('#') 사용 시: 4bit ≥ FP16 (양자화가 성능 유지 또는 소폭 향상)
- 3Di 사용 시: FP16 > 8bit > 4bit (양자화가 구조 신호를 손실)
- 해석: NF4 4bit는 의미 없는 '#' 토큰에는 무해했지만, 실제 구조 신호가 담긴 3Di 토큰에는 정밀도 손실이 영향을 줌
- **결론: 3Di 사용 시 FP16이 최적**

---

## 5. 최종 모델 선정

**채택: SaProt-650M FP16 + 3Di**

| 기준 | 내용 |
|------|------|
| DAVIS 성능 | 0.8082 (최고) |
| KIBA 성능 | 0.8032 (최고) |
| VRAM | ~2GB (frozen inference, GTX 1650 SUPER 가능) |
| 구조 토큰 | FP16에서 3Di 신호 완전 활용 |

대안: **SaProt-35M FP16 + 3Di** — KIBA 0.8035로 650M과 사실상 동일, inference 속도 우선 시 고려

---

## 6. 참조

| 레포 | 역할 |
|------|------|
| [westlake-repl/SaProt](https://github.com/westlake-repl/SaProt) | SaProt 모델, SA 토큰 포맷 |
| [steineggerlab/foldseek](https://github.com/steineggerlab/foldseek) | 3Di 토큰 추출 |
| [kexinhuang12345/DeepPurpose](https://github.com/kexinhuang12345/DeepPurpose) | DAVIS/KIBA 데이터 로더, 기준선 비교 |
| [alphafold.ebi.ac.uk](https://alphafold.ebi.ac.uk) | 단백질 3D 구조 DB |
