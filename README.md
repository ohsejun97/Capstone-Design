# Bio-AI Agent System for Drug–Target Interaction Analysis

> Capstone Design Project — Bio-AI pipeline that answers natural language queries about drug-target interactions using a frozen protein language model, structural databases, and LLM orchestration.

---

## What This Does

A user asks: *"Does Imatinib bind to BCR-ABL kinase?"*

Instead of requiring expert knowledge (UniProt IDs, SMILES strings, database queries), an **Agent AI** resolves the names, orchestrates the tools, and returns a structured answer with binding affinity prediction and 3D structural context.

```
"Does Imatinib bind to BCR-ABL?"
              │
              ▼
     ┌─────────────────────┐
     │  Agent (LLM)        │  ← smolagents orchestration
     └──┬──────────────┬───┘
        │              │
        ▼              ▼
 ┌────────────┐  ┌──────────────┐
 │ Drug Name  │  │Protein Name  │
 │ Resolver   │  │  Resolver    │
 │ (PubChem)  │  │(UniProt API) │
 └─────┬──────┘  └──────┬───────┘
  SMILES            UniProt ID + seq
        │              │
        ▼              ▼
 ┌─────────┐  ┌──────────────┐  ┌────────────┐
 │ Ligand  │  │  DTI Tool    │  │  Protein   │
 │  Tool   │  │              │  │  Tool      │
 │ (RDKit) │  │SaProt-650M   │  │(AlphaFold) │
 │ 3D SDF  │  │+ 3Di + MLP   │  │  3D PDB    │
 └─────────┘  └──────┬───────┘  └────────────┘
                     │
                     ▼
         "Predicted pKd: 8.7 (strong binding)
          3D structure: cache/alphafold/P00519.pdb
          Ligand conformation: cache/ligands/..."
```

---

## System Status

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | **DTI Model Development** | 1a–1i ✅ |
| ↳ 1a | DAVIS baseline benchmarking (4 model variants) | ✅ Complete |
| ↳ 1b | KIBA cross-dataset generalization validation | ✅ Complete |
| ↳ 1c | FoldSeek 3Di structural token integration | ✅ Complete |
| ↳ 1d | GNN drug encoder — from-scratch 실패 (DAVIS 68 drugs 부족) | ✅ Complete (failed) |
| ↳ 1e | ChemBERTa frozen — Morgan FP 미달 (DAVIS -0.019, KIBA -0.043) | ✅ Complete (failed) |
| ↳ 1f | **BindingDB + ChemBERTa r=0.8737, GNN r=0.8411** — 기준선(0.8082) 돌파 | ✅ Complete |
| ↳ 1g | **Transfer Learning** — BindingDB Head → DAVIS r=0.8166, KIBA r=0.8163 | ✅ Complete |
| ↳ 1h | **ChemBERTa Fine-tuning** — layers 4~5 unfreeze, BindingDB r=**0.8923** (+0.0186 vs frozen) | ✅ Complete |
| ↳ 1i | **ft ChemBERTa Transfer** — DAVIS r=**0.8677** (+0.051), KIBA r=**0.8594** (+0.043) | ✅ Complete |
| Phase 2 | **Agent Tools** (Tool 1–5 implementation) | ✅ Complete |
| Phase 3 | **Agent Orchestration** — smolagents ReAct | ⏳ Next |
| Phase 4 | **End-to-End Demo** | ⏳ Planned |

**Best DTI model (현재):** SaProt-650M FP16 + 3Di + ChemBERTa ft Transfer — DAVIS r=**0.8677**, KIBA r=**0.8594** (4GB VRAM, SOTA 대비 격차 0.02~0.04)
(see [Training Report](docs/PHASE1_TRAINING_EXPERIMENTS.md) | [ChemBERTa Fine-tuning](docs/PHASE1H_CHEMBERTA_UNFREEZE.md) | [ft Transfer](docs/PHASE1I_TRANSFER_FT.md))

---

## Tools

| # | Tool | Input | Output | Implementation |
|---|------|-------|--------|----------------|
| 1 | DTI Prediction | SMILES + AA sequence | pKd (binding affinity) | SaProt-650M FP16 + 3Di + ChemBERTa ft + MLP head (r=0.8923) |
| 2 | Protein Structure | UniProt ID | 3D PDB + pLDDT | AlphaFold DB REST API |
| 3 | Ligand Structure | SMILES | 3D SDF + properties | RDKit ETKDGv3 + MMFF94 |
| 4 | Drug Name Resolver | Drug name (e.g. "Imatinib") | SMILES | PubChem REST API |
| 5 | Protein Name Resolver | Gene/protein name (e.g. "EGFR") | UniProt ID + AA seq | UniProt Search API |

---

## Quick Start

```bash
# Setup
conda activate bioinfo

# ── Phase 1h: ChemBERTa fine-tune on BindingDB (최종 DTI 모델 학습) ──
python scripts/train_chemberta_unfreeze.py \
    --unfreeze 2 --split random --batch_size 32 --max_length 128

# ── Phase 1i: ft ChemBERTa Transfer → DAVIS / KIBA ──
python scripts/finetune_head_ft.py --target_dataset davis
python scripts/finetune_head_ft.py --target_dataset kiba

# ── DTI Tool 단독 테스트 (Phase 1h/1i 모델 사용) ──
python tools/dti_tool.py

# ── 개별 Agent Tool 테스트 ──
python tools/alphafold_tool.py P00533                        # Protein structure (EGFR)
python tools/rdkit_tool.py "CC(=O)Oc1ccccc1C(=O)O"         # Ligand 3D (Aspirin)

# ── 3Di 토큰 캐시 빌드 (prerequisite) ──
python scripts/build_3di_cache.py --dataset bindingdb --resume
```

---

## Architecture

### Research hypothesis and design rationale

**SaProt is not a DTI model.** SaProt is a Protein Language Model (PLM) — it encodes proteins only. This project uses SaProt as the *protein encoder component* of a DTI system, paired with a separate drug encoder.

A DTI prediction system requires two encoders:
```
Drug encoder   : drug chemical structure  → vector
Protein encoder: protein sequence + 3D   → vector
                                           → MLP Head → pKd (binding affinity)
```

**The two bottlenecks in existing DTI research:**

1. **Protein encoder lacks 3D structure.** Models like DeepPurpose use CNN or AAC descriptors that operate on sequence only. Binding sites are determined by 3D conformation, not sequence alone. SaProt (a structure-aware PLM) addresses this via FoldSeek 3Di tokens — but DTI-specific fine-tuning of SaProt requires >16GB VRAM (infeasible here). So SaProt is used **frozen**: its general protein representations are leveraged as-is.

2. **Drug encoder pretrained on the wrong objective.** Morgan Fingerprint has no trainable parameters. ChemBERTa (frozen) is pretrained on masked language modeling (MLM) over PubChem — optimizing for generic molecular property prediction, not binding affinity. The representations it produces are not aligned with pKd. **Solution:** fine-tune ChemBERTa's upper layers (4~5 of 6) on BindingDB DTI data, adapting the representations toward binding-relevant chemical features (hydrophobicity, H-bond donors/acceptors, shape complementarity) while keeping lower layers frozen to preserve fundamental chemical grammar.

**Why only upper layers?** Lower layers (0~3) encode universal chemical structure (atom types, bond order, functional groups) — useful for any task. Upper layers encode task-specific high-level features — these are what we adapt. Full fine-tuning risks overfitting (44M params, 56K training pairs) and destroying lower-layer representations.

**Core hypothesis:** Independently improving both encoders — protein via 3Di structural tokens, drug via ChemBERTa fine-tuning — closes the performance gap without full fine-tuning, achieving DAVIS r ≥ 0.85 within 4GB VRAM.

```
SMILES → ChemBERTa (layers 4~5 fine-tuned on BindingDB) → [768-dim]  ─┐
                                                                        ├→ MLP Head → pKd
AA seq → FoldSeek 3Di tokens → SaProt-650M (frozen, FP16) → [1280-dim] ─┘
```

- **SaProt-650M** (frozen, FP16): protein sequence + 3D structure via 3Di tokens. Full fine-tuning requires >16GB VRAM → frozen as a strong general encoder.
- **ChemBERTa** (upper 2 layers fine-tuned): pretrained on PubChem MLM → adapted to pKd prediction task on BindingDB 80K pairs. Lower 4 layers frozen to preserve fundamental chemical grammar.
- **MLP head** (~1.5M params): fine-tuned per dataset (BindingDB / DAVIS / KIBA)

### SA Token Format

SaProt's vocabulary includes both amino acid identity and FoldSeek 3Di structural tokens:

```python
# With AlphaFold + FoldSeek (current, Phase 3+)
sa_seq = "".join(aa.upper() + di.lower() for aa, di in zip(aa_seq, foldseek_3di))
# "MEVK" + "adcp" → "MaEdVcKp"

# Without structure info (Phase 1/2 baseline)
sa_seq = "".join(aa + "#" for aa in aa_seq)
# "MEVK" → "M#E#V#K#"
```

---

## Tech Stack

| Component | Stack |
|-----------|-------|
| Protein encoder | SaProt-650M AF2 (frozen, FP16) |
| Drug encoder | ChemBERTa-zinc-base (layers 4~5 fine-tuned on BindingDB, 768-dim) |
| Structural tokens | FoldSeek 3Di via AlphaFold DB PDB |
| DTI head | PyTorch MLP (~1.5M params), BindingDB → DAVIS/KIBA transfer |
| Agent framework | smolagents (Hugging Face, ReAct) |
| Protein structure | AlphaFold DB (EBI REST API) |
| Ligand structure | RDKit ETKDGv3 + MMFF94 force field |
| Name resolution | PubChem API + UniProt Search API |
| Hardware | GTX 1650 SUPER (4GB VRAM), WSL2, Python 3.10 |

---

## Results Summary

### Experiment Story

The experiments follow a clear progression: baseline → structural tokens → drug encoder → data scaling → transfer learning.

---

#### Step 1 — Baseline: SaProt + Morgan FP (Placeholder '#' tokens)

DAVIS (379 unique proteins, 68 unique drugs, 30K pairs):

| Model | Pearson r | RMSE | CI |
|---|---|---|---|
| SaProt-650M FP16 | 0.7855 | — | 0.8620 |
| SaProt-35M FP16  | 0.7832 | — | 0.8602 |
| SaProt-650M-8bit | 0.7812 | — | 0.8577 |
| SaProt-650M-4bit | 0.7914 | — | 0.8679 |

KIBA (229 unique proteins, 2068 unique drugs, 118K pairs):

| Model | Pearson r | RMSE | CI |
|---|---|---|---|
| SaProt-650M FP16 | 0.7987 | 0.5024 | 0.8304 |
| SaProt-35M FP16  | 0.7894 | — | — |
| SaProt-650M-8bit | 0.7916 | — | — |
| SaProt-650M-4bit | 0.7994 | — | — |

> RMSE/CI marked — were not recorded in early experiment logs.

---

#### Step 2 — FoldSeek 3Di Structural Tokens (+0.023 on DAVIS)

Replacing '#' placeholder with real FoldSeek 3Di tokens from AlphaFold DB structures.
DAVIS: 379/379 proteins (100%), KIBA: 228/229 (99.6%).

| Model | DAVIS r (Placeholder→3Di) | KIBA r (Placeholder→3Di) |
|---|---|---|
| **SaProt-650M FP16** | **0.7855 → 0.8082 (+0.023)** | **0.7987 → 0.8032 (+0.005)** |
| SaProt-35M FP16 | 0.7832 → 0.7996 (+0.017) | 0.7894 → 0.8035 (+0.014) |
| SaProt-650M-8bit | 0.7812 → 0.8027 (+0.022) | 0.7916 → 0.7997 (+0.008) |
| SaProt-650M-4bit | 0.7914 → 0.7977 (+0.006) | 0.7994 → 0.7935 (−0.006) |

**Finding:** 4-bit quantization degrades the 3Di structural signal → **FP16 selected as final protein encoder**.

---

#### Step 3 — Drug Encoder: GNN and ChemBERTa FAIL on Small Data

Replacing fixed Morgan FP with learnable drug encoders, trained on DAVIS (68 unique drugs):

| Drug Encoder | DAVIS r | RMSE | CI | vs Morgan FP |
|---|---|---|---|---|
| Morgan FP (baseline) | 0.8082 | — | — | — |
| GNN from-scratch | 0.5795 | 0.7618 | 0.7907 | −0.229 |
| ChemBERTa frozen (DAVIS) | 0.7915 | 0.5627 | 0.8608 | −0.017 |
| ChemBERTa frozen (KIBA) | 0.7667 | 0.5351 | 0.8148 | −0.037 |
| GNN from-scratch (KIBA) | 0.7191 | 0.5783 | 0.7828 | −0.084 |

**Diagnosis:** The problem is not the model architecture — it is the data. DAVIS has only 68 unique drugs. GNN needs thousands of diverse molecules to learn generalizable representations.

---

#### Step 4 — BindingDB Scaling: Drug Diversity Solves the Problem

**BindingDB preprocessing** (server, 500GB RAM):
- Source: BindingDB_All.tsv (7.9 GB, 3.17M rows) → 80,795 unique pairs | 32,480 drugs | 2,384 proteins
- FoldSeek 3Di cache: 2,309/2,384 proteins (96.9%)

| Drug Encoder | Split | Pearson r | RMSE | CI |
|---|---|---|---|---|
| **ChemBERTa** | **random** | **0.8737** | **0.7933** | **0.8633** |
| GNN | random | 0.8411 | 0.8842 | 0.8459 |
| ChemBERTa | cold_drug | 0.7083 | 1.2543 | 0.7473 |
| ChemBERTa | cold_protein | 0.6549 | 1.1840 | 0.7430 |

Zero-shot cross-dataset evaluation (cold_drug model → DAVIS/KIBA, no fine-tuning):

| Target | Pearson r | RMSE | CI | Note |
|---|---|---|---|---|
| DAVIS | 0.208 | 1.3029 | 0.5900 | domain shift (kinase-specific) |
| KIBA | 0.160 | 5.7871 | 0.5505 | label mismatch (KIBA score ≠ pKd) |

**Key finding:** GNN and ChemBERTa both surpass the Morgan FP baseline (0.8082) once trained on BindingDB. Zero-shot cross-dataset transfer fails due to domain shift and label scale mismatch.

---

#### Step 5 — Transfer Learning: BindingDB Head → DAVIS/KIBA

SaProt + ChemBERTa embeddings reused (cached). Only MLP Head fine-tuned per dataset.
KIBA labels z-score normalized to handle scale mismatch.

| Target | Pearson r | Spearman r | RMSE | CI | vs Direct Training |
|---|---|---|---|---|---|
| **DAVIS** | **0.8166** | 0.6794 | 0.5303 | 0.8747 | +0.0084 vs 0.8082 |
| **KIBA** | **0.8163** | 0.8114 | 0.4826 | 0.8414 | +0.0131 vs 0.8032 |

**Key finding:** Transfer Learning outperforms training directly on DAVIS/KIBA. BindingDB's 32K-drug diversity provides richer embeddings than the smaller benchmarks alone. Head re-calibration suffices — no encoder retraining needed.

---

#### Step 6 — ChemBERTa Fine-tuning: Adapting Drug Representations to pKd

ChemBERTa's upper 2 layers (4~5 of 6) fine-tuned on BindingDB DTI data. Lower layers frozen.
Differential learning rates: Head 5e-4 / ChemBERTa layers 1e-5.

| Method | BindingDB Pearson r | RMSE | CI |
|---|---|---|---|
| ChemBERTa frozen (baseline) | 0.8737 | 0.7933 | 0.8633 |
| **ChemBERTa fine-tuned (layers 4~5)** | **0.8923** | **0.7387** | **0.8770** |

**Key finding:** +0.0186 improvement. MLM pretraining objective (molecular property prediction) misaligns with pKd prediction — fine-tuning upper layers adapts representations toward binding-relevant chemical features. SOTA-level performance (DeepPurpose ~0.89) achieved at 885 MB VRAM.

---

#### Step 7 — ft ChemBERTa Transfer: Final Model (Phase 1i)

ft-ChemBERTa used to recompute DAVIS/KIBA drug embeddings → MLP Head fine-tuned per dataset.

| Target | Pearson r | Spearman r | RMSE | MAE | CI | vs Phase 1g |
|---|---|---|---|---|---|---|
| **DAVIS** | **0.8677** | 0.7021 | **0.4572** | 0.2514 | **0.8925** | **+0.0511** |
| **KIBA** | **0.8594** | 0.8464 | **0.4268** | 0.2578 | **0.8610** | **+0.0431** |

**Key finding:** ft-ChemBERTa drug representations transfer significantly better than frozen representations. DAVIS +5.1%, KIBA +4.3% improvement over Phase 1g. SOTA gap reduced to 0.02~0.03 at 4GB VRAM.

---

## Scope & Limitations

This system is optimized for **known drugs interacting with human protein targets**.

| Scope | Coverage |
|-------|----------|
| Drug input | Common name, generic name, some brand names (via PubChem) |
| Protein input | Human gene/protein names (via UniProt, organism: Homo sapiens) |
| Protein families | Human kinases — DAVIS (442 kinases) + KIBA (229 kinases) |
| Affinity metric | pKd / KIBA score (continuous regression) |

**Known limitations:**

**[Protein encoder] SaProt is frozen — no DTI-specific adaptation**
SaProt-650M full fine-tuning requires >16GB VRAM (infeasible). LoRA was attempted but abandoned (2.5h/epoch, no Tensor Cores on GTX 1650 SUPER). SaProt is therefore used as a general protein encoder — its 3Di-aware representations are strong, but not tuned for DTI specifically.

**[Drug encoder] ChemBERTa fine-tuning으로 해결 (Phase 1h/1i)**
GNN/ChemBERTa가 DAVIS(68약물)에서 실패한 원인은 데이터 부족 + pretraining 목적 불일치. BindingDB(32,480약물)로 ChemBERTa 상위 레이어를 fine-tune하여 r=0.8923(BindingDB), DAVIS r=0.8677, KIBA r=0.8594 달성. 상세: [Phase 1h](docs/PHASE1H_CHEMBERTA_UNFREEZE.md), [Phase 1i](docs/PHASE1I_TRANSFER_FT.md).

**[Evaluation] Cold-split evaluated (complete)**
Random split 결과(r=0.87~0.89)는 generalization을 과대추정할 수 있음. Cold-drug split(r=0.708)과 cold-protein split(r=0.655) 평가 완료. Bio-AI Agent 사용 시나리오(알려진 약물/단백질 질의)에서는 random split 성능이 실제 성능에 더 가까움.

**[Data] Human kinase-centric training data**
DAVIS and KIBA cover human kinases exclusively (442 and 229 kinases respectively). Predictions for viral/bacterial targets, GPCRs, proteases, and nuclear receptors are out-of-distribution and should not be trusted.

**[Usability] Non-English or vague inputs**
The LLM orchestrator must translate non-English drug names (e.g., "비아그라" → "Sildenafil") and vague target descriptions before Tool 4/5 can resolve them. This depends on the LLM's domain knowledge.

---

## Roadmap

| Phase | Task | Status |
|-------|------|--------|
| Phase 1a | DAVIS baseline benchmark — 4 model variants | ✅ Complete |
| Phase 1b | KIBA cross-dataset validation | ✅ Complete |
| Phase 1c | FoldSeek 3Di token integration + re-evaluation | ✅ Complete |
| Phase 1d | GNN drug encoder (from-scratch) — failed (68 drugs) | ✅ Complete |
| Phase 1e | ChemBERTa frozen drug encoder — failed (Morgan FP superior on DAVIS) | ✅ Complete |
| Phase 1f | BindingDB + ChemBERTa r=0.8737, GNN r=0.8411 — 기준선 돌파 | ✅ Complete |
| **Phase 1g** | **Transfer Learning: BindingDB→DAVIS r=0.8166, KIBA r=0.8163 — 직접 학습 초과** | **✅ Complete** |
| **Phase 1h** | **ChemBERTa fine-tune (layers 4~5): BindingDB r=0.8923 — SOTA 수준** | **✅ Complete** |
| **Phase 1i** | **ft ChemBERTa Transfer: DAVIS r=0.8677 (+0.051), KIBA r=0.8594 (+0.043)** | **✅ Complete** |
| Phase 2 | Agent Tools 1–5 implementation | ✅ Complete |
| Phase 3 | smolagents Agent orchestration | ⏳ Next |
| Phase 4 | End-to-end demo | ⏳ Planned |

---

## Documentation

| Document | Contents |
|----------|----------|
| [Phase 1 Training Report](docs/PHASE1_TRAINING_EXPERIMENTS.md) | 전체 실험 히스토리 (V1~V21), 모델 선정, 핵심 발견 |
| [Phase 1g Transfer Learning](docs/PHASE1G_TRANSFER_LEARNING.md) | Transfer Learning 실험 — BindingDB→DAVIS/KIBA (frozen ChemBERTa) |
| [Phase 1h ChemBERTa Fine-tuning](docs/PHASE1H_CHEMBERTA_UNFREEZE.md) | ChemBERTa unfreeze 근거, 실험 결과, 분석 |
| [Phase 1i ft Transfer](docs/PHASE1I_TRANSFER_FT.md) | ft ChemBERTa Transfer → DAVIS/KIBA, Phase 1 최종 결과 |
| [Phase 2 Agent Tools](docs/PHASE2_AGENT_TOOLS.md) | Tool 1–5 구현 상세 |
| [Evaluation Framework](docs/EVALUATION_FRAMEWORK.md) | 지표 정의(Pearson r / RMSE / CI), 비교 분석 프레임워크 |
| [Phase 1 Pipeline Design](docs/PHASE1_REFERENCE_PIPELINE.md) | Architecture rationale, why frozen encoder + MLP |
| [Phase 1 Experiment Log](docs/PHASE1_EXPERIMENT_LOG.md) | V1→V3 초기 실패 히스토리 |
| [Poster Brief](POSTER_BRIEF.md) | 학술 포스터 제작 지시서 (AI 프롬프트 포함) |
