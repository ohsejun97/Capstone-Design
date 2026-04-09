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
| Phase 1 | **DTI Model Development** | 1a–1g ✅ |
| ↳ 1a | DAVIS baseline benchmarking (4 model variants) | ✅ Complete |
| ↳ 1b | KIBA cross-dataset generalization validation | ✅ Complete |
| ↳ 1c | FoldSeek 3Di structural token integration | ✅ Complete |
| ↳ 1d | GNN drug encoder — from-scratch 실패 (DAVIS 68 drugs 부족) | ✅ Complete (failed) |
| ↳ 1e | ChemBERTa frozen — Morgan FP 미달 (DAVIS -0.019, KIBA -0.043) | ✅ Complete (failed) |
| ↳ 1f | **BindingDB + ChemBERTa r=0.8737, GNN r=0.8411** — 기준선(0.8082) 돌파 | ✅ Complete |
| ↳ 1g | **Transfer Learning** — BindingDB Head → DAVIS r=0.8166, KIBA r=0.8163 | ✅ Complete |
| Phase 2 | **Agent Tools** (Tool 1–5 implementation) | ✅ Complete |
| Phase 3 | **Agent Orchestration** — smolagents ReAct | ⏳ Next |
| Phase 4 | **End-to-End Demo** | ⏳ Planned |

**Best DTI model (현재):** SaProt-650M FP16 + 3Di + ChemBERTa — Transfer Learning (BindingDB→DAVIS r=0.8166, BindingDB→KIBA r=0.8163)
(see [Training Report](docs/PHASE1_TRAINING_EXPERIMENTS.md) | [Transfer Learning](docs/PHASE1G_TRANSFER_LEARNING.md))

---

## Tools

| # | Tool | Input | Output | Implementation |
|---|------|-------|--------|----------------|
| 1 | DTI Prediction | SMILES + AA sequence | pKd (binding affinity) | SaProt-650M FP16 + 3Di tokens + MLP head |
| 2 | Protein Structure | UniProt ID | 3D PDB + pLDDT | AlphaFold DB REST API |
| 3 | Ligand Structure | SMILES | 3D SDF + properties | RDKit ETKDGv3 + MMFF94 |
| 4 | Drug Name Resolver | Drug name (e.g. "Imatinib") | SMILES | PubChem REST API |
| 5 | Protein Name Resolver | Gene/protein name (e.g. "EGFR") | UniProt ID + AA seq | UniProt Search API |

---

## Quick Start

```bash
# Setup
conda activate bioinfo

# Train DTI model (with 3Di structural tokens)
python train_dti_saprot.py --dataset davis --encoder 650M --use_3di

# Test individual tools
python tools/alphafold_tool.py P00533                        # Protein structure (EGFR)
python tools/rdkit_tool.py "CC(=O)Oc1ccccc1C(=O)O"         # Ligand 3D (Aspirin)

# Build 3Di token cache (prerequisite for --use_3di)
python scripts/build_3di_cache.py --dataset davis --resume

# Evaluate all trained models
python experiments/evaluate_results.py
python experiments/visualize_results.py
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

2. **Drug encoder has no learnable representation.** Morgan Fingerprint is a deterministic bit vector computed from SMILES with no trainable parameters. It loses global molecular topology. This is the primary cause of the gap vs SOTA (r=0.81 vs 0.89). **Phase 1d plan:** replace with a GNN (AttentiveFP/MPNN) that learns directly from molecular graphs.

**Core hypothesis:** Independently improving both encoders — protein via 3Di structural tokens, drug via GNN — closes the performance gap without full fine-tuning, achieving DAVIS r ≥ 0.85 within 4GB VRAM.

```
SMILES → [Phase 1a–1c] Morgan FP (fixed)           ─┐
          [Phase 1d]    GNN encoder (learned)        ├→ MLP Head → pKd
AA seq → SA tokens → SaProt-650M (frozen, FP16)    ─┘
              ↑
  "MaEvKc..." (AA + FoldSeek 3Di structural tokens)
```

- **SaProt** (frozen): encodes protein sequence + 3D structure context via 3Di tokens
- **Drug encoder**: Morgan FP baseline → GNN replacement in Phase 1d
- **MLP head** (~2.4M params): the only component trained on DTI data (~3 min on DAVIS)

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
| Drug encoding | RDKit Morgan Fingerprint (2048-bit, radius=2) |
| DTI head | PyTorch MLP (~2.4M params), trained on DAVIS + KIBA |
| Structural tokens | FoldSeek 3Di via AlphaFold DB PDB |
| Agent framework | smolagents (Hugging Face) |
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

**[Drug encoder] BindingDB로 해결 완료 (Phase 1f)**
GNN/ChemBERTa가 DAVIS(68약물)에서 실패한 원인은 데이터 부족. BindingDB(32,480약물)로 재훈련 시 ChemBERTa r=0.8737, GNN r=0.8411로 기준선(0.8082) 돌파. 현재 cold-split + cross-dataset eval 진행 중.

**[Evaluation] Cold-split evaluation in progress**
Random split results (ChemBERTa r=0.8737) may overestimate generalization. Cold-drug split (test drugs never seen during training) and cold-protein split are now implemented and running. Cross-dataset evaluation (BindingDB-trained model tested on DAVIS/KIBA) is also running to validate generalization.

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
| Phase 2 | Agent Tools 1–5 implementation | ✅ Complete |
| Phase 3 | smolagents Agent orchestration | ⏳ Next |
| Phase 4 | End-to-end demo | ⏳ Planned |

---

## Documentation

| Document | Contents |
|----------|----------|
| [Evaluation Framework](docs/EVALUATION_FRAMEWORK.md) | 지표 정의(Pearson r / RMSE / CI), 비교 분석 프레임워크, 결과 서술 기준 |
| [Phase 1 Training Report](docs/PHASE1_TRAINING_EXPERIMENTS.md) | DAVIS/KIBA/3Di results, model selection, key findings |
| [Phase 1 Pipeline Design](docs/PHASE1_REFERENCE_PIPELINE.md) | Architecture rationale, why frozen encoder + MLP |
| [Phase 1 Experiment Log](docs/PHASE1_EXPERIMENT_LOG.md) | V1→V3 iteration history, failure analysis |
| [Phase 2 Agent Tools](docs/PHASE2_AGENT_TOOLS.md) | Tool 2–5 implementation details and test results |
| [Phase 1g Transfer Learning](docs/PHASE1G_TRANSFER_LEARNING.md) | Transfer Learning 실험 분석 — BindingDB→DAVIS/KIBA |
