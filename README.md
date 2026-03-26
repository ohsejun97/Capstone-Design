# Bio-AI Agent System for Drug–Target Interaction Analysis

> **Capstone Design Project**
> Building an Agent AI system that integrates quantized DTI prediction, protein 3D structure retrieval, and ligand structure generation — answering natural language queries about drug-target interactions end-to-end.

---

## Overview

Traditional DTI tools give you a number. This project gives you an **explanation**.

A user asks: *"How strongly does Imatinib bind to ABL1 kinase, and why?"*

Instead of running a single model, an **Agent AI** orchestrates three specialized tools:

1. **DTI Tool** — predicts binding affinity (pKd) using a quantized SaProt model
2. **Protein Tool** — retrieves the 3D structure of ABL1 from AlphaFold DB
3. **Ligand Tool** — generates the 3D conformation of Imatinib from its SMILES

The agent synthesizes these outputs into a human-readable response with structural context.

---

## The Three Data Sources — Roles and Connections

Understanding how DAVIS, KIBA, and AlphaFold DB fit together is key to understanding this project.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                             DATA SOURCES                                 │
│                                                                          │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐   │
│  │      DAVIS       │  │      KIBA        │  │    AlphaFold DB      │   │
│  │                  │  │                  │  │                      │   │
│  │ Drug + Protein   │  │ Drug + Protein   │  │ Protein 3D Structure │   │
│  │ → pKd score      │  │ → KIBA score     │  │ → PDB coordinates    │   │
│  │ ~30K pairs       │  │ ~118K pairs      │  │ ~200M proteins       │   │
│  │                  │  │                  │  │                      │   │
│  │ [1] Train DTI    │  │ [1] Validate     │  │ [Inference only]     │   │
│  │ [2] Candidate DB │  │ [2] Candidate DB │  │  Protein 3D lookup   │   │
│  └────────┬─────────┘  └────────┬─────────┘  └──────────┬───────────┘   │
└───────────┼────────────────────┼───────────────────────┼───────────────┘
            │ Training           │ Cross-validation       │ Structure
            ▼                    ▼                        ▼
     ┌─────────────────────────────┐              ┌───────────────┐
     │         DTI Tool            │              │ Protein Tool  │
     │  SaProt-650M-4bit + MLP     │              │  AlphaFold DB │
     │  (trained on DAVIS/KIBA)    │              │  API lookup   │
     └──────────────┬──────────────┘              └───────┬───────┘
                    │                                      │
     At inference:  │                                      │
     Search DAVIS+KIBA for drug candidates,               │
     predict pKd for each → rank by affinity              │
                    │                                      │
                    └────────────────┬─────────────────────┘
                                     │
                            ┌────────▼────────┐
                            │   Agent AI      │
                            │ (Orchestrator)  │
                            └────────┬────────┘
                                     │
                            ┌────────▼────────────────────┐
                            │  Final Response              │
                            │  "Top candidates: ...        │
                            │   Predicted pKd: 8.3         │
                            │   3D structure: [PDB]..."    │
                            └──────────────────────────────┘
```

### DAVIS — Training Benchmark & Candidate Database

| Property | Value |
|----------|-------|
| Content | Drug SMILES + Protein AA sequences + binding affinity (Kd) |
| Size | ~30,000 drug-protein pairs |
| Affinity metric | Kd (dissociation constant) → converted to pKd = -log10(Kd) |
| Proteins | 442 kinases |
| Role in this project | **Train/evaluate the DTI model + searchable candidate pool at inference** |

DAVIS serves two purposes: (1) it trains the DTI model to learn what "strong binding" looks like; (2) at inference time, when a user queries a target protein, the agent can search DAVIS to find known drug candidates and rank them by predicted pKd.

### KIBA — Generalization Validation & Extended Candidate Database

| Property | Value |
|----------|-------|
| Content | Drug SMILES + Protein AA sequences + bioactivity score |
| Size | ~118,000 drug-protein pairs |
| Affinity metric | KIBA score (composite of Ki, Kd, IC50) |
| Proteins | 229 kinases |
| Role in this project | **Cross-dataset validation + 4× larger candidate pool at inference** |

After training on DAVIS, we independently re-train and evaluate on KIBA to confirm generalization. KIBA is **not** a source of 3D protein structures — that role belongs exclusively to AlphaFold DB. At inference, KIBA extends the candidate search space with 118K drug-protein pairs across a wider range of kinases.

### AlphaFold DB — Structural Knowledge Base

| Property | Value |
|----------|-------|
| Content | 3D protein structures predicted by AlphaFold2 |
| Size | ~200 million protein structures |
| Format | PDB coordinates + per-residue pLDDT confidence scores |
| Role in this project | **Provide 3D structural context for Agent responses** |

AlphaFold DB is **not a DTI dataset** — it serves two purposes here:

1. **Protein Tool in the Agent**: When a user asks about a protein, the agent queries AlphaFold DB to retrieve its 3D structure for visualization and explanation.

2. **SA Token enrichment** (future): SaProt uses SA tokens that combine amino acid identity with FoldSeek 3Di structural tokens. Currently we use `'#'` placeholders (no structure). With AlphaFold structures + FoldSeek, we can generate real structural tokens — potentially improving DTI prediction quality.

```
Current  (no structure):  M#E#T#K#...   ← '#' = unknown structure
Future   (with AlphaFold): MaEvKcTp...  ← real FoldSeek 3Di tokens
```

---

## System Architecture

```
User Natural Language Query
          │
          ▼
   ┌─────────────────────────────────────────────────┐
   │              Agent AI (LLM Orchestrator)         │
   │           smolagents + Gemini / GPT              │
   │                                                  │
   │  Parses intent → selects tools → merges outputs  │
   └──────┬──────────────┬──────────────┬────────────┘
          │              │              │
          ▼              ▼              ▼
   ┌────────────┐ ┌────────────┐ ┌────────────────┐
   │ DTI Tool   │ │Protein Tool│ │  Ligand Tool   │
   │            │ │            │ │                │
   │ SaProt-4bit│ │AlphaFold DB│ │    RDKit       │
   │ + MLP Head │ │ API lookup │ │  3D conformer  │
   │            │ │            │ │  from SMILES   │
   │ Input:     │ │ Input:     │ │ Input:         │
   │  SMILES    │ │  UniProt ID│ │  SMILES        │
   │  AA seq    │ │  or name   │ │                │
   │            │ │            │ │                │
   │ Output:    │ │ Output:    │ │ Output:        │
   │  pKd score │ │  PDB struct│ │  3D coords     │
   └────────────┘ └────────────┘ └────────────────┘
          │              │              │
          └──────────────┴──────────────┘
                         │
                         ▼
          ┌──────────────────────────────┐
          │  Final Response              │
          │  "Predicted pKd: 8.3        │
          │   (strong binding)           │
          │   Key binding residues: ...  │
          │   3D visualization: ..."     │
          └──────────────────────────────┘
```

### DTI Tool — Internal Structure

```
SMILES  ──→ Morgan Fingerprint (2048-bit, RDKit)  ──────────────┐
                                                                  ├──→ MLP Head ──→ pKd
AA seq  ──→ SA tokens (AA + FoldSeek 3Di)                       │
            ──→ SaProt (frozen, NF4 4-bit quantized) ──→ emb   ─┘
```

The protein encoder (SaProt) is frozen — only the small MLP head (~2.6M params) is trained on DAVIS/KIBA. Quantization (NF4 4-bit) reduces VRAM to ~0.8GB while maintaining full-precision performance.

---

## Phase 1 Results — DTI Tool Benchmarking

### DAVIS (Primary Benchmark)

> SaProt frozen + MLP head trained on DAVIS. Comparing backbone size and quantization level.

| Model | Params | Quant | Test Pearson r | RMSE | CI | Train Time |
|-------|--------|-------|---------------|------|----|-----------|
| SaProt-650M | 652M | FP16 | 0.7855 | 0.5686 | 0.8620 | 59s |
| SaProt-35M | 34M | FP16 | 0.7832 | 0.5709 | 0.8602 | 55s |
| SaProt-650M-8bit | 652M | INT8 | 0.7812 | 0.5734 | 0.8577 | 62s |
| **SaProt-650M-4bit** | 652M | **NF4 4-bit** | **0.7914** | **0.5614** | **0.8679** | 198s |

**Key findings:**
- SaProt-35M (19× fewer params) achieves within **0.002 Pearson r** of 650M — effectively identical
- NF4 4-bit quantization **maintains performance** while reducing VRAM from ~2.5GB to ~0.8GB (regularization effect)
- All models: CI > 0.86 — correctly ranks drug binding strength 86% of the time

### KIBA (Cross-Dataset Generalization)

> Same pipeline (SaProt-650M-4bit frozen + MLP head), independently trained and evaluated on KIBA.

| Dataset | Pairs | Test Pearson r | Val r | Train Time |
|---------|-------|---------------|-------|-----------|
| DAVIS | 30,056 | 0.7914 | 0.8016 | 198s |
| **KIBA** | **118,254** | **0.7994** | **0.8106** | 206s |

**→ DAVIS results are not coincidental.** Identical pipeline on a completely different dataset (4× larger, different affinity metric) achieves equivalent performance. Both datasets reach val r ≥ 0.80.

**→ 4-bit quantized SaProt confirmed as viable real-time Agent Tool on GTX 1650 (4GB VRAM)**

---

## Roadmap

```
Phase 1  ✅  DAVIS benchmark — DTI model quantization comparison (4 models)
Phase 2a ✅  KIBA benchmark  — Cross-dataset generalization validation (r=0.7994)
Phase 2b ⏳  Protein Tool    — AlphaFold DB API integration
Phase 2c ⏳  Ligand Tool     — RDKit 3D conformer generation
Phase 3  ⏳  Agent           — smolagents orchestration layer
Phase 4  ⏳  Demo            — End-to-end natural language interface
```

---

## Experiment History

| Version | Method | Pearson r | Status |
|---------|--------|-----------|--------|
| V1 | SaProt-650M random head (CPU, 17h) | 0.030 | ❌ Wrong input format |
| V2 | SPRINT + MERGED weights | 0.141 | ❌ Out-of-distribution |
| V3-A | SaProt-650M frozen + MLP (DAVIS) | 0.7855 | ✅ |
| V3-B | SaProt-35M frozen + MLP (DAVIS) | 0.7832 | ✅ |
| V3-C | SaProt-650M-8bit frozen + MLP (DAVIS) | 0.7812 | ✅ |
| V3-D | SaProt-650M-4bit frozen + MLP (DAVIS) | **0.7914** | ✅ |
| V4 | SaProt-650M-4bit frozen + MLP (KIBA) | **0.7994** | ✅ |

---

## Quick Start

```bash
# Install environment
bash setup_env.sh          # Conda env: bioinfo (Python 3.10)

# Train DTI model (Phase 1)
python train_dti_saprot.py --encoder 650M                 # FP16 baseline
python train_dti_saprot.py --encoder 35M                  # Lightweight backbone
python train_dti_saprot.py --encoder 650M --quant 8bit    # INT8
python train_dti_saprot.py --encoder 650M --quant 4bit    # NF4 4-bit (best)

# Evaluate all models
python experiments/evaluate_results.py   # Pearson r, RMSE, MAE, CI

# Visualize results
python experiments/visualize_results.py  # outputs/figures/*.png
```

---

## Tech Stack

| Category | Stack |
|----------|-------|
| Protein encoder | SaProt (35M / 650M AF2), SA tokens (AA + FoldSeek 3Di) |
| Drug encoding | RDKit Morgan Fingerprint (radius=2, nBits=2048) |
| Quantization | bitsandbytes (NF4 4-bit, INT8) |
| ML framework | PyTorch 2.6, Hugging Face Transformers 5.3 |
| Agent | smolagents (Hugging Face) |
| Structural DB | AlphaFold DB (protein 3D), RDKit (ligand 3D) |
| Infrastructure | WSL2, Linux 32-core, GTX 1650 SUPER (4GB) |

---

## Environment Notes

- `torch >= 2.6.0` required (CVE-2025-32434)
- Do **not** install `torchvision` or `torchaudio` — version conflicts
- GPU: GTX 1650 SUPER (4GB VRAM, CUDA 12.6)
- Conda env: `bioinfo` (Python 3.10)

---

## Detailed Experiment Logs

- [Phase 1 Experiment Log](docs/PHASE1_EXPERIMENT_LOG.md)
- [Phase 1 Pipeline Design](docs/PHASE1_REFERENCE_PIPELINE.md)
- [Phase 1 Training Report](docs/PHASE1_TRAINING_EXPERIMENTS.md)
