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
| Phase 1 | **DTI Model Development** | 1a–1c ✅ / 1d 🔄 |
| ↳ 1a | DAVIS baseline benchmarking (4 model variants) | ✅ Complete |
| ↳ 1b | KIBA cross-dataset generalization validation | ✅ Complete |
| ↳ 1c | FoldSeek 3Di structural token integration | ✅ Complete |
| ↳ 1d | GNN drug encoder + cold-split evaluation | 🔄 Next |
| Phase 2 | **Agent Tools** (Tool 1–5 implementation) | ✅ Complete |
| Phase 3 | **Agent Orchestration** — smolagents ReAct | ⏳ Planned |
| Phase 4 | **End-to-End Demo** | ⏳ Planned |

**Best DTI model:** SaProt-650M FP16 + 3Di — DAVIS r=0.8082, KIBA r=0.8032
(see [Training Report](docs/PHASE1_TRAINING_EXPERIMENTS.md))

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

### Phase 3: Placeholder vs 3Di Structural Tokens

```
DAVIS
  Model        Placeholder    3Di     Delta
  650M          0.7855     0.8082  +0.0227
  35M           0.7832     0.7996  +0.0165
  650M-8bit     0.7812     0.8027  +0.0215
  650M-4bit     0.7914     0.7977  +0.0063

KIBA
  Model        Placeholder    3Di     Delta
  650M          0.7987       0.8032  +0.0045
  35M           0.7894     0.8035  +0.0141
  650M-8bit     0.7916     0.7997  +0.0081
  650M-4bit     0.7994     0.7935  -0.0059
```

**Key findings:**
- 3Di structural tokens improve performance across almost all models/datasets
- 4-bit quantization interferes with structural signal → FP16 is optimal when using 3Di tokens
- **Selected model: SaProt-650M FP16 + 3Di** (best average: DAVIS 0.8082 / KIBA 0.8032)

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

**[Drug encoder] Morgan FP — primary performance bottleneck**
Morgan Fingerprint is a deterministic bit vector with zero learnable parameters (`AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)`). It encodes local atomic neighbourhoods up to radius 2 but loses global molecular topology. SOTA DTI methods (MPNN_CNN r ≈ 0.89) use GNN drug encoders that learn directly from molecular graphs. The r=0.81 vs 0.89 gap is primarily attributable to this drug encoder gap. **Phase 1d:** replace with AttentiveFP or MPNN.

**[Evaluation] Random split may overestimate generalization**
Current train/val/test split is random 70/10/20. Because DAVIS has only 68 unique drugs and 442 unique proteins, the same drug–protein entities appear in both train and test sets. This risks overestimating performance on truly unseen molecules. Cold-drug split (test drugs never seen during training) and cold-target split (test proteins never seen) are planned for Phase 4 to establish more realistic generalization bounds.

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
| **Phase 1d** | **GNN drug encoder (AttentiveFP/MPNN) + cold-split evaluation** | **🔄 Next** |
| Phase 2 | Agent Tools 1–5 implementation | ✅ Complete |
| Phase 3 | smolagents Agent orchestration | ⏳ Planned |
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
