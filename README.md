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
| Phase 1 | DTI model benchmarking (DAVIS, 4 model variants) | ✅ Complete |
| Phase 2a | Cross-dataset generalization validation (KIBA) | ✅ Complete |
| Phase 2b | Agent Tools 1–5 implementation | ✅ Complete |
| Phase 3 | FoldSeek 3Di token integration + re-evaluation | ✅ Complete |
| Phase 4 | Agent orchestration — smolagents | 🔄 Next |
| Phase 5 | End-to-end demo | ⏳ Planned |

**Best DTI model:** SaProt-650M FP16 + 3Di — DAVIS r=0.8082, KIBA r=0.8032
(see [Phase 1 Training Report](docs/PHASE1_TRAINING_EXPERIMENTS.md))

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

### Why frozen SaProt + MLP head?

Full fine-tuning of SaProt-650M requires >4GB VRAM and was not feasible on the project hardware (GTX 1650 SUPER, 4GB). LoRA fine-tuning was attempted but abandoned due to training speed (~2.5h/epoch on a GPU without Tensor Cores).

Instead, this project treats SaProt as a **fixed protein encoder** and trains only a small MLP head to predict binding affinity. This separates concerns clearly:

```
SMILES  → Morgan Fingerprint (2048-bit, RDKit, fixed)    ─┐
                                                           ├→ MLP Head → pKd
AA seq  → SA tokens → SaProt-650M (frozen, FP16)          ─┘
               ↑
  "MaEvKc..." (AA + FoldSeek 3Di structural tokens)
```

- **SaProt** encodes protein sequence + 3D structural context (3Di tokens from FoldSeek)
- **Morgan FP** encodes drug chemical structure as a fixed bit vector
- **MLP head** (2.4M params) learns the binding affinity relationship between the two
- SaProt weights are never updated — only the MLP head is trained (~3 min on DAVIS)

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
  650M          N/A          0.8032
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

**[Model] Performance gap vs SOTA**
Current best: Pearson r ≈ 0.81. SOTA methods (DeepPurpose MPNN_CNN r ≈ 0.89, GraphDTA r ≈ 0.88) benefit from end-to-end fine-tuning and graph neural network drug encoders. The gap is primarily due to (1) frozen protein encoder and (2) Morgan FP drug encoding vs learned GNN representations. Full fine-tuning was hardware-constrained; this project's contribution is validating SaProt's 3Di structural tokens as effective protein representations in a DTI Agent system.

**[Model] Drug encoder asymmetry**
The protein encoder (SaProt, 650M params) is far more expressive than the drug encoder (Morgan Fingerprint, 2048-bit binary vector). Morgan FP is a fixed structural descriptor that does not learn from data. Replacing it with a graph neural network (e.g., MPNN) would better match the protein encoder's capacity.

**[Data] Human kinase-centric training data**
DAVIS and KIBA cover human kinases exclusively (442 and 229 kinases respectively). Predictions for viral/bacterial targets, GPCRs, proteases, and nuclear receptors are out-of-distribution and should not be trusted.

**[Usability] Non-English or vague inputs**
The LLM orchestrator must translate non-English drug names (e.g., "비아그라" → "Sildenafil") and vague target descriptions before Tool 4/5 can resolve them. This depends on the LLM's domain knowledge.

---

## Roadmap

| Phase | Task | Status |
|-------|------|--------|
| Phase 1 | DAVIS benchmark — 4 model variants | ✅ Complete |
| Phase 2a | KIBA cross-dataset validation | ✅ Complete |
| Phase 2b | Agent Tools 1–5 implementation | ✅ Complete |
| Phase 3 | FoldSeek 3Di token integration + re-evaluation | ✅ Complete |
| **Phase 4** | **smolagents orchestration + end-to-end demo** | **🔄 Next** |
| Phase 5 | BindingDB expansion (viral/non-kinase targets) | ⏳ Planned |

---

## Documentation

| Document | Contents |
|----------|----------|
| [Phase 1 Training Report](docs/PHASE1_TRAINING_EXPERIMENTS.md) | DAVIS/KIBA/3Di results, model selection, key findings |
| [Phase 1 Pipeline Design](docs/PHASE1_REFERENCE_PIPELINE.md) | Architecture rationale, why frozen encoder + MLP |
| [Phase 1 Experiment Log](docs/PHASE1_EXPERIMENT_LOG.md) | V1→V3 iteration history, failure analysis |
| [Phase 2 Agent Tools](docs/PHASE2_AGENT_TOOLS.md) | Tool 2–5 implementation details and test results |
