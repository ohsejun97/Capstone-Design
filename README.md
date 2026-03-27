# Bio-AI Agent System for Drug–Target Interaction Analysis

> Capstone Design Project — Bio-AI pipeline that answers natural language queries about drug-target interactions using quantized protein language models, structural databases, and LLM orchestration.

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
 │ (RDKit) │  │ SaProt-4bit  │  │(AlphaFold) │
 │ 3D SDF  │  │  + MLP Head  │  │  3D PDB    │
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
| Phase 2b | Protein Tool — AlphaFold DB API | ✅ Complete |
| Phase 2b | Ligand Tool — RDKit 3D conformer | ✅ Complete |
| Phase 2b | Drug Name Resolver — PubChem API | 🔄 In Progress |
| Phase 2b | Protein Name Resolver — UniProt Search API | 🔄 In Progress |
| Phase 3 | Agent orchestration — smolagents | ⏳ Planned |
| Phase 4 | End-to-end demo | ⏳ Planned |

**Best DTI model:** SaProt-650M-4bit — DAVIS r=0.7914, KIBA r=0.7994
(see [Phase 1 Training Report](docs/PHASE1_TRAINING_EXPERIMENTS.md))

---

## Tools

| # | Tool | Input | Output | Implementation |
|---|------|-------|--------|----------------|
| 1 | DTI Prediction | SMILES + AA sequence | pKd (binding affinity) | SaProt-650M (NF4 4-bit) + MLP head |
| 2 | Protein Structure | UniProt ID | 3D PDB + pLDDT | AlphaFold DB REST API |
| 3 | Ligand Structure | SMILES | 3D SDF + properties | RDKit ETKDGv3 + MMFF94 |
| 4 | Drug Name Resolver | Drug name (e.g. "Imatinib") | SMILES | PubChem REST API |
| 5 | Protein Name Resolver | Gene/protein name (e.g. "EGFR") | UniProt ID + AA seq | UniProt Search API |

---

## Quick Start

```bash
# Setup
conda activate bioinfo

# Train DTI model
python train_dti_saprot.py --dataset davis --encoder 650M --quant 4bit

# Test individual tools
python tools/alphafold_tool.py P00533                        # Protein structure (EGFR)
python tools/rdkit_tool.py "CC(=O)Oc1ccccc1C(=O)O"         # Ligand 3D (Aspirin)

# Evaluate all trained models
python experiments/evaluate_results.py
python experiments/visualize_results.py
```

---

## Tech Stack

| Component | Stack |
|-----------|-------|
| Protein encoder | SaProt-650M AF2 (frozen, NF4 4-bit via bitsandbytes) |
| Drug encoding | RDKit Morgan Fingerprint (2048-bit, radius=2) |
| DTI head | PyTorch MLP (~2.6M params), trained on DAVIS + KIBA |
| Agent framework | smolagents (Hugging Face) |
| Protein structure | AlphaFold DB (EBI REST API) |
| Ligand structure | RDKit ETKDGv3 + MMFF94 force field |
| Name resolution | PubChem API + UniProt Search API |
| Hardware | GTX 1650 SUPER (4GB VRAM), WSL2, Python 3.10 |

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

**[Model] SaProt 3D structure tokens not yet utilized**
SaProt's core design encodes both amino acid sequence and FoldSeek 3Di structural tokens (`"MaEvKc..."`). This project currently uses `'#'` placeholder tokens (`"M#E#T#..."`), meaning SaProt is operating without its structural advantage. This is the primary reason the current DTI performance (Pearson r ≈ 0.79) falls below SOTA methods such as DeepPurpose MPNN_CNN (r ≈ 0.89) and GraphDTA (r ≈ 0.88). The next planned experiment is to apply FoldSeek 3Di tokens using the AlphaFold structures already retrieved by Tool 2.

**[Model] Drug encoder asymmetry**
The protein encoder (SaProt, 650M params) is orders of magnitude more expressive than the drug encoder (Morgan Fingerprint, 2048-bit binary vector). Morgan FP is a fixed structural descriptor that does not learn from data. Replacing it with a graph neural network (e.g., MPNN) would better match the protein encoder's capacity.

**[Data] Human kinase-centric training data**
DAVIS and KIBA cover human kinases exclusively (442 and 229 kinases respectively). Predictions for viral/bacterial targets, GPCRs, proteases, and nuclear receptors are out-of-distribution and should not be trusted.

**[Usability] Non-English or vague inputs**
The LLM orchestrator must translate non-English drug names (e.g., "비아그라" → "Sildenafil") and vague target descriptions (e.g., "콜레스테롤 합성 효소" → gene name "HMGCR") before Tool 4/5 can resolve them. This depends on the LLM's domain knowledge.

---

## Roadmap

| Phase | Task | Status |
|-------|------|--------|
| Phase 1 | DAVIS benchmark — 4 model variants | ✅ Complete |
| Phase 2a | KIBA cross-dataset validation | ✅ Complete |
| Phase 2b | Agent Tools 1–5 implementation | ✅ Complete |
| **Phase 3** | **FoldSeek 3Di token integration + re-evaluation** | **🔄 Next** |
| Phase 4 | smolagents orchestration + end-to-end demo | ⏳ Planned |
| Phase 5 | BindingDB expansion (viral/non-kinase targets) | ⏳ Planned |

**Phase 3 detail — FoldSeek 3Di integration:**
```
AlphaFold PDB (Tool 2, already working)
      ↓
FoldSeek → per-residue 3Di tokens
      ↓
aa_to_sa(): "M#E#T#..." → "MaEvKc..."
      ↓
Re-embed DAVIS/KIBA proteins → retrain MLP head (~60s)
      ↓
Compare r: '#' baseline vs real 3Di
```
Expected outcome: performance improvement consistent with SaProt paper findings, or a concrete ablation result that quantifies the structural token contribution.

---

## Documentation

| Document | Contents |
|----------|----------|
| [Phase 1 Training Report](docs/PHASE1_TRAINING_EXPERIMENTS.md) | DAVIS/KIBA results, quantization comparison, key findings |
| [Phase 1 Pipeline Design](docs/PHASE1_REFERENCE_PIPELINE.md) | Why SaProt over DeepPurpose, architecture rationale |
| [Phase 1 Experiment Log](docs/PHASE1_EXPERIMENT_LOG.md) | V1→V3 iteration history, failure analysis |
| [Phase 2 Agent Tools](docs/PHASE2_AGENT_TOOLS.md) | Tool 2–5 implementation details and test results |
