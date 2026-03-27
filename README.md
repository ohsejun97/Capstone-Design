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

## Documentation

| Document | Contents |
|----------|----------|
| [Phase 1 Training Report](docs/PHASE1_TRAINING_EXPERIMENTS.md) | DAVIS/KIBA results, quantization comparison, key findings |
| [Phase 1 Pipeline Design](docs/PHASE1_REFERENCE_PIPELINE.md) | Why SaProt over DeepPurpose, architecture rationale |
| [Phase 1 Experiment Log](docs/PHASE1_EXPERIMENT_LOG.md) | V1→V3 iteration history, failure analysis |
| [Phase 2 Agent Tools](docs/PHASE2_AGENT_TOOLS.md) | Tool 2–5 implementation details and test results |
