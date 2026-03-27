# Phase 2 — Agent Tools 개발 일지

> **작성일시:** 2026년 3월 27일 KST
> **목표:** DTI 예측 Agent의 3개 Tool 구현 및 smolagents 오케스트레이션
> **현재 상태:** 🔄 진행 중 — Tool 1 (AlphaFold DB) 완료

---

## Agent 시스템 구조

```
User Query (자연어)
      ↓
Agent AI (smolagents + LLM)
      ↓
┌─────────────────────────────────┐
│ Tool 4: Drug Name Resolver  🔄  │ ← 약물 이름 → SMILES (PubChem API)
│ Tool 5: Protein Resolver    🔄  │ ← 단백질명 → UniProt ID + 서열 (UniProt API)
│ Tool 1: DTI Prediction      ✅  │ ← SaProt-650M-4bit + MLP (DAVIS/KIBA 학습 완료)
│ Tool 2: Protein Structure   ✅  │ ← AlphaFold DB API
│ Tool 3: Ligand Structure    ✅  │ ← RDKit 3D conformer
└─────────────────────────────────┘
      ↓
Final Response (pKd + 구조 정보)
```

---

## Tool 2: AlphaFold DB API Tool

> **파일:** `tools/alphafold_tool.py`
> **완료일:** 2026-03-27

### 기능

UniProt Accession ID를 입력받아 AlphaFold Protein Structure Database(EBI)에서
단백질 3D 구조(PDB)를 조회하고 로컬에 캐시합니다.

### 입출력

| 항목 | 내용 |
|------|------|
| **입력** | UniProt ID (예: `"P00533"`) |
| **출력** | gene, protein name, organism, seq_length, pLDDT, pdb_path |
| **캐시** | `cache/alphafold/{uniprot_id}.pdb` + `_meta.txt` |

### API

```
GET https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}
→ JSON 응답에서 pdbUrl 추출
→ PDB 파일 다운로드 및 로컬 저장
```

### 테스트 결과

```
$ python tools/alphafold_tool.py P00533

[AlphaFold Tool]
  Protein  : Epidermal growth factor receptor (EGFR)
  UniProt  : P00533  |  Entry: AF-P00533-F1
  Organism : Homo sapiens
  Length   : 1210 aa
  pLDDT    : 75.94 (global)  |  Very-high conf: 47.4%
  PDB      : cache/alphafold/P00533.pdb
```

**캐시 재호출:** API 재요청 없이 즉시 반환 확인 ✅
**에러 처리:** 존재하지 않는 ID(404), 네트워크 오류 모두 처리 ✅

### pLDDT 신뢰도 해석

| pLDDT 범위 | 의미 |
|-----------|------|
| > 90 | Very high confidence |
| 70~90 | Confident |
| 50~70 | Low confidence |
| < 50 | Very low (disordered region) |

EGFR(P00533) global pLDDT = 75.94 → Confident 수준

---

## Tool 1: DTI Prediction (Phase 1 완료)

> **파일:** `train_dti_saprot.py` (학습), `tools/dti_tool.py` (예정 — 추론 전용 래핑)

Phase 1에서 학습/검증 완료. Agent Tool로 사용하기 위한 추론 전용 래핑은
Tool 3 완료 후 Agent 오케스트레이션 단계에서 진행 예정.

| 모델 | DAVIS r | KIBA r |
|------|--------|--------|
| SaProt-650M-4bit | 0.7914 | 0.7994 |

---

## 다음 단계

### Tool 3: RDKit 3D Ligand Tool ✅ 완료

> **파일:** `tools/rdkit_tool.py` | **완료일:** 2026-03-27

| 항목 | 내용 |
|------|------|
| **입력** | SMILES 문자열 (예: `"CC(=O)Oc1ccccc1C(=O)O"`) |
| **처리** | ETKDGv3 embedding → MMFF94 force field 최적화 |
| **출력** | formula, mol_weight, n_atoms, n_bonds, n_rotatable_bonds, sdf_path |
| **캐시** | `cache/ligands/{smiles_hash}.sdf` |

**테스트 결과 (Aspirin):**
```
Formula : C9H8O4  |  MW: 180.159 g/mol
Atoms   : 21  |  Bonds: 21  |  Rotatable: 2
Status  : optimized
```
Cache hit/miss 모두 확인 ✅

### Tool 4: Drug Name Resolver (예정)

> **파일:** `tools/pubchem_tool.py`

| 항목 | 내용 |
|------|------|
| **입력** | 약물 이름 (예: `"Imatinib"`, `"아스피린"`) |
| **출력** | SMILES, CID, 분자식, MW |
| **API** | `pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/...` |

일반 사용자가 SMILES를 모르더라도 약물 이름으로 검색 가능하게 함.

### Tool 5: Protein Name Resolver (예정)

> **파일:** `tools/uniprot_tool.py`

| 항목 | 내용 |
|------|------|
| **입력** | 유전자명 또는 단백질명 (예: `"EGFR"`, `"BCR-ABL"`, `"COX-2"`) |
| **출력** | UniProt ID, AA sequence, organism |
| **API** | `rest.uniprot.org/uniprotkb/search?query=gene:{name}&organism_id=9606` |

전문가가 아니어도 단백질 이름으로 AlphaFold + DTI Tool 실행 가능하게 함.

**Tool 4+5 추가 시 가능해지는 쿼리 예시:**

| 쿼리 | Tool 4+5 없이 | Tool 4+5 있으면 |
|------|-------------|----------------|
| "이마티닙이 BCR-ABL에 결합하나요?" | ❌ | ✅ |
| "아스피린과 COX-2 결합력은?" | ❌ | ✅ |
| "게피티닙이 EGFR을 억제할 수 있나요?" | ❌ | ✅ |

### Agent 오케스트레이션 (예정)
- Framework: smolagents
- LLM: 미정 (로컬 경량 모델 또는 API)
- Tool 등록 → 자연어 쿼리 → Tool 선택 → 결과 종합

---

## 파일 구조

```
tools/
├── alphafold_tool.py   ✅  Tool 2: AlphaFold DB API
├── rdkit_tool.py       ✅  Tool 3: RDKit 3D Ligand
├── pubchem_tool.py     🔄  Tool 4: Drug Name Resolver (예정)
├── uniprot_tool.py     🔄  Tool 5: Protein Name Resolver (예정)
└── dti_tool.py         🔄  Tool 1: DTI 추론 래핑 (예정)

cache/
├── alphafold/
│   ├── P00533.pdb
│   └── P00533_meta.txt
└── ligands/
    └── {smiles_hash}.sdf
```
