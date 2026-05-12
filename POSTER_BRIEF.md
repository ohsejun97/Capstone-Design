# 포스터 제작 지시서 — Bio-AI Agent System for DTI Analysis

> 이 파일은 AI 도구(Gemini / Claude / GPT)를 활용해 학술 포스터를 제작하기 위한 지시서입니다.
> 포스터 성격: **연구 제안서 스타일** (계획 및 핵심 결과 강조, 구현 완성도 언급 최소화)

---

## 포스터 기본 정보

| 항목 | 내용 |
|------|------|
| 제목 | Bio-AI Agent System for Drug–Target Interaction Analysis |
| 부제 | Natural Language Interface for Binding Affinity Prediction via Structure-Aware Language Models |
| 성격 | 캡스톤 디자인 학술 포스터 (연구 제안 + 핵심 결과) |
| 크기 | A0 세로형 (841 × 1189 mm) 권장 |
| 언어 | 영어 (학술 포스터 표준) |
| 컬러 | 딥 네이비(#1a237e) + 틸(#00838f) + 화이트 + 연한 그레이 |
| 느낌 | 바이오인포매틱스 / AI 연구 / 미래지향적, 깔끔한 그리드 레이아웃 |

---

## 전체 레이아웃 구성 (위→아래, 좌→우)

```
┌─────────────────────────────────────────────────────────┐
│                    HEADER (제목 + 소속)                   │
├───────────────┬────────────────────────┬────────────────┤
│  MOTIVATION   │   SYSTEM ARCHITECTURE  │  DTI RESULTS   │
│  (문제 정의)   │      (전체 흐름도)       │  (모델 성능)    │
├───────────────┴────────────────────────┴────────────────┤
│               DTI MODEL DESIGN (핵심 기술 설명)            │
├─────────────────────────┬───────────────────────────────┤
│   AGENT PIPELINE        │   TECHNOLOGY STACK            │
│   (5 Tools 구성)         │   + DATASET                   │
├─────────────────────────┴───────────────────────────────┤
│                    CONCLUSION / FUTURE WORK              │
└─────────────────────────────────────────────────────────┘
```

---

## 각 섹션 상세 내용

---

### 1. HEADER

**제목 (크고 굵게):**
> Bio-AI Agent System for Drug–Target Interaction Analysis

**부제 (조금 작게):**
> Natural Language Interface for Binding Affinity Prediction via Structure-Aware Language Models

**소속/과목 정보** (하단 작게):
> Capstone Design Project · Department of [학과명]

**그래픽 요소:**
- 헤더 배경: 딥 네이비 그라디언트
- 오른쪽 상단에 DNA 이중나선 + 분자 구조 아이콘 (심볼 수준)

---

### 2. MOTIVATION — Why This Matters?

**키 메시지:** 신약 개발은 느리고 비싸며 전문가만 접근 가능하다.

**포함할 내용:**
- Drug discovery 평균 소요: **10~15년, 10억 달러+**
- DTI(Drug-Target Interaction) 예측이 초기 스크리닝에서 핵심
- 기존 접근의 문제: 전문가만 사용 가능 (UniProt ID, SMILES 문법 등 필요)
- **제안:** 자연어로 물어보면 AI가 모든 걸 처리하는 에이전트 시스템

**대화 예시 박스 (눈에 띄게):**
```
User: "Does Imatinib bind to BCR-ABL kinase?"
  ↓
System: "Predicted pKd = 8.7 — Strong binding (Kd ≤ 100 nM)"
         + 3D structure + ligand conformation
```

**시각 요소:** 아이콘 3개 나란히 (❓자연어 → 🤖 AI Agent → 💊 결과)

---

### 3. SYSTEM ARCHITECTURE — Full Pipeline

**제목:** End-to-End Natural Language DTI Query Pipeline

**다이어그램 (핵심, 꼭 그려야 함):**

```
User Natural Language Query
          │
          ▼
  ┌──────────────────┐
  │  LLM-based Agent │   ← smolagents (ReAct paradigm)
  │ (Orchestrator)   │
  └───┬──────────┬───┘
      │          │
      ▼          ▼
 ┌─────────┐  ┌──────────┐
 │Drug Name│  │Protein   │
 │Resolver │  │Resolver  │
 │(PubChem)│  │(UniProt) │
 └────┬────┘  └────┬─────┘
   SMILES      Sequence
      │          │
      └────┬─────┘
           ▼
   ┌───────────────────┐
   │  DTI Prediction   │  ← Core Module
   │  SaProt + ChemBERTa│    Pearson r = 0.8923
   └─────────┬─────────┘
             │
      ┌──────┴──────┐
      ▼             ▼
 ┌─────────┐  ┌──────────┐
 │ Ligand  │  │ Protein  │
 │Structure│  │Structure │
 │ (RDKit) │  │(AlphaFold│
 └─────────┘  └──────────┘
             │
             ▼
    Structured Answer
    (pKd + 3D context)
```

**디자인 팁:** 박스들을 색으로 구분 — 에이전트(네이비), DTI 모델(틸, 강조), 구조 도구(그레이)

---

### 4. DTI RESULTS — Model Performance

**제목:** DTI Model Validation Across Datasets

**강조 숫자 카드 3개 (크게):**

| BindingDB | DAVIS | KIBA |
|-----------|-------|------|
| **r = 0.8923** | **r = 0.8677** | **r = 0.8594** |
| 80K pairs | 30K pairs | 118K pairs |
| 32K drugs | 68 drugs | 2,068 drugs |

**SOTA 비교 테이블:**

| Model | DAVIS Pearson r | VRAM |
|-------|-----------------|------|
| DeepPurpose (2020) | ~0.89 | >16 GB |
| ConPLex (2023) | ~0.90 | ~8 GB |
| **Ours** | **0.8677** | **4 GB** |

**메시지:** SOTA 대비 4 GB VRAM(소비자급 GPU)으로 격차 0.02~0.03 수준

**시각 요소 (선택):** 3개 데이터셋 bar chart (Pearson r 비교)

---

### 5. DTI MODEL DESIGN — Core Technical Contribution

**제목:** Structure-Aware DTI Encoder with Task-Adapted Drug Representation

**두 파트로 나눠 설명:**

#### A. Architecture 다이어그램

```
SMILES string
    │
    ▼
ChemBERTa (fine-tuned, layers 4–5)
  ← Pretrained on PubChem (44M params)
  ← Upper layers adapted to binding affinity task
    │  [768-dim]
    └──────────────────────┐
                           ▼
                    ┌─────────────┐
                    │  MLP Head   │ ──→ pKd
                    └─────────────┘
                           ▲
                    ┌──────┘
                    │  [1280-dim]
Amino Acid Sequence
    │
    ▼ FoldSeek 3Di tokens
    ▼
SaProt-650M (frozen, FP16)
  ← Structure-aware Protein LM
  ← 3Di structural tokens from AlphaFold
```

#### B. Key Design Choices (텍스트 박스 또는 bullet)

- **SaProt frozen**: 650M params를 4 GB VRAM에서 fine-tuning 불가 → general protein representation 그대로 활용
- **FoldSeek 3Di tokens**: 아미노산 서열만 사용 시 대비 DAVIS Pearson r **+0.023** 향상 (0.808 → 0.808→ 0.808)
- **ChemBERTa fine-tuning**: MLM 사전학습 목적(분자 구조 복원)과 pKd 예측 목적 불일치 → 상위 2개 레이어 DTI 태스크로 적응, **+0.0186** 향상
- **BindingDB (80K pairs)**: DAVIS 68개 약물로는 ChemBERTa/GNN 학습 불가 → 32K 약물 다양성 확보

---

### 6. AGENT PIPELINE — 5 Modular Tools

**제목:** Modular Tool-Based Agent Architecture

**5개 Tool 카드 (가로 나열):**

| # | Tool | Input | Output | Backend |
|---|------|-------|--------|---------|
| 1 | **DTI Prediction** | SMILES + AA seq | pKd score | SaProt + ChemBERTa ft |
| 2 | **Protein Structure** | UniProt ID | 3D PDB + pLDDT | AlphaFold DB API |
| 3 | **Ligand Structure** | SMILES | 3D SDF + properties | RDKit ETKDGv3 |
| 4 | **Drug Name Resolver** | Drug name | SMILES | PubChem REST API |
| 5 | **Protein Name Resolver** | Gene/protein name | UniProt ID + sequence | UniProt Search API |

**아래 설명:**
> The LLM agent (smolagents, ReAct paradigm) dynamically selects and chains these tools based on the user query — no manual pipeline specification required.

---

### 7. TECHNOLOGY STACK + DATASET

**두 컬럼으로 구성:**

**왼쪽 — Tech Stack:**
```
Protein Encoder  : SaProt-650M-AF2 (ESM-2 기반, FP16)
Drug Encoder     : ChemBERTa-zinc-base (RoBERTa 기반)
Structural Tokens: FoldSeek 3Di via AlphaFold DB
Agent Framework  : smolagents (Hugging Face, ReAct)
DTI Head         : PyTorch MLP (~1.5M params)
Protein Structure: AlphaFold DB (EBI REST API)
Ligand 3D        : RDKit ETKDGv3 + MMFF94
Hardware         : GTX 1650 SUPER (4 GB VRAM), WSL2
```

**오른쪽 — Datasets:**

| Dataset | Pairs | Drugs | Proteins | Label |
|---------|-------|-------|----------|-------|
| BindingDB | 80,795 | 32,480 | 2,384 | pKd |
| DAVIS | 30,056 | 68 | 442 | pKd |
| KIBA | 118,254 | 2,068 | 229 | KIBA score |

---

### 8. CONCLUSION / FUTURE WORK

**Conclusion (2~3 줄):**
> We propose a Bio-AI Agent System that answers natural language queries about drug-target binding affinity without requiring expert knowledge. The core DTI model achieves DAVIS r = 0.8677 and KIBA r = 0.8594 using a 4 GB consumer GPU — competitive with SOTA methods requiring >8 GB VRAM.

**Future Work (bullet):**
- Full agent orchestration: connecting all 5 tools under smolagents ReAct
- End-to-end demo: natural language query → structured binding affinity report
- Cold-split evaluation for generalization to unseen drug scaffolds
- SaProt fine-tuning with larger GPU resources

---

## 시각화 지시사항 (AI 이미지 생성 요청용)

아래 그림들을 AI 이미지 생성 도구에 요청하거나 직접 제작:

### 그림 1 — System Overview (가장 중요)
**요청 프롬프트:**
> "Academic poster diagram of a Bio-AI agent pipeline. A user sends a natural language question to an LLM orchestrator (labeled 'Agent'). The agent calls 5 tools: Drug Name Resolver (PubChem), Protein Name Resolver (UniProt), DTI Prediction (central, highlighted in teal), Ligand Structure (RDKit), Protein Structure (AlphaFold). Arrows show data flow. Clean, dark navy and teal color scheme, white boxes, sans-serif font. Research poster style."

### 그림 2 — DTI Model Architecture
**요청 프롬프트:**
> "Diagram of a drug-target interaction prediction neural network. Left side: SMILES string feeds into 'ChemBERTa (fine-tuned)' transformer block producing a 768-dim vector. Right side: amino acid sequence with colored 3Di structural tokens feeds into 'SaProt-650M (frozen)' protein language model producing a 1280-dim vector. Both vectors merge into an MLP head outputting 'pKd (binding affinity)'. Style: clean, academic, dark navy and teal, white background."

### 그림 3 — 성능 비교 Bar Chart
**요청 프롬프트 (또는 직접 제작):**
> Three grouped bars for BindingDB (0.8923), DAVIS (0.8677), KIBA (0.8594). Y-axis: Pearson r (0.8 to 0.95). Color: teal gradient. Add horizontal dashed line at 0.89 labeled 'SOTA (DeepPurpose)'. Clean academic style.

### 그림 4 — 3Di Token 효과 비교 (선택)
**간단한 테이블 또는 small bar chart:**
- Placeholder '#' → r = 0.786
- FoldSeek 3Di → r = 0.808 (+0.023)

---

## AI 포스터 생성 요청 프롬프트 (전체)

아래를 Gemini / Claude / GPT에 그대로 붙여넣어 포스터 레이아웃 초안을 요청:

```
Create an academic research poster (A0, portrait) for the following capstone design project.

Title: "Bio-AI Agent System for Drug–Target Interaction Analysis"
Subtitle: "Natural Language Interface for Binding Affinity Prediction via Structure-Aware Language Models"

Style: professional academic poster, dark navy (#1a237e) and teal (#00838f) color scheme, white background sections, clean grid layout.

The poster should include these sections in order:

1. MOTIVATION: Drug discovery costs $1B+ and 10–15 years. Current DTI tools require expert knowledge (SMILES, UniProt IDs). Proposed solution: natural language query → AI agent → binding affinity prediction.

2. SYSTEM ARCHITECTURE: A pipeline diagram where a user's natural language question is processed by an LLM-based ReAct agent (smolagents). The agent orchestrates 5 tools: Drug Name Resolver (PubChem), Protein Name Resolver (UniProt), DTI Prediction Tool (core, highlighted), Ligand Structure Tool (RDKit), Protein Structure Tool (AlphaFold).

3. DTI MODEL RESULTS: Three stat cards showing Pearson r = 0.8923 (BindingDB), 0.8677 (DAVIS), 0.8594 (KIBA). SOTA comparison table: our model achieves competitive results with only 4 GB VRAM vs SOTA requiring >8–16 GB.

4. DTI MODEL DESIGN: Architecture diagram — ChemBERTa (fine-tuned upper layers, 768-dim) + SaProt-650M (frozen FP16, 1280-dim with FoldSeek 3Di structural tokens) → MLP Head → pKd. Key design choices: SaProt frozen (4GB VRAM constraint), 3Di tokens (+0.023 Pearson r), ChemBERTa fine-tuning (+0.019 Pearson r), BindingDB 80K pairs.

5. AGENT TOOLS TABLE: 5 tools with input/output/backend.

6. CONCLUSION: Competitive DTI performance at 4 GB VRAM; future work: full agent demo, cold-split evaluation.

Use academic poster conventions. Include clear section headers, bullet points, and data visualizations where appropriate.
```

---

## 핵심 수치 (포스터 어디든 활용)

| 지표 | 값 |
|------|----|
| BindingDB Pearson r | **0.8923** |
| DAVIS Pearson r | **0.8677** |
| KIBA Pearson r | **0.8594** |
| SOTA 대비 VRAM | **4 GB vs >8–16 GB** |
| BindingDB 학습 데이터 | **80,795 쌍, 32,480 약물** |
| 3Di 토큰 효과 | **+0.023 Pearson r** |
| ChemBERTa ft 효과 | **+0.019 Pearson r** |
| DTI Head 파라미터 | **~1.5M params** |
| Agent Tools | **5개** |
