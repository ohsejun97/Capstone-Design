# Commands & Frameworks Reference

## 환경

| 항목 | 값 |
|------|-----|
| 로컬 | WSL2, GTX 1650 SUPER 4GB, conda env `bioinfo` |
| 서버 | ABCLab `/disk2/biosj/`, conda env `osj_2026`, 500GB RAM |
| 서버 작업 디렉토리 | `/disk2/biosj/Capstone_Design/bindingdb_work/` |

---

## 로컬 — 학습 실행

```bash
conda activate bioinfo

# DAVIS (기준선)
python train_dti_saprot.py --dataset davis --encoder 650M --use_3di

# KIBA
python train_dti_saprot.py --dataset kiba --encoder 650M --use_3di

# BindingDB — drug encoder 변경 실험
python train_dti_saprot.py --dataset bindingdb --encoder 650M --use_3di --drug_encoder morgan
python train_dti_saprot.py --dataset bindingdb --encoder 650M --use_3di --drug_encoder chemberta
python train_dti_saprot.py --dataset bindingdb --encoder 650M --use_3di --drug_encoder gnn

# DAVIS + BindingDB 혼합
python train_dti_saprot.py --dataset davis+bindingdb --encoder 650M --use_3di --drug_encoder gnn
```

### 주요 인자

| 인자 | 선택지 | 설명 |
|------|--------|------|
| `--dataset` | davis / kiba / bindingdb / davis+bindingdb | 학습 데이터 |
| `--encoder` | 650M / 35M | SaProt 모델 크기 |
| `--quant` | none / 8bit / 4bit | 양자화 |
| `--use_3di` | flag | FoldSeek 3Di 구조 토큰 사용 |
| `--drug_encoder` | morgan / gnn / chemberta | Drug encoder 종류 |
| `--epochs` | int (default 50) | 학습 에포크 수 |
| `--patience` | int (default 10) | Early stopping patience |

---

## 로컬 — 3Di 캐시 빌드

```bash
# DAVIS (379개, 약 2~3시간)
python scripts/build_3di_cache.py --dataset davis --resume

# KIBA (229개)
python scripts/build_3di_cache.py --dataset kiba --resume

# BindingDB (2,384개) → 서버에서 실행 권장
python scripts/build_3di_cache.py --dataset bindingdb --resume
```

캐시 저장 위치: `cache/3di_tokens_{dataset}.json`

---

## 로컬 — 전처리 / 평가 / 시각화

```bash
# BindingDB 전처리 (서버에서 실행)
python scripts/preprocess_bindingdb.py --input ./BindingDB_All.tsv --output ./bindingdb_kd.csv

# 전체 모델 평가
python experiments/evaluate_results.py

# 시각화 (outputs/figures/ 생성)
python experiments/visualize_results.py
```

---

## 서버 (ABCLab) — BindingDB 작업

```bash
cd /disk2/biosj/Capstone_Design/bindingdb_work
export PATH=/disk2/biosj/Capstone_Design/foldseek/bin:$PATH

# 전처리 (uniprot_id 포함 버전)
python scripts/preprocess_bindingdb.py --input ./BindingDB_All.tsv --output ./bindingdb_kd.csv

# 3Di 캐시 빌드 (백그라운드)
nohup python scripts/build_3di_cache.py --dataset bindingdb --resume > 3di_build.log 2>&1 &
echo $!

# 진행 확인
tail -f 3di_build.log

# 프로세스 확인
ps aux | grep build_3di
```

### FoldSeek 설치 (서버, 1회)
```bash
cd ~
wget https://mmseqs.com/foldseek/foldseek-linux-avx2.tar.gz
tar -xzf foldseek-linux-avx2.tar.gz
echo 'export PATH=~/foldseek/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

---

## XFTP 파일 전송 정리

### 서버 → 로컬 (내려받기)
| 서버 | 로컬 |
|------|------|
| `bindingdb_work/bindingdb_kd.csv` | `data/BindingDB/bindingdb_kd.csv` |
| `bindingdb_work/cache/3di_tokens_bindingdb.json` | `cache/3di_tokens_bindingdb.json` |

### 로컬 → 서버 (올리기)
| 로컬 | 서버 |
|------|------|
| `scripts/preprocess_bindingdb.py` | `bindingdb_work/scripts/` |
| `scripts/build_3di_cache.py` | `bindingdb_work/scripts/` |
| `tools/alphafold_tool.py` | `bindingdb_work/tools/` |
| `tools/foldseek_tool.py` | `bindingdb_work/tools/` |

---

## 프레임워크 / 주요 라이브러리

| 라이브러리 | 용도 |
|-----------|------|
| `transformers` (HuggingFace) | SaProt-650M 로드 (`EsmModel`, `EsmTokenizer`) |
| `torch` (PyTorch) | 학습, 임베딩 캐시 |
| `bitsandbytes` | 4bit / 8bit 양자화 |
| `DeepPurpose` | DAVIS / KIBA 데이터 로드 |
| `rdkit` | Morgan Fingerprint, SMILES → 분자 그래프 |
| `FoldSeek` | 단백질 PDB → 3Di 토큰 추출 |
| `smolagents` | Phase 3 Agent 오케스트레이션 (예정) |
| `pandas` / `numpy` | 데이터 전처리 |
| `scipy` | Pearson r, CI 계산 |

---

## 핵심 모델 결과 (현재 기준선)

| 모델 | DAVIS r | KIBA r |
|------|---------|--------|
| SaProt-650M FP16 + 3Di + Morgan FP | **0.8082** | **0.8032** |
| SaProt-650M FP16 + 3Di + GNN (DAVIS 68약물) | 0.5795 | 0.7191 |
| SaProt-650M FP16 + 3Di + ChemBERTa mean (DAVIS) | 0.7915 | 0.7667 |
