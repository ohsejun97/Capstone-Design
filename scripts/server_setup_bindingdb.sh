#!/bin/bash
# server_setup_bindingdb.sh
# ==========================
# 서버(500GB RAM)에서 실행 — BindingDB 다운로드 → 압축 해제 → 전처리
#
# 사용법:
#   bash server_setup_bindingdb.sh
#
# 결과물:
#   ./bindingdb_kd.csv  (smiles, sequence, pkd 컬럼)
#   → 로컬 data/BindingDB/bindingdb_kd.csv 에 복사

set -e

WORKDIR="$(pwd)"
ZIP_FILE="BindingDB_All_202604_tsv.zip"
TSV_FILE="BindingDB_All.tsv"
OUT_FILE="bindingdb_kd.csv"

# Step 1: 다운로드 (이미 있으면 스킵)
if [ ! -f "$ZIP_FILE" ]; then
    echo "[1] Downloading BindingDB (525MB)..."
    wget "https://www.bindingdb.org/rwd/bind/downloads/BindingDB_All_202604_tsv.zip"
else
    echo "[1] $ZIP_FILE already exists, skipping download."
fi

# Step 2: 압축 해제 (이미 있으면 스킵)
if [ ! -f "$TSV_FILE" ]; then
    echo "[2] Extracting zip..."
    python -c "from zipfile import ZipFile; ZipFile('$ZIP_FILE').extractall('.')"
else
    echo "[2] $TSV_FILE already exists, skipping extraction."
fi

# Step 3: 전처리
echo "[3] Running preprocessing..."
python preprocess_bindingdb.py \
    --input  "./$TSV_FILE" \
    --output "./$OUT_FILE"

echo ""
echo "Done! Output: $WORKDIR/$OUT_FILE"
echo "XFTP로 이 파일을 로컬 data/BindingDB/bindingdb_kd.csv 에 복사하세요."
