"""
run_baseline_deeppurpose.py
===========================
DeepPurpose MPNN_CNN_DAVIS 사전학습 모델로 DAVIS 연속 pKd 기준값 생성

- 데이터: DeepPurpose 내장 DAVIS (연속 pKd, 30,056 샘플)
- 모델: MPNN_CNN_DAVIS (DeepPurpose 공식 사전학습 가중치)
- 지표: Pearson r  ← 이진 레이블이 아닌 연속값이므로 유효

역할:
  이 스크립트의 결과(Pearson r ≈ 0.88)가
  이후 SaProt-35M + 4-bit 양자화 모델과 비교할 '기준선'이 됩니다.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from datetime import datetime

# ── DeepPurpose ──────────────────────────────────────────────────────────────
from DeepPurpose.dataset import load_process_DAVIS, create_fold, data_process
from DeepPurpose import DTI as models

print("=" * 60)
print("  DeepPurpose MPNN_CNN_DAVIS — Reference Baseline")
print(f"  실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60, "\n")

# ══════════════════════════════════════════════════════════════════════════════
# [1] 데이터 로드 (연속 pKd, log 변환)
# ══════════════════════════════════════════════════════════════════════════════
print("[1] DAVIS 데이터 로드 (연속 pKd)...")
X_drugs, X_targets, y = load_process_DAVIS(
    path='./data',
    binary=False,
    convert_to_log=True,
    threshold=30,
)
print(f"    전체 샘플: {len(y):,}개")
print(f"    pKd 범위: {min(y):.2f} ~ {max(y):.2f}")
print(f"    pKd 평균: {np.mean(y):.2f}\n")

# ══════════════════════════════════════════════════════════════════════════════
# [2] 사전학습 모델 로드
# ══════════════════════════════════════════════════════════════════════════════
print("[2] MPNN_CNN_DAVIS 사전학습 모델 로드...")
net = models.model_pretrained(model='MPNN_CNN_DAVIS')
print("    ✅ 로드 완료\n")

# ══════════════════════════════════════════════════════════════════════════════
# [3] 전체 데이터 예측
# ══════════════════════════════════════════════════════════════════════════════
print("[3] 데이터 전처리 및 예측 중...")
drug_encoding  = net.drug_encoding
target_encoding = net.target_encoding

train_df, _, _ = data_process(
    X_drug=X_drugs,
    X_target=X_targets,
    y=y,
    drug_encoding=drug_encoding,
    target_encoding=target_encoding,
    split_method='random',
    frac=[1.0, 0, 0],   # 전체를 test set으로 사용
)

y_pred = net.predict(train_df)
print(f"    예측 완료: {len(y_pred):,}개\n")

# ══════════════════════════════════════════════════════════════════════════════
# [4] Pearson r 계산 및 저장
# ══════════════════════════════════════════════════════════════════════════════
r, p_val = pearsonr(y_pred, y)

df_out = pd.DataFrame({
    'smiles':     X_drugs,
    'target_seq': X_targets,
    'y_pred':     y_pred,
    'y_true':     y,
})
df_out.to_csv("baseline_deeppurpose_davis.csv", index=False)

print("[4] 결과 저장 → baseline_deeppurpose_davis.csv")
print(f"    샘플 수: {len(y_pred):,}개\n")

print("=" * 50)
print(f"  Pearson R : {r:+.4f}")
print(f"  p-value   : {p_val:.2e}")
if r >= 0.8:
    print("  ✅ 기준값 확보 완료! (r ≥ 0.8)")
print("=" * 50)
print(f"\n  → 이후 SaProt-35M 경량 모델의 목표: r ≥ {r * 0.9:.3f}")
print(f"    (기준값의 90% 이상 유지 = 경량화 성공)")
print("\n[완료]")
