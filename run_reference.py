import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
import os

# [1] 서버 자원 최적화
torch.set_num_threads(32)
device = torch.device('cpu')
model_name = "westlake-repl/SaProt_650M_AF2"

# [2] 데이터 로드 (첫 번째 컬럼은 인덱스이므로 무시)
df = pd.read_csv("davis_test.csv", index_col=0)
print(f"Dataset Loaded: {len(df)} samples.")

# [3] 모델 및 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, low_cpu_mem_usage=True, num_labels=1
).to(device)

# [4] 추론 (Inference)
results = []
model.eval()
print("Starting 17-hour journey... Good luck, SeJun!")

with torch.no_grad():
    for i, row in tqdm(df.iterrows(), total=len(df)):
        try:
            # SaProt 전용 토크나이징 (SA-Token을 그대로 입력)
            inputs = tokenizer(row['Target Sequence'], row['SMILES'], 
                               return_tensors="pt", truncation=True, max_length=512).to(device)
            score = model(**inputs).logits.item()
            
            results.append({
                'smiles': row['SMILES'],
                'reference_score': score,
                'label': row['Label'] # 실제값과 나중에 비교하기 위해 저장
            })
        except Exception as e:
            continue

# [5] 결과 저장
pd.DataFrame(results).to_csv("reference_scores_osj.csv", index=False)
print(f"\n[DONE] Results saved to reference_scores_osj.csv")
