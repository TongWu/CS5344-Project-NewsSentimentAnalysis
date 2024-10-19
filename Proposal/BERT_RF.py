#%%
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm  # 进度条

# 读取CSV文件的前1万行
file_path = "../ggg_sg.csv"
df = pd.read_csv(file_path, nrows=10000)

# 选择需要的列并清洗数据
selected_cols = ['Title', 'DateTime', 'DocTone', 'ContextualText']
df_cleaned = df[selected_cols].dropna()

# 生成情感标签
def sentiment_label(score):
    if score > 1.9910:
        return 2
    elif score < -2.0202:
        return 0
    else:
        return 1

df_cleaned["label"] = df_cleaned['DocTone'].apply(sentiment_label)

# Tokenizer and Model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').to('cuda')

texts = df_cleaned["ContextualText"].tolist()
labels = df_cleaned["label"].tolist()

# Tokenizer processing: Convert text to BERT input format (token ids, attention masks, etc.)
def tokenize_text(texts, tokenizer, max_length=512):  # Reduce max_length to save memory
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=max_length)
    return inputs['input_ids'], inputs['attention_mask']

# Tokenize all data
input_ids, attention_masks = tokenize_text(texts, tokenizer)

# Convert to TensorDataset
labels = torch.tensor(labels)
dataset = TensorDataset(input_ids, attention_masks, labels)

# Create DataLoader to batch the data
batch_size = 64  # Adjust batch size as needed to fit within GPU memory
dataloader = DataLoader(dataset, batch_size=batch_size)

# Collect embeddings from BERT in batches with progress bar
def get_bert_embeddings(dataloader, model):
    model.eval()
    all_embeddings = []
    total_batches = len(dataloader)  # 获取总批次数
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing Batches", total=total_batches, unit="batch"):
            input_ids_batch, attention_masks_batch, _ = [b.to('cuda') for b in batch]
            outputs = model(input_ids_batch, attention_mask=attention_masks_batch)
            # Get [CLS] token embeddings
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(embeddings)
    return np.vstack(all_embeddings)

# Get embeddings in batches
print("Starting to extract BERT embeddings...")
X = get_bert_embeddings(dataloader, model)
y = np.array(labels)

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 使用 RandomForestClassifier
print("Training Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)  # n_estimators 指定树的数量
rf_model.fit(X_train, y_train)


# Predict and evaluate the model
print("Evaluating model...")
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)

# %%
