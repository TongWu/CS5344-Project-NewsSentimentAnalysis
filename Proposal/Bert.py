#%%
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd


# 1. 加载数据
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long),
        }


# 2. 加载 BERT 模型和 tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

# 数据
file_path = "../ggg_sg.csv"
df = pd.read_csv(file_path, nrows=10000)
selected_cols = ["Title", "DateTime", "DocTone", "ContextualText"]
df_cleaned = df[selected_cols].dropna()

def sentiment_label(score):
    if score > 1.9910:
        return 2
    elif score < -2.0202:
        return 0
    else:
        return 1

df_cleaned["label"] = df_cleaned['DocTone'].apply(sentiment_label)

texts = df_cleaned["ContextualText"].tolist()
labels = df_cleaned["label"].tolist()

# 3. 数据集和数据加载器
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)
train_dataset = CustomDataset(X_train, y_train, tokenizer, max_len=512)
test_dataset = CustomDataset(X_test, y_test, tokenizer, max_len=512)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

#%%
# 4. 训练 BERT 模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
epochs = 3

import time

for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    # 记录 epoch 开始时间
    epoch_start_time = time.time()
    
    for batch_idx, batch in enumerate(train_loader):
        batch_start_time = time.time()  # 记录每个 batch 的开始时间
        
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        total_predictions += labels.size(0)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # 记录 batch 处理时间
        batch_end_time = time.time()
        batch_time = batch_end_time - batch_start_time

        # 输出每个 batch 的处理时间日志
        print(f"Batch {batch_idx + 1}/{len(train_loader)} processed in {batch_time:.2f} seconds.")

    # 记录 epoch 结束时间并计算耗时
    epoch_end_time = time.time()
    epoch_time = epoch_end_time - epoch_start_time

    # 输出每个 epoch 的耗时及准确率
    print(
        f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}, "
        f"Accuracy: {correct_predictions.double()/total_predictions:.4f}, "
        f"Epoch time: {epoch_time:.2f} seconds"
    )


# %%
# 5. 在测试集上评估模型
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        _, preds = torch.max(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {accuracy}")
print(classification_report(all_labels, all_preds))

# %%
