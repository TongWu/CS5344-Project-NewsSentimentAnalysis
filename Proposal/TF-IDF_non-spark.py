# %%
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split

# 读取CSV文件的前1万行
file_path = "../ggg_sg.csv"
df = pd.read_csv(file_path, nrows=10000)

# 选择需要的列并清洗数据
selected_cols = ["Title", "DateTime", "DocTone", "ContextualText"]
df_cleaned = df[selected_cols].dropna()


# 生成情感标签
def sentiment_label(score):
    if score > 1.9910:
        return 2
    elif score < -2.0202:
        return 0
    else:
        return 1


df_cleaned["label"] = df_cleaned["DocTone"].apply(sentiment_label)

# 初始化HashingVectorizer和SGDClassifier（用于逻辑回归）
vectorizer = HashingVectorizer(n_features=2**15)
model = SGDClassifier(loss="log_loss", n_jobs=-1)  # 使用随机梯度下降并模拟逻辑回归

# 提取文本特征和标签
X = vectorizer.fit_transform(df_cleaned["ContextualText"])
y = df_cleaned["label"].values

# 切分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 模型训练
model.fit(X_train, y_train)

# 预测并评估模型
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)

# %%
