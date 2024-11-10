# %%
import dask.dataframe as dd
import pandas as pd
import numpy as np
from dask.distributed import Client

pd.set_option("display.max_columns", None)  # 显示所有列
pd.set_option("display.max_rows", None)  # 显示所有行
pd.set_option("display.max_colwidth", None)  # 显示所有列宽

client = Client("tcp://127.0.0.1:8786")
TRAIN_FILE = "ggg_sg.csv"

# 加载数据
df = dd.read_csv(TRAIN_FILE, usecols=["ContextualText", "DocTone"]).dropna()

# %%
# 计算 33 分位数和 66 分位数
quantiles = df["DocTone"].quantile([0.33, 0.66]).compute()
q33, q66 = quantiles[0.33], quantiles[0.66]
print(f"33rd percentile: {q33}, 66th percentile: {q66}")


# %%
# 定义分类函数
def label_doctone(value):
    if value <= q33:
        return 0  # 低情绪
    elif value <= q66:
        return 1  # 中等情绪
    else:
        return 2  # 高情绪


# 应用分类函数，生成新的 Label 列
df["Label"] = df["DocTone"].map_partitions(lambda col: col.apply(label_doctone), meta=('Label', 'int64'))

# %%
import openai

# 假设嵌入维度为 384
embedding_dim = 384
embedding_columns = [f"embedding_{i}" for i in range(embedding_dim)]

def get_embedding(texts):
    client = openai.Client(api_key="magic", base_url="http://localhost:9997/v1")
    response = client.embeddings.create(model="bge-small-en-v1.5", input=texts)
    return np.vstack([embedding_obj.embedding for embedding_obj in response.data])


# 定义一个函数，用于将嵌入生成的结果添加到 DataFrame
def apply_embeddings_partition(df: pd.DataFrame):
    # 获取嵌入并生成 DataFrame
    embeddings = get_embedding(df["ContextualText"].tolist())
    
    embeddings_df = pd.DataFrame(embeddings, index=df.index)
    return pd.concat([df, embeddings_df], axis=1)

# 定义 meta 信息，用于明确输出的列结构和数据类型
meta = {col: "float64" for col in embedding_columns}
meta.update({"ContextualText": "object", "DocTone": "float64"})

df = df.map_partitions(apply_embeddings_partition, meta=meta)

# %%

import dask.dataframe as dd
import openai
import numpy as np
from dask_ml.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


df = df.persist()

# 特征和标签
X = df[[f"embedding_{i}" for i in range(384)]]
y = df["Label"]

# 训练集和测试集分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练随机森林
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 模型评估
accuracy = clf.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.2f}")
# %%
