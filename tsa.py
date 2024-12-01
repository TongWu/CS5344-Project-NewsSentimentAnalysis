# %%
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd
import seaborn as sns
from config import *

pd.set_option("display.max_columns", None)  # 显示所有列
pd.set_option("display.max_rows", None)  # 显示所有行
pd.set_option("display.max_colwidth", None)  # 显示所有列宽


# 文本数据：contextual context中包含多个句子

df = pd.read_csv(TRAIN_FILE)
df = df[["Title", "DateTime", "ContextualText", "DomainCountryCode", "DocTone"]]
df = df.dropna()
df.head(20)
from sentence_transformers import SentenceTransformer, util

# 加载模型并指定使用 GPU（如果可用）
model = SentenceTransformer("paraphrase-MiniLM-L6-v2", device="cuda")

# 定义参考句子并嵌入
reference_phrases = {
    "positive": "speaking positive to singapore",
    "negative": "speaking negative to singapore",
    "neutral": "speaking neutral to singapore",
}
reference_embeddings = {
    key: model.encode(value) for key, value in reference_phrases.items()
}

# 嵌入 ContextualText 列的所有文本
contextual_embeddings = model.encode(
    df["ContextualText"].tolist(), batch_size=256, show_progress_bar=True
)


# 定义分类函数，计算与参考句子的相似度
def classify_singapore_sentiment(embedding):
    similarities = {
        key: util.cos_sim(embedding, embedding_ref).item()
        for key, embedding_ref in reference_embeddings.items()
    }
    return max(similarities, key=similarities.get)


# 批量计算相似度并分类
df["SingaporeSentiment"] = [
    classify_singapore_sentiment(embedding)
    for embedding in tqdm(contextual_embeddings, desc="Classifying", ncols=100)
]

# 查看结果
df[["Title", "DateTime", "ContextualText", "SingaporeSentiment"]]


# %%
# 查看结果
df[["ContextualText", "SingaporeSentiment", "DocTone"]].head(20)

# DocTone是个数值，我们也把它划分成正面、负面和中性，以便和SingaporeSentiment进行比较, 这里需要测试不同的阈值，使得和SingaporeSentiment的结果尽量一致


# %%
import numpy as np
from sklearn.metrics import accuracy_score


# 定义 DocTone 划分函数
def classify_doc_tone(doc_tone, pos_threshold, neg_threshold):
    if doc_tone >= pos_threshold:
        return "positive"
    elif doc_tone <= neg_threshold:
        return "negative"
    else:
        return "neutral"


# 计算单个阈值组合的匹配率
def calculate_accuracy_for_thresholds(
    pos_threshold,
    neg_threshold,
    df,
    singapore_col="SingaporeSentiment",
    doc_tone_col="DocTone",
):
    df["DocToneSentiment"] = df[doc_tone_col].apply(
        lambda x: classify_doc_tone(x, pos_threshold, neg_threshold)
    )
    accuracy = accuracy_score(df[singapore_col], df["DocToneSentiment"])
    return pos_threshold, neg_threshold, accuracy


# 并行计算所有阈值组合的匹配率
def find_best_threshold_parallel(
    df, singapore_col="SingaporeSentiment", doc_tone_col="DocTone"
):
    pos_range = np.arange(-2, 5.1, 0.5)
    neg_range = np.arange(-5.0, 2, 0.5)

    results = Parallel(n_jobs=-1)(
        delayed(calculate_accuracy_for_thresholds)(
            pos, neg, df, singapore_col, doc_tone_col
        )
        for pos in pos_range
        for neg in neg_range
    )

    # 找到匹配率最高的阈值组合
    best_threshold = max(results, key=lambda x: x[2])
    best_pos_threshold, best_neg_threshold, best_accuracy = best_threshold
    print(
        f"最佳阈值组合：正面阈值 = {best_pos_threshold}, 负面阈值 = {best_neg_threshold}，匹配率 = {best_accuracy:.4f}"
    )
    df["DocToneSentiment"] = df[doc_tone_col].apply(
        lambda x: classify_doc_tone(x, best_pos_threshold, best_neg_threshold)
    )

    # 返回完整的结果表用于分析
    return pd.DataFrame(
        results, columns=["PositiveThreshold", "NegativeThreshold", "Accuracy"]
    )


# 运行阈值优化
results_df = find_best_threshold_parallel(df)

# 查看最佳阈值组合及对应的匹配率
results_df.sort_values(by="Accuracy", ascending=False).head()

# %%

# 绘制DocTone的分布，根据SingaporeSentiment分类
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
for sentiment in df["SingaporeSentiment"].unique():
    subset = df[df["SingaporeSentiment"] == sentiment]
    plt.hist(subset["DocTone"], bins=30, alpha=0.6, label=f"{sentiment}")

plt.xlabel("DocTone")
plt.ylabel("Frequency")
plt.title("DocTone Distribution by SingaporeSentiment")
plt.legend(title="SingaporeSentiment")
plt.show()

# %%
# 挑出几个分类错误的样本进行分析
df_errors = df[df["SingaporeSentiment"] != df["DocToneSentiment"]]
df_errors[["ContextualText", "SingaporeSentiment", "DocTone", "DocToneSentiment"]].head(
    20
)
