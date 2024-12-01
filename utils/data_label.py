import openai
import pandas as pd
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 设置 OpenAI 的 API 基础和自定义服务器
client = openai.Client(api_key="not empty", base_url="http://localhost:9997/v1")
# 读取CSV文件

file_path = "./ggg_sg_10k_train.csv"
df = pd.read_csv(file_path, nrows=3000)


# 定义一个函数来调用 OpenAI API，获取所需信息
def get_label(text):
    try:
        response = client.chat.completions.create(
            model="qwen2.5-instruct",
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant that analyzes media text about Singapore.",
                },
                {
                    "role": "user",
                    "content": (
                        f"Analyze the text to identify the main Singapore-related entity, sentiment, summary, and topic classification. You should give ONE main Singapore-related entity mentioned in the text, Sentiment classification towards the entity (positive, neutral, negative), A one-sentence summary of how the text describes the entity, The topic classification of the text (e.g., politics, economy, market, etc.). Give your answer with the form of "
                        f"\n\n###"
                        f"entity: xxx,\n"
                        f"sentiment: xxx,\n"
                        f"summary: xxx,\n"
                        f"topic: xxx\n"
                        f"###\n\n"
                        f"Please analyze the following text: {text}"
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=400,
        )

        # 获取模型的回复
        content = response.choices[0].message.content
        # content = response.choices[0].function.arguments
        # print(response)
        # 尝试将回复解析为JSON

        response = client.chat.completions.create(
            model="qwen2.5-instruct",
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant that converts infomation to JSON.",
                },
                {
                    "role": "user",
                    "content": f"Here is the info:\n\n{content}\n\n Please convert it to JSON format. Don't output ANY OTHER characters.",
                },
            ],
            max_tokens=400,
            temperature=0.0,
        )

        content = response.choices[0].message.content
        content = content.replace("```json\n", "").replace("\n```", "").replace("\\\\", "\\")
        result = json.loads(content)
        # print(response)
        return result

    except json.JSONDecodeError:
        # 如果无法解析为JSON，则返回空的结果，并记录原始输出
        print(response)
        return {
            "entity": "",
            "sentiment": "",
            "summary": "",
            "topic": "",
            "raw_output": content,
        }
    except Exception as e:
        print(f"Error: {e}")
        return None


# 并行处理函数
def process_row(row):
    return get_label(row["ContextualText"])


# 使用 ThreadPoolExecutor 进行并行处理，并显示进度条
def parallel_processing(df, max_workers=15):
    results = []

    # 使用 ThreadPoolExecutor 进行并行化
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_row, row): index for index, row in df.iterrows()
        }

        # 使用 tqdm 进度条监控
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Processing rows",
            unit="row",
        ):
            index = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error processing row {index}: {e}")
                results.append(None)

    return results


# 使用并行处理函数处理 DataFrame
df["Analysis"] = parallel_processing(df)

# 将解析的结果转换为独立的列
df["Entity"] = df["Analysis"].apply(lambda x: x.get("entity", "") if x else "")
df["Sentiment"] = df["Analysis"].apply(lambda x: x.get("sentiment", "") if x else "")
df["Summary"] = df["Analysis"].apply(lambda x: x.get("summary", "") if x else "")
df["Topic"] = df["Analysis"].apply(lambda x: x.get("topic", "") if x else "")
df["RawOutput"] = df["Analysis"].apply(lambda x: x.get("raw_output", "") if x else "")

# 保存带有标签的新 CSV 文件
df.to_csv("ggg_sg_labeled_3k.csv", index=False)
