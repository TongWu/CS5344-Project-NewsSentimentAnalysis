# %% 导入库和配置
import spacy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from config import *

# 加载语言模型
nlp = spacy.load("en_core_web_sm")

# 设置pandas显示选项
pd.set_option("display.max_columns", None)  # 显示所有列
pd.set_option("display.max_rows", None)  # 显示所有行
pd.set_option("display.max_colwidth", None)  # 显示所有列宽


# %% 数据加载与初步预处理
def load_and_clean_data(file_path):
    """加载CSV数据，保留所需列并去除缺失值"""
    df = pd.read_csv(file_path)
    df = df[
        ["Title", "DateTime", "ContextualText", "DomainCountryCode", "DocTone"]
    ].dropna()
    return df


df = load_and_clean_data(TRAIN_FILE)


# %% 数据过滤
def filter_countries_by_threshold(df, threshold=42):
    """过滤掉条目数少于指定阈值的国家代码"""
    country_counts = df.groupby("DomainCountryCode").size()
    valid_countries = country_counts[country_counts > threshold].index
    return df[df["DomainCountryCode"].isin(valid_countries)]


df = filter_countries_by_threshold(df)


# %% 分组统计分析
def calculate_avg_doctone(df):
    """计算每个国家代码的平均DocTone值"""
    return (
        df.groupby("DomainCountryCode")["DocTone"].mean().sort_values(ascending=False)
    )


def get_country_counts(df):
    """获取国家代码的条目数"""
    return df.groupby("DomainCountryCode").size().reindex(avg_doctone.index)


avg_doctone = calculate_avg_doctone(df)
country_code_counts = get_country_counts(df)


# %% 可视化国家代码的DocTone分布
def plot_doctone_distribution(avg_doctone, country_code_counts):
    """绘制按国家代码的DocTone平均值分布图"""
    plt.figure(figsize=(14, 8))
    sns.barplot(x=avg_doctone.index, y=avg_doctone.values, palette="viridis")

    # 添加标签显示平均值和条目数
    for index, value in enumerate(avg_doctone.values):
        plt.text(
            index,
            value,
            f"{value:.2f}\n({country_code_counts.iloc[index]})",
            ha="center",
            va="bottom",
        )

    plt.title("Average DocTone by Country Code")
    plt.xlabel("Country Code")
    plt.ylabel("Average DocTone")
    plt.show()


plot_doctone_distribution(avg_doctone, country_code_counts)


# %% 分析目标国家的DocTone年度变化
def plot_doctone_change_over_years(df, target_countries):
    """绘制目标国家的DocTone年度变化曲线"""
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df_target = df[df["DomainCountryCode"].isin(target_countries)]
    df_target["Year"] = df_target["DateTime"].dt.year

    # 按年和国家代码分组，计算平均DocTone
    avg_doctone_yearly = (
        df_target.groupby(["Year", "DomainCountryCode"])["DocTone"].mean().reset_index()
    )

    plt.figure(figsize=(14, 8))
    sns.lineplot(
        data=avg_doctone_yearly,
        x="Year",
        y="DocTone",
        hue="DomainCountryCode",
        marker="o",
    )

    plt.title("Average DocTone Change Over the Years by Country")
    plt.xlabel("Year")
    plt.ylabel("Average DocTone")
    plt.legend(title="Country Code")
    plt.show()


# 定义目标国家
target_countries = ["US", "SN", "CH", "MY", "IN"]
plot_doctone_change_over_years(df, target_countries)


# %% 分析目标国家的DocTone月度变化
def plot_doctone_change_over_months(df, target_countries, window=3):
    """绘制目标国家的DocTone月度变化曲线"""
    # 转换日期格式为"Year-Month"
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df["YearMonth"] = df["DateTime"].dt.to_period("M").astype(str)

    # 筛选目标国家的数据
    df_target = df[df["DomainCountryCode"].isin(target_countries)]

    # 按"YearMonth"和国家代码分组，计算平均DocTone
    avg_doctone_monthly = (
        df_target.groupby(["YearMonth", "DomainCountryCode"])["DocTone"]
        .mean()
        .reset_index()
    )
    smoothed_data = avg_doctone_monthly.groupby('DomainCountryCode').apply(
        lambda x: x.set_index('YearMonth')['DocTone'].rolling(window=window, min_periods=1).mean()
    ).reset_index()
    
    # 绘制DocTone月度变化曲线
    plt.figure(figsize=(14, 8))
    sns.lineplot(
        data=smoothed_data,
        x="YearMonth",
        y="DocTone",
        hue="DomainCountryCode",
        marker="o",
    )

    plt.title("Average DocTone Change Over the Months by Country")
    plt.xlabel("Year-Month")
    plt.ylabel("Average DocTone")
    plt.xticks(rotation=45)
    plt.legend(title="Country Code")
    plt.show()


# 定义目标国家
target_countries = ["SN", "CH", "MY"]
plot_doctone_change_over_months(df, target_countries)


