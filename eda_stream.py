# %% 导入库和配置
import dask.dataframe as dd
import matplotlib.pyplot as plt
import seaborn as sns
from dask.diagnostics import ProgressBar
from dask.distributed import Client
import pandas as pd
from config import *
import os

pd.set_option("display.max_columns", None)  # 显示所有列
pd.set_option("display.max_rows", None)  # 显示所有行
pd.set_option("display.max_colwidth", None)  # 显示所有列宽

client = Client("tcp://127.0.0.1:8786")
TRAIN_FILE = 'ggg_sg.csv'

# %% 数据加载与初步预处理
def load_and_clean_data(file_path):
    """加载CSV数据，保留所需列并去除缺失值"""
    df = dd.read_csv(file_path, usecols=["Title", "DateTime", "ContextualText", "DomainCountryCode", "DocTone"]).dropna()
    return df

df = load_and_clean_data(TRAIN_FILE)

#%%
# 查看每个国家的数据量
country_counts = df.groupby("DomainCountryCode").size().compute()
country_counts

#%%
# 按照数量排序
country_counts = country_counts.sort_values(ascending=False)
country_counts

# %% 数据过滤
def filter_countries_by_threshold(df, threshold=100000):
    """过滤掉条目数少于指定阈值的国家代码"""
    country_counts = df.groupby("DomainCountryCode").size()
    valid_countries = country_counts[country_counts > threshold].index.compute()
    return df[df["DomainCountryCode"].isin(valid_countries)]

with ProgressBar():
    df = filter_countries_by_threshold(df)

# %% 分组统计分析
def calculate_avg_doctone(df):
    """计算每个国家代码的平均DocTone值"""
    avg_doctone = df.groupby("DomainCountryCode")["DocTone"].mean().compute()
    return avg_doctone.sort_values(ascending=False)

def get_country_counts(df, avg_doctone):
    """获取国家代码的条目数"""
    country_counts = df.groupby("DomainCountryCode").size().compute()
    return country_counts.reindex(avg_doctone.index)

with ProgressBar():
    avg_doctone = calculate_avg_doctone(df)
    country_code_counts = get_country_counts(df, avg_doctone)

# %% 可视化国家代码的DocTone分布并保存到本地
def plot_doctone_distribution(avg_doctone, country_code_counts, filename="doctone_distribution.png"):
    """绘制按国家代码的DocTone平均值分布图并保存"""
    plt.figure(figsize=(14, 8))
    sns.barplot(x=avg_doctone.index, y=avg_doctone.values, palette="viridis")
    for index, value in enumerate(avg_doctone.values):
        plt.text(index, value, f"{value:.2f}\n({country_code_counts.iloc[index]})", ha="center", va="bottom")
    plt.title("Average DocTone by Country Code")
    plt.xlabel("Country Code")
    plt.ylabel("Average DocTone")
    plt.savefig(filename)  # 保存图片
    plt.close()

plot_doctone_distribution(avg_doctone, country_code_counts, 'graph/doctone_distribution.png')

# %% 分析目标国家的DocTone年度变化并保存到本地
def plot_doctone_change_over_years(df, target_countries, filename="doctone_yearly_change.png"):
    """绘制目标国家的DocTone年度变化曲线并保存"""
    df["DateTime"] = dd.to_datetime(df["DateTime"])
    df["Year"] = df["DateTime"].dt.year
    df_target = df[df["DomainCountryCode"].isin(target_countries)]

    avg_doctone_yearly = df_target.groupby(["Year", "DomainCountryCode"])["DocTone"].mean().compute().reset_index()
    
    plt.figure(figsize=(14, 8))
    sns.lineplot(data=avg_doctone_yearly, x="Year", y="DocTone", hue="DomainCountryCode", marker="o")
    plt.title("Average DocTone Change Over the Years by Country")
    plt.xlabel("Year")
    plt.ylabel("Average DocTone")
    plt.legend(title="Country Code")
    plt.savefig(filename)  # 保存图片
    plt.close()

target_countries = ["US", "SN", "CH", "MY", "IN"]
plot_doctone_change_over_years(df, target_countries, 'graph/doctone_yearly_change.png')

# %% 分析目标国家的DocTone月度变化并保存到本地
def plot_doctone_change_over_months(df, target_countries, window=6, filename="graph"):
    """绘制目标国家的DocTone月度变化曲线并保存"""
    df["DateTime"] = dd.to_datetime(df["DateTime"])
    df["YearMonth"] = df["DateTime"].dt.to_period("M").astype(str)
    df_target = df[df["DomainCountryCode"].isin(target_countries)]

    avg_doctone_monthly = df_target.groupby(["YearMonth", "DomainCountryCode"])["DocTone"].mean().compute().reset_index()
    smoothed_data = avg_doctone_monthly.groupby('DomainCountryCode').apply(
        lambda x: x.set_index('YearMonth')['DocTone'].rolling(window=window, min_periods=1).mean()
    ).reset_index()
    
    plt.figure(figsize=(14, 8))
    sns.lineplot(data=smoothed_data, x="YearMonth", y="DocTone", hue="DomainCountryCode", marker="o")
    plt.title("Average DocTone Change Over the Months by Country")
    plt.xlabel("Year-Month")
    plt.ylabel("Average DocTone")
    plt.xticks(rotation=45)
    plt.legend(title="Country Code")
    plt.savefig(filename)  # 保存图片
    plt.close()

target_countries = ["SN", "CH", "MY"]
plot_doctone_change_over_months(df, target_countries, filename='graph/doctone_monthly_change.png')


# %%
