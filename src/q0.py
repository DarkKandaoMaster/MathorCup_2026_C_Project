"""
数据预处理，把部分数据归一化
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

input_file = "../data/附件1：样例数据.xlsx"
raw_output_file = "../data/raw_data.csv"
output_file = "../data/preprocessed_data.csv"

# 1. 读取数据，然后立刻把读到的原始数据保存到本地（csv文件）
df = pd.read_excel(input_file)
df.to_csv(raw_output_file, index=False)

# 2. 定义各类型变量的列名
# 标签变量（Y）：不参与归一化
labels = ["高血脂症二分类标签", "血脂异常分型标签（确诊病例）"]

# 分类型变量：保持原样
categorical = ["性别", "吸烟史", "饮酒史", "年龄组", "体质标签"]

# 排除项：如样本ID，不具有特征意义
exclude = ["样本ID"]

# 连续型变量：剩余的所有列（血脂指标、空腹血糖、血尿酸、BMI、以及各项积分等）
continuous = [
    col for col in df.columns if col not in labels + categorical + exclude
]

# 3. 归一化
scaler = MinMaxScaler()  # Min-Max归一化（缩放到0-1）

# 4. 执行转换
df_preprocessed = df.copy()
df_preprocessed[continuous] = scaler.fit_transform(df[continuous])

# 导出到新的csv文件
df_preprocessed.to_csv(output_file, index=False)

# 5. 保存MinMaxScaler对象，供后续反归一化使用
scaler_output_file = "../data/minmax_scaler.pkl"
joblib.dump(scaler, scaler_output_file)
print(f"MinMaxScaler已保存至 {scaler_output_file}")
