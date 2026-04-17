"""
数据预处理，把部分数据归一化
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def preprocess_data(file_path):
    # 1. 读取数据
    df = pd.read_excel(file_path)

    # 2. 定义各类型变量的列名
    # 标签变量（Y）：不参与归一化
    labels = ["高血脂症二分类标签", "血脂异常分型标签（确诊病例）"]

    # 分类型变量：保持原样（根据数据实际情况，补充了年龄组和体质类别标签）
    categorical = ["性别", "吸烟史", "饮酒史", "年龄组", "体质标签"]

    # 排除项：如样本ID，不具有特征意义
    exclude = ["样本ID"]

    # 连续型变量：剩余的所有列（血脂指标、空腹血糖、血尿酸、BMI、以及各项积分等）
    continuous = [
        col for col in df.columns if col not in labels + categorical + exclude
    ]

    # 3. 归一化
    scaler = MinMaxScaler()  # Min-Max归一化 (缩放到0-1)

    # 4. 执行转换
    df_preprocessed = df.copy()
    df_preprocessed[continuous] = scaler.fit_transform(df[continuous])

    # 5. 返回并保存预处理后的数据
    return df_preprocessed


# ================= 使用示例 =================
if __name__ == "__main__":
    input_file = "../data/附件1：样例数据.xlsx"
    output_file = "../data/preprocessed_data.csv"

    # 调用函数（此处默认使用 z-score，若想用 Min-Max 改为 scaling_method='min-max'）
    df_clean = preprocess_data(input_file)

    # 导出到新的CSV文件
    df_clean.to_csv(output_file, index=False)
    print(f"数据预处理完成，已保存至：{output_file}")
