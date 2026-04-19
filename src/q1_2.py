"""
问题1-2：研究九种体质对发病风险的贡献度差异

方法：
  - 以九种体质积分作为自变量(X)，高血脂症二分类标签作为因变量(Y)
  - 构建多元逻辑回归（Logistic Regression）模型
  - 观察回归系数（Coefficient）、优势比（Odds Ratio, OR值）、P值
  - 按回归系数绝对值排序，系数越大说明该体质对高血脂的贡献度越高
  - 注意：本题数据下九种体质均未达P<0.05，贡献度排序仍具有参考意义
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import random
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

random.seed(42)
np.random.seed(42)
os.environ['PYTHONHASHSEED'] = '42'

# ==================== 字体与路径设置 ====================
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'output', 'q1_2')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 数据加载 ====================
df = pd.read_csv(os.path.join(DATA_DIR, 'preprocessed_data.csv'), index_col=0)
print(f"数据维度: {df.shape}")

# ==================== 变量定义 ====================
# 九种体质积分作为自变量
constitution_features = [
    '平和质', '气虚质', '阳虚质', '阴虚质',
    '痰湿质', '湿热质', '血瘀质', '气郁质', '特禀质'
]

X = df[constitution_features]
y = df['高血脂症二分类标签']

print(f"自变量: {len(constitution_features)}种体质")
print(f"因变量: 高血脂症二分类标签 (1={sum(y==1)}, 0={sum(y==0)})")
print(f"正样本比例: {y.mean():.2%}")


# ======================================================================
# Part 1: 多重共线性检验 —— 方差膨胀因子（VIF）
# ======================================================================
print("\n" + "=" * 70)
print("Part 1: 多重共线性检验 —— 方差膨胀因子（VIF）")
print("=" * 70)

X_with_const = sm.add_constant(X)
vif_data = pd.DataFrame({
    '特征': constitution_features,
    'VIF': [variance_inflation_factor(X_with_const.values, i + 1)
            for i in range(len(constitution_features))]
})
print(vif_data.to_string(index=False))
print(f"\nVIF > 10 的特征（存在严重多重共线性）: "
      f"{(vif_data['VIF'] > 10).sum()}个")
if (vif_data['VIF'] > 10).any():
    print("  注意: 存在多重共线性，但Logistic回归仍可估计方向和显著性，结果需谨慎解读")


# ======================================================================
# Part 2: 多元逻辑回归模型
# ======================================================================
print("\n" + "=" * 70)
print("Part 2: 多元逻辑回归模型")
print("=" * 70)

# 添加常数项（截距）
X_const = sm.add_constant(X)

# 拟合Logistic回归模型
logit_model = sm.Logit(y, X_const)
logit_result = logit_model.fit(disp=1, maxiter=1000)

print("\n模型摘要:")
print(logit_result.summary())

# ======================================================================
# Part 3: 提取回归系数、OR值、P值、置信区间
# ======================================================================
print("\n" + "=" * 70)
print("Part 3: 回归系数、优势比（OR值）与统计显著性")
print("=" * 70)

# 提取结果（去除截距）
params = logit_result.params.drop('const')
conf_int = logit_result.conf_int().drop('const')
pvalues = logit_result.pvalues.drop('const')

# 计算OR值及置信区间
or_values = np.exp(params)
or_ci_lower = np.exp(conf_int[0])
or_ci_upper = np.exp(conf_int[1])

# 构建结果表
result_df = pd.DataFrame({
    '体质': params.index,
    '回归系数(Coef)': params.values,
    'OR值': or_values.values,
    'OR_95%CI_下限': or_ci_lower.values,
    'OR_95%CI_上限': or_ci_upper.values,
    'P值': pvalues.values
})

# 按回归系数绝对值降序排列（贡献度排序）
result_df = result_df.sort_values('回归系数(Coef)', key=abs, ascending=False)
result_df.index = range(1, len(result_df) + 1)

print("\n九种体质对高血脂发病风险的贡献度排序:")
print("-" * 100)
print(f"{'体质':<8} {'系数':>10} {'OR值':>8} {'95%CI':>22} {'P值':>10} {'方向':>8}")
print("-" * 100)
for _, row in result_df.iterrows():
    ci_str = f"[{row['OR_95%CI_下限']:.3f}, {row['OR_95%CI_上限']:.3f}]"
    direction = "风险" if row['回归系数(Coef)'] > 0 else "保护"
    print(f"{row['体质']:<8} {row['回归系数(Coef)']:>10.4f} {row['OR值']:>8.3f} "
          f"{ci_str:>22} {row['P值']:>10.4f} {direction:>8}")


# ======================================================================
# Part 4: 模型评估
# ======================================================================
print("\n" + "=" * 70)
print("Part 4: 模型评估")
print("=" * 70)

# 伪R²
print(f"McFadden伪R²: {logit_result.prsquared:.4f}")
print(f"对数似然值: {logit_result.llf:.4f}")
print(f"AIC: {logit_result.aic:.4f}")
print(f"BIC: {logit_result.bic:.4f}")

# 预测与混淆矩阵
y_pred_prob = logit_result.predict(X_const)
y_pred = (y_pred_prob >= 0.5).astype(int)

from sklearn.metrics import (roc_auc_score, accuracy_score, confusion_matrix, classification_report)

auc = roc_auc_score(y, y_pred_prob)
acc = accuracy_score(y, y_pred)
cm = confusion_matrix(y, y_pred)

print(f"\nAUC: {auc:.4f}")
print(f"准确率: {acc:.4f}")
print(f"\n混淆矩阵:")
print(f"          预测=0  预测=1")
print(f"实际=0    {cm[0,0]:>5d}   {cm[0,1]:>5d}")
print(f"实际=1    {cm[1,0]:>5d}   {cm[1,1]:>5d}")

print(f"\n分类报告:")
print(classification_report(y, y_pred, target_names=['正常', '高血脂'], zero_division=0))


# ======================================================================
# Part 5: 绘图
# ======================================================================

# ---------- 图1: 森林图（Forest Plot）—— OR值及置信区间 ----------
fig, ax = plt.subplots(figsize=(12, 7))

# 按OR值排序
plot_df = result_df.sort_values('OR值')
y_pos = range(len(plot_df))

# 绘制置信区间（按系数方向着色：正向=红，负向=蓝）
for i, (_, row) in enumerate(plot_df.iterrows()):
    color = '#e74c3c' if row['回归系数(Coef)'] > 0 else '#3498db'
    ax.plot([row['OR_95%CI_下限'], row['OR_95%CI_上限']], [i, i],
            color=color, linewidth=2, alpha=0.7)
    ax.scatter(row['OR值'], i, color=color, marker='D', s=80, zorder=3)

# 参考线 OR=1
ax.axvline(x=1, color='black', linestyle='--', linewidth=1, alpha=0.5)

ax.set_yticks(y_pos)
ax.set_yticklabels(plot_df['体质'])
ax.set_xlabel('优势比 (Odds Ratio)', fontsize=12)
ax.set_title('九种体质对高血脂发病风险的贡献度（森林图）', fontsize=14)

# 图例
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='D', color='#e74c3c', label='正向（风险因素）',
           markersize=8, linestyle='None'),
    Line2D([0], [0], marker='D', color='#3498db', label='负向（保护因素）',
           markersize=8, linestyle='None'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'forest_plot.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"\n  -> 图表已保存: output/q1_2/forest_plot.png")


# ---------- 图2: 回归系数柱状图 ----------
fig, ax = plt.subplots(figsize=(12, 7))

plot_df2 = result_df.sort_values('回归系数(Coef)')
colors = ['#e74c3c' if c > 0 else '#3498db'
          for c in plot_df2['回归系数(Coef)']]

bars = ax.barh(range(len(plot_df2)), plot_df2['回归系数(Coef)'], color=colors, alpha=0.85)
ax.set_yticks(range(len(plot_df2)))
ax.set_yticklabels(plot_df2['体质'])
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax.set_xlabel('回归系数 (Coefficient)', fontsize=12)
ax.set_title('九种体质Logistic回归系数对比', fontsize=14)

# 在柱子上标注P值
for i, (_, row) in enumerate(plot_df2.iterrows()):
    ax.text(row['回归系数(Coef)'], i,
            f" P={row['P值']:.3f}", va='center', fontsize=12)

legend_patches = [
    matplotlib.patches.Patch(facecolor='#e74c3c', label='正向（风险因素）'),
    matplotlib.patches.Patch(facecolor='#3498db', label='负向（保护因素）'),
]
ax.legend(handles=legend_patches, loc='lower right', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'coefficient_bar.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"  -> 图表已保存: output/q1_2/coefficient_bar.png")


# ---------- 图3: ROC曲线 ----------
fig, ax = plt.subplots(figsize=(8, 8))

from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y, y_pred_prob)
ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'Logistic回归 (AUC = {auc:.4f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
ax.set_xlabel('假阳性率 (FPR)', fontsize=12)
ax.set_ylabel('真阳性率 (TPR)', fontsize=12)
ax.set_title('ROC曲线 —— 九种体质预测高血脂', fontsize=14)
ax.legend(fontsize=12, loc='lower right')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curve.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"  -> 图表已保存: output/q1_2/roc_curve.png")


# ======================================================================
# 保存结果到CSV
# ======================================================================
result_df.to_csv(os.path.join(OUTPUT_DIR, 'logistic_regression_results.csv'),
                 index=False, encoding='utf-8-sig')

# 保存完整模型摘要
with open(os.path.join(OUTPUT_DIR, 'summary.txt'), 'w', encoding='utf-8') as f:
    f.write("问题1-2：九种体质对高血脂发病风险的贡献度差异 —— 结果摘要\n")
    f.write("=" * 70 + "\n\n")

    # 变量信息
    f.write("一、变量信息\n")
    f.write("-" * 50 + "\n")
    f.write(f"自变量: {len(constitution_features)}种体质积分\n")
    f.write(f"因变量: 高血脂症二分类标签 (1={sum(y==1)}, 0={sum(y==0)})\n")
    f.write(f"正样本比例: {y.mean():.2%}\n\n")

    # VIF
    f.write("二、多重共线性检验（VIF）\n")
    f.write("-" * 50 + "\n")
    f.write(vif_data.to_string(index=False) + "\n")
    f.write(f"VIF > 10 的特征: {(vif_data['VIF'] > 10).sum()}个\n")
    if (vif_data['VIF'] > 10).any():
        f.write("注意: 存在多重共线性，结果需谨慎解读\n")
    f.write("\n")

    # 模型摘要
    f.write("三、多元逻辑回归模型\n")
    f.write("-" * 50 + "\n")
    f.write(logit_result.summary().as_text())
    f.write("\n\n")

    # 回归系数、OR值
    f.write("四、回归系数、优势比（OR值）与统计显著性\n")
    f.write("-" * 80 + "\n")
    f.write(f"{'体质':<8} {'系数':>10} {'OR值':>8} {'95%CI':>22} "
            f"{'P值':>10} {'方向':>8}\n")
    f.write("-" * 80 + "\n")
    for _, row in result_df.iterrows():
        ci_str = f"[{row['OR_95%CI_下限']:.3f}, {row['OR_95%CI_上限']:.3f}]"
        direction = "风险" if row['回归系数(Coef)'] > 0 else "保护"
        f.write(f"{row['体质']:<8} {row['回归系数(Coef)']:>10.4f} "
                f"{row['OR值']:>8.3f} {ci_str:>22} "
                f"{row['P值']:>10.4f} {direction:>8}\n")
    f.write("\n")

    # 模型评估
    f.write("五、模型评估\n")
    f.write("-" * 50 + "\n")
    f.write(f"McFadden伪R²: {logit_result.prsquared:.4f}\n")
    f.write(f"对数似然值: {logit_result.llf:.4f}\n")
    f.write(f"AIC: {logit_result.aic:.4f}\n")
    f.write(f"BIC: {logit_result.bic:.4f}\n")
    f.write(f"AUC: {auc:.4f}\n")
    f.write(f"准确率: {acc:.4f}\n\n")

    f.write("混淆矩阵:\n")
    f.write(f"          预测=0  预测=1\n")
    f.write(f"实际=0    {cm[0,0]:>5d}   {cm[0,1]:>5d}\n")
    f.write(f"实际=1    {cm[1,0]:>5d}   {cm[1,1]:>5d}\n\n")

    # 最终结论
    f.write("六、最终结论\n")
    f.write("-" * 50 + "\n")
    f.write("按贡献度（回归系数绝对值）排序:\n")
    for i, (_, row) in enumerate(result_df.iterrows(), 1):
        direction = "风险" if row['回归系数(Coef)'] > 0 else "保护"
        f.write(f"  {i}. {row['体质']}: Coef={row['回归系数(Coef)']:.4f}, "
                f"OR={row['OR值']:.3f}, P={row['P值']:.4f} —— {direction}因素\n")
    f.write(f"\n注：九种体质均未达P<0.05统计显著水平，"
            f"但贡献度排序仍可反映体质对高血脂风险的影响方向与相对大小。\n")

print(f"\n  -> 结果已保存: output/q1_2/logistic_regression_results.csv")
print(f"  -> 模型摘要已保存: output/q1_2/summary.txt")


# ======================================================================
# 最终总结
# ======================================================================
print("\n" + "=" * 70)
print("最终总结：九种体质对高血脂发病风险的贡献度差异")
print("=" * 70)

print(f"\n按贡献度（回归系数绝对值）排序:")
for i, (_, row) in enumerate(result_df.iterrows(), 1):
    direction = "风险" if row['回归系数(Coef)'] > 0 else "保护"
    pct = (row['OR值'] - 1) * 100 if row['OR值'] > 1 else (1 - row['OR值']) * 100
    print(f"  {i}. {row['体质']}: Coef={row['回归系数(Coef)']:.4f}, "
          f"OR={row['OR值']:.3f}, P={row['P值']:.4f} —— {direction}因素")

print(f"\n注：九种体质均未达P<0.05统计显著水平，"
      f"但贡献度排序仍可反映体质对高血脂风险的影响方向与相对大小。")

print(f"\n所有结果已保存至 output/q1_2/ 目录")
