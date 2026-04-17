"""
问题1-1：从血常规体检指标、中老年人活动量表评分中，
       筛选能有效表征痰湿体质严重程度、且能预警高血脂发病风险的关键指标

方法：
  - 痰湿积分（回归）：LASSO回归，L1正则化将不重要特征系数压缩为0
  - 发病风险（分类）：随机森林分类，特征重要性评分排序
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

from sklearn.linear_model import LassoCV, lasso_path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

# ==================== 字体与路径设置 ====================
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'output', 'q1_1')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 数据加载 ====================
df = pd.read_csv(os.path.join(DATA_DIR, 'preprocessed_data.csv'), index_col=0)
print(f"数据维度: {df.shape}")

# ==================== 特征定义 ====================
# 血常规体检指标
blood_indicators = [
    'HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）',
    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI'
]

# 中老年人活动量表评分
activity_scores = [
    'ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
    'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
    '活动量表总分（ADL总分+IADL总分）'
]

# 人口学控制变量
demographics = ['年龄组', '性别', '吸烟史', '饮酒史']

# 待筛选的特征（核心）
screening_features = blood_indicators + activity_scores
# 全部特征（含控制变量）
all_features = screening_features + demographics

# 目标变量
y_reg = df['痰湿质']
y_cls = df['高血脂症二分类标签']
X = df[all_features]

print(f"待筛选指标: {len(screening_features)}个 (血常规{len(blood_indicators)} + 活动量表{len(activity_scores)})")
print(f"控制变量: {len(demographics)}个")
print(f"回归目标(痰湿质): mean={y_reg.mean():.4f}, std={y_reg.std():.4f}")
print(f"分类目标(高血脂): 正={sum(y_cls==1)}, 负={sum(y_cls==0)}")


# ======================================================================
# Part 1: LASSO回归 —— 痰湿体质严重程度关键指标筛选
# ======================================================================
print("\n" + "=" * 70)
print("Part 1: LASSO回归 —— 痰湿体质严重程度关键指标筛选")
print("=" * 70)

# 5折交叉验证选择最优正则化参数alpha
lasso_cv = LassoCV(cv=5, random_state=42, alphas=100, max_iter=10000)
lasso_cv.fit(X, y_reg)

print(f"最优alpha: {lasso_cv.alpha_:.6f}")
print(f"交叉验证MSE: {lasso_cv.mse_path_.mean():.6f}")

# 非零系数 = 被选中的关键指标
lasso_coefs = pd.Series(lasso_cv.coef_, index=all_features)
selected_lasso = lasso_coefs[lasso_coefs != 0].sort_values(key=abs, ascending=False)

print(f"\nLASSO筛选出 {len(selected_lasso)} 个非零系数特征:")
for feat, coef in selected_lasso.items():
    cat = "血常规" if feat in blood_indicators else \
          "活动量表" if feat in activity_scores else "控制变量"
    direction = "正" if coef > 0 else "负"
    print(f"  {feat} ({cat}): {direction}向, 系数 = {coef:.6f}")


# ---------- 绘图: LASSO结果 ----------
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# (1a) LASSO非零系数柱状图
ax1 = axes[0]
color_map = {'血常规': '#e74c3c', '活动量表': '#3498db', '控制变量': '#95a5a6'}
bar_colors = [color_map['血常规' if f in blood_indicators else
                         '活动量表' if f in activity_scores else '控制变量']
              for f in selected_lasso.index]
selected_lasso.plot(kind='barh', color=bar_colors, ax=ax1)
ax1.set_title('LASSO回归非零系数特征（痰湿体质严重程度）', fontsize=14)
ax1.set_xlabel('系数值')
ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
from matplotlib.patches import Patch
legend_patches = [Patch(facecolor='#e74c3c', label='血常规指标'),
                  Patch(facecolor='#3498db', label='活动量表评分'),
                  Patch(facecolor='#95a5a6', label='控制变量')]
ax1.legend(handles=legend_patches, loc='lower right')

# (1b) LASSO正则化路径
ax2 = axes[1]
alphas_path, coefs_path, _ = lasso_path(X, y_reg, n_alphas=100, max_iter=10000)
for i, feat in enumerate(all_features):
    color = '#e74c3c' if feat in blood_indicators else \
            '#3498db' if feat in activity_scores else '#95a5a6'
    ax2.plot(-np.log10(alphas_path), coefs_path[i], label=feat, color=color, alpha=0.7)
ax2.axvline(x=-np.log10(lasso_cv.alpha_), color='red', linestyle='--',
            label=f'最优α={lasso_cv.alpha_:.4f}')
ax2.set_title('LASSO正则化路径', fontsize=14)
ax2.set_xlabel('-log₁₀(α)')
ax2.set_ylabel('系数值')
ax2.legend(fontsize=6, ncol=2, loc='best')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'lasso_results.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"  → 图表已保存: output/q1_1/lasso_results.png")


# ======================================================================
# Part 2: 随机森林分类 —— 高血脂发病风险关键指标筛选
# ======================================================================
print("\n" + "=" * 70)
print("Part 2: 随机森林分类 —— 高血脂发病风险关键指标筛选")
print("=" * 70)

# class_weight='balanced' 处理类别不平衡 (793 vs 207)
rf = RandomForestClassifier(
    n_estimators=500, random_state=42,
    class_weight='balanced', n_jobs=-1
)

# 5折分层交叉验证评估
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_auc = cross_val_score(rf, X, y_cls, cv=cv, scoring='roc_auc')
print(f"5折交叉验证AUC: {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")

# 训练最终模型
rf.fit(X, y_cls)

# 特征重要性（Mean Decrease in Impurity）
rf_importances = pd.Series(rf.feature_importances_, index=all_features)
rf_importances = rf_importances.sort_values(ascending=False)

print(f"\n随机森林特征重要性排序:")
for i, (feat, imp) in enumerate(rf_importances.items(), 1):
    cat = "血常规" if feat in blood_indicators else \
          "活动量表" if feat in activity_scores else "控制变量"
    print(f"  [{i:2d}] {feat} ({cat}): 重要性 = {imp:.4f}")


# ---------- 绘图: RF结果 ----------
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# (2a) 特征重要性柱状图
ax1 = axes[0]
bar_colors_rf = [color_map['血常规' if f in blood_indicators else
                            '活动量表' if f in activity_scores else '控制变量']
                 for f in rf_importances.index]
rf_importances.plot(kind='barh', color=bar_colors_rf, ax=ax1)
ax1.set_title('随机森林特征重要性（高血脂发病风险预警）', fontsize=14)
ax1.set_xlabel('特征重要性')
ax1.legend(handles=legend_patches, loc='lower right')

# (2b) 累积重要性
ax2 = axes[1]
cumulative = rf_importances.cumsum()
ax2.plot(range(1, len(cumulative) + 1), cumulative.values, 'bo-', markersize=5)
ax2.axhline(y=0.80, color='red', linestyle='--', alpha=0.7, label='80%累积阈值')
ax2.axhline(y=0.95, color='green', linestyle='--', alpha=0.7, label='95%累积阈值')
ax2.set_title('特征累积重要性', fontsize=14)
ax2.set_xlabel('特征数量（按重要性降序）')
ax2.set_ylabel('累积重要性')
ax2.set_xticks(range(1, len(cumulative) + 1))
ax2.set_xticklabels(rf_importances.index, rotation=60, ha='right', fontsize=7)
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'rf_results.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"  → 图表已保存: output/q1_1/rf_results.png")


# ======================================================================
# Part 3: 综合分析 —— 两种方法交叉验证
# ======================================================================
print("\n" + "=" * 70)
print("Part 3: 综合分析 —— 同时表征痰湿体质 & 预警高血脂的关键指标")
print("=" * 70)

# 构建综合评价表
combined = pd.DataFrame(index=all_features)
combined['类别'] = ['血常规' if f in blood_indicators else
                     '活动量表' if f in activity_scores else '控制变量'
                     for f in all_features]

# LASSO归一化重要性（绝对值归一化到0-1）
lasso_abs = lasso_coefs.abs()
combined['LASSO系数(绝对值)'] = lasso_abs
combined['LASSO归一化'] = lasso_abs / lasso_abs.sum()

# RF归一化重要性
combined['RF重要性'] = rf_importances
combined['RF归一化'] = rf_importances / rf_importances.sum()

# 综合得分 = 两种方法归一化重要性之和
combined['综合得分'] = combined['LASSO归一化'] + combined['RF归一化']
combined = combined.sort_values('综合得分', ascending=False)

# 只看待筛选指标（去除控制变量）
screening_result = combined[combined['类别'] != '控制变量']

print("\n待筛选指标综合排序:")
for i, (feat, row) in enumerate(screening_result.iterrows(), 1):
    lasso_mark = "✓" if row['LASSO系数(绝对值)'] > 0 else "✗"
    print(f"  [{i:2d}] {feat} ({row['类别']}): "
          f"综合={row['综合得分']:.4f}, "
          f"LASSO={lasso_mark}(归一化{row['LASSO归一化']:.4f}), "
          f"RF归一化={row['RF归一化']:.4f}")


# ---------- 绘图: 综合分析 ----------
fig, ax = plt.subplots(figsize=(14, 8))
key_data = screening_result.copy()
x = np.arange(len(key_data))
width = 0.35

bars1 = ax.bar(x - width/2, key_data['LASSO归一化'], width,
               label='LASSO归一化重要性', color='#e74c3c', alpha=0.8)
bars2 = ax.bar(x + width/2, key_data['RF归一化重要性'], width,
               label='随机森林归一化重要性', color='#3498db', alpha=0.8)

ax.set_title('关键指标综合分析（LASSO + 随机森林）', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(key_data.index, rotation=55, ha='right', fontsize=9)
ax.set_ylabel('归一化重要性')
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'combined_results.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"\n  → 图表已保存: output/q1_1/combined_results.png")


# ======================================================================
# 保存所有结果到CSV
# ======================================================================
# LASSO系数表
lasso_df = pd.DataFrame({
    '特征': all_features,
    '类别': combined.loc[all_features, '类别'].values,
    'LASSO系数': lasso_cv.coef_,
    'LASSO系数(绝对值)': np.abs(lasso_cv.coef_),
    '是否入选': lasso_cv.coef_ != 0
})
lasso_df.to_csv(os.path.join(OUTPUT_DIR, 'lasso_coefficients.csv'),
                index=False, encoding='utf-8-sig')

# RF重要性表
rf_df = pd.DataFrame({
    '特征': rf_importances.index,
    '类别': combined.loc[rf_importances.index, '类别'].values,
    'RF重要性': rf_importances.values,
    'RF排名': range(1, len(rf_importances) + 1)
})
rf_df.to_csv(os.path.join(OUTPUT_DIR, 'rf_feature_importance.csv'),
             index=False, encoding='utf-8-sig')

# 综合分析表
combined.to_csv(os.path.join(OUTPUT_DIR, 'combined_analysis.csv'),
                encoding='utf-8-sig')


# ======================================================================
# 最终总结
# ======================================================================
print("\n" + "=" * 70)
print("★ 最终总结：筛选出的关键指标")
print("=" * 70)

# LASSO选出的（仅血常规+活动量表）
lasso_key = [f for f in selected_lasso.index if f in screening_features]
# RF前N名（仅血常规+活动量表）
rf_screening = [f for f in rf_importances.index if f in screening_features]
rf_top_n = rf_screening[:10]
# 两种方法共同选中
both_key = [f for f in lasso_key if f in rf_top_n]

print(f"\n【LASSO筛选】痰湿体质关键指标（{len(lasso_key)}个）:")
for f in lasso_key:
    cat = "血常规" if f in blood_indicators else "活动量表"
    print(f"  · {f} ({cat})")

print(f"\n【随机森林】高血脂预警Top10指标（{len(rf_top_n)}个）:")
for f in rf_top_n:
    cat = "血常规" if f in blood_indicators else "活动量表"
    print(f"  · {f} ({cat})")

print(f"\n【综合】两种方法共同选中的关键指标（{len(both_key)}个）:")
for f in both_key:
    cat = "血常规" if f in blood_indicators else "活动量表"
    print(f"  ★ {f} ({cat})")

print(f"\n所有结果已保存至 output/q1_1/ 目录")
