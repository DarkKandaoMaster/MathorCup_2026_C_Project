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
import random
from sklearn.linear_model import LassoCV, Lasso, lasso_path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

random.seed(42)
np.random.seed(42)
os.environ['PYTHONHASHSEED'] = '42'

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

# --- 1.1 使用LassoCV交叉验证选择最优alpha ---
lasso_cv = LassoCV(cv=5, random_state=42, alphas=100, max_iter=10000)
lasso_cv.fit(X, y_reg)

print(f"[最小MSE准则] 最优alpha = {lasso_cv.alpha_:.6f}")
print(f"交叉验证MSE = {lasso_cv.mse_path_.mean():.6f}")

lasso_coefs_cv = pd.Series(lasso_cv.coef_, index=all_features)
n_selected_cv = (lasso_coefs_cv != 0).sum()
print(f"选中特征数 = {n_selected_cv}")

# --- 1.2 1-SE规则：选择更简练的模型 ---
# 在交叉验证MSE曲线中，找到距离最优MSE一个标准误以内最大的alpha
mean_mse = lasso_cv.mse_path_.mean(axis=1)
std_mse = lasso_cv.mse_path_.std(axis=1) / np.sqrt(5)  # 标准误
best_idx = np.argmin(mean_mse)
threshold = mean_mse[best_idx] + std_mse[best_idx]
# 找到MSE在该阈值之内、alpha最大的点（即模型最简的点）
se1_idx = np.where(mean_mse <= threshold)[0][-1]
alpha_1se = lasso_cv.alphas_[se1_idx]

print(f"\n[1-SE准则] alpha = {alpha_1se:.6f}")

lasso_1se = Lasso(alpha=alpha_1se, max_iter=10000, random_state=42)
lasso_1se.fit(X, y_reg)
lasso_coefs_1se = pd.Series(lasso_1se.coef_, index=all_features)
n_selected_1se = (lasso_coefs_1se != 0).sum()
print(f"选中特征数 = {n_selected_1se}")

# --- 1.3 宽松准则：使用最优alpha的1/10，保留更多潜在关联指标 ---
alpha_relaxed = lasso_cv.alpha_ * 0.1
lasso_relaxed = Lasso(alpha=alpha_relaxed, max_iter=10000, random_state=42)
lasso_relaxed.fit(X, y_reg)
lasso_coefs_relaxed = pd.Series(lasso_relaxed.coef_, index=all_features)
n_selected_relaxed = (lasso_coefs_relaxed != 0).sum()
print(f"\n[宽松准则 alpha*0.1] alpha = {alpha_relaxed:.6f}")
print(f"选中特征数 = {n_selected_relaxed}")

# --- 选用宽松准则的结果作为主要输出（保留更多有价值的指标）---
lasso_coefs = lasso_coefs_relaxed
selected_lasso = lasso_coefs[lasso_coefs != 0].sort_values(key=abs, ascending=False)

# 计算R²
r2_relaxed = 1 - np.sum((y_reg - lasso_relaxed.predict(X))**2) / np.sum((y_reg - y_reg.mean())**2)
print(f"\n宽松准则模型 R² = {r2_relaxed:.4f}")

print(f"\nLASSO筛选出 {len(selected_lasso)} 个非零系数特征（宽松准则）:")
for feat, coef in selected_lasso.items():
    cat = "血常规" if feat in blood_indicators else \
          "活动量表" if feat in activity_scores else "控制变量"
    direction = "正向" if coef > 0 else "负向"
    print(f"  {feat} ({cat}): {direction}, 系数 = {coef:.6f}")


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
    lw = 2.0 if lasso_coefs[feat] != 0 else 0.8
    ax2.plot(-np.log10(alphas_path), coefs_path[i], label=feat,
             color=color, alpha=0.7, linewidth=lw)
ax2.axvline(x=-np.log10(lasso_cv.alpha_), color='red', linestyle='--',
            label=f'CV最优alpha={lasso_cv.alpha_:.4f}')
ax2.axvline(x=-np.log10(alpha_relaxed), color='orange', linestyle='--',
            label=f'宽松alpha={alpha_relaxed:.4f}')
ax2.set_title('LASSO正则化路径', fontsize=14)
ax2.set_xlabel('-log10(alpha)')
ax2.set_ylabel('系数值')
ax2.legend(fontsize=5.8, ncol=3, loc='best') #？？字体大小居然可以设置成小数的？

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'lasso_results.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"\n  -> 图表已保存: output/q1_1/lasso_results.png")


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
cv_acc = cross_val_score(rf, X, y_cls, cv=cv, scoring='accuracy')
print(f"5折交叉验证 AUC: {cv_auc.mean():.4f} +/- {cv_auc.std():.4f}")
print(f"5折交叉验证 ACC: {cv_acc.mean():.4f} +/- {cv_acc.std():.4f}")

# 训练最终模型
rf.fit(X, y_cls)

# 特征重要性（Mean Decrease in Impurity）
rf_importances = pd.Series(rf.feature_importances_, index=all_features)
rf_importances = rf_importances.sort_values(ascending=False)

# 找到达到80%累积重要性的特征数
cumsum = rf_importances.cumsum()
n_top80 = (cumsum <= 0.80).sum() + 1
n_top95 = (cumsum <= 0.95).sum() + 1

print(f"\n随机森林特征重要性排序:")
for i, (feat, imp) in enumerate(rf_importances.items(), 1):
    cat = "血常规" if feat in blood_indicators else \
          "活动量表" if feat in activity_scores else "控制变量"
    print(f"  [{i:2d}] {feat} ({cat}): 重要性 = {imp:.4f}")

print(f"\n前{n_top80}个特征达到80%累积重要性")
print(f"前{n_top95}个特征达到95%累积重要性")


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
ax1.legend(handles=legend_patches, loc='upper right')

# (2b) 累积重要性
ax2 = axes[1]
ax2.plot(range(1, len(cumsum) + 1), cumsum.values, 'bo-', markersize=5)
ax2.axhline(y=0.80, color='red', linestyle='--', alpha=0.7, label='80%累积阈值')
ax2.axhline(y=0.95, color='green', linestyle='--', alpha=0.7, label='95%累积阈值')
ax2.set_title('特征累积重要性', fontsize=14)
ax2.set_xlabel('特征数量（按重要性降序）')
ax2.set_ylabel('累积重要性')
ax2.set_xticks(range(1, len(cumsum) + 1))
ax2.set_xticklabels(rf_importances.index, rotation=60, ha='right', fontsize=7)
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'rf_results.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"\n  -> 图表已保存: output/q1_1/rf_results.png")


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
combined['LASSO系数_绝对值'] = lasso_abs.values
combined['LASSO归一化'] = (lasso_abs / lasso_abs.sum()).values

# RF归一化重要性
combined['RF重要性'] = rf_importances.loc[all_features].values
combined['RF归一化'] = (rf_importances / rf_importances.sum()).loc[all_features].values

# 综合得分 = 两种方法归一化重要性之和
combined['综合得分'] = combined['LASSO归一化'] + combined['RF归一化']
combined = combined.sort_values('综合得分', ascending=False)

# 只看待筛选指标（去除控制变量）
screening_result = combined[combined['类别'] != '控制变量']

print("\n待筛选指标综合排序:")
for i, (feat, row) in enumerate(screening_result.iterrows(), 1):
    lasso_mark = "Y" if row['LASSO系数_绝对值'] > 0 else "N"
    print(f"  [{i:2d}] {feat} ({row['类别']}): "
          f"综合={row['综合得分']:.4f}, "
          f"LASSO={lasso_mark}(归一化{row['LASSO归一化']:.4f}), "
          f"RF归一化={row['RF归一化']:.4f}")


# ---------- 绘图: 综合分析 ----------
fig, ax = plt.subplots(figsize=(14, 8))
key_data = screening_result
x = np.arange(len(key_data))
width = 0.35

bars1 = ax.bar(x - width/2, key_data['LASSO归一化'], width,
               label='LASSO归一化重要性', color='#e74c3c', alpha=0.8)
bars2 = ax.bar(x + width/2, key_data['RF归一化'], width,
               label='随机森林归一化重要性', color='#3498db', alpha=0.8)

ax.set_title('关键指标综合分析（LASSO + 随机森林）', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(key_data.index, rotation=55, ha='right', fontsize=9)
ax.set_ylabel('归一化重要性')
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'combined_results.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"\n  -> 图表已保存: output/q1_1/combined_results.png")


# ======================================================================
# 保存所有结果到CSV
# ======================================================================
# LASSO系数表（含三种准则对比）
lasso_df = pd.DataFrame({
    '特征': all_features,
    '类别': combined.loc[all_features, '类别'].values,
    'LASSO_CV系数': lasso_coefs_cv.values,
    'LASSO_CV是否入选': lasso_coefs_cv.values != 0,
    'LASSO_1SE系数': lasso_coefs_1se.values,
    'LASSO_1SE是否入选': lasso_coefs_1se.values != 0,
    'LASSO_宽松系数': lasso_coefs_relaxed.values,
    'LASSO_宽松是否入选': lasso_coefs_relaxed.values != 0,
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
# 保存关键结果摘要到 TXT
# ======================================================================
with open(os.path.join(OUTPUT_DIR, 'summary.txt'), 'w', encoding='utf-8') as f:
    f.write("问题1-1：关键指标筛选 —— 结果摘要\n")
    f.write("=" * 70 + "\n\n")

    # Part 1: LASSO
    f.write("一、LASSO回归 —— 痰湿体质严重程度关键指标筛选\n")
    f.write("-" * 50 + "\n")
    f.write(f"CV最优alpha = {lasso_cv.alpha_:.6f}, 交叉验证MSE = {lasso_cv.mse_path_.mean():.6f}\n")
    f.write(f"  最小MSE准则: alpha={lasso_cv.alpha_:.6f}, 选中{n_selected_cv}个特征\n")
    f.write(f"  1-SE准则:    alpha={alpha_1se:.6f}, 选中{n_selected_1se}个特征\n")
    f.write(f"  宽松准则:    alpha={alpha_relaxed:.6f}, 选中{n_selected_relaxed}个特征\n")
    f.write(f"  宽松准则模型 R² = {r2_relaxed:.4f}\n\n")

    f.write(f"LASSO筛选出 {len(selected_lasso)} 个非零系数特征（宽松准则）:\n")
    for feat, coef in selected_lasso.items():
        cat = "血常规" if feat in blood_indicators else \
              "活动量表" if feat in activity_scores else "控制变量"
        direction = "正向" if coef > 0 else "负向"
        f.write(f"  {feat} ({cat}): {direction}, 系数 = {coef:.6f}\n")
    f.write("\n")

    # Part 2: RF
    f.write("二、随机森林分类 —— 高血脂发病风险关键指标筛选\n")
    f.write("-" * 50 + "\n")
    f.write(f"5折交叉验证 AUC: {cv_auc.mean():.4f} ± {cv_auc.std():.4f}\n")
    f.write(f"5折交叉验证 ACC: {cv_acc.mean():.4f} ± {cv_acc.std():.4f}\n\n")

    f.write("随机森林特征重要性排序:\n")
    for i, (feat, imp) in enumerate(rf_importances.items(), 1):
        cat = "血常规" if feat in blood_indicators else \
              "活动量表" if feat in activity_scores else "控制变量"
        f.write(f"  [{i:2d}] {feat} ({cat}): 重要性 = {imp:.4f}\n")
    f.write(f"\n前{n_top80}个特征达到80%累积重要性\n")
    f.write(f"前{n_top95}个特征达到95%累积重要性\n\n")

    # Part 3: 综合
    f.write("三、综合分析 —— 同时表征痰湿体质 & 预警高血脂的关键指标\n")
    f.write("-" * 50 + "\n")
    f.write("待筛选指标综合排序:\n")
    for i, (feat, row) in enumerate(screening_result.iterrows(), 1):
        lasso_mark = "Y" if row['LASSO系数_绝对值'] > 0 else "N"
        f.write(f"  [{i:2d}] {feat} ({row['类别']}): "
                f"综合={row['综合得分']:.4f}, "
                f"LASSO={lasso_mark}(归一化{row['LASSO归一化']:.4f}), "
                f"RF归一化={row['RF归一化']:.4f}\n")
    f.write("\n")

    # 最终结论
    lasso_key = [feat for feat in selected_lasso.index if feat in screening_features]
    rf_screening = [feat for feat in rf_importances.index if feat in screening_features]
    rf_top_n = rf_screening[:10]
    both_key = [feat for feat in lasso_key if feat in rf_top_n]

    f.write("四、最终结论\n")
    f.write("-" * 50 + "\n")
    f.write(f"[LASSO筛选] 痰湿体质关键指标（{len(lasso_key)}个）:\n")
    for feat_name in lasso_key:
        cat = "血常规" if feat_name in blood_indicators else "活动量表"
        f.write(f"  - {feat_name} ({cat})\n")
    f.write(f"\n[随机森林] 高血脂预警Top10指标（{len(rf_top_n)}个）:\n")
    for feat_name in rf_top_n:
        cat = "血常规" if feat_name in blood_indicators else "活动量表"
        f.write(f"  - {feat_name} ({cat})\n")
    f.write(f"\n[综合] 两种方法共同选中的关键指标（{len(both_key)}个）:\n")
    for feat_name in both_key:
        cat = "血常规" if feat_name in blood_indicators else "活动量表"
        f.write(f"  * {feat_name} ({cat})\n")

print(f"  -> 摘要已保存: output/q1_1/summary.txt")


# ======================================================================
# 最终总结
# ======================================================================
print("\n" + "=" * 70)
print("最终总结：筛选出的关键指标")
print("=" * 70)

# LASSO选出的（仅血常规+活动量表）
lasso_key = [f for f in selected_lasso.index if f in screening_features]
# RF前N名（仅血常规+活动量表）
rf_screening = [f for f in rf_importances.index if f in screening_features]
rf_top_n = rf_screening[:10]
# 两种方法共同选中
both_key = [f for f in lasso_key if f in rf_top_n]

print(f"\n[LASSO筛选] 痰湿体质关键指标（{len(lasso_key)}个）:")
for f in lasso_key:
    cat = "血常规" if f in blood_indicators else "活动量表"
    print(f"  - {f} ({cat})")

print(f"\n[随机森林] 高血脂预警Top10指标（{len(rf_top_n)}个）:")
for f in rf_top_n:
    cat = "血常规" if f in blood_indicators else "活动量表"
    print(f"  - {f} ({cat})")

print(f"\n[综合] 两种方法共同选中的关键指标（{len(both_key)}个）:")
for f in both_key:
    cat = "血常规" if f in blood_indicators else "活动量表"
    print(f"  * {f} ({cat})")

print(f"\n所有结果已保存至 output/q1_1/ 目录")
