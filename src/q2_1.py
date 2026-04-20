"""
问题2-1：构建融合多维度特征的风险预警模型
  - 第一步：逻辑回归概率预测模型
  - 第二步：三级风险阈值定义（低/中/高）
  - 第三步：决策树提取特征分层阈值选取依据
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import random
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (roc_curve, auc)
from sklearn.preprocessing import StandardScaler

random.seed(42)
np.random.seed(42)
os.environ['PYTHONHASHSEED'] = '42'

# ==================== 字体与路径设置 ====================
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'output', 'q2_1')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 数据加载 ====================
df = pd.read_csv(os.path.join(DATA_DIR, 'raw_data.csv'), index_col=0)
print(f"数据维度: {df.shape}")

# ==================== 特征定义 ====================
# 九种体质积分
constitution_scores = [
    '平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质',
    '湿热质', '血瘀质', '气郁质', '特禀质'
]

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

# 人口学变量
demographics = ['年龄组', '性别', '吸烟史', '饮酒史']

# 全部特征
all_features = constitution_scores + blood_indicators + activity_scores + demographics

# 目标变量
y = df['高血脂症二分类标签']
X = df[all_features]

print(f"特征数: {len(all_features)} (体质{len(constitution_scores)} + "
      f"血常规{len(blood_indicators)} + 活动量表{len(activity_scores)} + "
      f"人口学{len(demographics)})")
print(f"目标变量: 正例={sum(y == 1)}, 负例={sum(y == 0)}, "
      f"正例比例={y.mean():.2%}")


# ======================================================================
# 第一步：逻辑回归概率预测模型
# ======================================================================
print("\n" + "=" * 70)
print("第一步：逻辑回归概率预测模型")
print("=" * 70)

# 标准化特征（逻辑回归对尺度敏感）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 逻辑回归模型（L2正则化，class_weight处理类别不平衡）
lr = LogisticRegression(
    l1_ratio=0, C=1.0, solver='lbfgs',  # 这里需要把 penalty='l2' 改成 l1_ratio=0 ，不然会报一个未来版本弃用警告
    max_iter=5000, random_state=42,
    class_weight='balanced'
)

# 5折分层交叉验证
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_auc = cross_val_score(lr, X_scaled, y, cv=cv, scoring='roc_auc')
cv_acc = cross_val_score(lr, X_scaled, y, cv=cv, scoring='accuracy')

print(f"5折交叉验证 AUC: {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
print(f"5折交叉验证 ACC: {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")

# 训练最终模型（全量数据）
lr.fit(X_scaled, y)

# 输出每个样本的预测概率
y_prob = lr.predict_proba(X_scaled)[:, 1]  # 正类概率

print(f"\n预测概率统计:")
print(f"  全体: mean={y_prob.mean():.4f}, std={y_prob.std():.4f}, "
      f"min={y_prob.min():.4f}, max={y_prob.max():.4f}")
print(f"  正例: mean={y_prob[y == 1].mean():.4f}, std={y_prob[y == 1].std():.4f}")
print(f"  负例: mean={y_prob[y == 0].mean():.4f}, std={y_prob[y == 0].std():.4f}")

# 逻辑回归系数
lr_coefs = pd.Series(lr.coef_[0], index=all_features).sort_values(key=abs, ascending=False)
print(f"\n逻辑回归系数（按绝对值排序）:")
for feat, coef in lr_coefs.items():
    cat = ("体质" if feat in constitution_scores else
           "血常规" if feat in blood_indicators else
           "活动量表" if feat in activity_scores else "人口学")
    direction = "正向↑" if coef > 0 else "负向↓"
    print(f"  {feat} ({cat}): {direction}, 系数={coef:.4f}")

# 将概率添加到数据中
df['预测概率'] = y_prob


# ======================================================================
# 第二步：三级风险阈值定义
# ======================================================================
print("\n" + "=" * 70)
print("第二步：三级风险阈值定义")
print("=" * 70)

# --- 方法1：基于ROC曲线的分界点寻找 ---
fpr, tpr, thresholds_roc = roc_curve(y, y_prob)
roc_auc_val = auc(fpr, tpr)

# Youden's J = TPR - FPR，寻找最大J值对应的最优阈值
j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
best_threshold_roc = thresholds_roc[best_idx]
print(f"[ROC/Youden] 最优二分界点: {best_threshold_roc:.4f} (J={j_scores[best_idx]:.4f})")

# --- 方法2：基于概率分布的分位数 ---
prob_positive = y_prob[y == 1]  # 确诊患者概率
prob_negative = y_prob[y == 0]  # 健康人群概率

q25_pos = np.percentile(prob_positive, 25)
q75_neg = np.percentile(prob_negative, 75)
q50_all = np.median(y_prob)
q33_all = np.percentile(y_prob, 33)
q67_all = np.percentile(y_prob, 67)

print(f"\n[分位数分析]")
print(f"  全体概率: Q33={q33_all:.4f}, Q50={q50_all:.4f}, Q67={q67_all:.4f}")
print(f"  正例概率: Q25={q25_pos:.4f}, Q50={np.median(prob_positive):.4f}")
print(f"  负例概率: Q75={q75_neg:.4f}, Q50={np.median(prob_negative):.4f}")

# --- 方法3：综合确定三级阈值 ---
# 策略：
#   低/中分界：取负例Q75与健康正例Q25的较低值，确保大部分健康人归入低风险
#   中/高分界：取ROC最优阈值，确保高特异性
threshold_low = min(q75_neg, q33_all)   # 低/中分界
threshold_high = best_threshold_roc      # 中/高分界

# 确保低 < 高
if threshold_low >= threshold_high:
    # 回退到三分位数
    threshold_low = q33_all
    threshold_high = q67_all

print(f"\n>>> 最终三级风险阈值 <<<")
print(f"  低风险: P < {threshold_low:.4f}")
print(f"  中风险: {threshold_low:.4f} ≤ P ≤ {threshold_high:.4f}")
print(f"  高风险: P > {threshold_high:.4f}")


# --- 分配风险等级 ---
def assign_risk(prob, low_th, high_th):
    if prob < low_th:
        return '低风险'
    elif prob <= high_th:
        return '中风险'
    else:
        return '高风险'

df['风险等级'] = df['预测概率'].apply(assign_risk, args=(threshold_low, threshold_high))

# 统计各级别人数及实际患病率
risk_order = ['低风险', '中风险', '高风险']
print(f"\n风险等级分布:")
for level in risk_order:
    subset = df[df['风险等级'] == level]
    n = len(subset)
    n_pos = sum(subset['高血脂症二分类标签'] == 1)
    rate = n_pos / n if n > 0 else 0
    print(f"  {level}: n={n}, 确诊={n_pos}, 实际患病率={rate:.2%}")

# 编码风险等级为数值
risk_map = {'低风险': 0, '中风险': 1, '高风险': 2}
df['风险等级编码'] = df['风险等级'].map(risk_map)


# ---------- 绘图: 概率分布与阈值 ----------
# 图1: 概率直方图（按实际标签分组）
fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.hist(prob_negative, bins=40, alpha=0.6, color='#3498db', label='未确诊')
ax1.hist(prob_positive, bins=40, alpha=0.6, color='#e74c3c', label='确诊')
ax1.axvline(x=threshold_low, color='orange', linestyle='--', linewidth=2,
            label=f'低/中分界={threshold_low:.3f}')
ax1.axvline(x=threshold_high, color='red', linestyle='--', linewidth=2,
            label=f'中/高分界={threshold_high:.3f}')
ax1.set_title('逻辑回归预测概率分布', fontsize=14)
ax1.set_xlabel('预测概率 P(高血脂)')
ax1.set_ylabel('样本数')
ax1.legend(fontsize=10, loc='upper left')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'lr_probability_dist.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"\n  -> 图表已保存: output/q2_1/lr_probability_dist.png")

# 图2: ROC曲线
fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC曲线 (AUC={roc_auc_val:.4f})')
ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax2.scatter(fpr[best_idx], tpr[best_idx], color='red', s=100, zorder=5,
            label=f'Youden最优点 (阈值={best_threshold_roc:.3f})')
ax2.set_title('ROC曲线', fontsize=14)
ax2.set_xlabel('假阳性率 (FPR)')
ax2.set_ylabel('真阳性率 (TPR)')
ax2.legend(fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'lr_roc_curve.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"\n  -> 图表已保存: output/q2_1/lr_roc_curve.png")


# ======================================================================
# 第三步：决策树提取特征分层阈值选取依据
# ======================================================================
print("\n" + "=" * 70)
print("第三步：决策树提取特征分层阈值选取依据")
print("=" * 70)

# 使用风险等级编码作为目标变量
y_risk = df['风险等级编码']
X_tree = df[all_features]

# 训练浅层决策树（CART算法）
# max_depth控制树深度，min_samples_leaf防止过拟合
dt = DecisionTreeClassifier(
    max_depth=4,
    min_samples_leaf=20,
    min_samples_split=40,
    random_state=42,
    criterion='gini'
)
dt.fit(X_tree, y_risk)

# 决策树准确率
tree_acc = dt.score(X_tree, y_risk)
print(f"决策树训练准确率: {tree_acc:.4f}")

# 打印决策树文本规则
tree_text = export_text(dt, feature_names=list(all_features), show_weights=True)
print(f"\n决策树规则:\n{tree_text}")

# ---------- 提取决策路径中的显式规则 ----------
print("\n>>> 从决策树提取的特征分层阈值选取依据 <<<\n")


def extract_rules(tree, feature_names):
    """从决策树中提取所有叶节点的规则路径"""
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != -2 else "undefined!"
        for i in tree_.feature
    ]

    class_names = ['低风险', '中风险', '高风险']
    rules = []

    def recurse(node, path):
        if tree_.feature[node] != -2:  # 非叶节点
            name = feature_name[node]
            threshold = tree_.threshold[node]
            recurse(tree_.children_left[node], path + [f"{name} ≤ {threshold:.2f}"])
            recurse(tree_.children_right[node], path + [f"{name} > {threshold:.2f}"])
        else:  # 叶节点
            values = tree_.value[node][0]
            n_samples = tree_.n_node_samples[node]
            total_prob = values.sum()
            if total_prob > 0:
                counts = np.round(values / total_prob * n_samples).astype(int)
            else:
                counts = np.zeros(len(values), dtype=int)
            pred_class = np.argmax(values)
            pred_name = class_names[pred_class]
            confidence = values[pred_class] / total_prob if total_prob > 0 else 0
            rules.append({
                'path': path,
                'predicted': pred_name,
                'confidence': confidence,
                'samples': n_samples,
                'distribution': {class_names[i]: int(counts[i]) for i in range(len(class_names))}
            })

    recurse(0, [])
    return rules

rules = extract_rules(dt, list(all_features))

# 按风险等级分组展示
for level in ['低风险', '中风险', '高风险']:
    level_rules = [r for r in rules if r['predicted'] == level]
    print(f"【{level}】的判定规则:")
    for i, r in enumerate(level_rules, 1):
        conditions = ' 且 '.join(r['path'])
        dist_str = ', '.join([f"{k}:{v}" for k, v in r['distribution'].items() if v > 0])
        print(f"  规则{i} : {conditions}")
        print(f"    → {level} (置信度={r['confidence']:.2%}, 样本数={r['samples']}, 分布: {dist_str})")
    print()


# ---------- 绘图: 决策树 ----------
fig, ax = plt.subplots(figsize=(24, 14))
plot_tree(dt, feature_names=list(all_features),
          class_names=['低风险', '中风险', '高风险'],
          filled=True, rounded=True, fontsize=10, ax=ax)
ax.set_title('风险分级决策树（特征分层阈值提取）', fontsize=16)
plt.savefig(os.path.join(OUTPUT_DIR, 'decision_tree.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"  -> 图表已保存: output/q2_1/decision_tree.png")


# ======================================================================
# 补充分析：特征在各风险等级中的均值对比
# ======================================================================
print("\n" + "=" * 70)
print("补充分析：关键特征在各风险等级中的均值对比")
print("=" * 70)

# 选取题目中重点关注的关键特征
key_features = [
    '痰湿质', 'TC（总胆固醇）', 'TG（甘油三酯）',
    'LDL-C（低密度脂蛋白）', 'HDL-C（高密度脂蛋白）',
    '活动量表总分（ADL总分+IADL总分）', 'BMI', '空腹血糖'
]

print(f"\n{'特征':<30s} {'低风险':>10s} {'中风险':>10s} {'高风险':>10s}")
print("-" * 65)
for feat in key_features:
    vals = []
    for level in risk_order:
        subset = df[df['风险等级'] == level]
        vals.append(f"{subset[feat].mean():.4f}")
    print(f"{feat:<30s} {vals[0]:>10s} {vals[1]:>10s} {vals[2]:>10s}")


# ---------- 绘图: 关键特征箱线图 ----------
n_key = len(key_features)
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for i, feat in enumerate(key_features):
    ax = axes[i]
    data_by_risk = [df[df['风险等级'] == level][feat].values for level in risk_order]
    bp = ax.boxplot(data_by_risk, tick_labels=['低风险', '中风险', '高风险'],
                    patch_artist=True)
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_title(feat, fontsize=11)
    ax.set_ylabel('原始测量值')

plt.suptitle('关键特征在各风险等级中的分布', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'feature_boxplot_by_risk.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"\n  -> 图表已保存: output/q2_1/feature_boxplot_by_risk.png")


# ======================================================================
# 补充分析：逻辑回归系数可视化
# ======================================================================
fig, ax = plt.subplots(figsize=(12, 8))

color_map_lr = {'体质': '#9b59b6', '血常规': '#e74c3c',
                '活动量表': '#3498db', '人口学': '#95a5a6'}
bar_colors_lr = [color_map_lr['体质' if f in constitution_scores else
                             '血常规' if f in blood_indicators else
                             '活动量表' if f in activity_scores else '人口学']
                 for f in lr_coefs.index]

lr_coefs.plot(kind='barh', color=bar_colors_lr, ax=ax)
ax.set_title('逻辑回归系数（高血脂症风险预警模型）', fontsize=14)
ax.set_xlabel('系数值')
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

from matplotlib.patches import Patch
legend_patches_lr = [
    Patch(facecolor='#9b59b6', label='体质积分'),
    Patch(facecolor='#e74c3c', label='血常规指标'),
    Patch(facecolor='#3498db', label='活动量表评分'),
    Patch(facecolor='#95a5a6', label='人口学变量')
]
ax.legend(handles=legend_patches_lr, loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'lr_coefficients.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"  -> 图表已保存: output/q2_1/lr_coefficients.png")


# ======================================================================
# 保存结果
# ======================================================================
# 保存带风险等级的数据
output_df = df[['体质标签', '痰湿质', '活动量表总分（ADL总分+IADL总分）',
                'TC（总胆固醇）', 'TG（甘油三酯）', 'LDL-C（低密度脂蛋白）',
                'HDL-C（高密度脂蛋白）', 'BMI', '空腹血糖',
                '高血脂症二分类标签', '预测概率', '风险等级', '风险等级编码']]
output_df.to_csv(os.path.join(OUTPUT_DIR, 'risk_classification_results.csv'),
                 encoding='utf-8-sig')

# 保存决策树规则
rules_df = pd.DataFrame([
    {
        '风险等级': r['predicted'],
        '置信度': f"{r['confidence']:.4f}",
        '样本数': r['samples'],
        '规则条件': ' 且 '.join(r['path']),
        **{f'分布_{k}': v for k, v in r['distribution'].items()}
    }
    for r in rules
])
rules_df.to_csv(os.path.join(OUTPUT_DIR, 'decision_tree_rules.csv'),
                index=False, encoding='utf-8-sig')

# 保存阈值信息
thresholds_df = pd.DataFrame({
    '阈值类型': ['低/中分界', '中/高分界'],
    '概率阈值': [threshold_low, threshold_high],
    '确定依据': [
        f'取负例Q75({q75_neg:.4f})与全体Q33({q33_all:.4f})的较小值',
        f'ROC曲线Youden最优点(阈值={best_threshold_roc:.4f})'
    ]
})
thresholds_df.to_csv(os.path.join(OUTPUT_DIR, 'risk_thresholds.csv'),
                     index=False, encoding='utf-8-sig')


# ======================================================================
# 保存关键结果摘要到 TXT
# ======================================================================
with open(os.path.join(OUTPUT_DIR, 'summary.txt'), 'w', encoding='utf-8') as f:
    f.write("问题2-1：融合多维度特征的风险预警模型 —— 结果摘要\n")
    f.write("=" * 70 + "\n\n")

    # 逻辑回归
    f.write("一、逻辑回归概率预测模型\n")
    f.write("-" * 50 + "\n")
    f.write(f"5折交叉验证 AUC: {cv_auc.mean():.4f} ± {cv_auc.std():.4f}\n")
    f.write(f"5折交叉验证 ACC: {cv_acc.mean():.4f} ± {cv_acc.std():.4f}\n\n")

    f.write(f"预测概率统计:\n")
    f.write(f"  全体: mean={y_prob.mean():.4f}, std={y_prob.std():.4f}, "
            f"min={y_prob.min():.4f}, max={y_prob.max():.4f}\n")
    f.write(f"  正例: mean={y_prob[y == 1].mean():.4f}, std={y_prob[y == 1].std():.4f}\n")
    f.write(f"  负例: mean={y_prob[y == 0].mean():.4f}, std={y_prob[y == 0].std():.4f}\n\n")

    f.write("逻辑回归系数（按绝对值排序）:\n")
    for feat, coef in lr_coefs.items():
        cat = ("体质" if feat in constitution_scores else
               "血常规" if feat in blood_indicators else
               "活动量表" if feat in activity_scores else "人口学")
        direction = "正向↑" if coef > 0 else "负向↓"
        f.write(f"  {feat} ({cat}): {direction}, 系数={coef:.4f}\n")
    f.write("\n")

    # 三级阈值
    f.write("二、三级风险阈值\n")
    f.write("-" * 50 + "\n")
    f.write(f"[ROC/Youden] 最优二分界点: {best_threshold_roc:.4f} "
            f"(J={j_scores[best_idx]:.4f})\n")
    f.write(f"[分位数分析] 全体Q33={q33_all:.4f}, Q50={q50_all:.4f}, "
            f"Q67={q67_all:.4f}\n")
    f.write(f"  正例Q25={q25_pos:.4f}, 负例Q75={q75_neg:.4f}\n\n")
    f.write(f">>> 最终三级风险阈值 <<<\n")
    f.write(f"  低风险: P < {threshold_low:.4f}  "
            f"(依据: 负例Q75={q75_neg:.4f} 与 全体Q33={q33_all:.4f} 的较小值)\n")
    f.write(f"  中风险: {threshold_low:.4f} ≤ P ≤ {threshold_high:.4f}  "
            f"(依据: ROC曲线Youden最优点={best_threshold_roc:.4f})\n")
    f.write(f"  高风险: P > {threshold_high:.4f}\n\n")

    f.write("风险等级分布:\n")
    for level in risk_order:
        subset = df[df['风险等级'] == level]
        n = len(subset)
        n_pos = sum(subset['高血脂症二分类标签'] == 1)
        rate = n_pos / n if n > 0 else 0
        f.write(f"  {level}: n={n}, 确诊={n_pos}, 实际患病率={rate:.2%}\n")
    f.write("\n")

    # 决策树
    f.write("三、决策树特征分层阈值规则\n")
    f.write("-" * 50 + "\n")
    f.write(f"决策树训练准确率: {tree_acc:.4f}\n\n")
    f.write("决策树文本规则:\n")
    f.write(tree_text + "\n\n")

    for level in ['低风险', '中风险', '高风险']:
        level_rules = [r for r in rules if r['predicted'] == level]
        f.write(f"【{level}】的判定规则:\n")
        for i, r in enumerate(level_rules, 1):
            conditions = ' 且 '.join(r['path'])
            dist_str = ', '.join(
                [f"{k}:{v}" for k, v in r['distribution'].items() if v > 0])
            f.write(f"  规则{i} : {conditions}\n")
            f.write(f"    → {level} (置信度={r['confidence']:.2%}, "
                    f"样本数={r['samples']}, 分布: {dist_str})\n")
        f.write("\n")

    # 特征均值对比
    f.write("四、关键特征在各风险等级中的均值对比\n")
    f.write("-" * 50 + "\n")
    f.write(f"{'特征':<30s} {'低风险':>10s} {'中风险':>10s} {'高风险':>10s}\n")
    f.write("-" * 65 + "\n")
    for feat in key_features:
        vals = []
        for level in risk_order:
            subset = df[df['风险等级'] == level]
            vals.append(f"{subset[feat].mean():.4f}")
        f.write(f"{feat:<30s} {vals[0]:>10s} {vals[1]:>10s} {vals[2]:>10s}\n")
    f.write("\n")

    # ROC
    f.write("五、ROC曲线\n")
    f.write("-" * 50 + "\n")
    f.write(f"AUC = {roc_auc_val:.4f}\n")
    f.write(f"Youden最优点: 阈值={best_threshold_roc:.4f}, "
            f"TPR={tpr[best_idx]:.4f}, FPR={fpr[best_idx]:.4f}\n")

print(f"  -> 摘要已保存: output/q2_1/summary.txt")


# ======================================================================
# 最终总结
# ======================================================================
print("\n" + "=" * 70)
print("最终总结")
print("=" * 70)

print(f"\n1. 逻辑回归概率预测模型")
print(f"   5折CV AUC = {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
print(f"   5折CV ACC = {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")

print(f"\n2. 三级风险阈值")
print(f"   低风险: P < {threshold_low:.4f}")
print(f"   中风险: {threshold_low:.4f} ≤ P ≤ {threshold_high:.4f}")
print(f"   高风险: P > {threshold_high:.4f}")

for level in risk_order:
    subset = df[df['风险等级'] == level]
    n = len(subset)
    n_pos = sum(subset['高血脂症二分类标签'] == 1)
    rate = n_pos / n if n > 0 else 0
    print(f"   {level}: {n}人 (实际患病率={rate:.2%})")

print(f"\n3. 决策树提取的特征分层阈值规则（见上方输出）")

print(f"\n所有结果已保存至 output/q2_1/ 目录")
