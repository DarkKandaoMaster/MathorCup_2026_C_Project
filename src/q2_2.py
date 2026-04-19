"""
问题2-2：识别痰湿体质高风险人群的核心特征组合

方法：
  1. 筛选：体质标签=5（痰湿体质）且 风险等级=高风险
  2. 特征离散化：基于决策树阈值、临床参考、动态分位数
  3. Apriori关联规则挖掘：发现频繁共现的特征集合
  4. 对比分析：高风险 vs 非高风险（痰湿体质内），卡方检验
  5. 核心特征组合识别与解释
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import random
from scipy.stats import chi2_contingency
import joblib
from mlxtend.frequent_patterns import apriori, association_rules

random.seed(42)
np.random.seed(42)
os.environ['PYTHONHASHSEED'] = '42'

# ==================== 字体与路径设置 ====================
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
Q21_DIR = os.path.join(BASE_DIR, '..', 'output', 'q2_1')
OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'output', 'q2_2')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 数据加载 ====================
df = pd.read_csv(os.path.join(DATA_DIR, 'preprocessed_data.csv'), index_col=0)
risk_df = pd.read_csv(os.path.join(Q21_DIR, 'risk_classification_results.csv'), index_col=0)

df['风险等级'] = risk_df['风险等级']
df['预测概率'] = risk_df['预测概率']

# 加载MinMaxScaler（用于反归一化关联规则阈值）
mm_scaler = joblib.load(os.path.join(DATA_DIR, 'minmax_scaler.pkl'))

print(f"数据维度: {df.shape}")
print(f"全体样本数: {len(df)}")

# ==================== 目标群体筛选 ====================
df_tw = df[df['体质标签'] == 5].copy()
df_tw_high = df_tw[df_tw['风险等级'] == '高风险'].copy()
df_tw_nonhigh = df_tw[df_tw['风险等级'] != '高风险'].copy()

n_tw = len(df_tw)
n_high = len(df_tw_high)
n_nonhigh = len(df_tw_nonhigh)

print(f"\n目标群体:")
print(f"  痰湿体质(标签=5): {n_tw}人")
print(f"  痰湿体质+高风险: {n_high}人 ({n_high/n_tw:.1%})")
print(f"  痰湿体质+非高风险: {n_nonhigh}人 ({n_nonhigh/n_tw:.1%})")

# ==================== 特征离散化 ====================
# 策略：q2_1决策树阈值 > 临床参考 > 动态中位数

key_continuous = {
    # 注意：痰湿体质(标签=5)的痰湿积分最小值为0.8462，≥0.6阈值无区分度
    # 改用高风险组Q75(≈0.958)作为"极高痰湿积分"阈值
    '痰湿质': {'阈值': None, '方向': '高', '标签': '极高痰湿积分(≥Q75)',
              '来源': '动态Q75'},
    'TG（甘油三酯）': {'阈值': 0.48, '方向': '高', '标签': '高TG(>0.48)',
                     '来源': '决策树'},
    'TC（总胆固醇）': {'阈值': 0.64, '方向': '高', '标签': '高TC(>0.64)',
                     '来源': '决策树'},
    'LDL-C（低密度脂蛋白）': {'阈值': None, '方向': '高', '标签': '高LDL-C',
                           '来源': '动态中位数'},
    'HDL-C（高密度脂蛋白）': {'阈值': None, '方向': '低', '标签': '低HDL-C',
                           '来源': '动态中位数'},
    'BMI': {'阈值': None, '方向': '高', '标签': '高BMI', '来源': '动态中位数'},
    '活动量表总分（ADL总分+IADL总分）': {'阈值': 0.4, '方向': '低',
                                       '标签': '低活动量(<0.4)', '来源': '临床(<40分)'},
    '血尿酸': {'阈值': 0.63, '方向': '高', '标签': '高血尿酸(>0.63)',
             '来源': '决策树'},
    '空腹血糖': {'阈值': None, '方向': '高', '标签': '高血糖', '来源': '动态中位数'},
    'ADL总分': {'阈值': None, '方向': '低', '标签': '低ADL', '来源': '动态中位数'},
    'IADL总分': {'阈值': None, '方向': '低', '标签': '低IADL', '来源': '动态中位数'},
}

# 对没有预设阈值的特征，根据来源用中位数或Q75
print(f"\n离散化阈值:")
for feat, info in key_continuous.items():
    if info['阈值'] is None:
        if 'Q75' in info['来源']:
            q75_val = df_tw_high[feat].quantile(0.75)
            info['阈值'] = round(q75_val, 4)
            info['标签'] = info['标签'].replace('≥Q75', f'≥{q75_val:.2f}')
        else:
            median_val = df_tw_high[feat].median()
            info['阈值'] = round(median_val, 4)
        print(f"  {info['标签']}: 阈值={info['阈值']:.4f} ({info['来源']})")
    else:
        print(f"  {info['标签']}: 阈值={info['阈值']:.4f} ({info['来源']})")

# 反归一化：将阈值还原为原始物理尺度
print(f"\n反归一化阈值（原始物理尺度）:")
for feat, info in key_continuous.items():
    threshold = info['阈值']
    if feat in mm_scaler.feature_names_in_:
        idx = list(mm_scaler.feature_names_in_).index(feat)
        min_v = mm_scaler.data_min_[idx]
        max_v = mm_scaler.data_max_[idx]
        threshold_orig = threshold * (max_v - min_v) + min_v
        direction = '>' if info['方向'] == '高' else '<'
        print(f"  {feat}: {direction} {threshold:.4f}(归一化) → {direction} {threshold_orig:.2f}(原始尺度, "
              f"范围: [{min_v:.2f}, {max_v:.2f}])")
    else:
        print(f"  {feat}: 非连续型变量，阈值不变")


def discretize(data, feat_info):
    """将连续特征离散化为二元项目，返回布尔型DataFrame"""
    items = pd.DataFrame(index=data.index)

    for feat, info in feat_info.items():
        threshold = info['阈值']
        direction = info['方向']
        label = info['标签']

        if direction == '高':
            items[label] = data[feat] > threshold
        else:  # 低
            items[label] = data[feat] < threshold

    # 人口学特征
    items['老年(≥60岁)'] = data['年龄组'] >= 3
    items['男性'] = data['性别'] == 1
    items['有吸烟'] = data['吸烟史'] == 1
    items['有饮酒'] = data['饮酒史'] == 1

    return items


items_high = discretize(df_tw_high, key_continuous)
items_nonhigh = discretize(df_tw_nonhigh, key_continuous)

n_items = items_high.shape[1]
print(f"\n离散化后二元特征数: {n_items}")


# ======================================================================
# 一、各特征在高风险组的频率
# ======================================================================
print("\n" + "=" * 70)
print("一、各特征在痰湿体质高风险人群中的频率")
print("=" * 70)

freq_high = items_high.mean().sort_values(ascending=False)
freq_nonhigh = items_nonhigh.mean()

print(f"\n{'特征':<25s} {'高风险频率':>10s} {'非高风险频率':>12s} {'差异':>10s}")
print("-" * 60)
for feat in freq_high.index:
    f_h = freq_high[feat]
    f_nh = freq_nonhigh.get(feat, 0)
    diff = f_h - f_nh
    marker = " ★" if abs(diff) > 0.1 else "  "
    print(f"{marker}{feat:<25s} {f_h:>10.4f} {f_nh:>12.4f} {diff:>+10.4f}")


# ======================================================================
# 二、Apriori频繁项集挖掘
# ======================================================================
print("\n" + "=" * 70)
print("二、Apriori关联规则挖掘")
print("=" * 70)

min_support = 0.3
frequent_itemsets = apriori(
    items_high, min_support=min_support, use_colnames=True, max_len=4
)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(len)
frequent_itemsets = frequent_itemsets.sort_values('support', ascending=False)

print(f"\n最小支持度阈值: {min_support}")
print(f"发现的频繁项集数: {len(frequent_itemsets)}")

print(f"\n支持度Top-20频繁项集:")
print(f"{'支持度':>8s}  {'长度':>4s}  项集")
print("-" * 80)
for _, row in frequent_itemsets.head(20).iterrows():
    items_str = ' + '.join(sorted(row['itemsets']))
    print(f"{row['support']:>8.4f}  {row['length']:>4d}  {items_str}")

print(f"\n各长度频繁项集数量:")
for length in sorted(frequent_itemsets['length'].unique()):
    n = len(frequent_itemsets[frequent_itemsets['length'] == length])
    print(f"  长度{length}: {n}个")


# ======================================================================
# 三、关联规则提取
# ======================================================================
print("\n" + "=" * 70)
print("三、关联规则提取")
print("=" * 70)

min_confidence = 0.5
rules = association_rules(
    frequent_itemsets, metric="confidence", min_threshold=min_confidence
)
rules = rules.sort_values(['lift', 'confidence'], ascending=False)

print(f"\n最小置信度阈值: {min_confidence}")
print(f"发现的关联规则数: {len(rules)}")

if len(rules) > 0:
    print(f"\nTop-20关联规则（按Lift排序）:")
    print(f"{'前件':<45s} → {'后件':<20s} {'支持度':>6s} {'置信度':>6s} {'Lift':>6s}")
    print("-" * 100)
    for _, row in rules.head(20).iterrows():
        ant = ' + '.join(sorted(row['antecedents']))
        cons = ' + '.join(sorted(row['consequents']))
        print(f"{ant:<45s} → {cons:<20s} {row['support']:>6.4f} "
              f"{row['confidence']:>6.4f} {row['lift']:>6.4f}")
else:
    print("未发现满足条件的关联规则，降低置信度阈值至0.4...")
    min_confidence = 0.4
    rules = association_rules(
        frequent_itemsets, metric="confidence", min_threshold=min_confidence
    )
    rules = rules.sort_values(['lift', 'confidence'], ascending=False)
    print(f"重新发现的关联规则数: {len(rules)}")


# ======================================================================
# 四、对比分析：高风险 vs 非高风险（卡方检验）
# ======================================================================
print("\n" + "=" * 70)
print("四、对比分析：高风险 vs 非高风险（痰湿体质内）")
print("=" * 70)

chi2_results = []
for col in items_high.columns:
    # 构建列联表
    a = items_high[col].sum()      # 高风险+有该特征
    b = n_high - a                  # 高风险+无该特征
    c = items_nonhigh[col].sum()   # 非高风险+有该特征
    d = n_nonhigh - c               # 非高风险+无该特征

    ct = np.array([[a, b], [c, d]])

    try:
        chi2, p_val, dof, expected = chi2_contingency(ct, correction=False)
        n_total = a + b + c + d
        phi = np.sqrt(chi2 / n_total) if n_total > 0 else 0
    except:
        chi2, p_val, phi = 0, 1.0, 0

    chi2_results.append({
        '特征': col,
        '高风险频率': freq_high[col],
        '非高风险频率': freq_nonhigh.get(col, 0),
        '差异': freq_high[col] - freq_nonhigh.get(col, 0),
        '卡方值': chi2,
        'p值': p_val,
        'Phi系数': phi,
        '显著': '***' if p_val < 0.001 else '**' if p_val < 0.01 else
                '*' if p_val < 0.05 else 'ns'
    })

chi2_df = pd.DataFrame(chi2_results).sort_values('Phi系数', ascending=False, key=abs)

print(f"\n{'特征':<25s} {'高风险':>8s} {'非高风险':>10s} {'差异':>8s} "
      f"{'卡方':>8s} {'p值':>10s} {'Phi':>6s} {'显著':>4s}")
print("-" * 90)
for _, row in chi2_df.iterrows():
    print(f"{row['特征']:<25s} {row['高风险频率']:>8.4f} {row['非高风险频率']:>10.4f} "
          f"{row['差异']:>+8.4f} {row['卡方值']:>8.2f} {row['p值']:>10.6f} "
          f"{row['Phi系数']:>6.4f} {row['显著']:>4s}")


# ======================================================================
# 五、核心特征组合识别
# ======================================================================
print("\n" + "=" * 70)
print("五、核心特征组合识别")
print("=" * 70)

# 筛选高频特征组合（支持度≥0.35, 长度≥2）
core_combos = frequent_itemsets[
    (frequent_itemsets['length'] >= 2) &
    (frequent_itemsets['support'] >= 0.35)
].sort_values('support', ascending=False)

print(f"\n高频特征组合（支持度≥0.35, 长度≥2）:")
print(f"{'支持度':>8s}  {'长度':>4s}  特征组合")
print("-" * 80)
for _, row in core_combos.iterrows():
    items_str = ' + '.join(sorted(row['itemsets']))
    print(f"{row['support']:>8.4f}  {row['length']:>4d}  {items_str}")


# ======================================================================
# 六、特征组合的对比Lift分析
# ======================================================================
print("\n" + "=" * 70)
print("六、特征组合的对比Lift分析（高风险 vs 非高风险）")
print("=" * 70)

combo_lift_results = []
for _, row in core_combos.iterrows():
    itemset = row['itemsets']
    items_list = list(itemset)

    # 高风险组中的支持度
    mask_high = items_high[items_list].all(axis=1)
    support_high = mask_high.mean()

    # 非高风险组中的支持度
    mask_nonhigh = items_nonhigh[items_list].all(axis=1)
    support_nonhigh = mask_nonhigh.mean()

    # 对比Lift
    lift_vs_nonhigh = support_high / support_nonhigh if support_nonhigh > 0 else float('inf')

    # 高风险组中实际人数
    n_high_combo = mask_high.sum()

    combo_lift_results.append({
        '组合': ' + '.join(sorted(items_list)),
        '长度': len(items_list),
        '高风险支持度': support_high,
        '高风险人数': int(n_high_combo),
        '非高风险支持度': support_nonhigh,
        '对比Lift': lift_vs_nonhigh,
        '项集': itemset
    })

combo_lift_df = pd.DataFrame(combo_lift_results).sort_values('对比Lift', ascending=False)

print(f"\n{'特征组合':<60s} {'高风险':>8s} {'非高':>8s} {'Lift':>6s} {'人数':>4s}")
print("-" * 95)
for _, row in combo_lift_df.head(20).iterrows():
    print(f"{row['组合']:<60s} {row['高风险支持度']:>8.4f} "
          f"{row['非高风险支持度']:>8.4f} {row['对比Lift']:>6.2f} "
          f"{row['高风险人数']:>4d}")


# ======================================================================
# 七、最终核心特征组合排名
# ======================================================================
print("\n" + "=" * 70)
print("七、最终核心特征组合排名（综合得分）")
print("=" * 70)

# 综合得分 = 0.4*支持度 + 0.3*标准化Lift + 0.3*长度权重
final_combos = combo_lift_df.copy()
max_lift = final_combos['对比Lift'].replace([np.inf], 10).max()
final_combos['标准化Lift'] = np.minimum(
    final_combos['对比Lift'].replace([np.inf], max_lift + 1) / (max_lift + 0.01), 1.0
)
final_combos['综合得分'] = (
    final_combos['高风险支持度'] * 0.4 +
    final_combos['标准化Lift'] * 0.3 +
    (final_combos['长度'] / 4) * 0.3
)
final_combos = final_combos.sort_values('综合得分', ascending=False)

print(f"\n排名  {'特征组合':<55s} {'支持度':>6s} {'Lift':>6s} {'得分':>6s}")
print("-" * 85)
for i, (_, row) in enumerate(final_combos.head(15).iterrows(), 1):
    print(f"{i:>4d}  {row['组合']:<55s} {row['高风险支持度']:>6.4f} "
          f"{row['对比Lift']:>6.2f} {row['综合得分']:>6.4f}")


# ======================================================================
# 八、核心特征组合解释
# ======================================================================
print("\n" + "=" * 70)
print("八、核心特征组合的医学解释")
print("=" * 70)

explanations = {
    '高TG': '甘油三酯升高是高血脂症最直接的血液标志物，痰湿体质者水湿代谢障碍导致膏脂内停，TG升高尤为显著',
    '高TC': '总胆固醇升高反映整体脂代谢紊乱，痰湿内蕴致脾胃运化失常，膏脂积聚使TC升高',
    '高LDL-C': '低密度脂蛋白升高是动脉粥样硬化的核心危险因素，痰湿膏脂沉积脉道与LDL-C升高机制一致',
    '低HDL-C': '高密度脂蛋白降低削弱胆固醇逆向转运能力，痰湿体质气血运行不畅与HDL-C降低密切相关',
    '极高痰湿积分': '痰湿积分处于极高水平(Q75以上)反映痰湿体质偏颇极为显著，脾失健运程度深，为高血脂症的强体质基础',
    '低活动量(<0.4)': '活动量不足（<40分）导致气血津液运行滞缓，痰湿更易内停，且活动量少致能量消耗低、脂质堆积',
    '高BMI': 'BMI超标反映痰湿体质者形体肥胖、痰湿膏脂壅盛的体质特征，肥胖与胰岛素抵抗促进脂代谢紊乱',
    '高血尿酸(>0.63)': '高尿酸与痰湿体质水湿代谢障碍同源，嘌呤代谢异常与脂代谢紊乱常并存，均为代谢综合征组分',
    '高血糖': '血糖升高提示糖代谢异常，胰岛素抵抗促进肝脏VLDL合成增加，加重高甘油三酯血症',
    '低ADL': '日常躯体活动能力低下直接影响运动消耗，痰湿体质者活动减少形成"痰湿-少动-脂聚"恶性循环',
    '低IADL': '工具性日常活动能力低下反映综合生活自理能力不足，社会参与度低、饮食管理差加重代谢风险',
    '老年(≥60岁)': '中老年脏腑功能衰退，脾胃运化力下降，气血运行减缓，痰湿更易内生，脂质代谢能力降低',
    '男性': '男性内脏脂肪堆积倾向更强，雄激素水平变化影响脂代谢，痰湿体质男性更易出现高TG',
    '有吸烟': '烟草中有害物质损伤血管内皮，促进氧化应激，加重脂质过氧化，与痰湿体质瘀滞特征叠加',
    '有饮酒': '酒精促进肝脏VLDL合成，加重高甘油三酯血症，且酒为湿热之品助湿生痰',
}

# 为每个组合生成针对性解释
def generate_interpretation(combo_items):
    """根据组合中的特征类别生成针对性解释"""
    categories = {
        'lipid_high': [],    # 高血脂指标
        'lipid_low': [],     # 低保护性脂蛋白
        'activity_low': [],  # 低活动量
        'metabolic': [],     # 代谢异常
        'constitution': [],  # 体质特征
        'demographic': [],   # 人口学特征
    }

    for item in combo_items:
        if 'TG' in item:
            categories['lipid_high'].append('甘油三酯升高')
        elif 'TC' in item:
            categories['lipid_high'].append('总胆固醇升高')
        elif 'LDL-C' in item:
            categories['lipid_high'].append('低密度脂蛋白升高')
        elif 'HDL-C' in item:
            categories['lipid_low'].append('高密度脂蛋白降低')
        elif '痰湿积分' in item:
            categories['constitution'].append('痰湿偏颇极显著')
        elif '活动量' in item:
            categories['activity_low'].append('活动量不足')
        elif 'ADL' in item and '低' in item:
            categories['activity_low'].append('躯体活动能力低')
        elif 'IADL' in item and '低' in item:
            categories['activity_low'].append('工具性活动能力低')
        elif 'BMI' in item:
            categories['metabolic'].append('形体肥胖')
        elif '血尿酸' in item:
            categories['metabolic'].append('尿酸代谢异常')
        elif '血糖' in item:
            categories['metabolic'].append('糖代谢异常')
        elif '老年' in item:
            categories['demographic'].append('脏腑功能衰退')
        elif '男性' in item:
            categories['demographic'].append('男性脂代谢易损性')
        elif '饮酒' in item:
            categories['demographic'].append('酒精促脂合成')
        elif '吸烟' in item:
            categories['demographic'].append('烟草损伤血管')

    parts = []
    if categories['constitution']:
        parts.append(f"体质层面：{'、'.join(categories['constitution'])}构成内因基础")
    if categories['lipid_high']:
        parts.append(f"脂质层面：{'、'.join(categories['lipid_high'])}直接反映膏脂内停")
    if categories['lipid_low']:
        parts.append(f"保护缺失：{'、'.join(categories['lipid_low'])}削弱脂质清除能力")
    if categories['activity_low']:
        parts.append(f"活动层面：{'、'.join(categories['activity_low'])}致气血运行滞缓、脂质消耗不足")
    if categories['metabolic']:
        parts.append(f"代谢层面：{'、'.join(categories['metabolic'])}提示整体代谢紊乱")
    if categories['demographic']:
        parts.append(f"诱因层面：{'、'.join(categories['demographic'])}加重风险")

    return '；'.join(parts)

# 输出Top组合的解释
print("\n对Top-5核心特征组合的医学解释：\n")
for i, (_, row) in enumerate(final_combos.head(5).iterrows(), 1):
    combo_items = sorted(row['项集'])
    print(f"  【组合{i}】 {' + '.join(combo_items)}")
    print(f"    支持度={row['高风险支持度']:.4f}, 对比Lift={row['对比Lift']:.2f}")
    for item in combo_items:
        # 匹配解释（模糊匹配以适应动态标签）
        matched_exp = '该特征为高风险人群的常见伴随特征'
        for key, exp in explanations.items():
            if key in item:
                matched_exp = exp
                break
        print(f"    · {item}: {matched_exp}")
    interp = generate_interpretation(combo_items)
    print(f"    → 综合解释: {interp}。多维度特征共现表明该人群存在"
          f"「体质偏颇-脂代谢紊乱-活动/代谢异常」的链式病理机制。")
    print()


# ======================================================================
# 九、可视化
# ======================================================================
print("=" * 70)
print("九、可视化")
print("=" * 70)

# ---------- 图1: 特征频率对比 ----------
fig, ax = plt.subplots(figsize=(14, 8))

features_sorted = freq_high.sort_values(ascending=True)
y_pos = np.arange(len(features_sorted))
bar_height = 0.35

bars1 = ax.barh(y_pos - bar_height/2, features_sorted.values, bar_height,
                label='高风险', color='#e74c3c', alpha=0.8)
bars2 = ax.barh(y_pos + bar_height/2,
                [freq_nonhigh.get(f, 0) for f in features_sorted.index],
                bar_height, label='非高风险', color='#3498db', alpha=0.8)

ax.set_yticks(y_pos)
ax.set_yticklabels(features_sorted.index, fontsize=10)
ax.set_xlabel('特征频率（出现比例）', fontsize=12)
ax.set_title('痰湿体质：高风险 vs 非高风险 各特征频率对比', fontsize=14)
ax.legend(fontsize=11)
ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'feature_frequency_comparison.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print(f"  -> 图1已保存: output/q2_2/feature_frequency_comparison.png")

# ---------- 图2: Top频繁项集 ----------
top_itemsets = frequent_itemsets[frequent_itemsets['length'] >= 2].head(15)
if len(top_itemsets) > 0:
    fig, ax = plt.subplots(figsize=(14, 8))

    labels = [' + '.join(sorted(row['itemsets'])) for _, row in top_itemsets.iterrows()]
    supports = top_itemsets['support'].values
    lengths = top_itemsets['length'].values

    colors = ['#3498db' if l == 2 else '#e67e22' if l == 3 else '#e74c3c'
              for l in lengths]

    bars = ax.barh(range(len(labels)), supports, color=colors, alpha=0.8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('支持度 (Support)', fontsize=12)
    ax.set_title('高频特征组合Top-15（Apriori算法）', fontsize=14)
    ax.invert_yaxis()

    from matplotlib.patches import Patch
    legend_patches = [
        Patch(facecolor='#3498db', label='2项组合'),
        Patch(facecolor='#e67e22', label='3项组合'),
        Patch(facecolor='#e74c3c', label='4项组合')
    ]
    ax.legend(handles=legend_patches, fontsize=10)

    for bar, val in zip(bars, supports):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'top_frequent_itemsets.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> 图2已保存: output/q2_2/top_frequent_itemsets.png")

# ---------- 图3: 关联规则散点图 ----------
if len(rules) > 0:
    fig, ax = plt.subplots(figsize=(12, 8))

    scatter = ax.scatter(
        rules['support'], rules['confidence'],
        c=rules['lift'], cmap='YlOrRd', s=rules['lift'] * 30 + 20,
        alpha=0.7, edgecolors='gray', linewidth=0.5
    )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Lift值', fontsize=11)

    ax.set_xlabel('支持度 (Support)', fontsize=12)
    ax.set_ylabel('置信度 (Confidence)', fontsize=12)
    ax.set_title('关联规则分布：支持度 vs 置信度（颜色=Lift）', fontsize=14)

    # 标注Lift最高的几条规则
    top_rules = rules.nlargest(5, 'lift')
    for _, r in top_rules.iterrows():
        ant = ' + '.join(sorted(r['antecedents']))
        cons = ' + '.join(sorted(r['consequents']))
        label = f"{ant[:15]}...→{cons[:10]}"
        ax.annotate(label, (r['support'], r['confidence']),
                    fontsize=7, alpha=0.8, xytext=(5, 5),
                    textcoords='offset points')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'association_rules_scatter.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> 图3已保存: output/q2_2/association_rules_scatter.png")

# ---------- 图4: 特征共现热力图 ----------
# 选取频率最高的8-10个特征，绘制共现矩阵
top_features = freq_high.head(10).index.tolist()
cooccurrence = pd.DataFrame(0, index=top_features, columns=top_features, dtype=float)

for f1 in top_features:
    for f2 in top_features:
        if f1 == f2:
            cooccurrence.loc[f1, f2] = freq_high[f1]
        else:
            cooccurrence.loc[f1, f2] = (items_high[f1] & items_high[f2]).mean()

fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(cooccurrence.values, cmap='YlOrRd', aspect='auto')

ax.set_xticks(range(len(top_features)))
ax.set_yticks(range(len(top_features)))
ax.set_xticklabels(top_features, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(top_features, fontsize=9)

for i in range(len(top_features)):
    for j in range(len(top_features)):
        val = cooccurrence.iloc[i, j]
        color = 'white' if val > 0.5 else 'black'
        ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                fontsize=8, color=color)

ax.set_title('痰湿体质高风险人群：特征共现热力图', fontsize=14)
plt.colorbar(im, ax=ax, label='共现频率')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'feature_cooccurrence_heatmap.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print(f"  -> 图4已保存: output/q2_2/feature_cooccurrence_heatmap.png")

# ---------- 图5: 核心组合对比Lift ----------
top_combos_viz = combo_lift_df.head(12)
if len(top_combos_viz) > 0:
    fig, ax = plt.subplots(figsize=(14, 8))

    combo_labels = top_combos_viz['组合'].values
    lift_vals = top_combos_viz['对比Lift'].replace([np.inf], 10).values
    support_vals = top_combos_viz['高风险支持度'].values

    colors_bar = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(combo_labels))) # RdYlGn是Matplotlib里合法存在的colormap名称，只是它是“动态注册”的，静态检查有时识别不到。所以这里会报红色警告

    bars = ax.barh(range(len(combo_labels)), lift_vals, color=colors_bar, alpha=0.85)
    ax.set_yticks(range(len(combo_labels)))
    ax.set_yticklabels(combo_labels, fontsize=9)
    ax.set_xlabel('对比Lift（高风险/非高风险支持度比）', fontsize=12)
    ax.set_title('核心特征组合：对比Lift排名（高风险 vs 非高风险）', fontsize=14)
    ax.axvline(x=1.0, color='red', linestyle='--', linewidth=1.5,
               label='Lift=1（无差异基准）')
    ax.legend(fontsize=10)
    ax.invert_yaxis()

    for bar, sup in zip(bars, support_vals):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f'支持度={sup:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'combo_lift_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> 图5已保存: output/q2_2/combo_lift_comparison.png")


# ======================================================================
# 十、保存结果
# ======================================================================

# 保存高频组合详细结果
combo_output = combo_lift_df[['组合', '长度', '高风险支持度', '高风险人数',
                              '非高风险支持度', '对比Lift']].copy()
combo_output.to_csv(os.path.join(OUTPUT_DIR, 'core_combinations.csv'),
                    index=False, encoding='utf-8-sig')

# 保存卡方检验结果
chi2_df.to_csv(os.path.join(OUTPUT_DIR, 'chi2_comparison.csv'),
               index=False, encoding='utf-8-sig')

# 保存关联规则
if len(rules) > 0:
    rules_output = rules[['antecedents', 'consequents', 'support',
                          'confidence', 'lift']].copy()
    rules_output['antecedents'] = rules_output['antecedents'].apply(
        lambda x: ' + '.join(sorted(x)))
    rules_output['consequents'] = rules_output['consequents'].apply(
        lambda x: ' + '.join(sorted(x)))
    rules_output.to_csv(os.path.join(OUTPUT_DIR, 'association_rules.csv'),
                        index=False, encoding='utf-8-sig')

# 保存最终排名
final_output = final_combos[['组合', '长度', '高风险支持度', '高风险人数',
                             '非高风险支持度', '对比Lift', '综合得分']].copy()
final_output.to_csv(os.path.join(OUTPUT_DIR, 'final_ranking.csv'),
                    index=False, encoding='utf-8-sig')


# ======================================================================
# 十一、保存关键结果摘要
# ======================================================================
with open(os.path.join(OUTPUT_DIR, 'summary.txt'), 'w', encoding='utf-8') as f:
    f.write("问题2-2：识别痰湿体质高风险人群的核心特征组合 —— 结果摘要\n")
    f.write("=" * 70 + "\n\n")

    f.write("一、目标群体\n")
    f.write("-" * 50 + "\n")
    f.write(f"痰湿体质(标签=5): {n_tw}人\n")
    f.write(f"痰湿体质+高风险: {n_high}人 ({n_high/n_tw:.1%})\n")
    f.write(f"痰湿体质+非高风险: {n_nonhigh}人 ({n_nonhigh/n_tw:.1%})\n\n")

    f.write("二、离散化阈值\n")
    f.write("-" * 50 + "\n")
    for feat, info in key_continuous.items():
        threshold = info['阈值']
        if feat in mm_scaler.feature_names_in_:
            idx = list(mm_scaler.feature_names_in_).index(feat)
            min_v = mm_scaler.data_min_[idx]
            max_v = mm_scaler.data_max_[idx]
            threshold_orig = threshold * (max_v - min_v) + min_v
            f.write(f"  {info['标签']}: 阈值={threshold:.4f}(归一化) → "
                    f"{threshold_orig:.2f}(原始尺度) ({info['来源']})\n")
        else:
            f.write(f"  {info['标签']}: 阈值={threshold:.4f} ({info['来源']})\n")
    f.write("  老年(≥60岁): 年龄组≥3\n")
    f.write("  男性: 性别=1\n")
    f.write("  有吸烟: 吸烟史=1\n")
    f.write("  有饮酒: 饮酒史=1\n\n")

    f.write("三、各特征在高风险组的频率\n")
    f.write("-" * 50 + "\n")
    f.write(f"{'特征':<25s} {'高风险频率':>10s} {'非高风险频率':>12s} {'差异':>10s}\n")
    f.write("-" * 60 + "\n")
    for feat in freq_high.index:
        f_h = freq_high[feat]
        f_nh = freq_nonhigh.get(feat, 0)
        diff = f_h - f_nh
        f.write(f"{feat:<25s} {f_h:>10.4f} {f_nh:>12.4f} {diff:>+10.4f}\n")
    f.write("\n")

    f.write("四、Apriori频繁项集\n")
    f.write("-" * 50 + "\n")
    f.write(f"最小支持度: {min_support}\n")
    f.write(f"频繁项集总数: {len(frequent_itemsets)}\n\n")
    for _, row in frequent_itemsets[frequent_itemsets['length'] >= 2].head(20).iterrows():
        items_str = ' + '.join(sorted(row['itemsets']))
        f.write(f"  支持度={row['support']:.4f}, 长度={row['length']}: {items_str}\n")
    f.write("\n")

    f.write("五、关联规则\n")
    f.write("-" * 50 + "\n")
    f.write(f"最小置信度: {min_confidence}\n")
    f.write(f"关联规则数: {len(rules)}\n\n")
    if len(rules) > 0:
        for _, row in rules.head(15).iterrows():
            ant = ' + '.join(sorted(row['antecedents']))
            cons = ' + '.join(sorted(row['consequents']))
            f.write(f"  {ant} → {cons}  "
                    f"(支持度={row['support']:.4f}, "
                    f"置信度={row['confidence']:.4f}, Lift={row['lift']:.4f})\n")
    f.write("\n")

    f.write("六、卡方检验对比\n")
    f.write("-" * 50 + "\n")
    f.write(f"{'特征':<25s} {'高风险':>8s} {'非高风险':>10s} {'差异':>8s} "
            f"{'卡方':>8s} {'p值':>10s} {'Phi':>6s} {'显著':>4s}\n")
    f.write("-" * 90 + "\n")
    for _, row in chi2_df.iterrows():
        f.write(f"{row['特征']:<25s} {row['高风险频率']:>8.4f} "
                f"{row['非高风险频率']:>10.4f} {row['差异']:>+8.4f} "
                f"{row['卡方值']:>8.2f} {row['p值']:>10.6f} "
                f"{row['Phi系数']:>6.4f} {row['显著']:>4s}\n")
    f.write("\n")

    f.write("七、核心特征组合排名\n")
    f.write("-" * 50 + "\n")
    for i, (_, row) in enumerate(final_combos.iterrows(), 1):
        f.write(f"  排名{i}: {row['组合']}\n")
        f.write(f"    支持度={row['高风险支持度']:.4f}, "
                f"Lift={row['对比Lift']:.2f}, "
                f"综合得分={row['综合得分']:.4f}\n")
    f.write("\n")

    f.write("八、核心特征组合的医学解释\n")
    f.write("-" * 50 + "\n")
    for i, (_, row) in enumerate(final_combos.head(5).iterrows(), 1):
        combo_items = sorted(row['项集'])
        f.write(f"  【组合{i}】 {' + '.join(combo_items)}\n")
        f.write(f"    支持度={row['高风险支持度']:.4f}, 对比Lift={row['对比Lift']:.2f}\n")
        for item in combo_items:
            matched_exp = '该特征为高风险人群的常见伴随特征'
            for key, exp in explanations.items():
                if key in item:
                    matched_exp = exp
                    break
            f.write(f"    · {item}: {matched_exp}\n")
        interp = generate_interpretation(combo_items)
        f.write(f"    → 综合解释: {interp}。多维度特征共现表明该人群存在"
                f"「体质偏颇-脂代谢紊乱-活动/代谢异常」的链式病理机制。\n\n")

print(f"\n  -> 摘要已保存: output/q2_2/summary.txt")


# ======================================================================
# 最终总结
# ======================================================================
print("\n" + "=" * 70)
print("最终总结")
print("=" * 70)

print(f"\n1. 目标群体: 痰湿体质+高风险 = {n_high}人 (占痰湿体质{n_tw}人的{n_high/n_tw:.1%})")

print(f"\n2. 高频特征(高风险组中频率>50%):")
for feat in freq_high.index:
    if freq_high[feat] > 0.5:
        f_nh = freq_nonhigh.get(feat, 0)
        print(f"   {feat}: {freq_high[feat]:.2%} (非高风险: {f_nh:.2%})")

print(f"\n3. 核心特征组合Top-5:")
for i, (_, row) in enumerate(final_combos.head(5).iterrows(), 1):
    print(f"   组合{i}: {row['组合']}")
    print(f"          支持度={row['高风险支持度']:.4f}, "
          f"Lift={row['对比Lift']:.2f}, 得分={row['综合得分']:.4f}")

print(f"\n4. 关联规则Top-5:")
if len(rules) > 0:
    for _, row in rules.head(5).iterrows():
        ant = ' + '.join(sorted(row['antecedents']))
        cons = ' + '.join(sorted(row['consequents']))
        print(f"   {ant} → {cons} "
              f"(支持度={row['support']:.4f}, 置信度={row['confidence']:.4f}, Lift={row['lift']:.4f})")

print(f"\n所有结果已保存至 output/q2_2/ 目录")
