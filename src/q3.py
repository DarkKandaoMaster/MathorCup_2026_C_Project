"""问题3：痰湿体质患者6个月干预方案优化"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import os

# ==================== 字体与路径设置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'output', 'q3')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 数据加载 ====================
df = pd.read_csv(os.path.join(DATA_DIR, 'raw_data.csv'), index_col=0)
df_tw = df[df['体质标签'] == 5].copy()
print(f"痰湿体质患者数: {len(df_tw)}")

# ==================== 构建优化模型 ====================
BUDGET = 2000

def get_tcm_level(score):
    if score < 59: return 1
    elif score <= 61: return 2
    else: return 3

TCM_COST = {1: 30, 2: 80, 3: 130}
ACT_COST = {1: 3, 2: 5, 3: 8}  # 单次

def allowed_intensities(age_group, act_score):
    age_ok = {1,2,3} if age_group in [1,2] else {1,2} if age_group in [3,4] else {1}
    score_ok = {1} if act_score < 40 else {1,2} if act_score < 60 else {1,2,3}
    return sorted(age_ok & score_ok)

def simulate(s0, plan):
    """plan = [(intensity, freq), ...] 6个月"""
    score, cost = s0, 0
    details = []
    for m, (inten, freq) in enumerate(plan, 1):
        lv = get_tcm_level(score)
        mc = TCM_COST[lv] + freq * 4 * ACT_COST[inten]
        cost += mc
        rr = (inten * 0.03 + max(0, freq - 5) * 0.01) if freq >= 5 else 0.0
        ns = max(0, score * (1 - rr))
        details.append({'月份': m, '月初积分': round(score,2), 'TCM分级': lv,
                        '强度': inten, '频率': freq, '下降率': f'{rr:.0%}',
                        '月末积分': round(ns,2), '月成本': mc})
        score = ns
    return round(score, 4), cost, details

def optimize(s0, ag, asc):
    """暴力遍历固定方案(强度,频率不变的6个月计划)"""
    best, best_fs, best_cost, best_det = None, 1e9, 0, None
    for inten in allowed_intensities(ag, asc):
        for freq in range(1, 11):
            plan = [(inten, freq)] * 6
            fs, cost, det = simulate(s0, plan)
            if cost <= BUDGET and fs < best_fs:
                best, best_fs, best_cost, best_det = (inten, freq), fs, cost, det
    return best, best_fs, best_cost, best_det

# ==================== 既然我们都已经做出模型了，所以我也顺便把题目给的数据中278位确诊为“痰湿体质”的患者都找了个方案 ====================
rows = []
for idx, r in df_tw.iterrows():
    s0, ag, asc = r['痰湿质'], int(r['年龄组']), r['活动量表总分（ADL总分+IADL总分）']
    bp, fs, cost, _ = optimize(s0, ag, asc)
    if bp:
        rows.append({'样本ID': idx, '初始积分': s0, '年龄组': ag, '活动总分': asc,
                     '允许强度': str(allowed_intensities(ag, asc)),
                     '最优强度': bp[0], '最优频率': bp[1],
                     '最终积分': fs, '下降': round(s0-fs,2), '总成本': cost})

res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUTPUT_DIR, 'all_plans.csv'), index=False, encoding='utf-8-sig')
print(f"\n全部{len(res)}名患者求解完毕")
print(f"最优强度分布: {dict(res['最优强度'].value_counts().sort_index())}")
print(f"最优频率分布: {dict(res['最优频率'].value_counts().sort_index())}")

# ==================== “样本ID”为1、2、3的病患的详细方案 ====================
AGE_L = {1:'40-49',2:'50-59',3:'60-69',4:'70-79',5:'80-89'}
ACT_N = {1:'1级(低)',2:'2级(中)',3:'3级(高)'}
TCM_N = {1:'基础调理(1级)',2:'中度调理(2级)',3:'强化调理(3级)'}

for sid in [1, 2, 3]:
    r = df_tw.loc[sid]
    s0, ag, asc = r['痰湿质'], int(r['年龄组']), r['活动量表总分（ADL总分+IADL总分）']
    bp, fs, cost, det = optimize(s0, ag, asc)

    print(f"\n{'='*60}")
    print(f"样本{sid}: 积分={s0}, 年龄={AGE_L[ag]}, 活动总分={asc}, "
          f"允许强度={allowed_intensities(ag, asc)}")
    print(f"最优方案: 强度={bp[0]}({ACT_N[bp[0]]}), 频率={bp[1]}次/周")
    print(f"最终积分={fs}, 下降={s0-fs:.2f}({(s0-fs)/s0*100:.1f}%), 总成本={cost}元")
    print(f"{'月份':>4} {'月初':>6} {'TCM':>6} {'强度':>4} {'频率':>4} {'降率':>5} "
          f"{'月末':>6} {'月成本':>6}")
    for d in det:
        print(f"{d['月份']:>4} {d['月初积分']:>6} {d['TCM分级']:>6} {d['强度']:>4} "
              f"{d['频率']:>4} {d['下降率']:>5} {d['月末积分']:>6} {d['月成本']:>6}")

    pd.DataFrame(det).to_csv(os.path.join(OUTPUT_DIR, f'sample_{sid}_plan.csv'),
                             index=False, encoding='utf-8-sig')

# ==================== 总结出“什么样的患者-什么样的最优方案”的匹配规律 ====================
print(f"\n{'='*60}")
print("匹配规律分析")
print(f"{'='*60}")

res['能力组'] = res['允许强度'].map(lambda x: 'C(1-3级)' if '3' in x
                           else 'B(1-2级)' if '2' in x else 'A(仅1级)')
res['初始TCM'] = res['初始积分'].apply(get_tcm_level)

for grp in ['A(仅1级)', 'B(1-2级)', 'C(1-3级)']:
    sub = res[res['能力组'] == grp]
    if len(sub) == 0: continue
    print(f"\n{grp}: {len(sub)}人")
    for tcm in sorted(sub['初始TCM'].unique()):
        t = sub[sub['初始TCM'] == tcm]
        mi, mf = t['最优强度'].mode().iloc[0], t['最优频率'].mode().iloc[0]
        print(f"  TCM{tcm}级: {len(t)}人, 众数方案=强度{mi}+频率{mf}次/周, "
              f"均最终积分={t['最终积分'].mean():.1f}, 均成本={t['总成本'].mean():.0f}元")

print(f"\n核心规律: 在允许最大强度下选预算允许的最高频率; "
      f"初始TCM分级越高→月成本越高→可用频率越低")
print(f"结果已保存至 output/q3/")
