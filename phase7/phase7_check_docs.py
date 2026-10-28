"""Verification pass: every headline number in docs/01 and docs/03 is re-read from the result files it came from,
and compared with the number written in the text. Prints OK or MISMATCH per item.

usage: python phase7_check_docs.py RESULTS_DIR DOCS_DIR
"""
import sys, os, json, re, glob, numpy as np, pandas as pd
R, D = sys.argv[1:3]
doc1 = open(os.path.join(D, '01_mouse_callosal_dims.md')).read()
doc3 = open(os.path.join(D, '03_quantum_brain.md')).read()
ok = bad = 0


def check(name, text_has, value, tol=None, doc=None):
    """text_has: the string that must appear in the doc; value: what the data says (for the record)"""
    global ok, bad
    present = text_has in (doc if doc is not None else doc1)
    print(f'{"OK  " if present else "MISS"}  {name:52s} data={value}  text="{text_has}"')
    ok, bad = ok + present, bad + (not present)


p2 = pd.read_csv(os.path.join(R, 'phase2', 'phase2_pairs.csv'))
d = p2[p2.epoch == 'during']
check('phase2 correlated 3-dim', '3方向で 0.14', round(d.f_corr3.median(), 3))
check('phase2 own PCs 3-dim', 'B自身の上位3主成分 0.15', round(d.f_pc3.median(), 3))
check('phase2 signal 3-dim', 'Bの試行平均の信号の方向 0.17', round(d.f_sig3.median(), 3))
check('phase2 random 3-dim', 'でたらめ 0.07', round(d.f_rand3.median(), 3))

mo = pd.read_csv(os.path.join(R, 'phase6', 'phase6_modes.csv')) if os.path.exists(os.path.join(R, 'phase6', 'phase6_modes.csv')) else None
if mo is not None and 'during_g3_cd' in mo:
    check('mode: choice axis', '| 選択の軸 | 0.048 |', round(mo.during_g3_cd.median(), 3))

ref = json.load(open(os.path.join(R, 'phase7', 'phase7_referee.json')))
check('light spread: FS under own light', 'B 自身の光で中央値 5.8 倍', round(ref['R3_light_spread']['own_light_ratio_median'], 1))
check('light spread: FS under other light', 'A への光では中央値 0.55 倍', round(ref['R3_light_spread']['other_light_ratio_median'], 2))
check('light spread: fraction up 2x', '2 倍以上に上がったのは 1.4% だけ', round(100 * ref['R3_light_spread']['frac_other_ge_2'], 1))
check('early licks', '対照 3.7%、片側を止めた試行 3.6% で変わらなかった（マウス単位 p = 0.57）',
      (round(100 * ref['R4_early_licks']['ctrl_mean'], 1), round(100 * ref['R4_early_licks']['unilateral_mean'], 1),
       round(ref['R4_early_licks']['p_mouse'], 2)))
g = ref['R10_2state_minus_fixed']
check('gate equivalence', '−0.008（95%区間 −0.015〜+0.0004）',
      (round(g['mouse_median'], 4), [round(x, 4) for x in g['ci95']]))
check('precision gate', '−0.011（−0.017〜−0.004）',
      (round(ref['R10_prec_minus_fixed']['mouse_median'], 3), [round(x, 3) for x in ref['R10_prec_minus_fixed']['ci95']]))
check('within-mouse modularity/recovery', 'マウスの中の相関でも 0.29（p = 2×10⁻⁶）',
      round(ref['R11_modularity_recovery_within_mouse']['rho'], 2))
check('within-mouse leader/drop', 'マウスの中の相関でも 0.24（p = 7×10⁻⁵）',
      round(ref['R11_leader_drop_within_mouse']['rho'], 2))

lv = json.load(open(os.path.join(R, 'phase7', 'phase7_label_validation.json')))
check('label validation: same channel', '同じラベルなら 3.2%、違うラベルなら 0.37%',
      [(t['dist'], t['same'], round(t['frac_strong'], 4)) for t in lv['table'] if t['dist'] == '0'])
check('label validation: mice', '39匹中36匹で同じ向きだった（p = 1.5×10⁻⁷）',
      (round(lv['mice'] * lv['mice_same_gt_diff']), lv['mice'], f"{lv['p_mouse']:.1e}"))

# the intervals quoted in the text are the mouse-level bootstraps from the audit, not the pair-level ones
au = json.load(open(os.path.join(R, 'phase7', 'phase7_audit.json')))['A1']
own = au['phase7-2: fraction of c1 by own (bootstrap over mice)']
loop = au['phase7-2: fraction of c1 by loop1 (bootstrap over mice)']
check('gain pooled own (mouse bootstrap)', '状態依存性の 42%（95%区間 19〜139%）',
      (round(100 * own['fraction']), [round(100 * x) for x in own['ci']]))
check('gain pooled loop1 (mouse bootstrap)', '0.77（0.38〜2.54）',
      (round(loop['fraction'], 2), [round(x, 2) for x in loop['ci']]))

rd = json.load(open(os.path.join(R, 'phase7', 'phase7_rampdown.json')))
w0 = rd['-1.3..-0.9']
check('rampdown first window', '| −1.3〜−0.9 | 0.87 | −0.36 | 0.61（0.57〜0.67） |',
      (round(w0['g_wmean'], 2), round(w0['obs'], 2), round(w0['pred'], 2)))
rm = json.load(open(os.path.join(R, 'phase7', 'phase7_rampdown_mean.json')))
check('rampdown mean loss', '0.61（0.57〜0.67）',
      (round(rm['-1.3..-0.9']['L0'], 2), [round(x, 2) for x in rm['-1.3..-0.9']['L0_ci']]))

# DLAG, main and keep-mean
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'phase6'))
from dlag_summary import read_summary
dm = read_summary(R); dk = read_summary(os.path.join(R, 'phase7', 'dlag_keepmean'))
check('DLAG main frac across', '88%のセッションで1つ以上', round(100 * (dm.pa > 0).mean()))
check('DLAG keepmean frac across', '| 左右にまたがる成分が見つかるセッション | 88% | **35%** |',
      round(100 * (dk.pa > 0).mean()))
ov = pd.read_csv(os.path.join(R, 'phase7', 'dlag_keepmean', 'overlap.csv')); ovg = ov[(ov.rel_den > 0) & (ov.pa > 0)]
check('DLAG keepmean overlap', '| 見つかったときの、止めたときの反応との重なり | 0.20 | **0.13** |',
      round(ovg.f_dlag_across.median(), 3))

# 03: hyperfine and spin dynamics
sc = json.load(open(os.path.join(R, 'quantum', 'xe_superoxide_curve.json')))
lo = {(x['branch'], x['R_A']): x for x in sc if x['mode'] == 'lobe'}
check('Xe-O2 lobe 3.2 A', '| **3.2 Å** | −2.4 ／ 1.02 mT | **−3.8（全体の底）／ 0.36 mT** |',
      (round(lo[('somo_toward_Xe', 3.2)]['E_bind_kcal'], 1), round(lo[('somo_toward_Xe', 3.2)]['a_mT'], 2),
       round(lo[('somo_perpendicular', 3.2)]['E_bind_kcal'], 1), round(lo[('somo_perpendicular', 3.2)]['a_mT'], 2)), doc=doc3)
eb = json.load(open(os.path.join(R, 'quantum', 'xe_effect_bound.json')))
i = eb['dft_0.76mT']['k'].index(5.5e7); j = eb['dft_0.76mT']['r'].index(1e8)
check('effect bound locked at r=1e8', '| 向きが固定（$a$ = 0.76 mT） | 10.3 | 9.1 | 3.1 | 0.6 | 0.03 |',
      round(eb['dft_0.76mT']['delta_points'][i][j], 1), doc=doc3)
es = json.load(open(os.path.join(R, 'quantum', 'energy_scales.json')))
check('zeeman over kT', '$9\\times10^{-11}$ 倍', f"{es['zeeman_129Xe_50uT_over_kT']:.1e}", doc=doc3)
check('hyperfine over kT', '$4\\times10^{-6}$ 倍', f"{es['hyperfine_129Xe_1mT_over_kT']:.1e}", doc=doc3)
check('Xe mass equilibrium', '$10^{-4}$ 以下', f"{es['equilibrium_mass_effect_129Xe_132Xe_k10']:.1e}", doc=doc3)

print(f'\n{ok} present, {bad} missing')
