"""Fill the DLAG placeholders in the documents from results/phase6/phase6_dlag_summary.json (run once)."""
import json, sys

S = json.load(open('results/phase6/phase6_dlag_summary.json'))


def pct(x):
    return f'{100 * x:.0f}%'


def pv(p):
    m, e = f'{p:.0e}'.split('e')
    return f'{m}×10⁻{int(e[1:])}' if p < 0.01 else f'{p:.2f}'


DOC01 = f"""| 分かったこと | 値（{S['n_sessions']}セッション・{S['n_mice']}匹、中央値） |
| --- | --- |
| DLAG は、左右にまたがる成分を見つける | {pct(S['frac_pa_pos'])} のセッションで、またがる成分が1つ以上選ばれた（中央値 {S['pa_median']:.0f} 個）。左右が共有するゆらぎのうち、またがる成分の割合は {S['across_strength']:.2f} |
| 成分の多くに向きがある | またがる成分の {pct(S['frac_directional'])} で、遅れを0にすると交差検証の尤度が下がった。遅れの大きさの中央値 {S['med_abs_delay_dir']:.0f} ms、時間スケール {S['tau_median_ms']:.0f} ms。向きは左右で偏らない（code1 が先行 {pct(S['frac_code1_leads'])}） |
| **しかし、片側を止めたときの反応は、DLAG のまたがる部分空間にほとんど入らない** | 反応のうちの割合 {S['f_dlag_across']:.2f}（同じ次元の受信部分空間なら {S['f_rec_pa']:.2f}、でたらめなら {S['f_rand_pa']:.2f}。受信部分空間との差 p = {pv(S['p_across_vs_rec'])}） |
| 「AからBへ」の向きの成分に絞っても同じ | A先行 {S['f_dlag_AtoB']:.2f}、B先行 {S['f_dlag_BtoA']:.2f}（差なし、p = {pv(S['p_AtoB_vs_BtoA'])}） |
| 共有成分をすべて（またがる＋B内）使っても | {S['f_dlag_shared']:.2f}（同じ次元の受信部分空間なら {S['f_rec_shared']:.2f}） |
| **DLAG の推定する信号の向きは、因果の向きと無関係** | セッションごとの「流れの向き」（遅れの符号を分散で重み付け）と、片側を止めて測った因果の非対称の順位相関 {S['flow_rho']:+.2f}（p = {pv(S['flow_p'])}、{S['flow_n']}セッション） |

**読み方**：DLAG は、左右にまたがり、向きを持った共有成分をはっきり見つける。古典的な動的モデルで、左右の同時のゆらぎは表現できる（B4 の強化）。
しかし、その成分は「片側を失ったときに反対側で起きること」をほとんど含まず、その向きは因果の向きと一致しない。
DLAG の著者自身も、遅れは「共通の入力でも生じうる」と注意している。ここでは、**相関から推定できる最良の通信の推定でも、介入で測った通信を捉えられない**ことを、半球間で量として示した。Phase 2・3 の主張は、方法を最強にしても変わらなかった。"""

SUM01 = (f"相関から信号の流れを推定する最良の方法の一つ（DLAG、遅れと向きを推定する）を自作の実装で全セッションにかけても、"
         f"片側を止めたときの反応のうち DLAG の左右にまたがる成分に入るのは {S['f_dlag_across']:.2f}（受信部分空間なら {S['f_rec_pa']:.2f}）で、"
         f"推定された信号の向きは因果の向きと無関係だった（順位相関 {S['flow_rho']:+.2f}）（Phase 6-5）。")

DOC03 = (f"Gokcen ら（2022）の DLAG を PyTorch で書き直し（合成データで遅れと次元の復元を確認）、普通の試行にかけた（{S['n_sessions']}セッション）。\n"
         f"またがる成分は {pct(S['frac_pa_pos'])} のセッションで選ばれ、その {pct(S['frac_directional'])} は有意な遅れ（中央値 {S['med_abs_delay_dir']:.0f} ms）を持った。"
         f"古典的な動的モデルで、左右の同時のゆらぎとその時間の向きまで表現できる。\n"
         f"しかし、片側を止めたときの反応のうち、DLAG のまたがる部分空間に入るのは {S['f_dlag_across']:.2f}（受信部分空間なら {S['f_rec_pa']:.2f}、でたらめなら {S['f_rand_pa']:.2f}）で、"
         f"DLAG の推定する向きは、片側を止めて測った因果の向きと無関係だった（順位相関 {S['flow_rho']:+.2f}、p = {pv(S['flow_p'])}）。\n"
         f"**判定**：B4（古典的なモデルで統計は再現できる）は強まったが、同時に「統計を再現できるモデルが、通信を正しく表すとは限らない」ことも示された。Q4 の観点では、"
         f"脳のシミュレーションに足りないのは量子の計算資源ではなく、**介入のデータ**である。")

RD = (f"| Does DLAG (re-implemented in PyTorch; delays and dimensionality recovered on synthetic data) find the callosal channel in normal trials? (B4') | "
      f"It finds across-hemisphere latents in {pct(S['frac_pa_pos'])} of {S['n_sessions']} sessions, {pct(S['frac_directional'])} with significant delays "
      f"(median {S['med_abs_delay_dir']:.0f} ms). But they hold {S['f_dlag_across']:.2f} of the silencing response (receiving subspace of the same size: "
      f"{S['f_rec_pa']:.2f}; random {S['f_rand_pa']:.2f}), and the inferred direction of flow does not match the causal asymmetry measured by silencing "
      f"(ρ = {S['flow_rho']:+.2f}, p = {S['flow_p']:.2f}) |")

for path, key, txt in (('docs/01_mouse_callosal_dims.md', '{{DLAG_RESULTS}}', DOC01),
                       ('docs/01_mouse_callosal_dims.md', '{{DLAG_SUMMARY}}', SUM01),
                       ('docs/03_bold_hypotheses.md', '{{DLAG03}}', DOC03),
                       ('README.md', '{{DLAG_README}}', RD)):
    s = open(path).read()
    if key not in s:
        sys.exit(f'{key} not found in {path}')
    open(path, 'w').write(s.replace(key, txt))
print('filled')
