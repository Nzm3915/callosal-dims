# 分離脳ワールドモデル ― 二つの「半分の世界」を、どんな細いリンクで一つにできるか

> 修士課程での研究計画。前段の `01_mouse_callosal_dims.md`（マウスの脳梁の実測）で作った物差しを、
> 人工のワールドモデルに当てる。予備知識ゼロから読めるように、背景 → 基礎 → 候補の比較 → 最良案 → 進め方の順に書く。

---

## 0. 一枚でいうと

脳は、外の世界を内側で「描き直し」ながら見ている。渡邉正峰准教授はこの**生成の過程そのものが意識だ**と考える（生成モデル仮説）。
AIの世界では、この「内側で世界を描く仕組み」をワールドモデルと呼ぶ。2026年10月にはGoogle DeepMindがGenie 4を発表した。
Genie 4は操作できる仮想世界をその場で生成するが、場面の一貫性を保てるのは最大2分で、長くなると物体がずれていく。

本研究は、**左右半分ずつの視野だけを見る二つのワールドモデル**を作り、細いリンクでつなぐ。
そして、物体が正中線をまたいでも「同じ物体」として予測され続けるには、リンクが何を、どれだけ、どんな規則で運べばよいかを調べる。
これは、渡邉先生の「人工意識の主観テスト」（生体の半球と機械の半球をBMIでつなぐ）を、計算機の中で模した実験台になる。

中核の提案は、マウスの研究で立てた仮説 H4 を工学に移した**「精度ゲート付きリンク」**だ。
各半球が自分の推定の確からしさを持ち、相手の情報をどれだけ受け入れるかを、その確からしさで決める。

---

## 1. 背景知識

### 1.1 生成モデルとは

「データがどうやって生まれたか」を真似できるモデルのことだ。
たとえば顔写真の生成モデルは、「目・鼻・口の配置」のような隠れた要因 $z$ を決め、そこから画像 $x$ を描く。

$$
p(x)=\int p(x\mid z)\,p(z)\,dz
$$

$p(z)$ は隠れた要因の出方、$p(x\mid z)$ は要因から画像を描く仕組みである。
逆に画像から要因を推し量るのが「推論」$q(z\mid x)$ だ。
（入門：Lilian Weng「From Autoencoder to Beta-VAE」 https://lilianweng.github.io/posts/2018-08-12-vae/ ／
ニューラルネットの基礎：3Blue1Brown「Neural networks」 https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi ）

### 1.2 渡邉先生の生成モデル仮説と主観テスト

**生成モデル仮説**：脳は外界をそのまま受け取るのではなく、「たぶんこう見えるはずだ」という像を内側から生成し続けている。
先生はこの生成の過程そのものが意識だと考える。CGのレンダリングのように、脳が仮想現実を描いているというたとえで説明されている。

**主観テスト**：機械に意識があるかは、外から行動を見ても原理的に判定できない。
そこで人の片方の半球を機械に置き換え、BMIで生体の半球とつなぐ。
機械側が受け持つ視野の見え方が、本人の一つの意識に自然に溶け込むなら、機械にも意識があると考える。
判定するのが本人の主観なので「主観テスト」と呼ぶ。
（一次資料：Watanabe M. *From Biological to Artificial Consciousness*, Springer, 2022, 第6章「Uploading Human Consciousness」 https://doi.org/10.1007/978-3-030-91138-6_6 ／
一般向け：渡辺正峰『脳の意識 機械の意識』中公新書, 2017 ／ インタビュー https://levtech.jp/media/article/interview/detail_610/ ）

### 1.3 分離脳から分かっていること

脳梁を切った患者では、右視野（左半球）に見せた物の名前は言えるが、左視野（右半球）に見せた物は言えない、といった「分離」が起きる。
ただし近年は、意識そのものは一つのまま、知覚だけが分かれているという見方も強い（Pinto ら 2017）。
さらに2025年のPNAS論文は、脳梁後端（脳梁膨大）を約1 cm残すだけで、左右の統合が保たれることを示した。
**統合に必要なのは脳梁のごく一部かもしれない**。本研究の工学的な問いの出発点はここにある。
（動画：CGP Grey https://www.youtube.com/watch?v=wfYbgdo8e-8 ）

### 1.4 ワールドモデルの中身（RSSM）

Dreamer系のワールドモデルは、RSSM（recurrent state-space model）という構造を持つ。
決定的な記憶 $h_t$ と、確率的な状態 $z_t$ の二層だ。

$$
\begin{aligned}
h_t &= f_\theta\left(h_{t-1},\, z_{t-1},\, a_{t-1}\right) &&\text{（記憶の更新）}\\
z_t &\sim q_\theta\left(z_t \mid h_t,\, o_t\right) &&\text{（観測を見て状態を推す＝知覚）}\\
\hat z_t &\sim p_\theta\left(\hat z_t \mid h_t\right) &&\text{（観測なしで状態を予想する＝想像）}\\
\hat o_t &\sim p_\theta\left(o_t \mid h_t,\, z_t\right) &&\text{（状態から画像を描く＝生成）}
\end{aligned}
$$

学習では、「画像をうまく描けること」と「想像 $p$ が知覚 $q$ に近いこと」を同時に求める。

$$
\mathcal L=\sum_t\Big[-\log p_\theta(o_t\mid h_t,z_t)\;+\;\beta\,\mathrm{KL}\big(q_\theta(z_t\mid h_t,o_t)\,\big\|\,p_\theta(z_t\mid h_t)\big)\Big]
$$

ここで大事なのは、$q$ も $p$ も**平均と分散**を持つことだ。分散の逆数が、その半球の「確からしさ（精度）」としてそのまま使える。
（入門：Ha & Schmidhuber「World Models」の解説サイト https://worldmodels.github.io/ ／ DreamerV3 https://danijar.com/dreamerv3 ／
Genie https://sites.google.com/view/genie-2024/ ）

### 1.5 情報量で「リンクの太さ」を測る

リンクで送るメッセージ $m$ にノイズを足し、事前分布からのずれ（KL）に罰をかけると、
メッセージが運ぶ情報量の上限を「ナット（またはビット）毎ステップ」で測れる。これを**変分情報ボトルネック**と呼ぶ（Alemi ら 2017）。

$$
\mathcal L_{\rm link}=\mathcal L+\lambda\,\mathbb E\left[\mathrm{KL}\big(q(m\mid s_B)\,\|\,\mathcal N(0,I)\big)\right],
\qquad I(m;s_B)\le \mathbb E\left[\mathrm{KL}\right]
$$

$\lambda$ を変えると、リンクの太さと性能の関係（レート–性能曲線）が描ける。
マウス研究の「必要ビットレート」と同じ単位で比べられるのが利点だ。

### 1.6 スパイキングニューラルネットワーク（SNN）

通常のニューラルネットは連続値を一斉に計算する。SNNは、生体のように時間の中で0か1のスパイクを出す。
渡邉先生は、意識を問うなら生体に近いSNNが必要だと考えている。
（入門：snnTorchチュートリアル https://snntorch.readthedocs.io/ ）

---

## 2. 近い研究と、まだ空いているところ

| 分野 | 代表的な研究 | 本研究との違い |
| --- | --- | --- |
| 複数エージェントのワールドモデル | CoDreamer（Toledo & Prorok 2024, arXiv:2406.13600）、Decentralized Collective World Model（Nomura, Aoki, Taniguchi, Horii, ICDL 2025, arXiv:2504.03353）、Communicating Plans, Not Percepts（NeurIPS 2025, arXiv:2508.02912） | 協調や記号の創発が目的。リンクの次元・遅延・ゲートを、脳の実測と同じ物差しで系統的に振ってはいない |
| 集合的予測符号化 | Taniguchi ら, *Front Robot AI* 11:1353870 (2024) | 社会の中での記号創発の理論。二つの半球を一つの主体として統合する問題は扱っていない |
| グローバルワークスペース | Goyal ら ICLR 2022（arXiv:2103.01197）、VanRullen & Kanai, *Trends Neurosci* 44:692–704 (2021) | 多数のモジュールが容量の限られた共有領域を介して通信する。一対一の脳梁型リンクや、片側の損傷からの回復は扱っていない |
| 協調知覚（自動運転） | Where2comm（Hu ら, NeurIPS 2022, arXiv:2209.12836） | 「受け手が不確かな場所だけ送る」という工学的なゲートを持つ。精度ゲートの工学版として参考になる |
| スパイキング・ワールドモデル | Spiking World Model with Multi-Compartment Neurons（arXiv:2503.00713）、Towards biologically plausible Dreaming and Planning in recurrent spiking networks（arXiv:2205.10044） | 単一エージェントで、半球間のリンクは扱わない |
| 長時間の一貫性 | MBench（arXiv:2606.00793）、Video World Models with Long-term Spatial Memory（arXiv:2506.05284） | 一つのモデルの時間方向の一貫性。空間を分けた二つのモデルの間の一貫性ではない |

**空白**：二つのワールドモデルを脳梁のようにつなぐとき、
(1) **何を**（相関する成分か、修復を運ぶ成分か）、
(2) **どれだけ**（次元・ビットレート）、
(3) **どんな規則で**（固定か、状態で変わるゲートか）
運べば一つの世界が保たれるかを、**脳の実測と同じ物差しで**測った研究は見当たらない。

---

## 3. 候補アプローチの比較

| 案 | 中身 | 新規性 | 渡邉研への直結 | 一人での実現性 | 判定 |
| --- | --- | --- | --- | --- | --- |
| A 精度ゲート付き分離脳ワールドモデル | 半視野ずつの二つのRSSMを、次元・遅延・ゲートを制御したリンクでつなぐ。ゲートは各半球の精度で決める | 高い | 主観テストの計算モデル | 高い（小さな2次元の世界、GPU 1枚） | **最良** |
| B 学習する通信 | リンクを端から端まで学習させる（CoDreamer、集合的予測符号化型） | 中 | 中 | 高い | Aの比較対象として組み込む |
| C グローバルワークスペース型 | 多数のモジュールが容量の限られた共有領域で通信する | 中 | 中（意識理論とは近い） | 高い | Aの比較対象として組み込む |
| D 協調知覚型 | 受け手が不確かな場所だけ送る | 低〜中 | 低 | 高い | Aのゲートの一種として組み込む |
| E データ駆動の機械半球 | マウスALMの片側をRNNで学習し、もう片側の実測データとつなぐ | 高い | 非常に高い | 中（ALMは視覚ではない） | M2で取り組む発展課題 |
| F 大規模ワールドモデルの一貫性の解析 | Genie系の内部を調べる | 高い | 中 | 低い（モデルにアクセスできない） | 不採用 |

Aを主軸にし、B・C・Dを「比較対象のリンク」として同じ土俵で戦わせる。
こうすると、「脳型の精度ゲートは、学習まかせの通信や、ワークスペース型より少ないビットで統合を保てるか」という一つの問いにまとまる。

---

## 4. 最良案：精度ゲート付き分離脳ワールドモデル

### 4.1 問い

左右それぞれが半分の視野だけを見る二つのワールドモデルを、リンクでつなぐ。
物体が正中線をまたいでも同じ物体として予測され続け、片側が一時的に壊れても回復するには、
リンクは何を、どれだけ、どんな規則で運べばよいか。

### 4.2 リンクの式

半球Aの内部状態を $s_A=(h_A,z_A)$ とし、Bから届くメッセージを次のように作る。

$$
m_{B\to A}(t)=g_{B\to A}(t)\cdot W_{BA}\,s_B(t-L),\qquad \operatorname{rank}(W_{BA})\le d
$$

$d$ は運ぶ次元、$L$ は遅延、$g$ はゲートである。比べるゲートは次の4種類だ。

| ゲート | 式 | 対応するもの |
| --- | --- | --- |
| 切断 | $g=0$ | 完全な分離脳 |
| 固定 | $g=1$ | 太い管としてのBMI |
| 2状態 | 受け手の選択情報が閾値を超えたら $g=g_1$、下回ったら $g=g_0$ | Chen ら 2021 のモデル |
| 精度重み（提案） | $g_{B\to A}=\dfrac{\pi_B}{\pi_A+\pi_B},\ \ \pi=1/\sigma^2$ | 統計的に最適な合成（Ernst & Banks 2002） |

$\sigma^2$ は RSSM の事後分布 $q(z\mid h,o)$ の分散で、学習中に自然に得られる。追加の仕掛けはいらない。

### 4.3 仮説

| 仮説 | 内容 |
| --- | --- |
| H1 | 統合を保つのに必要な $d$ は小さく、運ぶべきは画素ではなく、物体の位置・速度・同一性のような潜在変数である |
| H2 | 同じ $d$ なら、「修復を運ぶ方向」を送るリンクのほうが、「相関が最大の方向」を送るリンクより統合を保つ。マウスの Phase 2 と同じ問いを人工系で問い直す |
| H3 | 精度ゲートは、片側に雑音や損傷を入れても誤りが反対側に広がらず（モジュール性）、損傷が終わると素早く回復する（修復）。この二つを、固定ゲートより少ないビットレートで両立する |
| H4 | 学習まかせの通信（案B）やワークスペース型（案C）と同じビットレートで比べても、精度ゲートは統合と頑健性で劣らない |
| H5 | 片方の半球をSNNに置き換えても、同じリンク仕様で統合が保たれる |

### 4.4 評価の物差し（マウス研究と共通）

| 指標 | 中身 | マウス研究との対応 |
| --- | --- | --- |
| 正中線の一貫性 | 物体が正中線を越える前後で、位置・速度・同一性の予測誤差が跳ね上がらないか | 統合 |
| 損傷からの回復 | 片側の状態を0.8秒相当ゼロにし、終了後に何ステップで元の予測に戻るか | 片側抑制後の回復（約0.5秒で80%） |
| 誤りの拡散 | 片側に雑音を入れたとき、反対側の予測がどれだけ崩れるか | モジュール性 |
| リンクのビットレート | 情報ボトルネックのKLから求める bit/ステップ | 必要ビットレート $R\approx d\cdot 2B\cdot b$ |
| 相関部分空間と受信部分空間の主角 | マウスの Phase 2 と同じ解析をモデルの内部状態にかける | 本研究の中心指標 |

---

## 5. 方法と進め方（修士2年）

| 時期 | やること | 成果物 |
| --- | --- | --- |
| M1 前期 | 物体が画面を横切る2次元の世界を作る。左右半視野ずつのRSSMを、リンクなし／太いリンクで学習し、「分離」と「統合」の両極を再現する | 正中線で予測が崩れる分離版と、崩れない統合版 |
| M1 後期 | $d$・$L$・ゲート4種類・情報ボトルネックの $\lambda$ を振る。案B・Cも同じ土俵で比較する | レート–統合曲線。H1〜H4の検証 |
| M2 前期 | 片方の半球をSNNに置き換える（代理勾配法）。同じリンク仕様で統合が保たれるかを見る | H5の検証。「生体半球＋機械半球」の計算モデル |
| M2 後期 | 研究室のマウスV1記録が使えれば、正中線をまたぐ刺激や錯視でモデルと生体の反応を比べる。案E（データ駆動の機械半球）にも取り組む | 修士論文 |

計算資源はGPU 1枚で足りる規模に抑える。Genie 4のような大規模モデルは、「一貫性の崩れ」という問題意識の出典として引用するにとどめる。

---

## 6. 注意点

- **意識の有無は判定しない。** 主張は「生成的な表現を一つに統合するためのリンク条件」に限る。
- **通常のニューラルネットへの先生の慎重さに配慮する。** 先生は通常のニューラルネットに意識を認めることに慎重なので、SNNへの置き換え（M2前期）を計画の柱として明記する。
- **小さな世界の結果が大きな世界に効くとは限らない。** 物体の数や世界の複雑さを段階的に増やし、結論がどこまで保たれるかを報告する。
- **精度ゲートが勝つとは限らない。** 負けた場合も、「脳が状態依存のゲートを持つ理由は、ビット効率以外にある」という結論になり、意味のある結果になる。

---

## 7. この研究がつなぐもの

```
マウスの実測（01）          人工のワールドモデル（02）           主観テスト（渡邉研の目標）
相関部分空間 vs 受信部分空間  →  同じ解析をモデル内部にかける     →  BMIが運ぶべき成分
状態依存のゲート            →  精度ゲート付きリンク             →  BMIのゲート規則
回復の時間（約0.5秒）        →  損傷からの回復ステップ数          →  BMIの遅延の許容範囲
```

---

## 8. 取ってきてほしい文献（本文を読めていないもの）

| 文献 | 書誌 | URL |
| --- | --- | --- |
| Pinto Y ほか. Split brain: divided perception but undivided consciousness | *Brain* **140**, 1231–1237 (2017) | https://doi.org/10.1093/brain/aww358 |
| Pinto Y ほか. The split-brain phenomenon revisited: a single conscious agent with split perception | *Trends Cogn Sci* **21**(11), 835–851 (2017) | https://doi.org/10.1016/j.tics.2017.09.003 |
| de Haan EHF ほか. Split-brain: what we know now and why this is important for understanding consciousness | *Neuropsychol Rev* **30**, 224–233 (2020) | https://doi.org/10.1007/s11065-020-09439-3 |
| VanRullen R, Kanai R. Deep learning and the Global Workspace Theory | *Trends Neurosci* **44**, 692–704 (2021) | https://doi.org/10.1016/j.tins.2021.04.005 |
| Hafner D ほか. Mastering diverse control tasks through world models（DreamerV3） | *Nature* **640**(8059), 647–653 (2025) | https://doi.org/10.1038/s41586-025-08744-2 |
| Watanabe M. Uploading Human Consciousness（*From Biological to Artificial Consciousness*, 第6章） | Springer, Cham (2022) | https://doi.org/10.1007/978-3-030-91138-6_6 |

次の文献は無料で読めるので、取得は不要だ。
Taniguchi ら *Front Robot AI* 11:1353870（2024）、arXiv の各論文（1803.10122、1811.04551、2301.04104、2402.15391、2103.01197、2209.12836、2406.13600、2504.03353、2508.02912、2503.00713）。

---

## 参考文献（本文で使ったもの）

- Ha D, Schmidhuber J. World Models. arXiv:1803.10122 (2018). https://arxiv.org/abs/1803.10122
- Hafner D ほか. Learning Latent Dynamics for Planning from Pixels（PlaNet/RSSM）. arXiv:1811.04551 (2018). https://arxiv.org/abs/1811.04551
- Hafner D ほか. Mastering diverse control tasks through world models. *Nature* **640**, 647–653 (2025). https://doi.org/10.1038/s41586-025-08744-2
- Bruce J ほか. Genie: Generative Interactive Environments. arXiv:2402.15391 (2024). https://arxiv.org/abs/2402.15391
- Assran M ほか. V-JEPA 2. arXiv:2506.09985 (2025). https://arxiv.org/abs/2506.09985
- Google's Genie 4 world model adds audio, more realistic worlds, and real-time steering. TechCrunch (2026-10-27). https://techcrunch.com/2026/10/27/googles-genie-4-world-model-adds-audio-more-realistic-worlds-and-real-time-steering/
- Goyal A ほか. Coordination Among Neural Modules Through a Shared Global Workspace. ICLR 2022. arXiv:2103.01197. https://arxiv.org/abs/2103.01197
- Hu Y ほか. Where2comm: Communication-Efficient Collaborative Perception via Spatial Confidence Maps. NeurIPS 2022. arXiv:2209.12836. https://arxiv.org/abs/2209.12836
- Toledo E, Prorok A. CoDreamer: Communication-Based Decentralised World Models. arXiv:2406.13600 (2024). https://arxiv.org/abs/2406.13600
- Nomura K, Aoki T, Taniguchi T, Horii T. Decentralized Collective World Model for Emergent Communication and Coordination. ICDL 2025. arXiv:2504.03353. https://arxiv.org/abs/2504.03353
- Taniguchi T ほか. Collective predictive coding hypothesis: symbol emergence as decentralized Bayesian inference. *Front Robot AI* **11**, 1353870 (2024). https://doi.org/10.3389/frobt.2024.1353870
- Santander T ほか. Full interhemispheric integration sustained by a fraction of posterior callosal fibers. *PNAS* **122**(43), e2520190122 (2025). https://doi.org/10.1073/pnas.2520190122
- Ernst MO, Banks MS. *Nature* **415**, 429–433 (2002). https://doi.org/10.1038/415429a
- Alemi AA, Fischer I, Dillon JV, Murphy K. Deep Variational Information Bottleneck. ICLR 2017. arXiv:1612.00410. https://arxiv.org/abs/1612.00410
- Chen G ほか. *Cell* **184**, 3717–3730 (2021). https://doi.org/10.1016/j.cell.2021.05.026
- Spiking World Model with Multi-Compartment Neurons for Model-based Reinforcement Learning. arXiv:2503.00713 (2025). https://arxiv.org/abs/2503.00713
