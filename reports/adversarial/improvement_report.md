# 改善レポート: SP500/DGS10 5年時系列生成

最終更新: 2026-07-10

## 目的

1260点 x 2次元の5年日次系列について、実データ `data/train_sp500_us10y.csv` の歴史的5年窓と見分けにくい生成データを作る。

ここで重視する「見分けにくさ」は、60年全体の平均統計に近いことではなく、実データから切り出した多数の5年窓の分布内に入ることとした。理由は、提出対象が単一の5年系列であり、5年ごとの市場レジーム差が大きいためである。

## 評価指標

主評価器は `pyfiles/evaluate_adversarial_detectors.py`。

主な評価軸:

- 周辺分布: mean, std, skewness, kurtosis, quantile, ES
- 時系列性: raw ACF, absolute ACF, squared ACF
- 累積パス形状: terminal, range, max drawdown/drawup
- 2変量依存: contemporaneous correlation, cross-lag correlation, rolling correlation, tail dependence
- DGS10固有性: 0.01刻み比率、ゼロ比率
- コピー検査: exact row match fraction, nearest raw window RMS
- 統合スコア: robust L2, Mahalanobis, nearest feature distance

解釈:

- `mahalanobis_pvalue` が高いほど、歴史的5年窓の特徴分布内にある。
- `outlier_rate_p05 = 0` が望ましい。
- `max_abs_z` は個別特徴の最大逸脱で、3未満を当面の目安にする。
- `exact_row_match_fraction = 0` は実データ丸写しでないことの最低条件。

## 改善履歴

### 0. 起点: 既存生成物と旧GANの評価

対象:

- teammate generated files
- 既存 QuantGAN
- 既存 MCGAN

結果:

| model | mahalanobis_pvalue | outlier_rate_p05 | max_abs_z | 備考 |
|---|---:|---:|---:|---|
| makieda | 0.000 | 1.000 | 3.770 | DGS10 zero/tick 周辺が弱い |
| hashimoto | 0.000 | 1.000 | 4.481 | SP500 vol persistence と DGS10 zero が弱い |
| setsu1 | 0.000 | 1.000 | 5.528 | DGS10 tail/ACF が弱い |
| setsu2 | 0.000 | 1.000 | 12.185 | 依存構造と tail dependence が過大 |
| QuantGAN | 0.000 | 1.000 | 13.191 | rolling vol ACF と tail dependence が不自然 |
| old MCGAN | 0.000 | 1.000 | 53.440 | 累積パスとスケールが破綻 |

判断:

既存生成物はすべて、歴史的5年窓の特徴分布から外れていた。特に以下が見破りやすい。

- DGS10が実データ同様の0.01刻みになっていない。
- DGS10のゼロ比率、中央値、符号比率がずれる。
- SP500/DGS10のcross-lag correlationが不自然。
- rolling volatility persistence が弱い、または過大。
- recursive GAN rolloutで累積パスが破綻する。

次の改善:

まず強い評価器を固定し、その評価器を通る生成方法を作る。純粋なGANだけにこだわるとDGS10離散性や5年レジームを壊しやすいため、GANに準ずるadversarial calibrationを導入する。

### 1. adversarial detector の実装

変更:

- `src/evaluation/adversarial.py` を追加。
- `pyfiles/evaluate_adversarial_detectors.py` を追加。
- 実データから642個の5年ローリング窓を作成し、特徴量分布を参照分布にした。

結果:

この評価器で既存生成物の弱点を特定できた。特に、単純な周辺統計だけでは見落とす以下を検出できた。

- DGS10 tick/zero fraction
- 20日・63日 cross-lag correlation
- rolling correlation の低/高ボラ局面差
- nearest feature distance による記憶疑い

次の改善:

評価器をそのまま生成器の選抜基準に使う。単一提出パスについては、生成候補を多数作り、評価器を通るものを選ぶadversarial rejection方式が有効と判断した。

### 2. Adversarial Block-GAN の導入

変更:

- `pyfiles/train_adversarial_block_gan.py` を追加。
- 実データの局所ブロックを時系列priorにしつつ、ノイズ、スケール揺らぎ、レジーム近傍混合、DGS10 0.01 tick復元を入れた。
- 2048候補からadversarial detectorで単一パスを選抜。

結果:

最終単一パス:

`data/generated/adversarial_block_gan/best_5y_1260.csv`

| model | mahalanobis_pvalue | outlier_rate_p05 | robust_l2 | max_abs_z | exact_row_match_fraction | nearest_raw_window_rms |
|---|---:|---:|---:|---:|---:|---:|
| adversarial_block_gan_best | 0.299 | 0.000 | 0.905 | 1.824 | 0.000 | 0.833 |

改善点:

- 既存生成物がすべて `mahalanobis_pvalue = 0` だったのに対し、0.299まで改善。
- `outlier_rate_p05 = 0`。
- 最大特徴逸脱 `max_abs_z = 1.824` で、主要特徴が歴史的5年窓の範囲内に収まった。
- `exact_row_match_fraction = 0` で丸写しではない。

限界:

128本ensembleとして見ると、feature classifier AUCが高く、無条件に大量生成した分布はまだ見破られる。

| ensemble | classifier_auc | classifier_accuracy | outlier_rate_p05 |
|---|---:|---:|---:|
| adversarial_block_gan_ensemble | 0.958 | 0.883 | 0.266 |

判断:

単一提出物としては現時点で最良。ただし「生成モデルとして大量に使う」観点では、選抜バイアスを検出される。

次の改善:

MPSがTerminalで使えることが確認できたため、GAN本体をより強く学習し、adversarial block selectionへの依存を下げる。ただし、GANのrecursive rolloutが破綻しやすい点に注意する。

### 3. MPS MCGAN 20k の実行

変更:

- Terminalで `--device mps` を使い、既存MCGANを20k step学習。
- Codex内ではMPSが見えないが、Terminalでは `mps available=True` を確認。

結果:

| model | mahalanobis_pvalue | outlier_rate_p05 | robust_l2 | max_abs_z |
|---|---:|---:|---:|---:|
| mps_mcgan_20k raw | 0.000 | 1.000 | 1.526 | 7.106 |

見破りポイント:

- `DGS10_positive_fraction`: z ≈ 7.00
- `DGS10_q_0.5`: z ≈ 6.19
- `DGS10_zero_fraction`: z ≈ 3.77
- `crosscorr_DGS10_to_sp500_lag20`: z ≈ 7.11

判断:

MPSによる学習実行は成功したが、生成品質は採用不可。DGS10が連続値になり、実データの0.01刻み・ゼロ比率を再現できていない。

次の改善:

DGS10の周辺分布はGAN出力に直接任せず、rank-map後処理で実データ5年窓のorder statisticへ写す。これにより、少なくともDGS10離散性と周辺分布を補正する。

### 4. rank-map 後処理

変更:

- `pyfiles/postprocess_rank_map.py` を追加。
- GAN出力の順位構造を残しつつ、各系列の周辺分布を歴史的5年窓に写す。
- DGS10は0.01刻みに丸める。

結果:

20k MCGANにrank-mapを適用:

| model | mahalanobis_pvalue | outlier_rate_p05 | robust_l2 | max_abs_z |
|---|---:|---:|---:|---:|
| mps_mcgan_20k_rankmap | 0.0016 | 1.000 | 1.113 | 6.797 |

改善点:

- DGS10 tick fraction は 1.0 に改善。
- DGS10 zero fraction も実データに近づいた。

残った問題:

- cross-lag correlation が依然として大きく外れる。
- rank-mapは周辺分布を直すだけなので、時系列順序構造の癖は残る。

判断:

rank-mapは必要だが十分ではない。GAN学習時点でcross-lag correlation、ACF、quantile/tailをlossに入れる必要がある。

次の改善:

MCGAN補助lossに、多ラグACF、cross-lag correlation、quantile、tail meanを追加する。

### 5. MCGAN feature-loss v2

変更:

`pyfiles/generate_sigcwgan_mcgan_1260.py` に以下を追加。

- `--raw-acf-weight`
- `--cross-lag-weight`
- `--quantile-weight`
- `--tail-weight`
- `--gan-weight`

狙い:

- discriminatorが強すぎるため、adversarial lossを弱める。
- 統計特徴matchingを主目的にする。

学習中ログの観察:

- 初期よりは改善したが、最終的に `d_loss < 0.01`, `d_fake ≈ 0`, `gan_loss ≈ 1` となり、discriminatorが再びほぼ勝つ状態になった。

結果:

| model | mahalanobis_pvalue | outlier_rate_p05 | robust_l2 | max_abs_z | 備考 |
|---|---:|---:|---:|---:|---|
| featureloss_v2 raw | 0.000 | 1.000 | 107.161 | 480.773 | recursive rolloutで極端値が爆発 |
| featureloss_v2 rankmap | 0.000 | 1.000 | 5.023 | 25.347 | 周辺補正後もラグ構造が破綻 |

raw版の基本統計:

| series | std | min | max |
|---|---:|---:|---:|
| sp500 | 0.277 | -2.574 | 2.396 |
| DGS10 | 3.552 | -34.675 | 30.295 |

rankmap後の主な見破りポイント:

- `DGS10_squared_acf_63`: z ≈ 25.35
- `crosscorr_sp500_to_DGS10_lag20`: z ≈ 22.21
- `crosscorr_DGS10_to_sp500_lag20`: z ≈ 19.20
- `DGS10_acf_20`: z ≈ 17.18

判断:

feature-loss自体よりも、1260点生成時の `recursive rollout` が問題。20日先を生成し、その出力を次の入力にして繰り返すため、誤差が累積し、極端値と長期ラグ構造が破綻した。

次の改善:

学習済みcheckpointを再利用し、生成方式だけ変える。

- `recursive`: synthetic pastを次の条件に使うため誤差累積が起きる。
- `historical_conditioned`: 各ブロックを実データの過去窓で条件付け直すため、局所状態が現実的になる。

### 6. historical-conditioned resampling

変更:

`pyfiles/generate_sigcwgan_mcgan_1260.py` に以下を追加。

- `--generation-mode historical_conditioned`
- `--sample-clip-quantile`
- `--load-generator-checkpoint`

狙い:

- v2の学習済みgeneratorを捨てず、生成時の誤差累積だけを取り除く。
- `--sample-clip-quantile 0.001` で極端なscaled outputを訓練データ分位点内に抑える。
- その後rank-mapでDGS10 tick/zeroを補正する。

実行コマンド:

```bash
venv/bin/python pyfiles/generate_sigcwgan_mcgan_1260.py \
  --algo MCGAN \
  --device mps \
  --steps 0 \
  --batch-size 512 \
  --hidden-dim 128 \
  --hidden-layers 4 \
  --mc-size 8 \
  --load-generator-checkpoint data/generated/mps_mcgan_featureloss_v2_20k/MCGAN/mcgan_5y_1260.generator.pt \
  --generation-mode historical_conditioned \
  --sample-clip-quantile 0.001 \
  --output-dir data/generated/mps_mcgan_featureloss_v2_resample
```

期待する改善:

- raw版の極端値爆発がなくなる。
- rankmap後の長期ACF/cross-lag z-scoreが、v2 rankmapの 20超から大きく低下する。

結果:

| model | mahalanobis_pvalue | outlier_rate_p05 | robust_l2 | max_abs_z | 備考 |
|---|---:|---:|---:|---:|---|
| featureloss_v2_resample raw | 0.000 | 1.000 | 0.784 | 3.770 | 爆発は解消。ただしDGS10 tick/zeroが未補正 |
| featureloss_v2_resample rankmap | 0.003 | 1.000 | 1.032 | 3.678 | v2 rankmapより大幅改善。ただし外れ値判定は残る |
| adversarial_block_gan_best | 0.299 | 0.000 | 0.905 | 1.824 | 依然として最良 |

raw版の基本統計:

| series | std | min | max |
|---|---:|---:|---:|
| sp500 | 0.008 | -0.037 | 0.045 |
| DGS10 | 0.060 | -0.321 | 0.350 |

v2 rawではDGS10標準偏差が3.552、最小値-34.675、最大値30.295だったため、`historical_conditioned` によって再帰生成の爆発は解消した。

rankmap後の主な見破りポイント:

- `sp500_roll252_vol_q05`: z ≈ 3.68
- `sp500_roll63_vol_q50`: z ≈ 3.15
- `sp500_roll63_vol_q05`: z ≈ 3.11
- `sp500_roll20_vol_q05`: z ≈ 2.81
- `DGS10_squared_acf_63`: z ≈ 2.47

判定基準との比較:

| 指標 | 目標 |
|---|---:|
| raw DGS10 std | 0.1未満: 達成。0.060 |
| rankmap max_abs_z | 3未満: 未達。3.678 |
| rankmap mahalanobis_pvalue | 0.05以上: 未達。0.003 |
| rankmap outlier_rate_p05 | 0: 未達。1.000 |

判断:

`historical_conditioned` は有効だった。recursive rollout由来の爆発を取り除き、rankmap後の `max_abs_z` を25.347から3.678まで下げた。

ただし、まだ歴史的5年窓の外れ値として判定される。残った問題は主にSP500のrolling volatility水準が低すぎること、特に20日・63日・252日rolling volatilityの下側分位が実データ5年窓から外れることだった。これはrank-mapが周辺分布を合わせても、時系列的なボラティリティ・クラスタリングを十分に回復しないためである。

次の改善:

MCGAN generator自体の採用優先度を下げる。ここからさらに改善するなら、次のいずれかが必要。

1. 生成後にvolatility clusteringを補正するブロック再配列または局所スケール補正を入れる。
2. MCGANの生成単位を20日から63日以上に伸ばし、rolling volatilityの下側分位を学習可能にする。
3. MCGANは補助成果として扱い、最終提出はAdversarial Block-GANを主成果に据える。

今回の目的は「現時点で最良の単一5年系列」を作ることなので、追加コストに対する改善期待を考えると、3を採るのが妥当である。

### 7. longblock MCGAN v4

変更:

MCGANの生成単位を `p=63, q=63` に伸ばした。狙いは、20日ブロックでは見えにくい63日ラグ・rolling volatilityを学習対象に入れること。

実行設定の要点:

- `--p 63`
- `--q 63`
- `--steps 12000`
- `--batch-size 256`
- `--mc-size 4`
- `--lr-d 3e-5`
- `--gan-weight 0.03`
- `--generation-mode historical_conditioned`
- `--sample-clip-quantile 0.001`

結果:

| model | mahalanobis_pvalue | outlier_rate_p05 | robust_l2 | max_abs_z | 備考 |
|---|---:|---:|---:|---:|---|
| longblock_v4 raw | 0.000 | 1.000 | 0.860 | 6.445 | rawは安定したがDGS10 tick/zero未補正、SP500 vol ACFが弱い |
| longblock_v4 rankmap | 0.003 | 1.000 | 1.000 | 5.020 | v2_resample rankmapより悪化 |
| featureloss_v2_resample rankmap | 0.003 | 1.000 | 1.032 | 3.678 | MCGAN系ではこちらの方が良い |
| adversarial_block_gan_best | 0.299 | 0.000 | 0.905 | 1.824 | 依然として最良 |

raw版の基本統計:

| series | std | min | max |
|---|---:|---:|---:|
| sp500 | 0.009 | -0.057 | 0.045 |
| DGS10 | 0.065 | -0.287 | 0.341 |

rankmap後の主な見破りポイント:

- `sp500_roll252_vol_acf1`: z ≈ 5.02
- `sp500_roll252_vol_q05`: z ≈ 3.17
- `sp500_roll20_vol_q50`: z ≈ 2.78
- `sp500_roll63_vol_q50`: z ≈ 2.73
- `sp500_q_0.1`: z ≈ 2.59

判断:

`p=q=63` に伸ばしても、SP500のrolling volatility persistenceが改善しなかった。むしろrankmap後の `max_abs_z` は3.678から5.020へ悪化した。これは、単純に生成ブロック長を伸ばしても、MCGAN出力の局所ボラティリティ水準と持続性が歴史的5年窓に合わないことを示している。

次の改善:

MCGANをさらに改善するなら、学習設定よりも生成後の volatility clustering 補正が必要である。具体的には、rank-map前またはrank-map後に、実データ5年窓のrolling volatility profileに合わせてSP500の局所スケールを補正する。ただし、この補正はGAN単体というよりhybrid生成になるため、主成果としてはAdversarial Block-GANの方が一貫している。

## 現時点の結論

現時点で最も強い単一提出データは:

`data/generated/adversarial_block_gan/best_5y_1260.csv`

理由:

- `mahalanobis_pvalue = 0.299`
- `outlier_rate_p05 = 0`
- `max_abs_z = 1.824`
- `exact_row_match_fraction = 0`

一方、MPS MCGAN系は実験価値はあるが、現時点ではAdversarial Block-GANを超えていない。最も良かったMPS MCGAN派生は `featureloss_v2_resample rankmap` であり、longblock v4はそれを超えなかった。

## 次に取るべき改善方針

優先順位:

1. 最終提出はAdversarial Block-GANを基本にする。
2. MCGAN改善実験は、MPSでGANを学習した補助成果として報告する。
3. MCGANをさらに試すなら、局所volatility profile補正を追加する。
4. ただし、最終提出品質を上げる目的では、MCGANではなくAdversarial Block-GAN側のensemble検出耐性を上げる。
5. 具体的には、候補選抜後の分布が狭くなりすぎないよう、レジーム・相関・ボラティリティ水準ごとのstratified selectionを導入する。

根拠:

- MCGAN rawは再帰生成で極端値が爆発しており、学習loss改善だけでは解決しない。
- rank-mapは周辺分布には効くが、時系列順序構造には効かない。
- Adversarial Block-GANは、現時点で唯一、評価器の統合スコアで歴史的5年窓内に入っている。
- historical-conditioned resamplingによりMCGANの爆発は解消したが、rolling volatility下側分位が残り、外れ値判定を抜けられなかった。
- longblock MCGANでもrolling volatility persistenceは改善せず、MCGAN系の主要な残課題はSP500の局所ボラティリティ構造だと分かった。
