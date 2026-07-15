# adversarial_block_gan の解説

## 概要

`adversarial_block_gan` は、通常のニューラルネットワーク型GANではなく、金融時系列向けに設計した **adversarial calibrated generator** です。

目的は、`sp500` リターンと `DGS10` 金利差からなる 1260 点、2 次元の5年時系列を、実データの5年窓と見分けにくい形で生成することです。

このモデルは大きく次の3つから構成されています。

1. 実データの局所ブロックを再構成する確率的生成器
2. 金融時系列の stylized facts を測る特徴量ベースの識別器
3. 多数候補から識別器を通過しやすい系列を選ぶ rejection/selection

実装ファイルは主に以下です。

- `pyfiles/train_adversarial_block_gan.py`
- `src/evaluation/adversarial.py`
- `pyfiles/evaluate_adversarial_detectors.py`

## なぜこの方式にしたか

最初に試した MCGAN / QuantGAN 系の生成データは、通常の平均・分散だけを見るとある程度近くても、次のような特徴で簡単に見破れました。

- `DGS10` が実データのような 0.01 刻みになっていない
- `DGS10` のゼロ比率や中央値が不自然
- ボラティリティクラスタリングが弱い
- rolling correlation の分布が狭すぎる
- cross-lag correlation が不自然
- tail dependence や上下極端値の出方が実データの5年窓と違う
- 累積パスのレンジや drawdown/drawup が不自然

そのため、単純にGANを長く学習するだけではなく、「見破る側」を明示的に作り、その評価を通過する生成方式にしました。

## 入力データ

入力は以下のCSVです。

```text
data/train_sp500_us10y.csv
```

列は次の2系列を使います。

- `sp500`: S&P500の日次リターン
- `DGS10`: 10年米国債利回りの日次差分

学習時には、実データ全体を 1260 点の5年窓に分割します。デフォルトでは 21 営業日刻みの rolling window を作るため、約642個の実5年窓が参照分布になります。

## 生成器の考え方

生成器は、実データの一部をそのままコピーして提出するものではありません。

実データの局所ブロックを時系列構造の prior として使い、そこに以下の変形を加えます。

- ランダムなブロック長で切り出す
- 近いレジームからブロックを選ぶ
- 一定確率で連続ブロックをつなげる
- パス全体・ブロック単位でスケールを少し変える
- 平均シフトを加える
- t分布ノイズを加える
- `DGS10` を 0.01 刻みに戻す

この結果、生成系列は実データの時系列構造を利用しつつ、単純な5年窓コピーにはならないようにしています。

## 主要パラメータ

`GeneratorParams` の主な意味は以下です。

| パラメータ | 意味 |
|---|---|
| `block_mean` | 平均ブロック長 |
| `block_min` | 最小ブロック長 |
| `block_max` | 最大ブロック長 |
| `noise_scale` | 実データ標準偏差に対するノイズ量 |
| `sp500_noise_floor` | `sp500` に最低限入れる連続ノイズ |
| `dgs10_noise_multiplier` | `DGS10` ノイズの倍率 |
| `scale_jitter` | パス・ブロック単位のスケール揺らぎ |
| `mean_jitter` | 平均シフトの大きさ |
| `regime_radius` | 参照する近傍レジーム範囲 |
| `continuation_prob` | 直前ブロックの続きを使う確率 |
| `tail_prob` | 極端値サンプルを混ぜる確率 |
| `dgs10_tick` | `DGS10` の刻み幅。実データに合わせて 0.01 |
| `profile` | balanced / calm / stress / rate_shock のレジーム指定 |

今回の最終実行で選ばれたパラメータは以下に保存されています。

```text
reports/adversarial/adversarial_block_gan/best_params.json
```

## 識別器・評価器

識別器はニューラルネットワークの1個の判別器ではなく、金融時系列の特徴量セットです。

実装は `src/evaluation/adversarial.py` にあります。

特徴量は主に次のカテゴリです。

### 周辺分布

- mean
- std
- skewness
- excess kurtosis
- min / max
- quantile
- VaR 相当の下側分位点
- expected shortfall
- positive fraction
- zero fraction
- sign switch fraction

### tail と極端値

- 0.1%, 1%, 5%, 95%, 99%, 99.9% 分位点
- lower expected shortfall
- upper expected shortfall
- max drawdown
- max drawup
- 累積パスの最大・最小・レンジ

### 時系列依存

- raw ACF
- absolute ACF
- squared ACF
- rolling volatility の平均・分散・分位点
- rolling volatility の自己相関

### 2系列間依存

- Pearson correlation
- Spearman correlation
- absolute correlation
- squared correlation
- cross-lag correlation
- rolling correlation の平均・分散・分位点
- 高ボラティリティ局面の rolling correlation
- downside / upside correlation
- lower / upper tail dependence

### DGS10 固有の検査

`DGS10` は実データでは 0.01 刻みです。そのため以下を明示的に評価します。

- 0.01 tick に乗っている割合
- 0 の割合
- 中央値
- 正負比率

GAN出力の多くはここで見破られます。

## スコアリング

実データの5年窓特徴量から robust reference distribution を作ります。

具体的には、各特徴量について以下を使います。

- median を中心
- IQR ベースの robust scale
- robust z-score
- shrinkage covariance による Mahalanobis distance
- robust L2 distance
- 最近傍特徴距離

主な評価指標は以下です。

| 指標 | 意味 |
|---|---|
| `mahalanobis` | 実5年窓の多変量特徴分布からの距離 |
| `mahalanobis_pvalue` | 実5年窓の中で、これ以上遠いものがどれくらいあるか |
| `outlier_rate_p05` | `mahalanobis_pvalue < 0.05` の割合 |
| `robust_l2` | robust z-score の平均的な大きさ |
| `max_abs_z` | 最も目立つ単一特徴の z-score |
| `nearest_feature_distance` | 実5年窓特徴量への最近傍距離 |
| `nearest_feature_pvalue` | 実窓に近すぎる、つまりコピー疑いの検査 |
| `exact_row_match_fraction` | 実データと完全一致する行の割合 |
| `nearest_raw_window_rms` | 実データの生5年窓との最小RMS距離 |

## 学習・選抜フロー

`train_adversarial_block_gan.py` の流れは次の通りです。

1. 実データから 1260 点の rolling windows を作る
2. 各実窓の特徴量を計算する
3. robust reference distribution を作る
4. 生成器パラメータをランダムに多数試す
5. 各パラメータで複数の生成パスを作る
6. 生成パスの特徴分布と実窓特徴分布の距離を計算する
7. 最も良いパラメータを選ぶ
8. そのパラメータで大量の候補パスを作る
9. `mahalanobis`, `robust_l2`, `max_abs_z`, コピー疑いスコアなどから最終候補を選ぶ
10. 最終の 1260 点CSVを保存する

今回の最終出力は以下です。

```text
data/generated/adversarial_block_gan/best_5y_1260.csv
```

複数候補の ensemble は以下です。

```text
data/generated/adversarial_block_gan/ensemble_paths.csv
```

## 最終結果

最終単一パスの主要スコアは以下でした。

```text
mahalanobis_pvalue        0.299
outlier_rate_p05          0.000
robust_l2                 0.905
max_abs_z                 1.824
exact_row_match_fraction  0.000
nearest_raw_window_rms    0.833
```

比較対象の MCGAN / QuantGAN / チーム生成物は、多くが以下のような結果でした。

```text
mahalanobis_pvalue  0.000
outlier_rate_p05    1.000
```

つまり、今回作った評価器では、最終単一パスは実データの5年窓分布の中に入っており、明確な外れ値としては検出されませんでした。

## MPSとの関係

このモデルは PyTorch の勾配学習モデルではありません。

主な処理は NumPy / Pandas による以下です。

- 候補パス生成
- 特徴量計算
- robust z-score 計算
- 候補ランキング

したがって、`--device mps` を指定して速くなるタイプではありません。

性能を上げるなら、MPS化よりも次が有効です。

1. `candidate_pool` を増やす
2. 候補生成・特徴抽出を multiprocessing でCPU並列化する
3. 生成器パラメータ探索をより広くする
4. MPS上で動くニューラル生成器に、この評価器の特徴量lossを組み込む

## 限界

この方式には明確な限界があります。

### 単一パス提出に強く、無条件大量生成には弱い

最終CSVは多数候補から選抜されています。そのため、単一の提出物としては強いですが、同じ生成器から大量に無条件生成した ensemble は discriminator に見破られやすくなります。

実際、`ensemble_paths.csv` に対しては MLP discriminator の AUC が高く出ています。

### 実データブロックへの依存がある

実データの局所ブロックを prior として使うため、完全なニューラル生成モデルよりも実データ構造に強く依存します。

ただし、ノイズ・スケール変形・ブロック再構成・選抜を行っており、実データの5年窓をそのまま提出しているわけではありません。

### 評価器への過適合リスク

今回作った特徴量評価器を通すように選抜しているため、この評価器に含まれていない特徴を使えば別の見破り方が存在する可能性があります。

そのため、今後追加すべき検査は以下です。

- signature kernel / path signature distance
- wavelet spectrum
- hidden Markov regime duration
- rolling tail dependence
- drawdown recovery time
- event clustering
- classifier two-sample test の交差検証強化

## 再実行例

最終モデルの再学習・再選抜は以下で実行できます。

```bash
venv/bin/python pyfiles/train_adversarial_block_gan.py \
  --trials 64 \
  --samples-per-trial 24 \
  --candidate-pool 2048 \
  --ensemble-size 128 \
  --report-dir reports/adversarial/adversarial_block_gan \
  --output-dir data/generated/adversarial_block_gan \
  --classifier-epochs 240
```

評価は以下です。

```bash
venv/bin/python pyfiles/evaluate_adversarial_detectors.py \
  --output-dir reports/adversarial/final_comparison \
  --generated \
    data/generated/adversarial_block_gan/best_5y_1260.csv \
    data/generated/adversarial_block_gan/ensemble_paths.csv
```

## 今後の改善方針

最も現実的な改善は、`candidate_pool` を大きくしてCPU並列化することです。

次に、MPSを活かす場合は、MCGAN/TCN generator に今回の特徴量評価器を differentiable loss として近似的に組み込みます。

特に追加すべきlossは以下です。

- `DGS10` の tick / zero 比率
- quantile matching
- raw / absolute / squared ACF
- rolling volatility distribution
- rolling correlation distribution
- cross-lag correlation
- cumulative path range
- tail dependence

この方向なら、MPSで大規模に学習する意味があります。
