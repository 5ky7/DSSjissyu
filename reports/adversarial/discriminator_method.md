# 金融時系列生成データの識別モデル・識別手法

## 位置づけ

本プロジェクトで用いた主な識別手法は、通常のGANにおけるニューラルネットワーク判別器ではない。実データから切り出した5年窓の特徴量分布を参照し、生成データがその分布内に入っているかを判定する、特徴量ベースの外れ値検出器である。

目的は、生成された1260点 x 2系列の5年時系列が、実データ `data/train_sp500_us10y.csv` の歴史的5年窓と見分けにくいかを評価することである。評価対象は `sp500` と `DGS10` の2系列である。

実装上の中心は以下である。

- `src/evaluation/adversarial.py`
- `pyfiles/evaluate_adversarial_detectors.py`

補助的に、生成窓が十分な数ある場合だけ、抽出済み特徴量を入力とする小さなMLP分類器も用いる。ただし、これはraw時系列を直接見るGAN discriminatorではなく、特徴量空間で実データ窓と生成データ窓を分類できるかを確認する補助評価である。

## 利用する特徴量

特徴量は、単純な平均・分散だけではなく、金融時系列のstylized factsと、DGS10固有の離散的性質を含むように設計した。

### 周辺分布の特徴量

各系列について、以下を計算する。

- 平均
- 標準偏差
- 歪度
- 超過尖度
- 最小値
- 最大値
- 正値比率
- ゼロ比率
- 符号反転比率
- 分位点
  - 0.1%
  - 1%
  - 5%
  - 10%
  - 50%
  - 90%
  - 95%
  - 99%
  - 99.9%
- 下側expected shortfall
  - 1%
  - 5%
- 上側expected shortfall
  - 1%
  - 5%

これにより、平均や分散が近いだけでなく、tail、極端値、非対称性まで実データと近いかを確認する。

### 時系列依存の特徴量

各系列について、以下のラグで自己相関を計算する。

- lag 1
- lag 5
- lag 20
- lag 63

対象は以下である。

- raw系列のACF
- 絶対値系列のACF
- 二乗系列のACF

raw ACFは線形自己相関、絶対値・二乗ACFはボラティリティクラスタリングの再現性を見るために用いる。

### Rolling volatility の特徴量

各系列について、以下のrolling windowを使う。

- 20日
- 63日
- 252日

それぞれについて、rolling volatilityの以下を計算する。

- 平均
- 標準偏差
- 5%分位点
- 50%分位点
- 95%分位点
- lag 1自己相関

これは、生成系列が局所的な低ボラ・高ボラ局面や、ボラティリティの持続性を再現できているかを識別するために重要である。MCGAN系の生成結果では、このrolling volatility関連の特徴量が主な見破りポイントになった。

### 累積パス形状の特徴量

各系列の累積和について、以下を計算する。

- 終端値
- 最小値
- 最大値
- レンジ
- 最大ドローダウン
- 最大ドローアップ

これは、日次リターンや金利差の局所統計が近くても、5年分を累積したときのパス形状が不自然でないかを見るための特徴量である。

### 2系列間依存の特徴量

`sp500` と `DGS10` の同時点依存として、以下を計算する。

- Pearson correlation
- Spearman correlation
- 絶対値系列の相関
- 二乗系列の相関

また、ラグ付き依存として、以下のラグでcross-correlationを計算する。

- lag 1
- lag 5
- lag 20
- lag 63

方向は両方を見る。

- `sp500` から将来の `DGS10` へのcross-correlation
- `DGS10` から将来の `sp500` へのcross-correlation

これにより、同時点の相関だけでなく、生成モデルが不自然な先行・遅行関係を作っていないかを確認する。

### Rolling correlation の特徴量

`sp500` と `DGS10` のrolling correlationについて、以下のwindowを使う。

- 20日
- 63日
- 252日

それぞれについて、以下を計算する。

- 平均
- 標準偏差
- 5%分位点
- 50%分位点
- 95%分位点
- `sp500` 高ボラティリティ局面での平均rolling correlation
- `sp500` 低ボラティリティ局面での平均rolling correlation
- rolling correlationと `sp500` rolling volatility の相関

金融時系列では、相関構造が市場局面によって変わるため、単一の全期間相関だけでは不十分である。

### Tail dependence と条件付き相関

下側・上側tailでの同時発生を評価する。

- lower tail dependence
  - 1%
  - 5%
- upper tail dependence
  - 1%
  - 5%
- `sp500` 下側tailかつ `DGS10` 上側tailの比率
  - 1%
  - 5%
- `sp500` downside局面での相関
- `sp500` upside局面での相関

これは、通常時の相関だけでなく、ストレス局面での依存関係が実データと整合的かを確認するためである。

### DGS10固有の特徴量

`DGS10` は実データ上、0.01刻みの値を持つ。そのため、通常の連続値生成モデルは、平均や分散が近くても容易に識別される。

明示的に以下を評価する。

- 0.01 tickに乗っている割合
- ゼロ比率
- 正値比率
- 中央値
- 分位点
- ACF
- rolling volatility

MCGANやQuantGANのraw出力では、`DGS10` が連続値になりやすく、tick率やゼロ比率のずれが強い識別特徴になった。

## 実際の識別への用い方

識別は、単一の特徴量で判定するのではなく、実データ5年窓の特徴量分布に対する距離として行う。

### 1. 実データから参照窓を作る

実データ `data/train_sp500_us10y.csv` から、1260点の5年窓をrolling windowとして切り出す。

現在の設定では、21営業日刻みで切り出し、642個の実データ5年窓を参照分布として使う。

```bash
python pyfiles/evaluate_adversarial_detectors.py \
  --real-csv data/train_sp500_us10y.csv \
  --generated data/generated/example.csv \
  --output-dir reports/adversarial/example_eval
```

### 2. 各窓から特徴量ベクトルを作る

実データ窓と生成データ窓のそれぞれについて、前節の特徴量をすべて計算する。出力される主なファイルは以下である。

- `real_window_features.csv`
- `generated_window_features.csv`

### 3. 実データ特徴量からrobust reference distributionを作る

実データ窓の特徴量から、各特徴量の中心とスケールを推定する。

- 中心: median
- スケール: IQRベースのrobust scale
- 補助スケール: 標準偏差
- 多変量距離: shrinkage covarianceに基づくMahalanobis distance

生成データの特徴量は、この参照分布に対するrobust z-scoreに変換される。

### 4. 生成データをスコアリングする

主に以下のスコアを見る。

| 指標 | 意味 | 判定の向き |
|---|---|---|
| `mahalanobis` | 実データ5年窓の多変量特徴分布からの距離 | 小さいほど自然 |
| `mahalanobis_pvalue` | 実データ窓の中で、同程度以上に遠い窓の割合 | 高いほど自然 |
| `outlier_rate_p05` | `mahalanobis_pvalue < 0.05` となる生成窓の割合 | 低いほど自然 |
| `robust_l2` | robust z-scoreの平均的な大きさ | 小さいほど自然 |
| `max_abs_z` | 最も外れた単一特徴量の絶対z-score | 小さいほど自然 |
| `nearest_feature_distance` | 実データ特徴量への最近傍距離 | 小さすぎるとコピー疑い |
| `nearest_feature_pvalue` | 実データ窓に近すぎないかの検査 | 低すぎるとコピー疑い |
| `exact_row_match_fraction` | 実データと完全一致する行の割合 | 0が望ましい |
| `nearest_raw_window_rms` | 生系列として最も近い実5年窓とのRMS距離 | 小さすぎるとコピー疑い |

基本的には、次の条件を満たすほど「識別されにくい」と判断する。

- `mahalanobis_pvalue` が十分に高い
- `outlier_rate_p05` が0に近い
- `max_abs_z` が3未満
- `exact_row_match_fraction` が0
- `nearest_feature_pvalue` が低すぎない

### 5. どの特徴量で見破られたかを確認する

`feature_zscores.csv` を見ると、生成データの各特徴量が実データ5年窓の分布からどれだけ外れているかが分かる。

例えば、`max_abs_z` が大きい場合は、絶対値が大きいz-scoreの列を確認することで、以下のような弱点を特定できる。

- DGS10のtick率が低い
- DGS10のゼロ比率が実データと違う
- cross-lag correlationが不自然
- rolling volatilityの下側分位が低すぎる
- volatility ACFが弱い
- tail dependenceが過大または過小
- 累積パスのレンジやdrawdownが不自然

### 6. 複数パスがある場合は補助分類器を使う

生成窓が12本以上ある場合は、実データ窓と生成データ窓の特徴量を入力として、小さなMLP分類器を学習する。

この分類器の出力は以下である。

- `classifier_auc`
- `classifier_accuracy`

解釈は次の通りである。

- AUCが0.5に近い: 特徴量空間では実データと生成データを識別しにくい
- AUCが1.0に近い: 特徴量空間で容易に識別できる

ただし、この分類器は生成データが十分な本数ある場合の補助評価である。単一提出パスの評価では、主に外れ値スコアとコピー検査を用いる。

## 識別結果の読み方

本プロジェクトの評価では、既存生成物、QuantGAN、old MCGAN、MPS MCGAN rawは、いずれも歴史的5年窓の特徴分布から大きく外れた。

典型的な見破りポイントは以下である。

- DGS10が0.01刻みになっていない
- DGS10のゼロ比率・中央値・正値比率がずれる
- SP500のrolling volatility persistenceが弱い
- cross-lag correlationが不自然
- tail dependenceが過大または過小
- 1260点のrecursive rolloutで極端値が爆発する

一方、選抜済みのブロック型生成系列は、単一パスとしては以下の値を示した。

| 指標 | 値 |
|---|---:|
| `mahalanobis_pvalue` | 0.299 |
| `outlier_rate_p05` | 0.000 |
| `robust_l2` | 0.905 |
| `max_abs_z` | 1.824 |
| `exact_row_match_fraction` | 0.000 |
| `nearest_raw_window_rms` | 0.833 |

これは、単一5年系列としては実データ5年窓の特徴分布内に入ったことを意味する。ただし、128本のensembleとして見ると補助分類器のAUCは高く、大量生成分布としてはまだ識別可能だった。したがって、主張できる範囲は「単一提出パスとしては識別されにくい」であり、「任意本数を生成しても識別不能」ではない。

## 注意点

この識別手法は、評価器に含めた特徴量に対しては強いが、含めていない性質は直接評価できない。また、生成器が同じ評価器に対して選抜・最適化されている場合、スコアが良くても汎化的な識別不能性を意味しない。

そのため、最終報告では「GAN discriminatorを欺いた」と表現するのではなく、「金融時系列特徴量に基づくadversarial detectorに対して、単一5年系列が歴史的5年窓の分布内に入った」と表現するのが正確である。
