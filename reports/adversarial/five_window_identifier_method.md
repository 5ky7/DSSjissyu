# 5本候補から実データ2本を識別する方法

## 問題設定

最終的に解くべき問題は、5本の2変量時系列から実データ2本と生成データ3本を識別することである。

- 各系列は `sp500` と `DGS10` の2次元
- 長さは5年分、1260点
- 実データは約60年の履歴から任意の5年分を切り出したもの
- 生成データは3本
- 生データの完全一致や最近傍raw window距離などのコピー指標は使わない

この設定では、単一系列ごとに「平均的な実データ窓に近いか」だけを見ると不十分である。生成器が評価器の中心に寄せてくる場合、実データよりも中心的に見える生成系列があり得るためである。

## 採用した識別器

実装は以下である。

- `src/evaluation/identification.py`
- `pyfiles/identify_real_vs_generated.py`

識別器は次の3段構成にした。

1. 実データの5年窓から特徴量参照分布を作る
2. 各候補について、実データからランダムに切り出された5年窓としてどれくらい典型的かを測る
3. 「5本中ちょうど2本が実データ」という制約を使い、2本の組として自然なペアを選ぶ

既知の生成データを背景負例として与えられる場合のみ、L2正則化ロジスティック分類器で補助的に校正する。ただし、最終候補そのものを背景負例に含めてはいけない。

## 特徴量

特徴量は既存の `src/evaluation/adversarial.py` と同じ金融時系列特徴量を用いる。

- 周辺分布
- quantile
- expected shortfall
- raw ACF
- absolute ACF
- squared ACF
- rolling volatility
- 累積パス形状
- contemporaneous correlation
- cross-lag correlation
- rolling correlation
- downside/upside correlation
- tail dependence
- DGS10の0.01 tick率、ゼロ比率、中央値など

一方で、以下は使わない。

- `exact_row_match_fraction`
- `nearest_raw_window_rms`
- 実データの生窓との直接照合

## スコアの考え方

各候補の特徴量を、実データ5年窓の分布に対するrobust z-scoreへ変換する。

中心とスケールは以下で推定する。

- center: median
- scale: IQR / 1.349
- IQRが使えない場合は標準偏差へフォールバック

その上で、以下の距離を計算する。

- `mahalanobis`
- `robust_l2`
- `max_abs_z`

従来の外れ値評価では、距離が大きすぎないかを見る `conformity_p` を主に見ていた。しかし、今回の最終問題では、生成データが特徴量空間の中心に寄りすぎる可能性もある。そのため、新しい識別器では、距離が実データ窓の経験分布としてどれくらい典型的かを表す `typicality_p` を重視する。

`typicality_p` は、距離分布の両側p値として計算する。

```text
typicality_p = 2 * min(P(real_distance <= d), P(real_distance >= d))
```

これにより、極端に外側の系列だけでなく、不自然に中心へ寄りすぎた系列も下げられる。

## 5本同時判定

最終判定では、候補を独立に分類するだけではなく、5本中2本が実データという制約を使う。

具体的には、候補5本から全ての2本ペアを作り、次を最大化するペアを実データと判定する。

```text
pair_objective
  = candidate_i_score
  + candidate_j_score
  + pair_weight * log(pair_typicality_p)
```

`pair_typicality_p` は、2本の候補の特徴量距離が、実データ5年窓同士の距離分布として自然かを表す。

このペア項により、個別にはそこそこ自然でも、2本の組として実データ同士に見えにくい組み合わせを下げる。

## 実行方法

5本の候補CSVを1つずつ渡す場合:

```bash
venv/bin/python pyfiles/identify_real_vs_generated.py \
  --candidates candidate_1.csv candidate_2.csv candidate_3.csv candidate_4.csv candidate_5.csv \
  --output-dir reports/adversarial/five_window_identification/final
```

`path_id` 列を持つ1つのCSVに5本入っている場合:

```bash
venv/bin/python pyfiles/identify_real_vs_generated.py \
  --candidates candidates_5_paths.csv \
  --output-dir reports/adversarial/five_window_identification/final
```

既知の生成データを背景負例として補助利用する場合:

```bash
venv/bin/python pyfiles/identify_real_vs_generated.py \
  --candidates candidates_5_paths.csv \
  --use-default-background \
  --output-dir reports/adversarial/five_window_identification/final_supervised
```

出力は以下である。

- `candidate_scores.csv`
- `pair_scores.csv`
- `report.md`

`candidate_scores.csv` の `predicted_class` が最終判定である。

## 検証

検証用に、実データ2本と生成データ3本からなる候補CSVを作成して動作確認した。

入力:

- `reports/adversarial/five_window_identification/demo_2real_3generated.csv`

コピー指標なし、背景生成データなしのone-class/pair識別:

- 出力: `reports/adversarial/five_window_identification/demo_oneclass/report.md`
- 実データである `demo_0`, `demo_1` を `real` と判定した

背景生成データを補助負例にした識別:

- 出力: `reports/adversarial/five_window_identification/demo_supervised/report.md`
- 同じく `demo_0`, `demo_1` を `real` と判定した

## 注意点

この識別器はコピー指標を使わないため、実データが既知履歴と完全一致していること自体は利用しない。これは最終問題の趣旨に合っている。

一方で、生成器がこの特徴量集合そのものに強く最適化されている場合、識別力は下がる。その場合は、評価器に使っていないholdout特徴量、周波数領域特徴、change point特徴、または別種の分類器を追加して検証する必要がある。
