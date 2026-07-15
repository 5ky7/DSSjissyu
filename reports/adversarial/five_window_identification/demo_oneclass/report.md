# 5本候補に対する実データ識別結果

## 目的

5本の2変量5年時系列のうち、実データ2本と生成データ3本を識別する。生データの完全一致、最近傍raw window距離などのコピー指標は使わない。

## 識別器

- 実データ約60年から1260点の5年窓を作り、金融時系列特徴量の参照分布を推定する。
- 各候補をrobust z-score化し、Mahalanobis距離、robust L2、max abs zの経験的typicalityで実データらしさを測る。
- 5本中ちょうど2本が実データという制約を使い、候補ペアの特徴量距離が実データ同士の距離分布として自然かも加味する。
- 既知の生成データを背景負例として与えた場合だけ、L2正則化ロジスティック分類器で補助的に校正する。

## 設定

- 実データ参照窓数: 642
- 背景生成窓数: 0
- 候補窓数: 5
- `supervised_weight`: 0.0
- `pair_weight`: 0.25

## 候補別スコア

| label | window_id | predicted_class | combined_real_probability | one_class_typicality_p | one_class_conformity_p | mahalanobis_conformity_p | max_abs_z | supervised_real_probability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| demo | demo_1 | real | 0.828386 | 0.828386 | 0.528304 | 0.63297 | 3.71723 |  |
| demo | demo_0 | real | 0.642528 | 0.642528 | 0.63516 | 0.505443 | 3.24438 |  |
| demo | demo_2 | generated | 0.083311 | 0.083311 | 0.041655 | 0.00311 | 4.48113 |  |
| demo | demo_4 | generated | 0.044595 | 0.044595 | 0.022297 | 0.001555 | 7.40678 |  |
| demo | demo_3 | generated | 0.035472 | 0.035472 | 0.017736 | 0.001555 | 13.1909 |  |

## ペアスコア上位

| rank | candidate_a | candidate_b | objective | pair_typicality_p | individual_logit_sum | selected_pair |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | demo_0 | demo_1 | 2.07955 | 0.723146 | 2.16059 | True |
| 2 | demo_1 | demo_2 | -0.905094 | 0.722845 | -0.823954 | False |
| 3 | demo_1 | demo_4 | -1.63136 | 0.568764 | -1.49029 | False |
| 4 | demo_1 | demo_3 | -1.75161 | 0.912277 | -1.72866 | False |
| 5 | demo_0 | demo_2 | -1.81876 | 0.972687 | -1.81183 | False |
| 6 | demo_0 | demo_4 | -2.51136 | 0.875662 | -2.47817 | False |
| 7 | demo_0 | demo_3 | -2.83867 | 0.613544 | -2.71654 | False |
| 8 | demo_2 | demo_4 | -5.49622 | 0.874535 | -5.46271 | False |
| 9 | demo_2 | demo_3 | -5.78315 | 0.720182 | -5.70108 | False |
| 10 | demo_3 | demo_4 | -6.76837 | 0.201126 | -6.36742 | False |

## 解釈

`predicted_class = real` の2本が、識別器が実データと判定した系列である。`one_class_conformity_p` は外れ値でない度合い、`one_class_typicality_p` はランダムな実データ窓として典型的な度合いを表す。最終判定では、個別スコアだけでなく、選ばれた2本のペアが実データ同士として自然な距離にあるかも使っている。

注意として、この識別器はコピー検査を使わないため、実データが既知の60年系列に完全一致している事実そのものは利用しない。一方で、生成器がこの特徴量集合に強く最適化されている場合は、別のholdout特徴量や人手での追加診断が必要である。
