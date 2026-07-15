# mixed_final_masked.csv：特徴量による実データ・生成データ識別結果

## 結論

実データの5年窓で推定した特徴量分布に対し、Mahalanobis距離の経験的p値を用いる一クラス識別器を適用した。
有意水準5%では、**実データらしい（棄却しない）**候補は **mask6, mask7, mask3, mask4**、
**生成データと判定した**候補は **mask10, mask8, mask11, mask9, mask14, mask12, mask1, mask2, mask5, mask13** である。

このファイルには各候補の正解ラベルが含まれないため、正解率・ROC-AUCは算出できない。ここでの「生成データ」は、既知の実データから得られる特徴量分布と統計的に整合しない、という意味での判定である。

## データと前処理

- 対象: `data/mixed_final_masked.csv`（1,260行 × 28列、欠損値 0件）
- `mask1`〜`mask14` の各SP500/DGS10列の組を、2変量・1,260観測（約5年）の候補時系列として扱った。
- 実データ参照: `data/train_sp500_us10y.csv`。1,260観測のローリング窓を21観測ごとに切り出し、642窓を作成した。

## 識別器

`src/evaluation/adversarial.py` の `FeatureConfig`、`feature_frame`、`fit_reference`、`ReferenceDistribution.score` を使用した。
特徴量は各系列の平均・分位点・裾・自己相関・ローリングボラティリティ・累積パス形状に加え、2系列間の相関・ラグ相関・裾依存を含む162個である。
各候補を実データ参照分布に対してrobust z-score化し、縮小共分散によるMahalanobis距離を計算した。`mahalanobis_pvalue < 0.05` を生成データ、0.05以上をreal_likeとした。

## 候補別結果

| label | predicted_class | mahalanobis | mahalanobis_pvalue | robust_l2 | robust_l2_pvalue | max_abs_z |
| --- | --- | --- | --- | --- | --- | --- |
| mask6 | real_like | 0.442213 | 0.676012 | 1.36341 | 0.233645 | 3.3972 |
| mask7 | real_like | 0.532493 | 0.180685 | 0.906319 | 0.661994 | 3.70339 |
| mask3 | real_like | 0.595966 | 0.104361 | 0.988305 | 0.570093 | 4.047 |
| mask4 | real_like | 0.62771 | 0.0825545 | 0.974672 | 0.5919 | 3.34477 |
| mask10 | generated | 0.964406 | 0.00311526 | 0.749329 | 0.900312 | 2.97818 |
| mask8 | generated | 1.41595 | 0.00155763 | 1.03477 | 0.496885 | 6.01389 |
| mask11 | generated | 1.32704 | 0.00155763 | 1.03525 | 0.496885 | 3.96604 |
| mask9 | generated | 1.22284 | 0.00155763 | 1.56232 | 0.21028 | 3.82863 |
| mask14 | generated | 7.94023 | 0 | 0.821208 | 0.791277 | 3.77028 |
| mask12 | generated | 8.13111 | 0 | 1.32301 | 0.244548 | 6.74887 |
| mask1 | generated | 1.73683 | 0 | 1.40593 | 0.222741 | 6.0335 |
| mask2 | generated | 2.44619 | 0 | 2.2958 | 0.00311526 | 8.3469 |
| mask5 | generated | 2.06106 | 0 | 2.40603 | 0.00155763 | 6.745 |
| mask13 | generated | 2.69632 | 0 | 2.71346 | 0.00155763 | 8.15098 |

p値は大きいほど実データの特徴量分布から外れていないことを表す。`robust_l2` と `max_abs_z` は補助診断であり、小さいほど参照分布に近い。

## 大きく乖離した特徴量（robust z-score）

| label | largest_robust_z_features |
| --- | --- |
| mask6 | sp500_roll20_vol_q95 (+3.40); sp500_q_0.1 (-3.25); sp500_q_0.05 (-3.14) |
| mask7 | DGS10_q_0.5 (-3.70); sp500_roll252_vol_q50 (+2.52); sp500_roll63_vol_q50 (+2.00) |
| mask3 | DGS10_zero_fraction (+4.05); DGS10_sign_switch_fraction (-2.68); abs_corr (-2.60) |
| mask4 | sp500_squared_acf_1 (+3.34); sp500_q_0.999 (+2.96); sp500_squared_acf_63 (+2.72) |
| mask10 | DGS10_max_drawdown (-2.98); DGS10_cumsum_min (-2.72); DGS10_cumsum_range (+2.42) |
| mask8 | abs_corr (-6.01); squared_corr (-3.80); sp500_excess_kurtosis (+2.95) |
| mask11 | crosscorr_DGS10_to_sp500_lag20 (+3.97); abs_corr (-3.82); DGS10_squared_acf_63 (+3.72) |
| mask9 | DGS10_q_0.001 (-3.83); DGS10_roll252_vol_q50 (+3.81); sp500_cumsum_range (+3.80) |
| mask14 | DGS10_zero_fraction (-3.77); DGS10_positive_fraction (+2.36); crosscorr_sp500_to_DGS10_lag20 (+2.32) |
| mask12 | crosscorr_DGS10_to_sp500_lag1 (+6.75); sp500_roll63_vol_acf1 (-5.21); sp500_roll252_vol_acf1 (-4.40) |
| mask1 | sp500_abs_acf_63 (+6.03); sp500_excess_kurtosis (+4.45); DGS10_squared_acf_20 (+3.92) |
| mask2 | sp500_max_drawdown (-8.35); sp500_cumsum_range (+8.11); sp500_cumsum_min (-7.58) |
| mask5 | DGS10_q_0.9 (+6.74); DGS10_q_0.95 (+6.27); sp500_cumsum_min (-5.79) |
| mask13 | sp500_max_drawdown (-8.15); sp500_cumsum_range (+7.87); sp500_cumsum_min (-7.86) |

## 解釈と限界

- `mask6` はMahalanobis p値が最も高く、参照分布との整合性が最も高い。`mask7`、`mask3`、`mask4`も5%水準では棄却されない。
- `mask1`、`mask2`、`mask5`、`mask8`〜`mask14`はp値が5%未満であり、少なくとも一部の特徴量が実データ参照分布から有意に離れている。
- これは特徴量ベースの分布適合性判定であり、ファイル内の真の生成元を直接観測した教師あり分類ではない。生成器が同じ特徴量に最適化されている場合や、実データの市場局面が参照期間と異なる場合には誤判定の可能性がある。
- 完全一致や生系列の最近傍距離は使用していないため、コピー検査とは独立した評価である。

## 出力ファイル

- `candidate_features.csv`: 各候補の162特徴量
- `candidate_scores.csv`: 識別スコアと判定
- `top_feature_deviations.csv`: 候補ごとの乖離上位特徴量

## 再現方法

```bash
<bundled-python> pyfiles/analyze_mixed_final_adversarial.py
```