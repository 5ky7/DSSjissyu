# DGS10 0.01 Tick 丸め後の識別評価

## 目的

DGS10 が 0.01 刻みに乗っていないことが識別の大きな要因になっている可能性を切り分けるため、既存生成物、QuantGAN、old MCGAN、MPS MCGAN の DGS10 を 0.01 刻みに四捨五入したうえで、同じ特徴量ベース識別器にかけ直した。

比較対象には、Block generator の生成データと、実データから固定seedでサンプリングした5年窓5本も含めた。

## 設定

- 実データ参照分布: `data/train_sp500_us10y.csv` から21営業日刻みで作った642個の5年窓
- 生成データ窓長: 1260点
- DGS10丸め: `round(DGS10 / 0.01) * 0.01`
- 実データサンプル開始位置: `[6975, 8034, 9224, 10494, 12572]`

## 結果

| label | n_windows | dgs10_tick_fraction_before | dgs10_tick_fraction_after | min_mahalanobis_pvalue | outlier_rate_p05 | median_robust_l2 | median_max_abs_z | median_nearest_feature_pvalue | median_exact_row_match_fraction | median_nearest_raw_window_rms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| real_sample_5_windows | 5 | 1 | 1 | 0.077882 | 0 | 1.05971 | 3.3972 | 0.048287 | 1 | 0 |
| block_generator | 1 |  |  | 0.299065 | 0 | 0.905481 | 1.8244 | 0.97352 | 0 | 0.832579 |
| makieda_round001 | 1 | 0 | 1 | 0.003115 | 1 | 0.640246 | 2.06487 | 0.998442 | 0 | 1.14958 |
| hashimoto_round001 | 1 | 0 | 1 | 0.001558 | 1 | 1.13786 | 4.48113 | 0.998442 | 0 | 0.844925 |
| murayama_round001 | 11 | 0.000136 | 1 | 0 | 1 | 1.62705 | 6.27308 | 0.998442 | 0.02381 | 1.39125 |
| setsu1_round001 | 1 | 0 | 1 | 0.001558 | 1 | 1.45131 | 5.57484 | 0.998442 | 0 | 1.3157 |
| quantgan_round001 | 1 | 0 | 1 | 0 | 1 | 1.69216 | 13.1909 | 0.998442 | 0 | 1.21188 |
| mps_mcgan_20k_round001 | 1 | 0 | 1 | 0 | 1 | 1.46684 | 7.40678 | 0.998442 | 0 | 0.885755 |
| setsu2_round001 | 1 | 0 | 1 | 0 | 1 | 1.38475 | 7.56397 | 0.998442 | 0 | 1.21127 |
| old_mcgan_round001 | 1 | 0 | 1 | 0 | 1 | 11.0824 | 53.4402 | 1 | 0 | 3.51038 |

## 読み方

- `min_mahalanobis_pvalue` が高いほど、歴史的5年窓の特徴分布内にある。
- `outlier_rate_p05` は、5%水準で外れ値と判定された窓の割合で、0に近いほどよい。
- `median_max_abs_z` は最も外れた個別特徴量の大きさで、3未満を目安にした。
- `median_exact_row_match_fraction` が高い場合は、実データそのものを含む可能性がある。実データサンプルでは1になる。

## 丸め後に残った主な識別要因

| label | largest_abs_z_features |
| --- | --- |
| hashimoto_round001 | sp500_roll20_vol_acf1=4.481; DGS10_zero_fraction=4.289; sp500_roll63_vol_acf1=4.163; DGS10_roll20_vol_q05=2.155; crosscorr_sp500_to_DGS10_lag63=2.100 |
| makieda_round001 | abs_corr=2.065; DGS10_roll252_vol_acf1=1.961; DGS10_roll63_vol_q05=1.682; squared_corr=1.657; DGS10_roll20_vol_q05=1.528 |
| murayama_round001 | DGS10_q_0.001=3.660; DGS10_q_0.999=2.356; DGS10_max=2.231; DGS10_roll63_vol_q95=2.026; DGS10_roll20_vol_std=2.019 |
| setsu1_round001 | DGS10_q_0.999=5.575; DGS10_squared_acf_63=5.180; crosscorr_DGS10_to_sp500_lag20=3.827; DGS10_acf_20=3.797; DGS10_es_high_0.01=3.669 |
| setsu2_round001 | sp500_roll252_vol_acf1=7.564; sp500_roll63_vol_acf1=4.909; DGS10_q_0.001=4.291; DGS10_q_0.999=3.562; sp500_roll20_vol_acf1=3.509 |
| quantgan_round001 | sp500_roll252_vol_acf1=13.191; sp500_roll63_vol_acf1=9.880; sp500_roll20_vol_acf1=5.918; sp500_roll63_vol_q05=3.214; DGS10_q_0.999=2.598 |
| old_mcgan_round001 | sp500_cumsum_range=53.440; DGS10_max_drawdown=43.832; DGS10_cumsum_range=43.586; sp500_max_drawup=41.263; DGS10_cumsum_min=30.423 |
| mps_mcgan_20k_round001 | DGS10_q_0.5=7.407; crosscorr_DGS10_to_sp500_lag20=7.157; DGS10_cumsum_range=5.532; DGS10_max_drawup=5.263; DGS10_cumsum_max=5.162 |
| block_generator | crosscorr_sp500_to_DGS10_lag20=1.824; DGS10_q_0.999=1.784; DGS10_acf_1=1.738; roll63_corr_low_sp500_vol_mean=1.666; roll20_corr_q05=1.640 |
| real_sample_5_windows | crosscorr_sp500_to_DGS10_lag20=1.862; roll20_corr_low_sp500_vol_mean=1.365; upper_tail_dependence_0.01=1.349; roll20_corr_std=1.311; sp500_cumsum_range=1.280 |

## 要約

DGS10 を 0.01 刻みに丸めると、tick率そのものは改善する。しかし、多くの生成データでは `mahalanobis_pvalue` と `outlier_rate_p05` が改善しきらず、rolling volatility、cross-lag correlation、tail、累積パス形状などの別特徴で引き続き識別される。

Block generator は単一パスとして `outlier_rate_p05 = 0` を満たす。一方、実データからサンプリングした5年窓は当然ながら参照分布に近いが、コピー検査では `exact_row_match_fraction = 1` となるため、生成データとしては不適切である。
