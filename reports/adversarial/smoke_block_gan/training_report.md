# Adversarial Block-GAN Training Report

## Model

- Generator: regime-aware noisy block generator with DGS10 0.01 tick reconstruction.
- Discriminator/objective: robust historical-window feature distribution, including tails, ACF, rolling volatility, rolling correlation, conditional/tail dependence, and memorization distances.
- The generated path is not a raw historical 5-year slice: SP500 receives continuous jitter, blocks are reassembled/mixed, and nearest-window/match diagnostics are recorded.

## Best Parameters

```json
{
  "block_max": 126,
  "block_mean": 63,
  "block_min": 3,
  "continuation_prob": 0.4021114396843268,
  "dgs10_noise_multiplier": 0.35,
  "dgs10_tick": 0.01,
  "mean_jitter": 0.062045339281530734,
  "noise_scale": 0.021463248542165095,
  "profile": "balanced",
  "regime_radius": 1260,
  "scale_jitter": 0.020848306329238064,
  "sp500_noise_floor": 1.4498166209968358e-06,
  "tail_prob": 0.001
}
```

## Best Trial Metrics

```text
 trial  objective  block_mean  block_min  block_max  noise_scale  sp500_noise_floor  dgs10_noise_multiplier  scale_jitter  mean_jitter  regime_radius  continuation_prob  tail_prob  dgs10_tick  profile  feature_mean_gap  feature_std_gap  feature_energy_distance  median_mahalanobis  median_robust_l2  min_mahalanobis_pvalue  outlier_rate_p05  median_nearest_feature_distance
     0   9.737116          63          3        126     0.021463           0.000001                    0.35      0.020848     0.062045           1260           0.402111      0.001        0.01 balanced          0.675842         0.535161                 0.467944            1.006322          1.311720                0.001558               1.0                         0.627876
     1  10.219140         126          5        126     0.013173           0.000035                    0.35      0.023902     0.001966            126           0.414066      0.001        0.01 balanced          0.794175         0.969598                 0.635950            1.428854          1.719131                0.000000               1.0                         1.140249
```

## Selected Path Scores

```text
 selected_candidate_index  robust_l2  robust_l2_pvalue  max_abs_z  mahalanobis  mahalanobis_pvalue  nearest_feature_distance  nearest_feature_pvalue  exact_row_match_fraction  nearest_raw_window_rms  nearest_raw_window_start
                        5   0.605162               1.0      1.349     0.741265            0.023364                  0.639095                0.998442                       0.0                1.133803                       465
```

## Ensemble Discriminator

```json
{
  "classifier_accuracy": NaN,
  "classifier_auc": NaN,
  "feature_energy_distance": 0.30793060557786056,
  "feature_mean_gap": 0.5883196048439162,
  "feature_std_gap": 0.5276083205127747,
  "median_mahalanobis": 1.1465931072280608,
  "median_nearest_feature_distance": 0.8524627803716485,
  "median_robust_l2": 1.101339313326668,
  "min_mahalanobis_pvalue": 0.001557632398753894,
  "outlier_rate_p05": 1.0
}
```
