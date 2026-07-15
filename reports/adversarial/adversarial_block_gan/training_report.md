# Adversarial Block-GAN Training Report

## Model

- Generator: regime-aware noisy block generator with DGS10 0.01 tick reconstruction.
- Discriminator/objective: robust historical-window feature distribution, including tails, ACF, rolling volatility, rolling correlation, conditional/tail dependence, and memorization distances.
- The generated path is not a raw historical 5-year slice: SP500 receives continuous jitter, blocks are reassembled/mixed, and nearest-window/match diagnostics are recorded.

## Best Parameters

```json
{
  "block_max": 126,
  "block_mean": 126,
  "block_min": 5,
  "continuation_prob": 0.8403339256014136,
  "dgs10_noise_multiplier": 0.35,
  "dgs10_tick": 0.01,
  "mean_jitter": 0.06560096522882178,
  "noise_scale": 0.009264337344563264,
  "profile": "balanced",
  "regime_radius": 126,
  "scale_jitter": 0.0019043086274097899,
  "sp500_noise_floor": 1.5054683891985551e-06,
  "tail_prob": 0.0
}
```

## Best Trial Metrics

```text
 trial  objective  block_mean  block_min  block_max  noise_scale  sp500_noise_floor  dgs10_noise_multiplier  scale_jitter  mean_jitter  regime_radius  continuation_prob  tail_prob  dgs10_tick  profile  feature_mean_gap  feature_std_gap  feature_energy_distance  median_mahalanobis  median_robust_l2  min_mahalanobis_pvalue  outlier_rate_p05  median_nearest_feature_distance
    33   8.100200         126          5        126     0.009264           0.000002                    0.35      0.001904     0.065601            126           0.840334     0.0000        0.01 balanced          0.464668         0.474820                 0.161456            1.001202          1.188928                0.000000          0.833333                         0.618404
    26   8.159002         126          2        126     0.009932           0.000031                    0.05      0.020324     0.031933           2520           0.729839     0.0005        0.01 balanced          0.304429         0.323513                 0.095898            0.921655          1.110556                0.001558          0.875000                         0.571159
    22   8.586424          63          3        126     0.001660           0.000006                    0.35      0.056798     0.026833           2520           0.761703     0.0005        0.01 balanced          0.341338         0.308753                 0.113005            1.129095          1.206217                0.000000          0.916667                         0.746398
    32   9.116723         126          2        252     0.009979           0.000002                    0.20      0.085767     0.042112           2520           0.050030     0.0005        0.01 balanced          0.254455         0.324679                 0.100595            0.984659          1.046854                0.001558          1.000000                         0.724978
    57   9.156697         126          5        126     0.001775           0.000065                    0.05      0.065949     0.054351            252           0.829182     0.0005        0.01 balanced          0.264169         0.382219                 0.083746            0.929512          1.141051                0.000000          1.000000                         0.557109
     6   9.200889          63          3         63     0.002181           0.000010                    0.00      0.057738     0.001127           1260           0.533674     0.0000        0.01 balanced          0.313002         0.377712                 0.100191            0.995316          1.022942                0.001558          1.000000                         0.678513
    37   9.260660          40          2         40     0.033735           0.000002                    0.10      0.064876     0.132851           1260           0.834793     0.0000        0.01 balanced          0.323022         0.467731                 0.102693            1.032859          1.056229                0.001558          1.000000                         0.703615
    58   9.268482          63          5         63     0.048858           0.000005                    0.05      0.066701     0.135196           1260           0.365923     0.0005        0.01 balanced          0.353413         0.331493                 0.129312            1.157902          1.140276                0.001558          1.000000                         0.751798
     4   9.308322          10          5         20     0.016901           0.000003                    0.00      0.080759     0.050443           2520           0.645292     0.0005        0.01 balanced          0.364239         0.417513                 0.147210            1.134597          1.054152                0.001558          1.000000                         0.855772
    19   9.362347          63          5        126     0.048945           0.000036                    0.05      0.035240     0.080957            504           0.581801     0.0000        0.01 balanced          0.428942         0.377447                 0.160025            1.086900          1.197422                0.001558          1.000000                         0.689703
```

## Selected Path Scores

```text
 selected_candidate_index  robust_l2  robust_l2_pvalue  max_abs_z  mahalanobis  mahalanobis_pvalue  nearest_feature_distance  nearest_feature_pvalue  exact_row_match_fraction  nearest_raw_window_rms  nearest_raw_window_start
                     1766   0.905481          0.661994   1.824398     0.499486            0.299065                  0.298359                 0.97352                       0.0                0.832579                     11670
```

## Ensemble Discriminator

```json
{
  "classifier_accuracy": 0.8701298701298701,
  "classifier_auc": 0.9601889338731444,
  "feature_energy_distance": 0.16777104344220017,
  "feature_mean_gap": 0.4628137319595298,
  "feature_std_gap": 0.5833612022954978,
  "median_mahalanobis": 0.6064441447280067,
  "median_nearest_feature_distance": 0.2864690150154108,
  "median_robust_l2": 0.8647844309812405,
  "min_mahalanobis_pvalue": 0.00778816199376947,
  "outlier_rate_p05": 0.265625
}
```
