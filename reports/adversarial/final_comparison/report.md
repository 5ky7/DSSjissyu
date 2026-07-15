# Adversarial Evaluation Report

## Evaluation Design

- Historical reference: 642 rolling 5-year windows.
- Window length: 1260 observations.
- Feature families: marginal tails, quantiles, ACF/absolute ACF, rolling volatility, cumulative path shape, cross-correlation, rolling correlation, conditional/tail dependence, and DGS10 tick/zero fractions.
- Low `mahalanobis_pvalue` means the path sits outside the historical 5-year feature distribution.
- Low `nearest_feature_pvalue` means the path is unusually close to one historical feature vector, which is a memorization warning.

## Summary

```text
                               label  n_windows  median_mahalanobis  min_mahalanobis_pvalue  outlier_rate_p05  median_robust_l2  median_max_abs_z  median_nearest_feature_distance  median_nearest_feature_pvalue  median_exact_row_match_fraction  median_nearest_raw_window_rms  adversarial_block_gan_ensemble_paths:classifier_auc  adversarial_block_gan_ensemble_paths:classifier_accuracy
  adversarial_block_gan_best_5y_1260          1            0.499486                0.299065          0.000000          0.905481          1.824398                         0.298359                       0.973520                              0.0                       0.832579                                                  NaN                                                       NaN
adversarial_block_gan_ensemble_paths        128            0.606444                0.007788          0.265625          0.864784          2.594211                         0.286469                       0.971963                              0.0                       0.986165                                             0.957592                                                  0.883117
                             makieda          1            7.931714                0.000000          1.000000          0.741482          3.770282                         0.839006                       0.998442                              0.0                       1.147999                                                  NaN                                                       NaN
                              setsu1          1            8.022822                0.000000          1.000000          1.498887          5.527623                         1.278666                       0.998442                              0.0                       1.326880                                                  NaN                                                       NaN
                           hashimoto          1            8.024221                0.000000          1.000000          1.149498          4.481134                         1.100375                       0.998442                              0.0                       0.844136                                                  NaN                                                       NaN
                              setsu2          1            8.611729                0.000000          1.000000          2.769993         12.184687                         2.332818                       1.000000                              0.0                       0.934800                                                  NaN                                                       NaN
                    quantgan_5y_1260          1            8.651911                0.000000          1.000000          3.007611         13.190909                         2.575779                       1.000000                              0.0                       0.997734                                                  NaN                                                       NaN
                       mcgan_5y_1260          1           12.535985                0.000000          1.000000         11.099665         53.440175                        10.085132                       1.000000                              0.0                       3.510613                                                  NaN                                                       NaN
```
