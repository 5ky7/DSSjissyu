# Adversarial Evaluation Report

## Evaluation Design

- Historical reference: 642 rolling 5-year windows.
- Window length: 1260 observations.
- Feature families: marginal tails, quantiles, ACF/absolute ACF, rolling volatility, cumulative path shape, cross-correlation, rolling correlation, conditional/tail dependence, and DGS10 tick/zero fractions.
- Low `mahalanobis_pvalue` means the path sits outside the historical 5-year feature distribution.
- Low `nearest_feature_pvalue` means the path is unusually close to one historical feature vector, which is a memorization warning.

## Summary

```text
                             label  n_windows  median_mahalanobis  min_mahalanobis_pvalue  outlier_rate_p05  median_robust_l2  median_max_abs_z  median_nearest_feature_distance  median_nearest_feature_pvalue  median_exact_row_match_fraction  median_nearest_raw_window_rms
adversarial_block_gan_best_5y_1260          1            0.499486                0.299065               0.0          0.905481          1.824398                         0.298359                       0.973520                              0.0                       0.832579
             mcgan_5y_1260_rankmap          1            1.029044                0.003115               1.0          1.031909          3.677914                         0.675342                       0.998442                              0.0                       1.258671
                     mcgan_5y_1260          1            7.940539                0.000000               1.0          0.783996          3.770282                         0.949036                       0.998442                              0.0                       1.038937
```
