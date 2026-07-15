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
mcgan_5y_1260          1            8.071371                     0.0               1.0          1.526045           7.10631                         1.409783                       0.998442                              0.0                       0.892913
```
