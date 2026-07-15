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
         makieda          1            7.931714                     0.0               1.0          0.741482          3.770282                         0.839006                       0.998442                              0.0                       1.147999
        murayama         11            7.998820                     0.0               1.0          1.672032          6.273082                         1.217917                       0.998442                              0.0                       1.411439
          setsu1          1            8.022822                     0.0               1.0          1.498887          5.527623                         1.278666                       0.998442                              0.0                       1.326880
       hashimoto          1            8.024221                     0.0               1.0          1.149498          4.481134                         1.100375                       0.998442                              0.0                       0.844136
          setsu2          1            8.611729                     0.0               1.0          2.769993         12.184687                         2.332818                       1.000000                              0.0                       0.934800
quantgan_5y_1260          1            8.651911                     0.0               1.0          3.007611         13.190909                         2.575779                       1.000000                              0.0                       0.997734
   mcgan_5y_1260          2           10.279403                     0.0               1.0          6.299276         29.483899                         5.681899                       0.999221                              0.0                       2.418747
```
