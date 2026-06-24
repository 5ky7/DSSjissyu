# 時系列生成 MVP レポート

## データ定義

- 入力: `data/train_sp500_us10y.csv`
- 期間: 1966-01-03 から 2025-10-02
- 観測数: 14,734
- `sp500`: 日次単純リターンとして扱った
- `DGS10`: 入力値の分布から日次金利差として扱った
- `bond_return_proxy`: `-8.0 * dgs10_diff / 100` として作成した

## 実装済みパイプライン

1. `scripts/00_prepare_data.py`
   - `data/processed/daily_features.csv` を作成
2. `scripts/01_diagnostics_real_data.py`
   - 周辺分布、ACF、leverage、依存構造の診断表を作成
   - rolling correlation 図を作成
3. `scripts/02_fit_bootstrap.py`
   - iid bootstrap、moving block bootstrap、stationary bootstrap、circular block bootstrap を生成
4. `scripts/03_fit_marginals.py`
   - GARCH、GJR-GARCH、EGARCH の軽量 QMLE 推定
   - 標準化残差と条件付きボラティリティを出力
5. `scripts/04_fit_dcc.py`
   - 2 変量 DCC を推定し、5 年日次パスを生成
6. `scripts/05_fit_adcc.py`
   - 2 変量 ADCC を推定し、5 年日次パスを生成
7. `scripts/06_fit_copula.py`
   - Gaussian copula、t-copula、dynamic Gaussian copula を推定し、5 年日次パスを生成
8. `scripts/08_fit_regime.py`
   - volatility/correlation regime による empirical regime model を推定し、5 年日次パスを生成
9. `scripts/07_evaluate_generated.py`
   - 実データと生成データの比較表を作成

## 主要出力

- `data/processed/daily_features.csv`
- `data/generated/bootstrap/generated_paths.csv`
- `data/generated/dcc/generated_paths.csv`
- `data/generated/adcc/generated_paths.csv`
- `data/generated/copula/generated_paths.csv`
- `data/generated/regime/generated_paths.csv`
- `reports/tables/real_data_stylized_facts.csv`
- `reports/tables/marginal_model_comparison.csv`
- `reports/tables/dcc_params.csv`
- `reports/tables/adcc_params.csv`
- `reports/tables/copula_params.csv`
- `reports/tables/regime_params.csv`
- `reports/tables/regime_transition_matrix.csv`
- `reports/tables/model_comparison.csv`
- `reports/figures/real_data_rolling_corr.png`

## 推定結果メモ

- DCC: `a=0.0289`, `b=0.9658`
- ADCC: `a=0.0289`, `b=0.9658`, `g=0.000001`
- 今回のデータでは ADCC の非対称項はほぼ 0 に推定された
- Gaussian/t copula の copula-score correlation: `0.0301`
- t-copula の自由度: `10.0`
- empirical regime model の regime 数: 4

## 比較サマリ

`reports/tables/model_comparison.csv` の Version B 比較では、bootstrap 系は周辺標準偏差と downside/upside correlation を比較的よく保持した。copula 系は周辺分布を empirical quantile で保持しながら依存構造を別途生成するが、tail や high-volatility 条件付き相関の再現は限定的だった。empirical regime model は high-volatility 条件付き相関を DCC/ADCC より実データ方向に近づけたが、まだ実データほど強い負相関は出ていない。

## 限界

- `arch` パッケージを使わず SciPy の軽量 QMLE として実装しているため、skew-t 尤度や厳密な GARCH-t 推定ではない
- DCC/ADCC の生成では条件付きボラティリティを実データ推定値から再サンプリングしている
- dynamic copula は rolling Gaussian copula の簡易版であり、厳密な dynamic t-copula 推定ではない
- regime model は明示的な HMM 推定ではなく、volatility/correlation による empirical Markov model である
- 深層生成モデルは未実装
- 本番設定の 10,000 パスではなく、検証用に 100 パスを生成した

## 次の拡張

- `arch` 依存を追加して EGARCH/GJR-GARCH skew-t を厳密推定する
- DCC/ADCC simulation を GARCH volatility recursion と完全に結合する
- Version A と Version B の評価を同時に出す
- dynamic t-copula と rotated Clayton/Gumbel copula を追加する
- HMM ベースの regime-switching copula を追加する
