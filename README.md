# SABR-style Two Asset Generator

2つのリターン時系列からSABR風の確率ボラティリティモデルをフィットし、合成リターン系列と価格系列を生成します。

## セットアップ

```bash
python -m venv venv
venv/bin/python -m pip install -r requirements.txt
```

## 実行例

```bash
venv/bin/python sabr_generator.py \
  --input train_sp500_us10y.csv \
  --asset-columns sp500,DGS10 \
  --steps 252 \
  --paths 100 \
  --seed 7 \
  --output generated_sabr_series.csv \
  --params-output sabr_params.json
```

学習期間を区切る場合は次のように指定できます。

```bash
venv/bin/python sabr_generator.py \
  --input train_sp500_us10y.csv \
  --asset-columns sp500,DGS10 \
  --start-date 1966-01-01 \
  --end-date 2026-12-31
```

`generated_sabr_series.csv` は long format で、各パス・各ステップについて2資産のリターンを出力します。`sabr_params.json` にはフィットされた各資産の `mu`, `alpha`, `beta`, `nu`, `rho` と、2資産間を含む4次元ブラウン運動の相関行列を保存します。

## モデルの扱い

学習データはリターン系列なので、オプション市場のSABRキャリブレーションではなく、リターンの平均、標準偏差、ローリング実現ボラティリティ、標準化リターンショックとボラティリティショックの相関からパラメータを推定します。

生成時の価格過程は次の形です。

```text
dS_i = mu_i S_i dt + alpha_i S_i^beta_i dW_i
d alpha_i = nu_i alpha_i dZ_i
```

2資産間の相関は、各資産の価格ショックとボラティリティショックをまとめた4次元相関行列として反映します。

## 拡張の方針

将来的に期間ごとにフィットする場合は、`fit_two_asset_sabr_by_periods()` に `("1966-01-01", "1975-12-31")` のような期間ペアを渡すと、期間ごとの `TwoAssetSabrParams` を得られます。
