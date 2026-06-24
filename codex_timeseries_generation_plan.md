# SP500 リターン・DGS10 金利差の時系列生成プロジェクト実装計画

## 1. 目的

SP500 の日次リターンと DGS10 の日次金利差を入力データとして、実データと同様の統計的性質を持つ 2 変量日次時系列を生成する。

生成対象は **5年分の日次データ** とする。営業日ベースなら、おおよそ 1260 点を 1 パスとする。

本プロジェクトでは、単なる予測ではなく、以下の stylized facts を再現する **シナリオ生成モデル** を構築する。

---

## 2. 入力データ

想定入力データ:

- `date`
- `sp500`
- `dgs10`

期間:

- 1966-01-03 から 2025-10-02 まで

派生系列:

$$
r^S_t = \frac{P^S_t}{P^S_{t-1}} - 1
$$

$$
\Delta y_t = y_t - y_{t-1}
$$

また、解釈用に duration-adjusted bond return も作る。

$$
r^B_t \approx -D_t \Delta y_t
$$

ここで $D_t$ は 10 年債の近似 duration とする。最初の実装では定数 duration を使ってよい。

---

## 3. 再現したい性質

### 3.1 各系列について再現したい性質

- ファットテール
- リターン分布の非対称性
- ボラティリティ・クラスタリング
- 資産リターンと将来ボラティリティの逆相関
- 特に SP500 についての leverage effect

### 3.2 2 系列間で再現したい性質

- 相関の時間依存性
- 相関の非対称性
- 高ボラティリティ局面で相関が強まる性質
- 株式・金利関係の regime dependence
- 可能なら tail dependence

---

## 4. 重要な設計判断

### 4.1 DGS10 は価格ではなく利回りである

DGS10 は 10 年債価格ではなく、10 年国債利回りである。

したがって、DGS10 に対して通常の価格リターンを計算してはいけない。基本系列は金利差である。

$$
\Delta y_t = y_t - y_{t-1}
$$

ただし、株式リターンと比較する場合は、金利差そのものと債券リターン近似の両方を作る。

### 4.2 2 種類の分析用データセットを作る

#### Version A: 株式リターンと金利差

$$
x_t =
\begin{pmatrix}
r^S_t \\
\Delta y_t
\end{pmatrix}
$$

用途:

- 株価変化と金利変化の関係を見る
- インフレショック・金融政策ショックの分析に向く

#### Version B: 株式リターンと債券リターン近似

$$
x_t =
\begin{pmatrix}
r^S_t \\
r^B_t
\end{pmatrix}
$$

用途:

- 株債相関を見る
- リスクオフ局面、flight-to-quality、ポートフォリオ分散効果の分析に向く

原則として、両方を実装・評価する。

---

## 5. 手法の優先順位

### 5.1 MVP で必ず実装するモデル

1. Block bootstrap
2. GJR-GARCH or EGARCH marginal model
3. DCC-GARCH-t
4. ADCC-GARCH-skew-t, または ADCC-GARCH-t

### 5.2 拡張候補

5. GARCH marginal + dynamic copula
6. Regime-switching copula
7. Regime-switching ADCC

### 5.3 研究的比較対象

8. QuantGAN
9. Tail-GAN
10. Sig-WGAN
11. Diffusion model

深層生成モデルは最初から本命にしない。まず計量モデルで強いベースラインを作る。

---

## 6. 推奨するメインモデル

最初の本命は次の構成とする。

$$
x_t =
\begin{pmatrix}
r^S_t \\
r^B_t
\end{pmatrix}
$$

または

$$
x_t =
\begin{pmatrix}
r^S_t \\
\Delta y_t
\end{pmatrix}
$$

各系列の周辺モデル:

$$
x_{i,t} = \mu_i + \sigma_{i,t} z_{i,t}
$$

SP500:

$$
r^S_t \sim \mathrm{EGARCH}\text{-}\mathrm{skew}\text{-}t
$$

DGS10 金利差または債券リターン近似:

$$
x^B_t \sim \mathrm{GARCH}\text{-}\mathrm{skew}\text{-}t
$$

依存構造:

$$
(z^S_t, z^B_t) \sim \mathrm{ADCC}\text{-}t
$$

または

$$
(u^S_t, u^B_t) \sim \mathrm{dynamic}\ \mathrm{skew}\text{-}t\ \mathrm{copula}
$$

---

## 7. 作業フロー

## Phase 0: リポジトリ初期化

### 目的

後でモデルを増やしても壊れない構成にする。

### 推奨ディレクトリ構成

```text
timeseries-generation/
  README.md
  pyproject.toml
  configs/
    base.yaml
    dcc_garch.yaml
    adcc_garch.yaml
    bootstrap.yaml
    copula.yaml
  data/
    raw/
    processed/
    generated/
  reports/
    figures/
    tables/
    diagnostics/
  src/
    data/
      load.py
      preprocess.py
      transforms.py
    diagnostics/
      stylized_facts.py
      dependence.py
      plots.py
    models/
      bootstrap.py
      marginal_garch.py
      dcc.py
      adcc.py
      copula.py
      regime.py
      deep/
    simulation/
      simulate_paths.py
      initial_state.py
    evaluation/
      metrics.py
      compare.py
      scoring.py
    utils/
      random.py
      io.py
  scripts/
    00_prepare_data.py
    01_diagnostics_real_data.py
    02_fit_bootstrap.py
    03_fit_marginals.py
    04_fit_dcc.py
    05_fit_adcc.py
    06_generate_paths.py
    07_evaluate_generated.py
  tests/
```

### 成果物

- プロジェクト構造
- 設定ファイル
- 実行スクリプト雛形
- 乱数 seed 固定機構

---

## Phase 1: データ読み込み・前処理

### 実装対象

- CSV 読み込み
- 日付パース
- 営業日 alignment
- 欠損処理
- SP500 単純リターン計算
- DGS10 金利差計算
- duration-adjusted bond return 作成

### チェック項目

- DGS10 の欠損日
- SP500 と DGS10 の休日不一致
- 金利単位が `%` か `decimal` か
- 金利差の単位が `percentage point` か `basis point` か
- 入力期間が 1966-01-03 から 2025-10-02 までであること

### 出力

- `data/processed/daily_features.csv`

カラム例:

```text
date
sp500
dgs10
sp500_return
dgs10_diff
bond_return_proxy
```

---

## Phase 2: 実データの stylized facts 計測

### 目的

生成モデルの比較対象となる実データの統計量を固定する。

### 周辺分布の指標

各系列について以下を計算する。

- mean
- standard deviation
- skewness
- excess kurtosis
- min / max
- empirical quantiles
- VaR
- ES
- Hill tail index
- Jarque-Bera statistic
- QQ plot 用データ

### 時系列依存の指標

- ACF of $x_t$
- ACF of $|x_t|$
- ACF of $x_t^2$
- volatility clustering 指標
- volatility cluster length distribution

### leverage effect

以下を複数 lag について計算する。

$$
L_i(k) = \mathrm{Corr}(x_{i,t}, x_{i,t+k}^2)
$$

必要なら推定 volatility を使う。

$$
L_i^{\sigma}(k) = \mathrm{Corr}(x_{i,t}, \hat{\sigma}_{i,t+k}^2)
$$

### 2 変量依存の指標

- full-sample correlation
- rolling correlation
- downside correlation
- upside correlation
- high-volatility conditional correlation
- empirical copula distance
- lower-tail dependence
- upper-tail dependence
- conditional correlation by volatility regime

特に重要な指標:

$$
\rho_{\mathrm{highvol}}
=
\mathrm{Corr}(x^S_t, x^B_t \mid \hat{\sigma}^{S}_t > q_{0.8})
$$

$$
\rho_{\mathrm{down}}
=
\mathrm{Corr}(x^S_t, x^B_t \mid x^S_t < q_{0.1})
$$

$$
\rho_{\mathrm{up}}
=
\mathrm{Corr}(x^S_t, x^B_t \mid x^S_t > q_{0.9})
$$

### 出力

- `reports/tables/real_data_stylized_facts.csv`
- `reports/figures/real_data_acf_*.png`
- `reports/figures/real_data_rolling_corr.png`
- `reports/figures/real_data_tail_dependence.png`

---

## Phase 3: Baseline 1 - Block bootstrap

### 目的

最小限の生成 baseline を作る。

### 実装する手法

- iid bootstrap
- moving block bootstrap
- stationary bootstrap
- circular block bootstrap

### 入力

- Version A: $(r^S_t, \Delta y_t)$
- Version B: $(r^S_t, r^B_t)$

### パラメータ

- block length: 5, 10, 20, 60, 120
- number of paths: 1000 以上
- path length: 1260

### 評価

- 周辺分布
- ACF
- rolling correlation
- high-volatility conditional correlation
- downside/upside correlation

### 想定される限界

- 新しい構造を生成できない
- ブロック境界で不連続が出る
- 長期 regime change を自然に生成できない

### 成果物

- `data/generated/bootstrap/*.csv`
- `reports/diagnostics/bootstrap_report.md`

---

## Phase 4: 周辺 GARCH モデル

### 目的

各系列の fat tail, skewness, volatility clustering, leverage effect を再現する。

### 候補モデル

SP500:

- GARCH-t
- GJR-GARCH-t
- EGARCH-t
- GJR-GARCH-skew-t
- EGARCH-skew-t

DGS10 金利差または債券リターン近似:

- GARCH-t
- GJR-GARCH-t
- EGARCH-t
- skew-t innovation を使ったモデル

### 選択基準

- AIC / BIC
- 標準化残差の ACF
- 標準化残差二乗の ACF
- QQ plot
- VaR / ES backtest
- leverage effect の再現度

### 出力

- fitted parameters
- conditional volatility series
- standardized residuals
- PIT residuals

### 成果物

- `reports/tables/marginal_model_comparison.csv`
- `data/processed/standardized_residuals.csv`
- `data/processed/pit_residuals.csv`

---

## Phase 5: DCC-GARCH-t

### 目的

時間変化する相関を再現する。

### モデル

条件付き共分散を以下の形にする。

$$
H_t = D_t R_t D_t
$$

ここで $D_t$ は単変量 GARCH から得た条件付き標準偏差、$R_t$ は時変相関行列。

DCC の状態更新:

$$
Q_t
=
(1-a-b)\bar{Q}
+
a z_{t-1} z_{t-1}^{\top}
+
b Q_{t-1}
$$

$$
R_t
=
\mathrm{diag}(Q_t)^{-1/2}
Q_t
\mathrm{diag}(Q_t)^{-1/2}
$$

### 実装上の注意

- $a \ge 0$
- $b \ge 0$
- $a+b < 1$
- $R_t$ が正定値になるように数値安定化する
- likelihood の最適化が不安定な場合は parameter transform を使う

### 評価

- rolling correlation distribution
- DCC-implied correlation distribution
- high-volatility conditional correlation
- downside/upside correlation
- generated path の stylized facts

### 限界

DCC は相関の時間変化を扱えるが、相関の非対称性は弱い。

### 成果物

- `models/dcc.py`
- `reports/diagnostics/dcc_report.md`

---

## Phase 6: ADCC-GARCH-t or ADCC-GARCH-skew-t

### 目的

相関の非対称性を再現する。

### モデル

DCC に negative shock term を追加する。

$$
n_t = z_t \odot \mathbf{1}_{\{z_t < 0\}}
$$

$$
Q_t
=
(1-a-b-g)\bar{Q}
+
a z_{t-1} z_{t-1}^{\top}
+
b Q_{t-1}
+
g n_{t-1} n_{t-1}^{\top}
$$

### 重要な注意点

DGS10 は金利差なので、単純に $\Delta y_t < 0$ を downside とすると金融的解釈がずれる可能性がある。

そのため、ADCC は以下の両方で試す。

1. $(r^S_t, \Delta y_t)$
2. $(r^S_t, r^B_t)$

特に $(r^S_t, r^B_t)$ の方が、downside shock の符号解釈は自然である。

### 評価

以下の再現度を重視する。

$$
\rho_{\mathrm{down}}
$$

$$
\rho_{\mathrm{up}}
$$

$$
\rho_{\mathrm{highvol}}
$$

また、実データと生成データで DCC/ADCC implied correlation の分布を比較する。

### 限界

- regime switching は弱い
- joint tail dependence は copula model の方が自然な場合がある
- 2 変量では過剰に複雑な ADCC パラメータは不安定になる可能性がある

### 成果物

- `models/adcc.py`
- `reports/diagnostics/adcc_report.md`

---

## Phase 7: Dynamic copula model

### 目的

周辺分布と依存構造を分離して、tail dependence と非対称依存をより柔軟に扱う。

### 手順

1. 周辺 GARCH/EGARCH モデルを fitting
2. 標準化残差 $z_{i,t}$ を取得
3. PIT residual を作る

$$
u_{i,t} = F_i(z_{i,t})
$$

4. $(u_{1,t}, u_{2,t})$ に copula を fitting
5. copula から $u_t$ を simulation
6. marginal inverse transform で $z_t$ に戻す
7. GARCH volatility recursion と組み合わせて $x_t$ を生成

### 候補 copula

- Gaussian copula
- t-copula
- skew-t copula
- Clayton copula
- Gumbel copula
- rotated Clayton
- rotated Gumbel
- regime-switching copula

### 優先順位

最初は以下だけでよい。

1. Gaussian copula
2. t-copula
3. dynamic t-copula
4. rotated Clayton or Gumbel

### 評価

- empirical copula distance
- lower-tail dependence
- upper-tail dependence
- high-volatility conditional correlation
- downside/upside correlation

### 限界

- モデル選択が難しい
- 推定が不安定になる可能性がある
- dynamic copula は DCC より実装が重い

### 成果物

- `models/copula.py`
- `reports/diagnostics/copula_report.md`

---

## Phase 8: Regime-switching extension

### 目的

株債相関の符号変化や macro regime dependence を再現する。

### 想定 regime

例:

1. normal
2. inflation shock
3. flight-to-quality
4. liquidity crisis

### 実装候補

- Markov-switching Gaussian/t distribution
- Markov-switching copula
- volatility-threshold regime model
- rolling correlation clustering による empirical regime model

### MVP での簡易版

まずは明示的な HMM ではなく、実データを以下で分割する。

- SP500 volatility high / low
- DGS10 volatility high / low
- rolling correlation positive / negative

その後、regime ごとに相関・分布を推定して、semi-parametric に生成する。

### 限界

- regime 数の選択が難しい
- regime の経済的意味付けが恣意的になりやすい
- rare regime の推定誤差が大きい

### 成果物

- `models/regime.py`
- `reports/diagnostics/regime_report.md`

---

## Phase 9: 深層生成モデルは後回し

### 目的

計量モデルを超える nonlinear benchmark として利用する。

### 候補

- QuantGAN
- Tail-GAN
- Sig-WGAN
- TimeGAN
- diffusion model

### 実装方針

最初に実装しない。以下の条件を満たした後に着手する。

- block bootstrap が完成
- marginal GARCH が完成
- DCC が完成
- ADCC が完成
- evaluation pipeline が完成

### 深層生成モデルの注意点

- stylized facts の保証はない
- 学習が不安定
- 2 変量 16000 点程度では過学習しやすい
- tail event を過小生成する可能性がある
- 生成結果がよく見えても risk measure が崩れている可能性がある

### 使うなら優先すべきもの

通常の TimeGAN よりも、金融時系列向けの目的関数を持つ Tail-GAN, Sig-WGAN, QuantGAN を優先する。

---

## 8. 生成処理

### 入力

- fitted marginal models
- fitted dependence model
- initial volatility state
- path length
- number of paths
- random seed

### 推奨設定

```yaml
simulation:
  path_length: 1260
  n_paths: 10000
  seed: 42
  initial_state: "sample_from_real_data"
```

### 初期状態

初期 volatility state は固定値にしない。実データ中のランダムな時点からサンプリングする。

### 出力

生成データは long format を推奨する。

```text
model
path_id
t
date_index
sp500_return
dgs10_diff
bond_return_proxy
conditional_vol_sp500
conditional_vol_dgs10
conditional_corr
regime
```

---

## 9. 評価パイプライン

### 9.1 評価の基本方針

生成データは見た目ではなく、実データの stylized facts との距離で評価する。

### 9.2 評価対象

各モデルについて以下を評価する。

- block bootstrap
- DCC-GARCH-t
- ADCC-GARCH-t
- ADCC-GARCH-skew-t
- dynamic copula
- optional: regime-switching copula

### 9.3 周辺評価

- mean error
- volatility error
- skewness error
- kurtosis error
- Hill tail index error
- Wasserstein distance
- quantile error
- VaR error
- ES error
- QQ plot

### 9.4 時系列評価

- ACF error of $x_t$
- ACF error of $|x_t|$
- ACF error of $x_t^2$
- volatility clustering score
- leverage correlation curve error

### 9.5 2 変量依存評価

- unconditional correlation error
- rolling correlation distribution error
- DCC-implied correlation distribution error
- downside correlation error
- upside correlation error
- high-volatility conditional correlation error
- empirical copula distance
- lower-tail dependence error
- upper-tail dependence error

### 9.6 重点指標

以下は必ず report に出す。

$$
\rho_{\mathrm{highvol}}
=
\mathrm{Corr}(x^S_t, x^B_t \mid \hat{\sigma}^{S}_t > q_{0.8})
$$

$$
\rho_{\mathrm{down}}
=
\mathrm{Corr}(x^S_t, x^B_t \mid x^S_t < q_{0.1})
$$

$$
\rho_{\mathrm{up}}
=
\mathrm{Corr}(x^S_t, x^B_t \mid x^S_t > q_{0.9})
$$

### 9.7 モデル選択

総合 score は一つにまとめてもよいが、最初は dashboard 的に並べる方がよい。

重視順位:

1. tail risk
2. volatility clustering
3. leverage effect
4. rolling correlation distribution
5. high-volatility conditional correlation
6. downside/upside correlation
7. unconditional moments

unconditional correlation だけが合っていても合格にしない。

---

## 10. モデル比較表

| Model | Fat tail | Skewness | Leverage | Time-varying correlation | Asymmetric correlation | High-vol correlation | Interpretability | Priority |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| iid bootstrap | Yes | Yes | No | No | No | No | High | Low |
| block bootstrap | Yes | Yes | Partial | Partial | Partial | Partial | High | Medium |
| EGARCH/GJR only | Yes | Yes | Yes | No | No | No | High | Medium |
| DCC-GARCH-t | Yes | Partial | Yes | Yes | Partial | Partial | High | High |
| ADCC-GARCH-skew-t | Yes | Yes | Yes | Yes | Yes | Yes | High | Highest |
| GARCH + dynamic copula | Yes | Yes | Yes | Yes | Strong | Strong | Medium | High |
| regime-switching copula | Yes | Yes | Yes | Strong | Strong | Strong | Medium | High |
| stochastic volatility | Yes | Yes | Strong | Yes | Yes | Yes | Medium | Medium |
| QuantGAN / Tail-GAN | Yes | Partial | Yes | Yes | Partial | Yes | Low | Medium |
| diffusion model | Yes | Partial | Partial | Yes | Partial | Yes | Low | Medium |

---

## 11. 実装上のリスク

### 11.1 DGS10 の符号解釈

$\Delta y_t$ と $r^B_t$ は符号が逆になる。

生成結果の相関を解釈するときは、必ずどちらの系列を使ったかを明示する。

### 11.2 GARCH 推定の不安定性

skew-t, EGARCH, ADCC を同時に複雑化すると推定が不安定になりやすい。

対策:

- まず GARCH-t
- 次に GJR-GARCH-t
- 次に EGARCH-t
- 最後に skew-t
- DCC の後に ADCC

### 11.3 過剰適合

1966-01-03 から 2025-10-02 までの日次データは約 14700 点程度であり、2 変量モデルには十分だが、深層生成モデルには多いとは言えない。

深層モデルは必ず out-of-sample diagnostics と stylized facts comparison を行う。

### 11.4 評価指標の過剰集約

単一スコアだけで選ばない。

tail, volatility, dependence, regime behavior を分けて評価する。

---

## 12. 受け入れ条件

MVP の受け入れ条件:

- `data/processed/daily_features.csv` が作成できる
- 実データの diagnostics report が作成できる
- block bootstrap で 5 年日次パスを生成できる
- DCC-GARCH-t で 5 年日次パスを生成できる
- ADCC-GARCH-t で 5 年日次パスを生成できる
- 各モデルについて同一の evaluation pipeline が走る
- 実データと生成データの stylized facts 比較表が出力される

最終的な合格条件:

- SP500 の fat tail, skewness, volatility clustering が再現されている
- SP500 の leverage effect が再現されている
- DGS10 金利差または債券リターン近似の fat tail が再現されている
- rolling correlation distribution が実データに近い
- high-volatility conditional correlation が実データに近い
- downside/upside correlation の非対称性が実データに近い
- 生成パスが 5 年日次系列として破綻していない

---

## 13. 実装の推奨順序

### Round 1: diagnostics first

1. データ読み込み
2. 特徴量作成
3. 実データ diagnostics
4. 可視化

### Round 2: simple generation

5. iid bootstrap
6. block bootstrap
7. bootstrap evaluation

### Round 3: marginal models

8. GARCH-t
9. GJR-GARCH-t
10. EGARCH-t
11. residual diagnostics

### Round 4: dependence models

12. DCC-GARCH-t
13. ADCC-GARCH-t
14. simulation
15. evaluation

### Round 5: extensions

16. skew-t innovation
17. dynamic copula
18. regime model
19. optional deep generative model

---

## 14. Codex への実装指示

まず以下だけを実装する。

1. `src/data/preprocess.py`
   - SP500 return
   - DGS10 difference
   - bond return proxy
   - missing value handling

2. `src/diagnostics/stylized_facts.py`
   - moments
   - quantiles
   - VaR / ES
   - ACF
   - leverage correlation
   - rolling correlation
   - downside/upside/high-vol correlation

3. `src/models/bootstrap.py`
   - iid bootstrap
   - moving block bootstrap
   - stationary bootstrap

4. `src/models/marginal_garch.py`
   - GARCH-t
   - GJR-GARCH-t
   - EGARCH-t
   - standardized residual output

5. `src/models/dcc.py`
   - DCC recursion
   - parameter estimation
   - simulation

6. `src/models/adcc.py`
   - ADCC recursion
   - parameter estimation
   - simulation

7. `src/evaluation/compare.py`
   - real vs generated metrics comparison
   - tables
   - plots

深層生成モデルはこの段階では実装しない。

---

## 15. 最初に確認すべき質問

実装前にデータについて以下を確認する。

1. `sp500` は価格指数か、すでにリターンか
2. `dgs10` は % 表記か decimal 表記か
3. DGS10 の欠損日はどのように処理されているか
4. 入力データ期間は 1966-01-03 から 2025-10-02 まででよいか
5. 生成したいのは金利差そのものか、債券リターン近似か、それとも両方か
6. 5 年分は calendar day か business day か
7. 生成パス数は何本必要か

デフォルトでは、business day 1260 点、10000 パス、Version A と Version B の両方を実装する。

---

## 16. 最終成果物

最終的に以下を出力する。

```text
data/processed/daily_features.csv
data/generated/bootstrap/generated_paths.csv
data/generated/dcc/generated_paths.csv
data/generated/adcc/generated_paths.csv
reports/tables/real_data_stylized_facts.csv
reports/tables/model_comparison.csv
reports/figures/*.png
reports/diagnostics/final_report.md
```

`final_report.md` には以下を含める。

- データ定義
- モデル一覧
- 各モデルの推定結果
- 実データと生成データの stylized facts 比較
- モデルごとの限界
- 採用モデル
- 今後の拡張案
