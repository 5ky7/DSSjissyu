# Block generator 3本 vs 実データ2本の識別検証

## 概要

Block generator で提出用の生成データ3本を作成し、実データからランダムに抽出した5年窓2本と混ぜた5本候補を、コピー指標なしの5本識別器で判定した。

## データ作成

- 実データ参照窓数: 642
- Block generator候補プール: 1024本
- 生成データ選抜: Mahalanobis/L2/max-z/相関特徴の中心性に基づく非コピー指標スコア上位3本
- 実データサンプル開始位置: `[3268, 10963]`

### Block generator パラメータ

```json
{
  "block_max": 126,
  "block_mean": 126,
  "block_min": 5,
  "continuation_prob": 0.8403339256,
  "dgs10_noise_multiplier": 0.35,
  "dgs10_tick": 0.01,
  "mean_jitter": 0.0656009652,
  "noise_scale": 0.0092643373,
  "profile": "balanced",
  "regime_radius": 126,
  "scale_jitter": 0.0019043086,
  "sp500_noise_floor": 1.5055e-06,
  "tail_prob": 0.0
}
```

### 選抜された生成候補

| output_id | candidate_index | submission_rank | mahalanobis | mahalanobis_pvalue | robust_l2 | max_abs_z |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 114 | 0.475958 | 0.4988 | 0.300623 | 0.711893 | 2.14377 |
| 1 | 193 | 0.48916 | 0.501894 | 0.28972 | 0.842788 | 2.03158 |
| 2 | 367 | 0.496158 | 0.494251 | 0.314642 | 0.917029 | 2.57804 |

### 5本候補の正解

| window_id | true_class | source_window_id | source_start | source_end |
| --- | --- | --- | --- | --- |
| candidate_0 | generated | block_submit_1 | 0 | 1259 |
| candidate_1 | generated | block_submit_0 | 0 | 1259 |
| candidate_2 | real | real_sample_1 | 10963 | 12222 |
| candidate_3 | generated | block_submit_2 | 0 | 1259 |
| candidate_4 | real | real_sample_0 | 3268 | 4527 |

## 識別結果: one-class/pair

- 真の実データ2本を正しく選べたか: `False`

| window_id | true_class | predicted_class | is_correct | combined_real_probability | one_class_typicality_p | one_class_conformity_p | mahalanobis_conformity_p | max_abs_z | supervised_real_probability | source_window_id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| candidate_3 | generated | real | False | 0.590988 | 0.590988 | 0.490769 | 0.315708 | 2.57804 |  | block_submit_2 |
| candidate_2 | real | real | True | 0.419973 | 0.419973 | 0.73517 | 0.608087 | 2.11383 |  | real_sample_1 |
| candidate_0 | generated | generated | True | 0.389307 | 0.389307 | 0.51407 | 0.290824 | 2.03158 |  | block_submit_1 |
| candidate_1 | generated | generated | True | 0.256948 | 0.256948 | 0.562859 | 0.301711 | 2.14377 |  | block_submit_0 |
| candidate_4 | real | generated | False | 0.125336 | 0.125336 | 0.062668 | 0.083981 | 7.40678 |  | real_sample_0 |

### ペアスコア上位

| rank | candidate_a | candidate_b | objective | pair_typicality_p | selected_pair | is_true_real_pair |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | candidate_2 | candidate_3 | -0.224824 | 0.339606 | True | False |
| 2 | candidate_1 | candidate_3 | -0.794459 | 0.668666 | False | False |
| 3 | candidate_0 | candidate_3 | -0.819013 | 0.052478 | False | False |
| 4 | candidate_0 | candidate_2 | -1.10171 | 0.268631 | False | False |
| 5 | candidate_0 | candidate_1 | -1.6542 | 0.56647 | False | False |
| 6 | candidate_1 | candidate_2 | -1.65536 | 0.338809 | False | False |
| 7 | candidate_3 | candidate_4 | -1.95622 | 0.217465 | False | False |
| 8 | candidate_0 | candidate_4 | -2.79827 | 0.197733 | False | False |
| 9 | candidate_2 | candidate_4 | -2.88499 | 0.08399 | False | True |
| 10 | candidate_1 | candidate_4 | -3.28432 | 0.326814 | False | False |

## 識別結果: 既知生成物を負例にした補助教師あり校正

- 背景生成窓数: 149
- 真の実データ2本を正しく選べたか: `False`

| window_id | true_class | predicted_class | is_correct | combined_real_probability | one_class_typicality_p | one_class_conformity_p | mahalanobis_conformity_p | max_abs_z | supervised_real_probability | source_window_id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| candidate_4 | real | real | True | 0.937125 | 0.125336 | 0.062668 | 0.083981 | 7.40678 | 0.999908 | real_sample_0 |
| candidate_3 | generated | real | False | 0.712451 | 0.590988 | 0.490769 | 0.315708 | 2.57804 | 0.746216 | block_submit_2 |
| candidate_2 | real | generated | False | 0.545021 | 0.419973 | 0.73517 | 0.608087 | 2.11383 | 0.732416 | real_sample_1 |
| candidate_1 | generated | generated | True | 0.291899 | 0.256948 | 0.562859 | 0.301711 | 2.14377 | 0.586964 | block_submit_0 |
| candidate_0 | generated | generated | True | 0.153445 | 0.389307 | 0.51407 | 0.290824 | 2.03158 | 0.074798 | block_submit_1 |

### ペアスコア上位

| rank | candidate_a | candidate_b | objective | pair_typicality_p | selected_pair | is_true_real_pair |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | candidate_3 | candidate_4 | 3.22755 | 0.217465 | True | False |
| 2 | candidate_2 | candidate_4 | 2.26297 | 0.08399 | False | True |
| 3 | candidate_1 | candidate_4 | 1.53589 | 0.326814 | False | False |
| 4 | candidate_2 | candidate_3 | 0.817899 | 0.339606 | False | False |
| 5 | candidate_0 | candidate_4 | 0.588621 | 0.197733 | False | False |
| 6 | candidate_1 | candidate_3 | -0.079479 | 0.668666 | False | False |
| 7 | candidate_1 | candidate_2 | -0.976187 | 0.338809 | False | False |
| 8 | candidate_0 | candidate_3 | -1.53736 | 0.052478 | False | False |
| 9 | candidate_0 | candidate_2 | -1.85587 | 0.268631 | False | False |
| 10 | candidate_0 | candidate_1 | -2.7361 | 0.56647 | False | False |

## 解釈

この検証では、Block generatorの3本が提出用に特徴量中心へ寄せて選抜されているため、単純な外れ値検出だけでは見破りにくい。one-class/pair識別と、既知生成物を負例にした補助教師あり校正の双方で、真の実データ2本を選べたかを確認する。

識別器は `exact_row_match_fraction` や `nearest_raw_window_rms` を使っていない。したがって、実データ2本が元の60年系列と一致するという情報は判定に使っていない。
