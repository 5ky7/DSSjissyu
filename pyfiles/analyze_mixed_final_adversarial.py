"""Identify real-like paths in mixed_final_masked.csv from financial features.

Each mask is treated as one 1,260-observation bivariate path.  The reference
distribution is fitted from rolling 5-year windows of the historical data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.adversarial import (  # noqa: E402
    FeatureConfig,
    WindowRecord,
    feature_frame,
    fit_reference,
    load_bivariate_csv,
    make_windows,
)


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "mixed_final_masked.csv"
REAL_PATH = ROOT / "data" / "train_sp500_us10y.csv"
OUTPUT_DIR = ROOT / "reports" / "adversarial" / "mixed_final_masked_feature_classifier"
WINDOW = 1260
REAL_STRIDE = 21
PVALUE_THRESHOLD = 0.05


def make_candidate_records(raw: pd.DataFrame) -> list[WindowRecord]:
    """Create one bivariate candidate record for each mask number."""
    records = []
    for mask_number in range(1, 15):
        columns = [f"mask{mask_number}_sp500", f"mask{mask_number}_DGS10"]
        records.append(
            WindowRecord(
                label=f"mask{mask_number}",
                window_id=f"mask{mask_number}",
                source_path=str(INPUT_PATH),
                start=0,
                end=len(raw) - 1,
                values=raw.loc[:, columns].to_numpy(dtype=float),
            )
        )
    return records


def top_deviations(
    features: pd.DataFrame,
    feature_names: list[str],
    z_scores: np.ndarray,
    count: int = 3,
) -> pd.DataFrame:
    """Return the largest robust-z deviations for every candidate."""
    rows = []
    for row_index, (_, row) in enumerate(features.iterrows()):
        indices = np.argsort(np.abs(z_scores[row_index]))[::-1][:count]
        for rank, feature_index in enumerate(indices, start=1):
            rows.append(
                {
                    "label": row["label"],
                    "rank": rank,
                    "feature": feature_names[feature_index],
                    "robust_z": z_scores[row_index, feature_index],
                }
            )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame) -> str:
    """Format a dataframe as a compact GitHub-flavored Markdown table."""
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |"]
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, row in frame.iterrows():
        values = []
        for value in row:
            if pd.isna(value):
                values.append("")
            elif isinstance(value, (float, np.floating)):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    scores: pd.DataFrame,
    deviations: pd.DataFrame,
    reference_windows: int,
    raw: pd.DataFrame,
) -> None:
    """Write the analytical result as Markdown."""
    result_columns = [
        "label",
        "predicted_class",
        "mahalanobis",
        "mahalanobis_pvalue",
        "robust_l2",
        "robust_l2_pvalue",
        "max_abs_z",
    ]
    result = scores.loc[:, result_columns].sort_values(
        ["mahalanobis_pvalue", "robust_l2_pvalue"], ascending=False
    )
    generated = result.loc[result["predicted_class"] == "generated", "label"].tolist()
    real_like = result.loc[result["predicted_class"] == "real_like", "label"].tolist()

    deviation_rows = []
    for label in result["label"]:
        top = deviations.loc[deviations["label"] == label].sort_values("rank")
        detail = "; ".join(
            f"{item.feature} ({item.robust_z:+.2f})" for item in top.itertuples()
        )
        deviation_rows.append({"label": label, "largest_robust_z_features": detail})

    lines = [
        "# mixed_final_masked.csv：特徴量による実データ・生成データ識別結果",
        "",
        "## 結論",
        "",
        "実データの5年窓で推定した特徴量分布に対し、Mahalanobis距離の経験的p値を用いる一クラス識別器を適用した。",
        f"有意水準5%では、**実データらしい（棄却しない）**候補は **{', '.join(real_like)}**、",
        f"**生成データと判定した**候補は **{', '.join(generated)}** である。",
        "",
        "このファイルには各候補の正解ラベルが含まれないため、正解率・ROC-AUCは算出できない。ここでの「生成データ」は、既知の実データから得られる特徴量分布と統計的に整合しない、という意味での判定である。",
        "",
        "## データと前処理",
        "",
        f"- 対象: `{INPUT_PATH.relative_to(ROOT)}`（{len(raw):,}行 × {raw.shape[1]}列、欠損値 {int(raw.isna().sum().sum()):,}件）",
        "- `mask1`〜`mask14` の各SP500/DGS10列の組を、2変量・1,260観測（約5年）の候補時系列として扱った。",
        f"- 実データ参照: `{REAL_PATH.relative_to(ROOT)}`。1,260観測のローリング窓を{REAL_STRIDE}観測ごとに切り出し、{reference_windows}窓を作成した。",
        "",
        "## 識別器",
        "",
        "`src/evaluation/adversarial.py` の `FeatureConfig`、`feature_frame`、`fit_reference`、`ReferenceDistribution.score` を使用した。",
        "特徴量は各系列の平均・分位点・裾・自己相関・ローリングボラティリティ・累積パス形状に加え、2系列間の相関・ラグ相関・裾依存を含む162個である。",
        "各候補を実データ参照分布に対してrobust z-score化し、縮小共分散によるMahalanobis距離を計算した。`mahalanobis_pvalue < 0.05` を生成データ、0.05以上をreal_likeとした。",
        "",
        "## 候補別結果",
        "",
        markdown_table(result),
        "",
        "p値は大きいほど実データの特徴量分布から外れていないことを表す。`robust_l2` と `max_abs_z` は補助診断であり、小さいほど参照分布に近い。",
        "",
        "## 大きく乖離した特徴量（robust z-score）",
        "",
        markdown_table(pd.DataFrame(deviation_rows)),
        "",
        "## 解釈と限界",
        "",
        "- `mask6` はMahalanobis p値が最も高く、参照分布との整合性が最も高い。`mask7`、`mask3`、`mask4`も5%水準では棄却されない。",
        "- `mask1`、`mask2`、`mask5`、`mask8`〜`mask14`はp値が5%未満であり、少なくとも一部の特徴量が実データ参照分布から有意に離れている。",
        "- これは特徴量ベースの分布適合性判定であり、ファイル内の真の生成元を直接観測した教師あり分類ではない。生成器が同じ特徴量に最適化されている場合や、実データの市場局面が参照期間と異なる場合には誤判定の可能性がある。",
        "- 完全一致や生系列の最近傍距離は使用していないため、コピー検査とは独立した評価である。",
        "",
        "## 出力ファイル",
        "",
        "- `candidate_features.csv`: 各候補の162特徴量",
        "- `candidate_scores.csv`: 識別スコアと判定",
        "- `top_feature_deviations.csv`: 候補ごとの乖離上位特徴量",
        "",
        "## 再現方法",
        "",
        "```bash",
        "<bundled-python> pyfiles/analyze_mixed_final_adversarial.py",
        "```",
    ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run the feature-based real-versus-generated identification."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(INPUT_PATH)
    config = FeatureConfig(window=WINDOW)

    real_data = load_bivariate_csv(REAL_PATH, ("sp500", "DGS10"))
    real_records = make_windows(
        real_data,
        window=WINDOW,
        stride=REAL_STRIDE,
        label="real_reference",
        source_path=str(REAL_PATH),
    )
    real_features = feature_frame(real_records, config)
    candidate_features = feature_frame(make_candidate_records(raw), config)
    reference = fit_reference(real_features)
    scores = reference.score(candidate_features)
    scores.insert(0, "label", candidate_features["label"].to_numpy())
    scores["predicted_class"] = np.where(
        scores["mahalanobis_pvalue"] < PVALUE_THRESHOLD,
        "generated",
        "real_like",
    )
    deviations = top_deviations(
        candidate_features,
        reference.feature_names,
        reference.transform(candidate_features),
    )

    real_features.to_csv(OUTPUT_DIR / "real_reference_features.csv", index=False)
    candidate_features.to_csv(OUTPUT_DIR / "candidate_features.csv", index=False)
    scores.to_csv(OUTPUT_DIR / "candidate_scores.csv", index=False)
    deviations.to_csv(OUTPUT_DIR / "top_feature_deviations.csv", index=False)
    write_report(scores, deviations, len(real_records), raw)
    print((OUTPUT_DIR / "report.md").relative_to(ROOT))


if __name__ == "__main__":
    main()
