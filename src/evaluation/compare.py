"""Compare generated paths with real stylized-fact metrics."""

from __future__ import annotations

import pandas as pd

from src.diagnostics.stylized_facts import DiagnosticConfig, dependence_metrics, marginal_metrics


def summarize_generated_paths(
    generated: pd.DataFrame,
    x_col: str,
    y_col: str,
    config: DiagnosticConfig | None = None,
) -> pd.DataFrame:
    """Calculate path-level metrics and average them by model."""
    config = config or DiagnosticConfig()
    rows = []
    for (model, path_id), path in generated.groupby(["model", "path_id"]):
        marginal = marginal_metrics(path, [x_col, y_col], config)
        dependence = dependence_metrics(path, x_col, y_col, config)
        row = {"model": model, "path_id": path_id}
        for _, metric_row in marginal.iterrows():
            series = metric_row["series"]
            for key, value in metric_row.items():
                if key != "series":
                    row[f"{series}_{key}"] = value
        for key, value in dependence.iloc[0].items():
            if key not in {"x", "y"}:
                row[f"dep_{key}"] = value
        rows.append(row)
    path_metrics = pd.DataFrame(rows)
    return path_metrics.groupby("model").mean(numeric_only=True).reset_index()


def real_metric_vector(
    real: pd.DataFrame,
    x_col: str,
    y_col: str,
    config: DiagnosticConfig | None = None,
) -> pd.DataFrame:
    """Flatten real-data metrics into one row."""
    config = config or DiagnosticConfig()
    marginal = marginal_metrics(real, [x_col, y_col], config)
    dependence = dependence_metrics(real, x_col, y_col, config)
    row = {"model": "real"}
    for _, metric_row in marginal.iterrows():
        series = metric_row["series"]
        for key, value in metric_row.items():
            if key != "series":
                row[f"{series}_{key}"] = value
    for key, value in dependence.iloc[0].items():
        if key not in {"x", "y"}:
            row[f"dep_{key}"] = value
    return pd.DataFrame([row])


def comparison_table(
    real: pd.DataFrame,
    generated: pd.DataFrame,
    x_col: str,
    y_col: str,
    config: DiagnosticConfig | None = None,
) -> pd.DataFrame:
    """Build a table with real metrics, generated metrics, and errors."""
    real_row = real_metric_vector(real, x_col, y_col, config)
    generated_summary = summarize_generated_paths(generated, x_col, y_col, config)
    real_values = real_row.iloc[0].drop(labels=["model"])
    rows = [real_row.iloc[0].to_dict()]
    for _, model_row in generated_summary.iterrows():
        row = model_row.to_dict()
        for key, real_value in real_values.items():
            if key in row:
                row[f"{key}_error"] = row[key] - real_value
        rows.append(row)
    return pd.DataFrame(rows)
