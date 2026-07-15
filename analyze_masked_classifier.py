"""masked_final_masked.csv に対する2値分類と報告書の生成。"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix,
                             f1_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, GroupKFold,
                                     GroupShuffleSplit, cross_validate)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "data" / "mixed_final_masked.csv"
OUTPUT_PATH = ROOT / "mixed_final_masked_classifier_report.md"
RANDOM_STATE = 42


def fmt_score(values):
  """平均と標準偏差を小数3桁で整形する。"""
  return f"{np.mean(values):.3f} ± {np.std(values, ddof=1):.3f}"


def main():
  raw = pd.read_csv(INPUT_PATH)
  mask_numbers = range(1, 15)
  sp500 = raw[[f"mask{i}_sp500" for i in mask_numbers]].copy()
  dgs10 = raw[[f"mask{i}_DGS10" for i in mask_numbers]].copy()
  feature_names = [f"mask{i}" for i in mask_numbers]
  sp500.columns = feature_names
  dgs10.columns = feature_names

  # 同一行の2系列を一つの観測日グループとして扱い、分割時の情報漏洩を防ぐ。
  X = pd.concat([sp500, dgs10], ignore_index=True)
  y = np.array(["SP500"] * len(sp500) + ["DGS10"] * len(dgs10))
  groups = np.tile(np.arange(len(raw)), 2)

  train_index, test_index = next(GroupShuffleSplit(
      n_splits=1, test_size=0.20, random_state=RANDOM_STATE).split(X, y, groups))
  X_train, X_test = X.iloc[train_index], X.iloc[test_index]
  y_train, y_test = y[train_index], y[test_index]
  group_train = groups[train_index]

  numeric_pipeline = Pipeline([
      ("imputer", SimpleImputer(strategy="median")),
      ("scaler", StandardScaler()),
  ])
  preprocess = ColumnTransformer([("numeric", numeric_pipeline, feature_names)])
  models = {
      "多数派ベースライン": DummyClassifier(strategy="most_frequent"),
      "ロジスティック回帰": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
      "RBF-SVM": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
      "ランダムフォレスト": RandomForestClassifier(
          n_estimators=600, min_samples_leaf=3, class_weight="balanced",
          random_state=RANDOM_STATE, n_jobs=-1),
  }

  cv = GroupKFold(n_splits=5)
  scoring = {
      "accuracy": "accuracy",
      "balanced_accuracy": "balanced_accuracy",
      "f1": "f1_macro",
      "roc_auc": "roc_auc",
  }
  cv_rows = []
  for name, model in models.items():
    pipeline = Pipeline([("preprocess", preprocess), ("model", model)])
    scores = cross_validate(
        pipeline, X_train, y_train, groups=group_train, cv=cv, scoring=scoring,
        n_jobs=-1, error_score="raise")
    cv_rows.append({
        "識別器": name,
        "正解率": fmt_score(scores["test_accuracy"]),
        "Balanced accuracy": fmt_score(scores["test_balanced_accuracy"]),
        "Macro F1": fmt_score(scores["test_f1"]),
        "ROC-AUC": fmt_score(scores["test_roc_auc"]),
    })

  svm_pipeline = Pipeline([("preprocess", preprocess), ("model", models["RBF-SVM"])])
  search = GridSearchCV(
      svm_pipeline,
      param_grid={"model__C": [0.1, 1, 10, 100], "model__gamma": ["scale", 0.01, 0.1, 1]},
      scoring="roc_auc", cv=cv, n_jobs=-1, refit=True, error_score="raise")
  search.fit(X_train, y_train, groups=group_train)
  best_model = search.best_estimator_
  predicted = best_model.predict(X_test)
  probabilities = best_model.predict_proba(X_test)[:, list(best_model.classes_).index("SP500")]
  cm = confusion_matrix(y_test, predicted, labels=["SP500", "DGS10"])
  test_metrics = {
      "accuracy": accuracy_score(y_test, predicted),
      "balanced_accuracy": balanced_accuracy_score(y_test, predicted),
      "macro_f1": f1_score(y_test, predicted, average="macro"),
      "roc_auc": roc_auc_score(y_test == "SP500", probabilities),
  }

  importance = permutation_importance(
      best_model, X_test, y_test, scoring="roc_auc", n_repeats=30,
      random_state=RANDOM_STATE, n_jobs=-1)
  importance_df = pd.DataFrame({
      "特徴量": feature_names,
      "ROC-AUC低下量（平均）": importance.importances_mean,
      "標準偏差": importance.importances_std,
  }).sort_values("ROC-AUC低下量（平均）", ascending=False)

  report_lines = [
      "# mixed_final_masked.csv：識別器の適用結果",
      "",
      "## 結論",
      "",
      f"14種類のマスク値を説明変数として、各観測を **SP500** または **DGS10** に識別するRBF-SVMを適用した。独立テスト集合での正解率は **{test_metrics['accuracy']:.3f}**、ROC-AUCは **{test_metrics['roc_auc']:.3f}** であった。",
      "この評価は同じ元行に属するSP500/DGS10のペアが学習・テストにまたがらないよう、行単位で分割している。",
      "",
      "## データと目的変数の定義",
      "",
      f"- 入力ファイル: `{INPUT_PATH.name}`",
      f"- 元データ: {len(raw):,}行 × {raw.shape[1]}列（欠損値: {int(raw.isna().sum().sum())}件）",
      "- 1行につき、`mask1`〜`mask14` のSP500系列とDGS10系列をそれぞれ1観測へ再構成した。",
      f"- 分類用データ: {len(X):,}観測 × {len(feature_names)}特徴量。クラス数はSP500={sum(y == 'SP500'):,}、DGS10={sum(y == 'DGS10'):,}で均衡している。",
      "- 目的変数: 列名の接尾辞（`_sp500` / `_DGS10`）から作成した系列ラベル。",
      "",
      "## 評価方法",
      "",
      "- テスト集合: 元行を単位に20%を保持（乱数シード42）。学習=2,016観測、テスト=504観測。",
      "- 交差検証: 学習集合で5分割のGroupKFold。元行をグループとし、同一行のペアが別foldに分かれないようにした。",
      "- 前処理: 中央値補完（本データでは欠損なし）と標準化。",
      "- 比較識別器: 多数派ベースライン、ロジスティック回帰、RBF-SVM、ランダムフォレスト。SVMのCとgammaは学習集合内のGroupKFoldでROC-AUCを基準に選択した。",
      "",
      "## 交差検証結果（学習集合、平均 ± 標準偏差）",
      "",
      pd.DataFrame(cv_rows).to_markdown(index=False),
      "",
      "## 最終モデルの独立テスト結果",
      "",
      f"選択モデルはRBF-SVM（`C={search.best_params_['model__C']}`, `gamma={search.best_params_['model__gamma']}`）である。",
      "",
      "| 指標 | 値 |",
      "|---|---:|",
      f"| 正解率 | {test_metrics['accuracy']:.3f} |",
      f"| Balanced accuracy | {test_metrics['balanced_accuracy']:.3f} |",
      f"| Macro F1 | {test_metrics['macro_f1']:.3f} |",
      f"| ROC-AUC（SP500を陽性） | {test_metrics['roc_auc']:.3f} |",
      "",
      "混同行列（行=実測、列=予測）:",
      "",
      "| 実測 \\ 予測 | SP500 | DGS10 |",
      "|---|---:|---:|",
      f"| SP500 | {cm[0, 0]} | {cm[0, 1]} |",
      f"| DGS10 | {cm[1, 0]} | {cm[1, 1]} |",
      "",
      "クラス別指標:",
      "",
      "```text",
      classification_report(y_test, predicted, digits=3),
      "```",
      "",
      "## 重要なマスク特徴量",
      "",
      "最終モデルに対して、テスト集合で特徴量をランダムに入れ替えたときのROC-AUC低下量を測定した。値が大きいほど、その特徴量を崩すと識別性能が下がることを示す。",
      "",
      importance_df.head(8).to_markdown(index=False, floatfmt=("", ".4f", ".4f")),
      "",
      "## 解釈と注意点",
      "",
      "- この結果は、マスク後の数値分布・変換の違いから系列ラベルを識別できる度合いを示すものであり、将来の市場価格や金利を予測するモデルではない。",
      "- ラベルは列名に由来するため、本検証は列名の意味を既知とする教師あり分類である。匿名化の強度を評価する用途では、列名・生成過程・時点情報を識別器に渡さない条件を厳密に保つ必要がある。",
      "- 同一行ペアをグループ分割したため、単純な観測単位のランダム分割よりも情報漏洩を抑えた評価になっている。ただし、時系列の将来汎化を評価するには、時点順の訓練・テスト分割が別途必要である。",
      "",
      "## 再現手順",
      "",
      "```bash",
      "<bundled-python> analyze_masked_classifier.py",
      "```",
      "",
      "乱数シードは42で固定している。",
  ]
  OUTPUT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
  print(f"report: {OUTPUT_PATH}")
  print(pd.DataFrame(cv_rows).to_string(index=False))
  print(test_metrics)
  print("best params:", search.best_params_)
  print("confusion matrix:\n", cm)
  print(importance_df.head(8).to_string(index=False))


if __name__ == "__main__":
  main()
