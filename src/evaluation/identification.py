"""Identify real windows among a small set of candidate financial time series."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.evaluation.adversarial import (
    FeatureConfig,
    ReferenceDistribution,
    WindowRecord,
    feature_frame,
    fit_reference,
)


@dataclass(frozen=True)
class IdentificationModel:
    """Feature-based real/fake identifier for fixed-length bivariate windows.

    This model intentionally does not use row-level copy checks or raw nearest
    historical-window distances.  It only uses stylized-fact feature vectors.
    """

    config: FeatureConfig
    reference: ReferenceDistribution
    hist_mahalanobis: np.ndarray
    hist_robust_l2: np.ndarray
    hist_max_abs_z: np.ndarray
    hist_pair_distance: np.ndarray
    supervised: "LogisticRealClassifier | None" = None
    supervised_weight: float = 0.5
    pair_weight: float = 0.25

    @classmethod
    def fit(
        cls,
        real_records: list[WindowRecord],
        config: FeatureConfig,
        fake_records: list[WindowRecord] | None = None,
        supervised_weight: float = 0.5,
        pair_weight: float = 0.25,
        seed: int = 7,
    ) -> "IdentificationModel":
        """Fit the real-window reference and optional supervised calibration."""
        real_features = feature_frame(real_records, config)
        reference = fit_reference(real_features)
        real_z = reference.transform(real_features)
        real_scores = _distance_scores(real_z, reference)
        hist_pair_distance = _pair_distances(real_z)

        supervised = None
        if fake_records:
            fake_features = feature_frame(fake_records, config)
            fake_z = reference.transform(fake_features)
            supervised = LogisticRealClassifier.fit(
                real_z,
                fake_z,
                seed=seed,
            )

        return cls(
            config=config,
            reference=reference,
            hist_mahalanobis=real_scores["mahalanobis"],
            hist_robust_l2=real_scores["robust_l2"],
            hist_max_abs_z=real_scores["max_abs_z"],
            hist_pair_distance=hist_pair_distance,
            supervised=supervised,
            supervised_weight=supervised_weight,
            pair_weight=pair_weight,
        )

    def score_candidates(self, candidate_records: list[WindowRecord]) -> pd.DataFrame:
        """Score each candidate as a random real-window draw."""
        features = feature_frame(candidate_records, self.config)
        z = self.reference.transform(features)
        scores = _distance_scores(z, self.reference)

        maha_upper = _upper_tail_pvalue(scores["mahalanobis"], self.hist_mahalanobis)
        l2_upper = _upper_tail_pvalue(scores["robust_l2"], self.hist_robust_l2)
        maxz_upper = _upper_tail_pvalue(scores["max_abs_z"], self.hist_max_abs_z)

        maha_typical = _two_sided_pvalue(scores["mahalanobis"], self.hist_mahalanobis)
        l2_typical = _two_sided_pvalue(scores["robust_l2"], self.hist_robust_l2)
        maxz_typical = _two_sided_pvalue(scores["max_abs_z"], self.hist_max_abs_z)

        typicality_p = _geometric_mean(
            np.vstack([maha_typical, l2_typical, maxz_typical]).T,
            weights=np.array([0.45, 0.35, 0.20]),
        )
        conformity_p = _geometric_mean(
            np.vstack([maha_upper, l2_upper, maxz_upper]).T,
            weights=np.array([0.45, 0.35, 0.20]),
        )
        one_class_logit = _logit(typicality_p)

        supervised_prob = np.full(len(features), np.nan)
        supervised_logit = np.zeros(len(features))
        if self.supervised is not None:
            supervised_prob = self.supervised.predict_real_probability(z)
            supervised_logit = _logit(supervised_prob)

        combined_logit = one_class_logit + self.supervised_weight * supervised_logit
        real_probability = _sigmoid(combined_logit)

        out = features.loc[:, ["label", "window_id", "source_path", "start", "end"]].copy()
        out["mahalanobis"] = scores["mahalanobis"]
        out["mahalanobis_conformity_p"] = maha_upper
        out["mahalanobis_typicality_p"] = maha_typical
        out["robust_l2"] = scores["robust_l2"]
        out["robust_l2_conformity_p"] = l2_upper
        out["robust_l2_typicality_p"] = l2_typical
        out["max_abs_z"] = scores["max_abs_z"]
        out["max_abs_z_conformity_p"] = maxz_upper
        out["max_abs_z_typicality_p"] = maxz_typical
        out["one_class_typicality_p"] = typicality_p
        out["one_class_conformity_p"] = conformity_p
        out["supervised_real_probability"] = supervised_prob
        out["combined_real_probability"] = real_probability
        out["combined_logit"] = combined_logit
        return out.sort_values("combined_real_probability", ascending=False).reset_index(drop=True)

    def score_pairs(
        self,
        candidate_records: list[WindowRecord],
        candidate_scores: pd.DataFrame,
        num_real: int = 2,
    ) -> pd.DataFrame:
        """Score candidate subsets under the exactly-``num_real`` constraint."""
        if num_real != 2:
            raise ValueError("Only num_real=2 is currently implemented for pair scoring.")
        features = feature_frame(candidate_records, self.config)
        z = self.reference.transform(features)
        score_by_window_id = candidate_scores.set_index("window_id")["combined_logit"].to_dict()
        label_by_window_id = candidate_scores.set_index("window_id")["label"].to_dict()

        rows = []
        for i, j in combinations(range(len(features)), 2):
            id_i = str(features.iloc[i]["window_id"])
            id_j = str(features.iloc[j]["window_id"])
            pair_distance = float(np.sqrt(np.mean((z[i] - z[j]) ** 2)))
            pair_typicality = float(_two_sided_pvalue(np.array([pair_distance]), self.hist_pair_distance)[0])
            individual_score = float(score_by_window_id[id_i] + score_by_window_id[id_j])
            objective = individual_score + self.pair_weight * float(np.log(max(pair_typicality, 1e-12)))
            rows.append(
                {
                    "candidate_a": id_i,
                    "candidate_b": id_j,
                    "label_a": label_by_window_id[id_i],
                    "label_b": label_by_window_id[id_j],
                    "individual_logit_sum": individual_score,
                    "pair_feature_distance": pair_distance,
                    "pair_typicality_p": pair_typicality,
                    "objective": objective,
                }
            )
        out = pd.DataFrame(rows).sort_values("objective", ascending=False).reset_index(drop=True)
        out["rank"] = np.arange(1, len(out) + 1)
        out["selected_pair"] = out["rank"] == 1
        return out

    def predict(
        self,
        candidate_records: list[WindowRecord],
        num_real: int = 2,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return candidate and pair scores with the selected real windows marked."""
        candidate_scores = self.score_candidates(candidate_records)
        pair_scores = self.score_pairs(candidate_records, candidate_scores, num_real=num_real)
        selected = set(pair_scores.iloc[0][["candidate_a", "candidate_b"]].astype(str))
        candidate_scores["predicted_is_real"] = candidate_scores["window_id"].astype(str).isin(selected)
        candidate_scores["predicted_class"] = np.where(candidate_scores["predicted_is_real"], "real", "generated")
        return candidate_scores, pair_scores


@dataclass(frozen=True)
class LogisticRealClassifier:
    """Small L2-regularized logistic classifier in robust feature space."""

    weights: np.ndarray
    intercept: float
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(
        cls,
        real_z: np.ndarray,
        fake_z: np.ndarray,
        l2: float = 2.0,
        seed: int = 7,
    ) -> "LogisticRealClassifier":
        """Fit a regularized real-vs-generated logistic classifier."""
        rng = np.random.default_rng(seed)
        x = np.vstack([real_z, fake_z])
        y = np.concatenate([np.ones(len(real_z)), np.zeros(len(fake_z))])
        order = rng.permutation(len(y))
        x = x[order]
        y = y[order]

        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std = np.where(std > 1e-12, std, 1.0)
        x = (x - mean) / std

        n_real = max(float(y.sum()), 1.0)
        n_fake = max(float(len(y) - y.sum()), 1.0)
        sample_weight = np.where(y > 0.5, 0.5 / n_real, 0.5 / n_fake)

        def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
            weights = params[:-1]
            intercept = params[-1]
            logits = x @ weights + intercept
            prob = _sigmoid(logits)
            eps = 1e-12
            loss = -np.sum(sample_weight * (y * np.log(prob + eps) + (1.0 - y) * np.log(1.0 - prob + eps)))
            loss += 0.5 * l2 * float(np.mean(weights**2))

            error = sample_weight * (prob - y)
            grad_w = x.T @ error + (l2 / len(weights)) * weights
            grad_b = np.array([error.sum()])
            return float(loss), np.concatenate([grad_w, grad_b])

        initial = np.zeros(x.shape[1] + 1)
        result = minimize(
            fun=lambda params: objective(params)[0],
            x0=initial,
            jac=lambda params: objective(params)[1],
            method="L-BFGS-B",
            options={"maxiter": 500},
        )
        params = result.x
        return cls(
            weights=params[:-1],
            intercept=float(params[-1]),
            mean=mean,
            std=std,
        )

    def predict_real_probability(self, z: np.ndarray) -> np.ndarray:
        """Return calibrated-ish probabilities of being a real historical window."""
        x = (z - self.mean) / self.std
        return _sigmoid(x @ self.weights + self.intercept)


def _distance_scores(z: np.ndarray, reference: ReferenceDistribution) -> dict[str, np.ndarray]:
    centered = z - reference.mean_z
    mahalanobis = np.sqrt(
        np.maximum(np.einsum("ij,jk,ik->i", centered, reference.cov_inv, centered), 0.0)
        / max(z.shape[1], 1)
    )
    return {
        "mahalanobis": mahalanobis,
        "robust_l2": np.sqrt(np.mean(z**2, axis=1)),
        "max_abs_z": np.max(np.abs(z), axis=1),
    }


def _pair_distances(z: np.ndarray) -> np.ndarray:
    distances = []
    for i, j in combinations(range(len(z)), 2):
        distances.append(float(np.sqrt(np.mean((z[i] - z[j]) ** 2))))
    if not distances:
        return np.array([1.0])
    return np.array(distances, dtype=float)


def _upper_tail_pvalue(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference, dtype=float)
    return np.array([(np.sum(reference >= value) + 1.0) / (len(reference) + 1.0) for value in values], dtype=float)


def _lower_tail_pvalue(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference, dtype=float)
    return np.array([(np.sum(reference <= value) + 1.0) / (len(reference) + 1.0) for value in values], dtype=float)


def _two_sided_pvalue(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    lower = _lower_tail_pvalue(values, reference)
    upper = _upper_tail_pvalue(values, reference)
    return np.minimum(1.0, 2.0 * np.minimum(lower, upper))


def _geometric_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weights = weights / weights.sum()
    clipped = np.clip(values, 1e-12, 1.0)
    return np.exp(np.sum(np.log(clipped) * weights[None, :], axis=1))


def _logit(probability: np.ndarray) -> np.ndarray:
    p = np.clip(probability, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-values))
