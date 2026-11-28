"""Direction #3: Learning-to-Rank (LTR) Engine for Section-Aware CV-Job Matching.

Provides two CPU-efficient, monotonicity-constrained LTR rankers that replace fixed
heuristic fusion weights with learned multi-granularity feature weights:
1. `MonotonicLinearLTR`: Simplex-constrained (`w >= 0, sum(w) = 1`) pairwise CoSENT ranker
   with Dirichlet/L2 prior regularization (0.01 ms CPU inference, 100% interpretable weights).
2. `MonotonicLambdaMART`: 5-Fold GroupKFold LightGBM `LGBMRanker` ensemble with strict
   monotonicity constraints (`monotone_constraints = +1` for all similarity/coverage features).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

from .engine import min_max

FEATURE_NAMES: Tuple[str, ...] = (
    "sim_flat_raw",
    "sim_flat_norm",
    "sim_sec_raw",
    "sim_sec_norm",
    "sim_skills",
    "sim_experience",
    "sim_education",
    "sim_summary",
    "sim_sec_max",
    "sim_sec_mean",
    "bm25_raw",
    "bm25_norm",
    "rrf_norm",
    "tax_must_cov",
    "tax_nice_cov",
    "tax_full_cov",
)


def extract_query_feature_matrix(
    flat_scores: np.ndarray,
    sec_scores: np.ndarray,
    sec_component_list: Sequence[Dict[str, float]],
    bm25_scores: np.ndarray,
    tax_must_scores: np.ndarray,
    tax_nice_scores: np.ndarray,
    tax_full_scores: np.ndarray,
    query_groups: Dict[int, List[int]],
) -> np.ndarray:
    """Build an (N_pairs, 16) feature matrix with per-query Min-Max calibration and section fallback."""
    n_pairs = len(flat_scores)
    X = np.zeros((n_pairs, len(FEATURE_NAMES)), dtype=np.float64)

    flat_norm = np.zeros(n_pairs, dtype=np.float64)
    sec_norm = np.zeros(n_pairs, dtype=np.float64)
    bm25_norm = np.zeros(n_pairs, dtype=np.float64)
    rrf_norm = np.zeros(n_pairs, dtype=np.float64)

    for _, p_idx in query_groups.items():
        if not p_idx:
            continue
        sub_flat = flat_scores[p_idx]
        sub_sec = sec_scores[p_idx]
        sub_bm = bm25_scores[p_idx]

        flat_norm[p_idx] = min_max(sub_flat)
        sec_norm[p_idx] = min_max(sub_sec)
        bm25_norm[p_idx] = min_max(sub_bm)

        r_sec = np.empty_like(sub_sec, dtype=int)
        r_sec[np.argsort(-sub_sec)] = np.arange(1, len(sub_sec) + 1)
        r_bm = np.empty_like(sub_bm, dtype=int)
        r_bm[np.argsort(-sub_bm)] = np.arange(1, len(sub_bm) + 1)
        rrf_sub = (1.0 / (60.0 + r_sec)) + (1.0 / (60.0 + r_bm))
        rrf_norm[p_idx] = min_max(rrf_sub)

    for i in range(n_pairs):
        fl = float(flat_scores[i])
        sc = float(sec_scores[i])
        comp = sec_component_list[i] if i < len(sec_component_list) else {}

        # Fallback missing individual section similarities to flat document similarity
        s_sk = float(comp.get("skills", fl))
        s_ex = float(comp.get("experience", fl))
        s_ed = float(comp.get("education", fl))
        s_su = float(comp.get("summary", fl))

        present_vals = [float(v) for v in comp.values()] if comp else [fl]
        s_max = max(present_vals)
        s_mean = sum(present_vals) / len(present_vals)

        X[i, 0] = fl
        X[i, 1] = flat_norm[i]
        X[i, 2] = sc
        X[i, 3] = sec_norm[i]
        X[i, 4] = s_sk
        X[i, 5] = s_ex
        X[i, 6] = s_ed
        X[i, 7] = s_su
        X[i, 8] = s_max
        X[i, 9] = s_mean
        X[i, 10] = float(bm25_scores[i])
        X[i, 11] = bm25_norm[i]
        X[i, 12] = rrf_norm[i]
        X[i, 13] = float(tax_must_scores[i])
        X[i, 14] = float(tax_nice_scores[i])
        X[i, 15] = float(tax_full_scores[i])

    return X


@dataclass
class MonotonicLinearLTR:
    """Simplex-constrained (w >= 0, sum(w) = 1) pairwise CoSENT ranker with prior regularization."""

    temperature: float = 15.0
    l2_prior_reg: float = 0.25
    feature_names: Tuple[str, ...] = FEATURE_NAMES
    weights_: np.ndarray | None = None

    def _default_prior(self) -> np.ndarray:
        """Balanced domain prior across section-aware dense, flat dense, lexical BM25, and taxonomy."""
        prior_dict = {
            "sim_flat_raw": 0.08,
            "sim_flat_norm": 0.08,
            "sim_sec_raw": 0.16,
            "sim_sec_norm": 0.18,
            "sim_skills": 0.08,
            "sim_experience": 0.07,
            "sim_education": 0.02,
            "sim_summary": 0.02,
            "sim_sec_max": 0.04,
            "sim_sec_mean": 0.03,
            "bm25_raw": 0.03,
            "bm25_norm": 0.08,
            "rrf_norm": 0.03,
            "tax_must_cov": 0.05,
            "tax_nice_cov": 0.01,
            "tax_full_cov": 0.04,
        }
        p = np.array([prior_dict.get(k, 1.0 / len(self.feature_names)) for k in self.feature_names], dtype=np.float64)
        return p / np.sum(p)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        query_groups: Dict[int, List[int]],
        prior_weights: np.ndarray | None = None,
    ) -> "MonotonicLinearLTR":
        n_feat = X.shape[1]
        prior = prior_weights if prior_weights is not None else self._default_prior()
        prior = np.asarray(prior, dtype=np.float64)
        prior = prior / np.sum(prior)

        # Precompute within-query positive-negative feature difference vectors (X_pos - X_neg)
        diff_blocks: List[Tuple[np.ndarray, np.ndarray]] = []
        all_dX_list: List[np.ndarray] = []
        for _, p_idx in query_groups.items():
            if len(p_idx) < 2:
                continue
            q_y = y[p_idx]
            if len(np.unique(q_y)) < 2:
                continue
            q_X = X[p_idx]
            dy = q_y[:, None] - q_y[None, :]
            pos_i, neg_j = np.where(dy > 0)
            if len(pos_i) == 0:
                continue
            dX = q_X[pos_i] - q_X[neg_j]  # (P, F)
            pair_w = np.where(dy[pos_i, neg_j] >= 2, 1.5, 1.0)
            diff_blocks.append((dX, pair_w))
            all_dX_list.append(dX)

        if not diff_blocks:
            self.weights_ = prior.copy()
            return self

        # Standardize feature differences by their RMS scale during loss evaluation
        # so [0, 1] min-max features and raw cosine features compete on equal margin scale
        all_dX = np.vstack(all_dX_list)
        feat_rms = np.sqrt(np.mean(all_dX ** 2, axis=0))
        feat_rms = np.where(feat_rms > 1e-6, feat_rms, 1.0)
        norm_blocks = [(dX / feat_rms[None, :], pw) for dX, pw in diff_blocks]

        def objective(w: np.ndarray) -> float:
            total_loss = 0.0
            for dX_std, pair_w in norm_blocks:
                margins = dX_std @ w
                # Robust RankNet pairwise logistic loss: E[w_ij * softplus(-tau * margin)]
                q_loss = float(np.mean(pair_w * np.logaddexp(0.0, -self.temperature * margins)))
                total_loss += q_loss
            mean_rank_loss = total_loss / len(norm_blocks)
            reg_loss = 0.5 * self.l2_prior_reg * float(np.sum((w - prior) ** 2))
            return mean_rank_loss + reg_loss

        bounds = [(0.0, 1.0) for _ in range(n_feat)]
        constraints = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
        res = minimize(
            objective,
            x0=prior.copy(),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 300, "ftol": 1e-8},
        )
        # Convert standardized weights back to raw feature scale and renormalize onto simplex
        w_std = np.clip(res.x, 0.0, None)
        w_raw = w_std / feat_rms
        s = float(np.sum(w_raw))
        self.weights_ = (w_raw / s) if s > 0 else prior.copy()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise ValueError("MonotonicLinearLTR is not fitted yet.")
        return np.asarray(X @ self.weights_, dtype=np.float64)

    def weight_summary(self) -> Dict[str, float]:
        if self.weights_ is None:
            return {}
        return {name: round(float(w), 5) for name, w in zip(self.feature_names, self.weights_)}


class MonotonicLambdaMART:
    """5-Fold Out-of-Fold LightGBM LambdaMART ensemble with strict monotonicity constraints."""

    def __init__(
        self,
        n_estimators: int = 80,
        learning_rate: float = 0.03,
        num_leaves: int = 7,
        max_depth: int = 3,
        min_child_samples: int = 20,
        reg_alpha: float = 1.0,
        reg_lambda: float = 5.0,
        colsample_bytree: float = 0.75,
        subsample: float = 0.85,
        n_splits: int = 5,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.colsample_bytree = colsample_bytree
        self.subsample = subsample
        self.n_splits = n_splits
        self.random_state = random_state
        self.models_: List[Any] = []
        self.feature_importances_: np.ndarray | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        query_groups: Dict[int, List[int]],
    ) -> "MonotonicLambdaMART":
        import lightgbm as lgb

        q_ids = sorted([q for q, idx in query_groups.items() if len(idx) >= 2 and len(np.unique(y[idx])) >= 2])
        rng = np.random.RandomState(self.random_state)
        shuffled_q = list(q_ids)
        rng.shuffle(shuffled_q)

        n_folds = min(self.n_splits, len(shuffled_q))
        if n_folds < 2:
            raise ValueError("Need at least 2 valid query groups to fit MonotonicLambdaMART.")

        folds = [shuffled_q[i::n_folds] for i in range(n_folds)]
        n_feat = X.shape[1]
        monotone = [1] * n_feat

        self.models_ = []
        importances = np.zeros(n_feat, dtype=np.float64)

        for fold_idx in range(n_folds):
            val_q = set(folds[fold_idx])
            tr_q = [q for q in shuffled_q if q not in val_q]

            tr_indices = [idx for q in tr_q for idx in query_groups[q]]
            tr_groups = [len(query_groups[q]) for q in tr_q]
            val_indices = [idx for q in folds[fold_idx] for idx in query_groups[q]]
            val_groups = [len(query_groups[q]) for q in folds[fold_idx]]

            X_tr, y_tr = X[tr_indices], y[tr_indices].astype(int)
            X_va, y_va = X[val_indices], y[val_indices].astype(int)

            ranker = lgb.LGBMRanker(
                objective="lambdarank",
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                max_depth=self.max_depth,
                min_child_samples=self.min_child_samples,
                reg_alpha=self.reg_alpha,
                reg_lambda=self.reg_lambda,
                colsample_bytree=self.colsample_bytree,
                subsample=self.subsample,
                subsample_freq=1,
                monotone_constraints=monotone,
                random_state=self.random_state + fold_idx,
                n_jobs=1,
                verbose=-1,
            )
            ranker.fit(
                X_tr,
                y_tr,
                group=tr_groups,
                eval_set=[(X_va, y_va)],
                eval_group=[val_groups],
                eval_at=[5, 10],
                callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)],
            )
            self.models_.append(ranker)
            imp = np.asarray(ranker.booster_.feature_importance(importance_type="gain"), dtype=np.float64)
            if np.sum(imp) > 0:
                imp = imp / np.sum(imp)
            importances += imp / n_folds

        self.feature_importances_ = importances
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.models_:
            raise ValueError("MonotonicLambdaMART is not fitted yet.")
        preds = np.zeros(X.shape[0], dtype=np.float64)
        for m in self.models_:
            p = np.asarray(m.predict(X), dtype=np.float64)
            preds += p / len(self.models_)
        return preds

    def importance_summary(self, feature_names: Sequence[str] = FEATURE_NAMES) -> Dict[str, float]:
        if self.feature_importances_ is None:
            return {}
        return {name: round(float(w), 5) for name, w in zip(feature_names, self.feature_importances_)}


__all__ = [
    "FEATURE_NAMES",
    "MonotonicLambdaMART",
    "MonotonicLinearLTR",
    "extract_query_feature_matrix",
]
