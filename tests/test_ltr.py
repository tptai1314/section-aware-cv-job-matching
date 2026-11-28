"""Unit tests for Direction #3 Learning-to-Rank (`cvjd.ranking.ltr`)."""

from __future__ import annotations

import numpy as np

from cvjd.ranking.ltr import (
    FEATURE_NAMES,
    MonotonicLambdaMART,
    MonotonicLinearLTR,
    extract_query_feature_matrix,
)


def _make_synthetic_ranking_data():
    rng = np.random.RandomState(42)
    n_queries = 10
    cands_per_q = 12
    n_pairs = n_queries * cands_per_q

    y = np.zeros(n_pairs, dtype=int)
    query_groups = {}
    for q in range(n_queries):
        idx = list(range(q * cands_per_q, (q + 1) * cands_per_q))
        query_groups[q] = idx
        y[idx[:3]] = 2
        y[idx[3:7]] = 1
        y[idx[7:]] = 0

    signal = y.astype(float) / 2.0
    flat_scores = np.clip(0.5 * signal + 0.5 * rng.uniform(0, 1, n_pairs), 0, 1)
    sec_scores = np.clip(0.75 * signal + 0.25 * rng.uniform(0, 1, n_pairs), 0, 1)
    bm25_scores = np.clip(0.45 * signal + 0.55 * rng.uniform(0, 1, n_pairs), 0, 1)
    tax_must = np.clip(0.6 * signal + 0.4 * rng.uniform(0, 1, n_pairs), 0, 1)
    tax_nice = np.clip(0.3 * signal + 0.7 * rng.uniform(0, 1, n_pairs), 0, 1)
    tax_full = 0.85 * tax_must + 0.15 * tax_nice
    sec_comps = [{"skills": float(sec_scores[i]), "experience": float(flat_scores[i])} for i in range(n_pairs)]

    X = extract_query_feature_matrix(
        flat_scores, sec_scores, sec_comps, bm25_scores, tax_must, tax_nice, tax_full, query_groups
    )
    return X, y, query_groups


def test_extract_query_feature_matrix_shape_and_fallback():
    X, y, query_groups = _make_synthetic_ranking_data()
    assert X.shape == (len(y), len(FEATURE_NAMES))
    assert not np.any(np.isnan(X))
    # Missing education/summary should fallback to flat_scores (col 0)
    assert np.allclose(X[:, 6], X[:, 0])
    assert np.allclose(X[:, 7], X[:, 0])


def test_monotonic_linear_ltr_simplex_constraints():
    X, y, query_groups = _make_synthetic_ranking_data()
    ltr = MonotonicLinearLTR(temperature=15.0, l2_prior_reg=0.2).fit(X, y, query_groups)
    assert ltr.weights_ is not None
    assert len(ltr.weights_) == len(FEATURE_NAMES)
    assert np.all(ltr.weights_ >= -1e-9)
    assert np.isclose(np.sum(ltr.weights_), 1.0, atol=1e-5)

    preds = ltr.predict(X)
    assert preds.shape == (len(y),)
    summary = ltr.weight_summary()
    assert set(summary.keys()) == set(FEATURE_NAMES)


def test_monotonic_lambdamart_fit_and_predict():
    X, y, query_groups = _make_synthetic_ranking_data()
    mart = MonotonicLambdaMART(n_estimators=20, min_child_samples=3, n_splits=3).fit(X, y, query_groups)
    preds = mart.predict(X)
    assert preds.shape == (len(y),)
    imp = mart.importance_summary()
    assert set(imp.keys()) == set(FEATURE_NAMES)
