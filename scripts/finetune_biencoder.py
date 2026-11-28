"""Direction #1: Leakage-Free Section-Aware Bi-Encoder Fine-Tuning & Scientific Evaluation.

Protocol:
1. Leakage Prevention:
   - Removes all CVs and JDs from `train.csv` that appear in `test.csv` (68 overlapping CVs removed -> 5,432 clean train pairs, 280 JDs, 574 CVs).
   - Splits the 280 clean train JDs strictly by query group (`job_description_text`) into:
     * Train Split: 230 JDs
     * Validation Split: 50 JDs (used exclusively for checkpoint selection / early stopping)
2. Joint Multi-Granularity Within-Query Ranking Loss (Section-Aware CoSENT):
   - Optimizes `sentence-transformers/all-MiniLM-L6-v2` end-to-end on both:
     * Document-level cosine similarity: cos(v_JD, v_CV)
     * Section-Aware weighted cosine similarity: sum_m w_m * cos(v_JD, v_CV^(m)) / sum_m w_m
   - Uses query-grouped CoSENT pairwise ranking loss across graded relevance labels {0: No Fit, 1: Potential Fit, 2: Good Fit}.
3. Scientific Evaluation on Frozen `test.csv`:
   - Evaluates Frozen MiniLM vs. Fine-Tuned MiniLM (`models/minilm-cvjd-ft`) across:
     * Single-Vector Dense
     * Section-Aware Dense
     * Dense + BM25 (No Taxonomy)
     * Calibrated Hybrid (Dense + BM25 + 3-Tier Taxonomy)
     * Parameter-Free RRF + Taxonomy
   - Reports Setting A (All 71 queries), Setting B (rel >= 1: 61 queries; rel == 2: 28 queries), and Clean Held-out CVs (409 CVs).
   - Computes 1,000-resample Paired Bootstrap 95% CI, Paired t-test, and Wilcoxon signed-rank test.
"""

from __future__ import annotations

from collections import defaultdict
import copy
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_rel, wilcoxon
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from cvjd.data.preprocessing import sections_to_dict, split_sections
from cvjd.data.taxonomy import get_taxonomy
from cvjd.ranking.engine import min_max

DATA_DIR = REPO_ROOT / "data" / "public_benchmark" / "resume-job-description-fit"
MODELS_DIR = REPO_ROOT / "models"
OUTPUT_MODEL_DIR = MODELS_DIR / "minilm-cvjd-ft"
RESULTS_DIR = REPO_ROOT / "results"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LABEL_MAP = {"No Fit": 0, "Potential Fit": 1, "Good Fit": 2}
SEC_KEYS = ["skills", "experience", "education", "summary"]
SEC_WEIGHTS = {"skills": 0.40, "experience": 0.35, "education": 0.15, "summary": 0.10}


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==============================================================================
# 1. METRICS & STATISTICAL TESTS
# ==============================================================================

def dcg_at_k(rels: Sequence[float | int], k: int) -> float:
    score = 0.0
    for i in range(min(k, len(rels))):
        score += (math.pow(2.0, float(rels[i])) - 1.0) / math.log2(i + 2.0)
    return score


def ndcg_at_k(ranked_rels: Sequence[float | int], ideal_rels: Sequence[float | int], k: int) -> float:
    idcg = dcg_at_k(ideal_rels, k)
    if idcg <= 0.0:
        return 0.0
    return min(1.0, max(0.0, dcg_at_k(ranked_rels, k) / idcg))


def reciprocal_rank(ranked_rels: Sequence[float | int], threshold: int) -> float:
    for rank, r in enumerate(ranked_rels, start=1):
        if r >= threshold:
            return 1.0 / rank
    return 0.0


def average_precision(ranked_rels: Sequence[float | int], threshold: int) -> float:
    total_rel = sum(1 for r in ranked_rels if r >= threshold)
    if total_rel <= 0:
        return 0.0
    hits = 0
    p_sum = 0.0
    for rank, r in enumerate(ranked_rels, start=1):
        if r >= threshold:
            hits += 1
            p_sum += hits / rank
    return p_sum / total_rel


def precision_at_k(ranked_rels: Sequence[float | int], k: int, threshold: int) -> float:
    if k <= 0 or len(ranked_rels) == 0:
        return 0.0
    return sum(1 for r in ranked_rels[:k] if r >= threshold) / float(k)


def evaluate_query_with_tie_breaking(
    scores: np.ndarray,
    rels: np.ndarray,
    k_vals: Tuple[int, int] = (5, 10),
    thresholds: Tuple[int, int] = (1, 2),
    n_tie_shuffles: int = 50,
    rng: np.random.RandomState | None = None,
) -> Dict[str, float]:
    if rng is None:
        rng = np.random.RandomState(42)

    n_cands = len(scores)
    ideal_rels = sorted(rels, reverse=True)
    metrics_accum: Dict[str, float] = defaultdict(float)

    for _ in range(n_tie_shuffles):
        jitter = rng.uniform(-1e-9, 1e-9, size=n_cands)
        shuffled_order = np.argsort(-(scores + jitter))
        ranked_rels = rels[shuffled_order]

        for k in k_vals:
            metrics_accum[f"ndcg@{k}"] += ndcg_at_k(ranked_rels, ideal_rels, k)

        for th in thresholds:
            metrics_accum[f"mrr_th{th}"] += reciprocal_rank(ranked_rels, th)
            metrics_accum[f"map_th{th}"] += average_precision(ranked_rels, th)
            for k in k_vals:
                metrics_accum[f"p@{k}_th{th}"] += precision_at_k(ranked_rels, k, th)

    return {m: val / n_tie_shuffles for m, val in metrics_accum.items()}


def paired_bootstrap_ci(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    n_boot: int = 1000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    arr_a = np.asarray(scores_a, dtype=float)
    arr_b = np.asarray(scores_b, dtype=float)
    diffs = arr_a - arr_b
    mean_diff = float(np.mean(diffs))
    rng = np.random.RandomState(seed)
    n = len(diffs)
    boot_means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boot_means[i] = np.mean(diffs[idx])
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    return mean_diff, float(ci_low), float(ci_high)


# ==============================================================================
# 2. DIFFERENTIABLE FORWARD & WITHIN-QUERY COSENT LOSS
# ==============================================================================

def encode_differentiable(
    model: SentenceTransformer,
    texts: List[str],
    device: torch.device,
) -> torch.Tensor:
    preprocess_fn = getattr(model, "preprocess", model.tokenize)
    features = preprocess_fn(texts)
    features = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in features.items()
    }
    out = model(features)["sentence_embedding"]
    return F.normalize(out, p=2, dim=-1)


def within_query_cosent_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 20.0,
) -> torch.Tensor:
    """CoSENT pairwise ranking loss within a single JD query.

    For all candidate pairs (i, j) where label[i] > label[j]:
        L = log(1 + sum_{y_i > y_j} weight_ij * exp(temperature * (s_j - s_i)))
    """
    diff_label = labels.unsqueeze(1) - labels.unsqueeze(0)  # (N, N)
    mask = diff_label > 0
    if not torch.any(mask):
        return torch.zeros((), device=scores.device, dtype=scores.dtype)

    # Pairwise score differences s_j - s_i where y_i > y_j
    score_diff = scores.unsqueeze(0) - scores.unsqueeze(1)  # s_j - s_i at [i, j]
    active_diffs = score_diff[mask] * temperature
    # Weight by label gap (1.0 for gap=1, 1.5 for gap=2)
    weights = torch.where(diff_label[mask] >= 2, 1.5, 1.0)
    log_terms = active_diffs + torch.log(weights)
    # log(1 + sum(exp(log_terms))) via logsumexp
    zero = torch.zeros((1,), device=scores.device, dtype=scores.dtype)
    all_terms = torch.cat([zero, log_terms], dim=0)
    return torch.logsumexp(all_terms, dim=0)


# ==============================================================================
# 3. FAST BATCH EVALUATION HELPER
# ==============================================================================

def compute_dense_scores_for_split(
    model: SentenceTransformer,
    unique_jobs: List[str],
    unique_cvs: List[str],
    cv_parsed: List[Dict[str, Any]],
    pairs: List[Tuple[int, int]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (flat_scores, section_aware_scores) for all pairs."""
    job_embs = model.encode(unique_jobs, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    cv_embs = model.encode(unique_cvs, batch_size=64, normalize_embeddings=True, show_progress_bar=False)

    flat_scores = np.array([float(np.dot(job_embs[j], cv_embs[c])) for j, c in pairs], dtype=float)

    all_sec_texts = []
    sec_meta = []
    for c_idx, sd in enumerate(cv_parsed):
        for s_name in SEC_KEYS:
            if s_name in sd and len(sd[s_name].text.strip()) > 5:
                sec_meta.append((c_idx, s_name))
                all_sec_texts.append(sd[s_name].text[:1024])

    sec_vecs_flat = model.encode(all_sec_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    cv_sec_vecs: List[Dict[str, np.ndarray]] = [{} for _ in range(len(unique_cvs))]
    for (c_idx, s_name), vec in zip(sec_meta, sec_vecs_flat):
        cv_sec_vecs[c_idx][s_name] = vec

    sec_scores = np.empty(len(pairs), dtype=float)
    for idx, (j, c) in enumerate(pairs):
        jv = job_embs[j]
        c_secs = cv_sec_vecs[c]
        w_sim, tot_w = 0.0, 0.0
        for sn, w in SEC_WEIGHTS.items():
            if sn in c_secs:
                w_sim += w * float(np.dot(jv, c_secs[sn]))
                tot_w += w
        sec_scores[idx] = (w_sim / tot_w) if tot_w > 0.0 else flat_scores[idx]

    return flat_scores, sec_scores


def quick_val_metrics(
    flat_scores: np.ndarray,
    sec_scores: np.ndarray,
    bm25_scores: np.ndarray,
    tax_scores: np.ndarray,
    query_groups: Dict[int, List[int]],
    y_true: np.ndarray,
) -> Dict[str, float]:
    norm_sec = np.zeros_like(sec_scores)
    norm_bm25 = np.zeros_like(bm25_scores)
    for q, p_idx in query_groups.items():
        norm_sec[p_idx] = min_max(sec_scores[p_idx])
        norm_bm25[p_idx] = min_max(bm25_scores[p_idx])
    dense_bm25 = 0.60 * norm_sec + 0.40 * norm_bm25
    hybrid = 0.60 * dense_bm25 + 0.40 * tax_scores

    rng = np.random.RandomState(42)
    flat_n10, sec_n10, hyb_n10, sec_map, hyb_map = [], [], [], [], []
    for q, p_idx in query_groups.items():
        rels = y_true[p_idx]
        if not any(r >= 1 for r in rels):
            continue
        r_flat = evaluate_query_with_tie_breaking(flat_scores[p_idx], rels, n_tie_shuffles=20, rng=rng)
        r_sec = evaluate_query_with_tie_breaking(sec_scores[p_idx], rels, n_tie_shuffles=20, rng=rng)
        r_hyb = evaluate_query_with_tie_breaking(hybrid[p_idx], rels, n_tie_shuffles=20, rng=rng)
        flat_n10.append(r_flat["ndcg@10"])
        sec_n10.append(r_sec["ndcg@10"])
        hyb_n10.append(r_hyb["ndcg@10"])
        sec_map.append(r_sec["map_th1"])
        hyb_map.append(r_hyb["map_th1"])

    return {
        "flat_ndcg@10": float(np.mean(flat_n10)),
        "sec_ndcg@10": float(np.mean(sec_n10)),
        "hybrid_ndcg@10": float(np.mean(hyb_n10)),
        "sec_map": float(np.mean(sec_map)),
        "hybrid_map": float(np.mean(hyb_map)),
        "selection_score": float(0.45 * np.mean(sec_n10) + 0.35 * np.mean(hyb_n10) + 0.20 * np.mean(flat_n10)),
    }


# ==============================================================================
# 4. MAIN FINE-TUNING & BENCHMARK PIPELINE
# ==============================================================================

def main() -> None:
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80, flush=True)
    print(f"DIRECTION #1: SECTION-AWARE BI-ENCODER FINE-TUNING (Device: {device})", flush=True)
    print("=" * 80, flush=True)

    # 1. Load & Filter Clean Train Data (0% overlap with test.csv)
    train_df = pd.read_csv(DATA_DIR / "train.csv", encoding="latin1")
    test_df = pd.read_csv(DATA_DIR / "test.csv", encoding="latin1")
    train_df["rel"] = train_df["label"].map(LABEL_MAP)
    test_df["rel"] = test_df["label"].map(LABEL_MAP)

    test_cv_set = set(test_df["resume_text"].unique())
    test_jd_set = set(test_df["job_description_text"].unique())

    clean_train_df = train_df[
        (~train_df["resume_text"].isin(test_cv_set))
        & (~train_df["job_description_text"].isin(test_jd_set))
    ].copy()

    all_clean_jds = sorted(clean_train_df["job_description_text"].unique().tolist())
    rng_split = np.random.RandomState(42)
    shuffled_jds = list(all_clean_jds)
    rng_split.shuffle(shuffled_jds)

    val_jds_list = shuffled_jds[:50]
    tr_jds_list = shuffled_jds[50:]

    tr_sub_df = clean_train_df[clean_train_df["job_description_text"].isin(set(tr_jds_list))].copy()
    val_sub_df = clean_train_df[clean_train_df["job_description_text"].isin(set(val_jds_list))].copy()

    print(f"Clean Train Split: {len(tr_jds_list)} JDs, {tr_sub_df['resume_text'].nunique()} CVs, {len(tr_sub_df)} pairs", flush=True)
    print(f"Clean Val Split  : {len(val_jds_list)} JDs, {val_sub_df['resume_text'].nunique()} CVs, {len(val_sub_df)} pairs", flush=True)

    # 2. Pre-parse CVs in Train and Val
    print("\n--> Pre-parsing CV sections for Train & Validation splits ...", flush=True)
    all_tr_val_cvs = clean_train_df["resume_text"].unique().tolist()
    cv_sec_cache: Dict[str, Dict[str, str]] = {}
    for c in all_tr_val_cvs:
        sd = sections_to_dict(split_sections(c, is_jd=False))
        cv_sec_cache[c] = {
            sn: sd[sn].text[:1024]
            for sn in SEC_KEYS
            if sn in sd and len(sd[sn].text.strip()) > 5
        }

    # Build training query groups (only queries with >= 2 distinct labels can form ranking pairs)
    train_query_records: List[Dict[str, Any]] = []
    for jd_text, grp in tr_sub_df.groupby("job_description_text"):
        labels = grp["rel"].tolist()
        if len(set(labels)) < 2:
            continue
        cvs = grp["resume_text"].tolist()
        train_query_records.append({
            "jd": jd_text[:1500],
            "cvs": [c[:1500] for c in cvs],
            "cv_raw": cvs,
            "labels": labels,
        })
    print(f"Active Train Query Groups with >=2 relevance levels: {len(train_query_records)} / {len(tr_jds_list)}", flush=True)

    # Build validation structures
    val_unique_jobs = val_jds_list
    val_job_to_idx = {j: i for i, j in enumerate(val_unique_jobs)}
    val_unique_cvs = val_sub_df["resume_text"].unique().tolist()
    val_cv_to_idx = {c: i for i, c in enumerate(val_unique_cvs)}
    val_cv_parsed = [sections_to_dict(split_sections(c, is_jd=False)) for c in val_unique_cvs]

    val_pairs: List[Tuple[int, int]] = []
    val_y_true: List[int] = []
    val_query_groups: Dict[int, List[int]] = defaultdict(list)
    for _, row in val_sub_df.iterrows():
        j_i = val_job_to_idx[row["job_description_text"]]
        c_i = val_cv_to_idx[row["resume_text"]]
        p_i = len(val_pairs)
        val_query_groups[j_i].append(p_i)
        val_pairs.append((j_i, c_i))
        val_y_true.append(int(row["rel"]))
    val_y_arr = np.array(val_y_true, dtype=int)

    # Validation BM25 & Taxonomy
    taxonomy = get_taxonomy()
    val_vec = TfidfVectorizer(max_features=10000, stop_words="english", sublinear_tf=True)
    val_vec.fit(val_unique_jobs + val_unique_cvs)
    val_j_tfidf = val_vec.transform(val_unique_jobs)
    val_c_tfidf = val_vec.transform(val_unique_cvs)
    val_bm25 = np.array([float((val_j_tfidf[j].multiply(val_c_tfidf[c])).sum()) for j, c in val_pairs])

    val_j_skills = [taxonomy.classify_jd_skills(j) for j in val_unique_jobs]
    val_c_skills = [set(taxonomy.extract_skills_from_text(c)) for c in val_unique_cvs]
    val_tax = []
    for j, c in val_pairs:
        must, nice = val_j_skills[j]
        c_sk = val_c_skills[c]
        if not must and not nice:
            val_tax.append(1.0)
            continue
        cov_m = sum(
            1.0 if m in c_sk else max([taxonomy.get_relation_weight(cs, m) for cs in c_sk], default=0.0)
            for m in must
        )
        m_sc = cov_m / len(must) if must else 1.0
        n_sc = sum(1.0 for n in nice if n in c_sk) / len(nice) if nice else 0.0
        val_tax.append(0.85 * m_sc + 0.15 * n_sc)
    val_tax_arr = np.array(val_tax, dtype=float)

    # 3. Initialize Model & Evaluate Epoch 0 (Frozen Baseline on Validation)
    base_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(base_model_name, device=str(device))
    model.max_seq_length = 256

    val_flat_0, val_sec_0 = compute_dense_scores_for_split(
        model, val_unique_jobs, val_unique_cvs, val_cv_parsed, val_pairs
    )
    val_res_0 = quick_val_metrics(val_flat_0, val_sec_0, val_bm25, val_tax_arr, val_query_groups, val_y_arr)
    print(
        f"\n[Epoch 0 - Frozen Baseline] Val Flat nDCG@10: {val_res_0['flat_ndcg@10']:.4f} | "
        f"Val Sec nDCG@10: {val_res_0['sec_ndcg@10']:.4f} | "
        f"Val Hybrid nDCG@10: {val_res_0['hybrid_ndcg@10']:.4f} | "
        f"Val Hybrid MAP: {val_res_0['hybrid_map']:.4f}",
        flush=True,
    )

    best_val_score = val_res_0["selection_score"]
    best_state_dict = copy.deepcopy(model.state_dict())
    best_epoch = 0
    training_history = [{"epoch": 0, **val_res_0}]

    # 4. Fine-Tuning Loop (Query-Grouped Joint Flat + Section-Aware CoSENT Loss)
    epochs = 4
    lr = 1.5e-5
    max_cands_per_step = 12
    grad_accum_steps = 4

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = (len(train_query_records) * epochs) // grad_accum_steps + 1
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    print(f"\n--> Starting Fine-Tuning ({epochs} epochs, lr={lr}, max_cands={max_cands_per_step}) ...", flush=True)
    for epoch in range(1, epochs + 1):
        t_ep = time.perf_counter()
        model.train()
        random.shuffle(train_query_records)
        optimizer.zero_grad()
        running_loss = 0.0
        valid_steps = 0

        for step_idx, rec in enumerate(train_query_records, start=1):
            cvs = rec["cvs"]
            cv_raw = rec["cv_raw"]
            labels = rec["labels"]

            # Subsample candidates if query has many CVs while preserving label diversity
            if len(cvs) > max_cands_per_step:
                pos_idx = [i for i, y in enumerate(labels) if y >= 1]
                neg_idx = [i for i, y in enumerate(labels) if y == 0]
                n_pos = min(len(pos_idx), max_cands_per_step // 2 + 1)
                n_neg = min(len(neg_idx), max_cands_per_step - n_pos)
                chosen = random.sample(pos_idx, n_pos) + random.sample(neg_idx, n_neg)
                if len(set(labels[i] for i in chosen)) < 2:
                    continue
            else:
                chosen = list(range(len(cvs)))

            sub_cvs = [cvs[i] for i in chosen]
            sub_raw = [cv_raw[i] for i in chosen]
            sub_labels = torch.tensor([labels[i] for i in chosen], dtype=torch.float32, device=device)

            # Collect section texts for chosen candidates
            sec_texts: List[str] = []
            sec_map: List[Tuple[int, float]] = []  # (cand_local_idx, weight)
            for local_i, raw_c in enumerate(sub_raw):
                s_dict = cv_sec_cache[raw_c]
                for sn, w in SEC_WEIGHTS.items():
                    if sn in s_dict:
                        sec_texts.append(s_dict[sn])
                        sec_map.append((local_i, w))

            # Batch encode: [JD] + sub_cvs + sec_texts in one forward pass
            all_texts = [rec["jd"]] + sub_cvs + sec_texts
            embs = encode_differentiable(model, all_texts, device)

            jd_emb = embs[0]  # (d,)
            cv_embs = embs[1 : 1 + len(sub_cvs)]  # (M, d)
            sec_embs = embs[1 + len(sub_cvs) :]  # (S, d)

            # 1) Flat similarity scores
            flat_sim = torch.mv(cv_embs, jd_emb)  # (M,)

            # 2) Section-Aware weighted similarity scores
            if len(sec_map) > 0:
                sec_sims = torch.mv(sec_embs, jd_emb)  # (S,)
                sec_num = torch.zeros(len(sub_cvs), device=device, dtype=flat_sim.dtype)
                sec_den = torch.zeros(len(sub_cvs), device=device, dtype=flat_sim.dtype)
                for s_idx, (c_local, w) in enumerate(sec_map):
                    sec_num[c_local] = sec_num[c_local] + w * sec_sims[s_idx]
                    sec_den[c_local] = sec_den[c_local] + w
                has_sec = sec_den > 0
                sec_score = torch.where(has_sec, sec_num / torch.clamp(sec_den, min=1e-6), flat_sim)
            else:
                sec_score = flat_sim

            loss_flat = within_query_cosent_loss(flat_sim, sub_labels, temperature=20.0)
            loss_sec = within_query_cosent_loss(sec_score, sub_labels, temperature=20.0)
            loss = (0.45 * loss_flat + 0.55 * loss_sec) / grad_accum_steps

            if loss.requires_grad and loss.item() > 0.0:
                loss.backward()
                running_loss += loss.item() * grad_accum_steps
                valid_steps += 1

            if step_idx % grad_accum_steps == 0 or step_idx == len(train_query_records):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        # Evaluate at end of epoch on Validation Split
        model.eval()
        val_flat, val_sec = compute_dense_scores_for_split(
            model, val_unique_jobs, val_unique_cvs, val_cv_parsed, val_pairs
        )
        val_res = quick_val_metrics(val_flat, val_sec, val_bm25, val_tax_arr, val_query_groups, val_y_arr)
        avg_loss = running_loss / max(1, valid_steps)
        ep_time = time.perf_counter() - t_ep

        print(
            f"[Epoch {epoch}/{epochs} - {ep_time:.1f}s] Loss: {avg_loss:.4f} | "
            f"Val Flat nDCG@10: {val_res['flat_ndcg@10']:.4f} | "
            f"Val Sec nDCG@10: {val_res['sec_ndcg@10']:.4f} | "
            f"Val Hybrid nDCG@10: {val_res['hybrid_ndcg@10']:.4f} | "
            f"Val Hybrid MAP: {val_res['hybrid_map']:.4f}",
            flush=True,
        )
        training_history.append({"epoch": epoch, "train_loss": round(avg_loss, 4), **val_res})

        if val_res["selection_score"] > best_val_score:
            best_val_score = val_res["selection_score"]
            best_state_dict = copy.deepcopy(model.state_dict())
            best_epoch = epoch

    print(f"\n--> Restoring best checkpoint from Epoch {best_epoch} (Val Score: {best_val_score:.4f}) ...", flush=True)
    model.load_state_dict(best_state_dict)
    model.eval()
    model.save(str(OUTPUT_MODEL_DIR))
    print(f"Saved fine-tuned Bi-Encoder to: {OUTPUT_MODEL_DIR}", flush=True)

    # ==============================================================================
    # 5. SCIENTIFIC BENCHMARK EVALUATION ON FROZEN TEST.CSV
    # ==============================================================================
    print("\n" + "=" * 80, flush=True)
    print("EVALUATING FROZEN vs. FINE-TUNED BI-ENCODER ON TEST.CSV (1,759 PAIRS, 71 JDs)", flush=True)
    print("=" * 80, flush=True)

    train_cv_set = set(train_df["resume_text"].unique())
    test_jobs = test_df["job_description_text"].unique().tolist()
    test_job_to_idx = {j: i for i, j in enumerate(test_jobs)}
    test_cvs = test_df["resume_text"].unique().tolist()
    test_cv_to_idx = {c: i for i, c in enumerate(test_cvs)}
    overlap_cv_indices = set(i for i, c in enumerate(test_cvs) if c in train_cv_set)

    test_pairs: List[Tuple[int, int]] = []
    test_y_true: List[int] = []
    test_query_groups: Dict[int, List[int]] = defaultdict(list)

    for _, row in test_df.iterrows():
        j_i = test_job_to_idx[row["job_description_text"]]
        c_i = test_cv_to_idx[row["resume_text"]]
        p_i = len(test_pairs)
        test_query_groups[j_i].append(p_i)
        test_pairs.append((j_i, c_i))
        test_y_true.append(int(row["rel"]))

    test_y_arr = np.array(test_y_true, dtype=int)
    n_test_queries = len(test_query_groups)
    pos_queries_th1 = [q for q, p_idx in test_query_groups.items() if any(test_y_arr[p] >= 1 for p in p_idx)]
    pos_queries_th2 = [q for q, p_idx in test_query_groups.items() if any(test_y_arr[p] == 2 for p in p_idx)]

    test_cv_parsed = [sections_to_dict(split_sections(c, is_jd=False)) for c in test_cvs]

    # BM25 & Taxonomy on test.csv
    t_vec = TfidfVectorizer(max_features=10000, stop_words="english", sublinear_tf=True)
    t_vec.fit(test_jobs + test_cvs)
    t_j_tfidf = t_vec.transform(test_jobs)
    t_c_tfidf = t_vec.transform(test_cvs)
    test_bm25 = np.array([float((t_j_tfidf[j].multiply(t_c_tfidf[c])).sum()) for j, c in test_pairs])

    t_j_skills = [taxonomy.classify_jd_skills(j) for j in test_jobs]
    t_c_skills = [set(taxonomy.extract_skills_from_text(c)) for c in test_cvs]
    test_tax = []
    for j, c in test_pairs:
        must, nice = t_j_skills[j]
        c_sk = t_c_skills[c]
        if not must and not nice:
            test_tax.append(1.0)
            continue
        cov_m = sum(
            1.0 if m in c_sk else max([taxonomy.get_relation_weight(cs, m) for cs in c_sk], default=0.0)
            for m in must
        )
        m_sc = cov_m / len(must) if must else 1.0
        n_sc = sum(1.0 for n in nice if n in c_sk) / len(nice) if nice else 0.0
        test_tax.append(0.85 * m_sc + 0.15 * n_sc)
    test_tax_arr = np.array(test_tax, dtype=float)

    # Frozen MiniLM scores on test.csv
    frozen_model = SentenceTransformer(base_model_name, device=str(device))
    frozen_model.max_seq_length = 256
    frozen_flat, frozen_sec = compute_dense_scores_for_split(
        frozen_model, test_jobs, test_cvs, test_cv_parsed, test_pairs
    )

    # Fine-Tuned MiniLM scores on test.csv (loaded from saved checkpoint)
    ft_model = SentenceTransformer(str(OUTPUT_MODEL_DIR), device=str(device))
    ft_flat, ft_sec = compute_dense_scores_for_split(
        ft_model, test_jobs, test_cvs, test_cv_parsed, test_pairs
    )

    def build_hybrid_variants(sec_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        norm_sec = np.zeros_like(sec_scores)
        norm_bm25 = np.zeros_like(test_bm25)
        rrf_raw = np.zeros_like(sec_scores)
        for q, p_idx in test_query_groups.items():
            sub_sec = sec_scores[p_idx]
            sub_bm = test_bm25[p_idx]
            norm_sec[p_idx] = min_max(sub_sec)
            norm_bm25[p_idx] = min_max(sub_bm)

            r_sec = np.empty_like(sub_sec, dtype=int)
            r_sec[np.argsort(-sub_sec)] = np.arange(1, len(sub_sec) + 1)
            r_bm = np.empty_like(sub_bm, dtype=int)
            r_bm[np.argsort(-sub_bm)] = np.arange(1, len(sub_bm) + 1)
            rrf_sub = (1.0 / (60 + r_sec)) + (1.0 / (60 + r_bm))
            rrf_raw[p_idx] = min_max(rrf_sub)

        dense_bm25 = 0.60 * norm_sec + 0.40 * norm_bm25
        cal_hybrid = 0.60 * dense_bm25 + 0.40 * test_tax_arr
        rrf_hybrid = 0.60 * rrf_raw + 0.40 * test_tax_arr
        return dense_bm25, cal_hybrid, rrf_hybrid

    fr_dense_bm25, fr_hybrid, fr_rrf = build_hybrid_variants(frozen_sec)
    ft_dense_bm25, ft_hybrid, ft_rrf = build_hybrid_variants(ft_sec)

    all_models: Dict[str, np.ndarray] = {
        "Single-Vector Dense (Frozen MiniLM)": frozen_flat,
        "Single-Vector Dense (Fine-Tuned MiniLM)": ft_flat,
        "Section-Aware Dense (Frozen MiniLM)": frozen_sec,
        "Section-Aware Dense (Fine-Tuned MiniLM)": ft_sec,
        "Dense + BM25 (Frozen MiniLM)": fr_dense_bm25,
        "Dense + BM25 (Fine-Tuned MiniLM)": ft_dense_bm25,
        "Calibrated Hybrid (Frozen MiniLM)": fr_hybrid,
        "Calibrated Hybrid (Fine-Tuned MiniLM)": ft_hybrid,
        "Parameter-Free RRF + Tax (Frozen MiniLM)": fr_rrf,
        "Parameter-Free RRF + Tax (Fine-Tuned MiniLM)": ft_rrf,
    }

    rng_eval = np.random.RandomState(42)

    def evaluate_suite(target_queries: List[int], query_filter_fn=None) -> Dict[str, Any]:
        suite_res = {}
        q_to_pairs = {}
        for q in target_queries:
            p_list = test_query_groups[q]
            if query_filter_fn:
                p_list = [p for p in p_list if query_filter_fn(test_pairs[p])]
            q_to_pairs[q] = p_list

        for m_name, m_scores in all_models.items():
            per_query_metrics = []
            per_query_rhos = []
            for q in target_queries:
                p_idx = q_to_pairs[q]
                if not p_idx:
                    continue
                q_sc = m_scores[p_idx]
                q_rel = test_y_arr[p_idx]
                q_res = evaluate_query_with_tie_breaking(q_sc, q_rel, n_tie_shuffles=50, rng=rng_eval)
                per_query_metrics.append(q_res)

                if len(p_idx) >= 2 and len(np.unique(q_rel)) > 1 and len(np.unique(q_sc)) > 1:
                    r_val, _ = spearmanr(q_sc, q_rel)
                    if not np.isnan(r_val):
                        per_query_rhos.append(r_val)

            avg_metrics = {k: float(np.mean([r[k] for r in per_query_metrics])) for k in per_query_metrics[0].keys()}
            all_target_pairs = [p for q in target_queries for p in q_to_pairs[q]]
            rho_mic, _ = spearmanr(m_scores[all_target_pairs], test_y_arr[all_target_pairs])
            rho_mac = float(np.mean(per_query_rhos)) if per_query_rhos else 0.0

            suite_res[m_name] = {
                "metrics": avg_metrics,
                "spearman_micro": round(float(rho_mic), 4),
                "spearman_macro": round(float(rho_mac), 4),
                "per_query_n10": [r["ndcg@10"] for r in per_query_metrics],
                "per_query_map1": [r["map_th1"] for r in per_query_metrics],
            }
        return suite_res

    all_q = list(range(n_test_queries))
    eval_setting_a = evaluate_suite(all_q)
    eval_setting_b_th1 = evaluate_suite(pos_queries_th1)
    eval_setting_b_th2 = evaluate_suite(pos_queries_th2)
    eval_clean_cvs = evaluate_suite(all_q, query_filter_fn=lambda pair: pair[1] not in overlap_cv_indices)

    # Paired Statistical Comparison (Fine-Tuned vs Frozen) on Setting A & Setting B (rel >= 1)
    comparison_pairs = [
        ("Single-Vector Dense", "Single-Vector Dense (Fine-Tuned MiniLM)", "Single-Vector Dense (Frozen MiniLM)"),
        ("Section-Aware Dense", "Section-Aware Dense (Fine-Tuned MiniLM)", "Section-Aware Dense (Frozen MiniLM)"),
        ("Dense + BM25", "Dense + BM25 (Fine-Tuned MiniLM)", "Dense + BM25 (Frozen MiniLM)"),
        ("Calibrated Hybrid", "Calibrated Hybrid (Fine-Tuned MiniLM)", "Calibrated Hybrid (Frozen MiniLM)"),
        ("Parameter-Free RRF + Tax", "Parameter-Free RRF + Tax (Fine-Tuned MiniLM)", "Parameter-Free RRF + Tax (Frozen MiniLM)"),
    ]

    stat_comparisons = {}
    for label, ft_key, fr_key in comparison_pairs:
        ft_n10 = eval_setting_a[ft_key]["per_query_n10"]
        fr_n10 = eval_setting_a[fr_key]["per_query_n10"]
        diff_n10, ci_l_n10, ci_h_n10 = paired_bootstrap_ci(ft_n10, fr_n10)
        _, p_t_n10 = ttest_rel(ft_n10, fr_n10)
        try:
            _, p_w_n10 = wilcoxon(ft_n10, fr_n10)
        except ValueError:
            p_w_n10 = 1.0

        ft_map = eval_setting_a[ft_key]["per_query_map1"]
        fr_map = eval_setting_a[fr_key]["per_query_map1"]
        diff_map, ci_l_map, ci_h_map = paired_bootstrap_ci(ft_map, fr_map)
        _, p_t_map = ttest_rel(ft_map, fr_map)
        try:
            _, p_w_map = wilcoxon(ft_map, fr_map)
        except ValueError:
            p_w_map = 1.0

        stat_comparisons[label] = {
            "ndcg@10_delta": round(diff_n10, 4),
            "ndcg@10_ci95": [round(ci_l_n10, 4), round(ci_h_n10, 4)],
            "ndcg@10_ttest_p": round(float(p_t_n10), 5),
            "ndcg@10_wilcoxon_p": round(float(p_w_n10), 5),
            "map_delta": round(diff_map, 4),
            "map_ci95": [round(ci_l_map, 4), round(ci_h_map, 4)],
            "map_ttest_p": round(float(p_t_map), 5),
            "map_wilcoxon_p": round(float(p_w_map), 5),
        }

    # Print Summary Table
    print("\n" + "-" * 105, flush=True)
    print(f"{'Model / Pipeline (Setting A: N=71)':<45} | {'nDCG@5':>7} | {'nDCG@10':>7} | {'MRR':>7} | {'MAP':>7} | {'P@5':>7} | {'Spearman':>8}", flush=True)
    print("-" * 105, flush=True)
    for m_name, res in eval_setting_a.items():
        mt = res["metrics"]
        print(
            f"{m_name:<45} | {mt['ndcg@5']:>7.4f} | {mt['ndcg@10']:>7.4f} | "
            f"{mt['mrr_th1']:>7.4f} | {mt['map_th1']:>7.4f} | {mt['p@5_th1']:>7.4f} | {res['spearman_macro']:>8.4f}",
            flush=True,
        )
    print("-" * 105, flush=True)

    print("\nPaired Statistical Improvements (Fine-Tuned vs. Frozen on Setting A):", flush=True)
    for label, st in stat_comparisons.items():
        print(
            f"  * {label:<26}: d_nDCG@10 = {st['ndcg@10_delta']:+.4f} "
            f"(95% CI [{st['ndcg@10_ci95'][0]:+.4f}, {st['ndcg@10_ci95'][1]:+.4f}], p={st['ndcg@10_ttest_p']:.4f}) | "
            f"d_MAP = {st['map_delta']:+.4f} "
            f"(95% CI [{st['map_ci95'][0]:+.4f}, {st['map_ci95'][1]:+.4f}], p={st['map_ttest_p']:.4f})",
            flush=True,
        )

    output_report = {
        "model_base": base_model_name,
        "output_checkpoint": str(OUTPUT_MODEL_DIR),
        "device": str(device),
        "best_epoch": best_epoch,
        "clean_train_stats": {
            "total_clean_pairs": len(clean_train_df),
            "train_jds": len(tr_jds_list),
            "val_jds": len(val_jds_list),
            "active_ranking_train_jds": len(train_query_records),
        },
        "training_history": training_history,
        "paired_statistical_tests_setting_A": stat_comparisons,
        "setting_A_all_71_queries": {
            k: {mk: mv for mk, mv in v.items() if not mk.startswith("per_query_")}
            for k, v in eval_setting_a.items()
        },
        "setting_B_filtered_rel_ge_1": {
            k: {mk: mv for mk, mv in v.items() if not mk.startswith("per_query_")}
            for k, v in eval_setting_b_th1.items()
        },
        "setting_B_filtered_rel_eq_2": {
            k: {mk: mv for mk, mv in v.items() if not mk.startswith("per_query_")}
            for k, v in eval_setting_b_th2.items()
        },
        "clean_cvs_setting_A": {
            k: {mk: mv for mk, mv in v.items() if not mk.startswith("per_query_")}
            for k, v in eval_clean_cvs.items()
        },
    }

    out_json = RESULTS_DIR / "finetune_biencoder_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output_report, f, indent=2)
    print(f"\nSaved full evaluation report to: {out_json}", flush=True)


if __name__ == "__main__":
    main()
