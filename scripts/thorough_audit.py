import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_rel, wilcoxon
from collections import defaultdict
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'src'))

df = pd.read_csv(REPO_ROOT / 'data/public_benchmark/resume-job-description-fit/test.csv', encoding='latin1')
label_map = {'No Fit': 0, 'Potential Fit': 1, 'Good Fit': 2}
df['rel'] = df['label'].map(label_map)

unique_jobs = df['job_description_text'].unique().tolist()
job_to_idx = {j: i for i, j in enumerate(unique_jobs)}
unique_res = df['resume_text'].unique().tolist()
res_to_idx = {r: i for i, r in enumerate(unique_res)}

query_groups = defaultdict(list)
pairs = []
y_true = []
for idx, row in df.iterrows():
    j_idx = job_to_idx[row['job_description_text']]
    r_idx = res_to_idx[row['resume_text']]
    query_groups[j_idx].append(len(pairs))
    pairs.append((j_idx, r_idx))
    y_true.append(row['rel'])
y_true = np.array(y_true)

from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from cvjd.data.preprocessing import sections_to_dict, split_sections
from cvjd.data.taxonomy import get_taxonomy
from cvjd.ranking.metrics import ndcg_at_k, reciprocal_rank, average_precision, precision_at_k
from cvjd.ranking.engine import min_max

# 1. Lexical BM25
vec = TfidfVectorizer(max_features=10000, stop_words='english', sublinear_tf=True)
vec.fit(unique_jobs + unique_res)
job_vecs = vec.transform(unique_jobs)
res_vecs = vec.transform(unique_res)
bm25_scores = np.array([float((job_vecs[j].multiply(res_vecs[r])).sum()) for j, r in pairs])

# 2. Dense Single-Vector
dense_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
job_dense_embs = dense_model.encode(unique_jobs, batch_size=32, normalize_embeddings=True)
res_dense_embs = dense_model.encode(unique_res, batch_size=32, normalize_embeddings=True)
single_dense_scores = np.array([float(np.dot(job_dense_embs[j], res_dense_embs[r])) for j, r in pairs])

# 3. Section-Aware
res_parsed = [sections_to_dict(split_sections(r)) for r in unique_res]
res_sec_vecs = []
for sd in res_parsed:
    vd = {}
    for sn, s_obj in sd.items():
        if len(s_obj.text.strip()) > 10:
            vd[sn] = dense_model.encode(s_obj.text[:512], normalize_embeddings=True)
    res_sec_vecs.append(vd)

sec_weights = {'skills': 0.40, 'experience': 0.35, 'education': 0.15, 'summary': 0.10}
sec_aware_scores = []
for j, r in pairs:
    jv = job_dense_embs[j]
    rvs = res_sec_vecs[r]
    if not rvs:
        sec_aware_scores.append(single_dense_scores[len(sec_aware_scores)])
        continue
    w_sim, tot_w = 0.0, 0.0
    for sn, w in sec_weights.items():
        if sn in rvs:
            w_sim += w * float(np.dot(jv, rvs[sn]))
            tot_w += w
    sim = w_sim / tot_w if tot_w > 0 else single_dense_scores[len(sec_aware_scores)]
    sec_aware_scores.append(0.65 * sim + 0.35 * single_dense_scores[len(sec_aware_scores)])
sec_aware_scores = np.array(sec_aware_scores)

# 4. Taxonomy
taxonomy = get_taxonomy()
job_skills = [taxonomy.classify_jd_skills(j) for j in unique_jobs]
res_skills = [taxonomy.extract_skills_from_text(r) for r in unique_res]
tax_scores = []
for j, r in pairs:
    must, nice = job_skills[j]
    c_sk = res_skills[r]
    if not must:
        tax_scores.append(1.0)
        continue
    cov = 0.0
    for m in must:
        if m in c_sk: cov += 1.0
        else:
            mx = max([taxonomy.get_relation_weight(cs, m) for cs in c_sk], default=0.0)
            cov += mx
    must_cov = cov / len(must)
    nice_cov = sum(1.0 for ns in nice if ns in c_sk) / len(nice) if nice else 0.0
    tax_scores.append(0.85 * must_cov + 0.15 * nice_cov)
tax_scores = np.array(tax_scores)

# Normalization per query
norm_sec = np.zeros_like(sec_aware_scores)
norm_bm25 = np.zeros_like(bm25_scores)
for q, p_idx in query_groups.items():
    norm_sec[p_idx] = min_max(sec_aware_scores[p_idx])
    norm_bm25[p_idx] = min_max(bm25_scores[p_idx])

# Dense + BM25 (no taxonomy)
dense_plus_bm25 = 0.60 * norm_sec + 0.40 * norm_bm25
# Calibrated Hybrid (Dense + BM25 + Taxonomy)
hybrid_full = 0.60 * dense_plus_bm25 + 0.40 * tax_scores

# Random floor & Oracle ceiling
def get_oracle_and_random():
    rng = np.random.RandomState(42)
    # Oracle
    o_scores = y_true.astype(float) + rng.uniform(0, 1e-6, size=len(y_true))
    # Random
    r_scores = rng.uniform(0, 1, size=len(y_true))
    return r_scores, o_scores

rand_scores, oracle_scores = get_oracle_and_random()

models = {
    'Floor: Random Ranking': rand_scores,
    'BM25 (Lexical)': bm25_scores,
    'Single-Vector Dense': single_dense_scores,
    'Section-Aware Dense': sec_aware_scores,
    'Dense + BM25 (No Tax)': dense_plus_bm25,
    'Taxonomy Rules Alone': tax_scores,
    'Hybrid (Dense+BM25+Tax)': hybrid_full,
    'Ceiling: Oracle Ranking': oracle_scores,
}

def evaluate_full(scores, thresh):
    n5, n10, mrr, m_ap, p5 = [], [], [], [], []
    per_q_rhos = []
    for q, p_idx in query_groups.items():
        qrels = {str(pairs[p][1]): float(y_true[p]) for p in p_idx}
        c_ids = [str(pairs[p][1]) for p in p_idx]
        order = np.argsort(-scores[p_idx])
        ranked = [c_ids[i] for i in order]
        n5.append(ndcg_at_k(ranked, qrels, k=5))
        n10.append(ndcg_at_k(ranked, qrels, k=10))
        mrr.append(reciprocal_rank(ranked, qrels, threshold=thresh))
        m_ap.append(average_precision(ranked, qrels, threshold=thresh))
        p5.append(precision_at_k(ranked, qrels, k=5, threshold=thresh))
        # Spearman per query
        if len(p_idx) >= 2 and len(np.unique(y_true[p_idx])) > 1 and len(np.unique(scores[p_idx])) > 1:
            r, _ = spearmanr(scores[p_idx], y_true[p_idx])
            if not np.isnan(r): per_q_rhos.append(r)
    rho_micro, _ = spearmanr(scores, y_true)
    rho_macro = np.mean(per_q_rhos) if per_q_rhos else 0.0
    return {
        'n5': np.mean(n5), 'n10': np.mean(n10), 'mrr': np.mean(mrr), 'map': np.mean(m_ap), 'p5': np.mean(p5),
        'rho_micro': rho_micro, 'rho_macro': rho_macro, 'n10_list': n10, 'n5_list': n5, 'map_list': m_ap
    }

for thresh in [1, 2]:
    desc = 'rel >= 1 (Potential Fit + Good Fit)' if thresh == 1 else 'rel == 2 (Good Fit only)'
    print(f'\n=== THRESHOLD: {desc} ===')
    for name, sc in models.items():
        res = evaluate_full(sc, thresh)
        n5_val = res['n5']
        n10_val = res['n10']
        mrr_val = res['mrr']
        map_val = res['map']
        p5_val = res['p5']
        mic_val = res['rho_micro']
        mac_val = res['rho_macro']
        print(f"{name:26s} | n5: {n5_val:.4f} | n10: {n10_val:.4f} | MRR: {mrr_val:.4f} | MAP: {map_val:.4f} | P@5: {p5_val:.4f} | Rho_mic: {mic_val:.4f} | Rho_mac: {mac_val:.4f}")

# Statistical Significance test
res_single = evaluate_full(single_dense_scores, 1)
res_sec = evaluate_full(sec_aware_scores, 1)
res_hyb = evaluate_full(hybrid_full, 1)
res_no_tax = evaluate_full(dense_plus_bm25, 1)

t_n10, p_n10 = ttest_rel(res_single['n10_list'], res_sec['n10_list'])
w_n10, wp_n10 = wilcoxon(res_single['n10_list'], res_sec['n10_list'])
print(f'\n--- Statistical Testing (71 queries) ---')
print(f'Single-Vector vs Section-Aware nDCG@10: t-test p={p_n10:.4f}, Wilcoxon p={wp_n10:.4f}')

t_tax, p_tax = ttest_rel(res_no_tax['map_list'], res_hyb['map_list'])
print(f'Dense+BM25 vs Dense+BM25+Tax MAP: t-test p={p_tax:.4f}')
