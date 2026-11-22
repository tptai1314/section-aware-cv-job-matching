import json

with open("results/scientific_benchmark_results.json", "r", encoding="utf-8") as f:
    d = json.load(f)

print("=== SETTING B: rel >= 1 (N = 61 queries) ===")
s1 = d.get("setting_B_filtered_rel_ge_1", {})
for k, v in s1.items():
    if "metrics" in v:
        m = v["metrics"]
        print(f"{k:45s} | nDCG@10: {m.get('ndcg@10', 0):.4f} | MRR: {m.get('mrr_th1', 0):.4f} | MAP: {m.get('map_th1', 0):.4f} | P@5: {m.get('p@5_th1', 0):.4f}")

print("\n=== SETTING B: rel == 2 (N = 28 queries) ===")
s2 = d.get("setting_B_filtered_rel_eq_2", {})
for k, v in s2.items():
    if "metrics" in v:
        m = v["metrics"]
        print(f"{k:45s} | nDCG@10: {m.get('ndcg@10', 0):.4f} | MRR: {m.get('mrr_th2', 0):.4f} | MAP: {m.get('map_th2', 0):.4f} | P@5: {m.get('p@5_th2', 0):.4f}")
