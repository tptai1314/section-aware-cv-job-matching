import json

with open("results/statistical_significance_family.json", "r", encoding="utf-8") as f:
    d = json.load(f)

set_b = d["Setting B (Filtered 61)"]
print("=== SETTING B (Filtered 61) ===")
for k, v in set_b.items():
    print(f"{v['rq']:4s} | {v['model_a']:18s} vs {v['model_b']:18s} | {v['metric']:8s} | Delta={v['mean_diff']:+7.4f} | CI={v['ci_str']:22s} | raw_p={v['raw_t_p']:7.4f} | adj_p={v['adj_p']:7.4f}")
