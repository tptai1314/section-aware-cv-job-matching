import json
from pathlib import Path

with open("results/scientific_benchmark_results.json", "r", encoding="utf-8") as f:
    d = json.load(f)

for setting_key in ["setting_A_all_71_queries", "setting_B_filtered_rel_ge_1"]:
    print(f"\n--- {setting_key} ---")
    sub = d[setting_key]
    for model_name, data in sub.items():
        pq_n10 = data.get("per_query_n10", [])
        pq_map = data.get("per_query_map1", [])
        print(f"  {model_name[:35]:35s} | per_query_n10 len={len(pq_n10)} | per_query_map1 len={len(pq_map)}")
