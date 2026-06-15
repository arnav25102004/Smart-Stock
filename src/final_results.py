"""
SMARTSTOCK / RAL-ATCNN — Final Results Summary
Run: python src/final_results.py
"""

import pandas as pd
import os

print("=" * 70)
print("SMARTSTOCK / RAL-ATCNN -- FINAL RESULTS SUMMARY")
print("=" * 70)

files_to_check = {
    "Forecasting Comparison": "results/tables/forecasting_comparison.csv",
    "AT-CNN Results": "results/tables/atcnn_results.csv",
    "Full Comparison with RAL": "results/tables/full_comparison_with_ral.csv",
    "NLP Rule-Based": "results/tables/nlp_rule_based.csv",
    "NLP TF-IDF+SVM": "results/tables/nlp_tfidf_svm.csv",
    "Integration Demo": "results/tables/integration_demo.csv",
}

for name, path in files_to_check.items():
    print(f"\n{'='*70}")
    print(f"{name} ({path})")
    print('='*70)
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(df.to_string(index=False))
    else:
        print("  [FILE NOT FOUND]")

print(f"\n{'='*70}")
print("ALL PLOTS GENERATED:")
print('='*70)
plot_dir = "results/plots"
if os.path.exists(plot_dir):
    for f in sorted(os.listdir(plot_dir)):
        if f.endswith('.png'):
            size_kb = os.path.getsize(os.path.join(plot_dir, f)) / 1024
            print(f"  {f}  ({size_kb:.1f} KB)")

print(f"\n{'='*70}")
print("MODEL FILES:")
print('='*70)
model_dir = "results/models"
if os.path.exists(model_dir):
    for f in sorted(os.listdir(model_dir)):
        path = os.path.join(model_dir, f)
        if os.path.isfile(path):
            size_kb = os.path.getsize(path) / 1024
            print(f"  {f}  ({size_kb:.1f} KB)")

print(f"\n{'='*70}")
print("COPY EVERYTHING ABOVE FROM 'SMARTSTOCK / RAL-ATCNN' TO HERE")
print("AND PASTE INTO CLAUDE CHAT")
print('='*70)
