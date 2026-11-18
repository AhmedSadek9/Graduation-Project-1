import csv
from pathlib import Path
from statistics import mean

from src.utils.evaluation import (
    calculate_corpus_cer,
    calculate_corpus_wer,
)

# ============================================================
# PATHS
# ============================================================

results_path = Path(
    "data/outputs/evaluation_run_02/review_subset_44_results.csv"
)

issues_path = Path(
    "data/outputs/evaluation_run_02/evaluation_issues.csv"
)

summary_path = Path(
    "data/outputs/evaluation_run_02/evaluation_summary.csv"
)

# ============================================================
# LOAD EXISTING OCR RESULTS
# ============================================================

if not results_path.exists():
    raise FileNotFoundError(results_path)

with open(
    results_path,
    "r",
    encoding="utf-8-sig",
    newline="",
) as file:
    all_results = list(csv.DictReader(file))

# ============================================================
# LOAD EXISTING ISSUES
# ============================================================

if issues_path.exists():
    with open(
        issues_path,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        failed_results = list(csv.DictReader(file))
else:
    failed_results = []

# ============================================================
# CALCULATE SUMMARY
# ============================================================

summary_results = []

engines = ["Tesseract", "PaddleOCR"]

for engine_name in engines:

    engine_results = [
        item for item in all_results
        if item["Engine"] == engine_name
        and item["Status"] == "success"
        and item["Extracted Text"].strip()
    ]

    engine_failures = [
        item for item in failed_results
        if item["Engine"] == engine_name
    ]

    if not engine_results:
        print(f"{engine_name}: No successful results")
        continue

    avg_confidence = mean(
        float(item["Confidence (%)"])
        for item in engine_results
    )

    avg_cer = mean(
        float(item["CER"])
        for item in engine_results
    )

    avg_wer = mean(
        float(item["WER"])
        for item in engine_results
    )

    pairs = [
        (
            item["Ground Truth"],
            item["Extracted Text"],
        )
        for item in engine_results
    ]

    corpus_cer = calculate_corpus_cer(pairs) * 100
    corpus_wer = calculate_corpus_wer(pairs) * 100

    print("\n" + "=" * 50)
    print(engine_name)
    print("Successful images:", len(engine_results))
    print("Failed images:", len(engine_failures))
    print(f"Average Confidence: {avg_confidence:.2f}%")
    print(f"Average CER: {avg_cer:.2f}%")
    print(f"Average WER: {avg_wer:.2f}%")
    print(f"Corpus CER: {corpus_cer:.2f}%")
    print(f"Corpus WER: {corpus_wer:.2f}%")

    summary_results.append({
        "Engine": engine_name,
        "Successful Images": len(engine_results),
        "Failed Images": len(engine_failures),
        "Average Confidence (%)": round(avg_confidence, 2),
        "Average CER (%)": round(avg_cer, 2),
        "Average WER (%)": round(avg_wer, 2),
        "Corpus CER (%)": round(corpus_cer, 2),
        "Corpus WER (%)": round(corpus_wer, 2),
    })

# ============================================================
# SAVE SUMMARY
# ============================================================

summary_fields = [
    "Engine",
    "Successful Images",
    "Failed Images",
    "Average Confidence (%)",
    "Average CER (%)",
    "Average WER (%)",
    "Corpus CER (%)",
    "Corpus WER (%)",
]

# Refuse to overwrite an existing summary
if summary_path.exists():
    raise FileExistsError(
        f"Summary already exists: {summary_path}"
    )

with open(
    summary_path,
    "w",
    newline="",
    encoding="utf-8-sig",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=summary_fields,
    )

    writer.writeheader()
    writer.writerows(summary_results)

print("\nSummary created successfully:")
print(summary_path)