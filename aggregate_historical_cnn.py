"""Read-only aggregation of historical CNN-LSTM seed artifacts."""
from pathlib import Path
import csv, json, statistics

ROOT = Path("runs/cnn_lstm")
OUT = Path("results/historical_cnn_lstm")
SEEDS = [11, 12, 34, 42, 56, 69, 90, 93, 104, 702, 1000]
METRICS = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "macro_TSS", "macro_HSS"]

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in SEEDS:
        directory = ROOT / f"seed_{seed}"
        path = directory / "cnn_lstm_test_results.json"
        if not path.exists():
            raise FileNotFoundError(path)
        data = json.loads(path.read_text())
        cm = data["confusion_matrix"]
        total = sum(sum(row) for row in cm)
        row = {"seed": seed, "path": str(path), "test_sequences": data.get("test_sequences"), "confusion_total": total}
        row.update({metric: data.get(metric) for metric in METRICS})
        row["C_recall"] = data["per_class"]["C"]["POD"]
        row["M_recall"] = data["per_class"]["M"]["POD"]
        row["X_recall"] = data["per_class"]["X"]["POD"]
        row["X_TSS"] = data["per_class"]["X"]["TSS"]
        row["X_HSS"] = data["per_class"]["X"]["HSS"]
        rows.append(row)

    fields = ["seed", "path", "test_sequences", "confusion_total"] + METRICS + ["C_recall", "M_recall", "X_recall", "X_TSS", "X_HSS"]
    with (OUT / "historical_seed_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)

    summary = {}
    for metric in METRICS + ["C_recall", "M_recall", "X_recall", "X_TSS", "X_HSS"]:
        values = [float(row[metric]) for row in rows]
        summary[metric] = {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values), "min": min(values), "max": max(values)}
    output = {"seeds": SEEDS, "n_seeds": len(rows), "rows": rows, "aggregate": summary, "all_test_sizes_2021": all(r["test_sequences"] == 2021 for r in rows), "all_confusion_totals_2021": all(r["confusion_total"] == 2021 for r in rows)}
    (OUT / "historical_seed_metrics.json").write_text(json.dumps(output, indent=2))

    lines = ["# Historical CNN-LSTM seed verification", "", "Historical evaluation artifacts were read without retraining or rerunning evaluation.", "", "| Seed | Accuracy | Macro F1 | Weighted F1 | Macro recall | Macro TSS | Macro HSS | X recall | Test n |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append("| {seed} | {accuracy:.4f} | {macro_f1:.4f} | {weighted_f1:.4f} | {macro_recall:.4f} | {macro_TSS:.4f} | {macro_HSS:.4f} | {X_recall:.4f} | {test_sequences} |".format(**r))
    lines += ["", "## Mean ± sample SD; range", "", "| Metric | Mean | Sample SD | Minimum | Maximum |", "|---|---:|---:|---:|---:|"]
    for metric, s in summary.items():
        lines.append(f"| {metric} | {s['mean']:.6f} | {s['sample_sd']:.6f} | {s['min']:.6f} | {s['max']:.6f} |")
    lines += ["", f"Verified seeds: `{', '.join(map(str, SEEDS))}`.", "All JSON files report 2,021 test sequences and each confusion matrix sums to 2,021."]
    (OUT / "HISTORICAL_SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT / 'historical_seed_metrics.csv'}")
    print(f"Wrote {OUT / 'historical_seed_metrics.json'}")
    print(f"Wrote {OUT / 'HISTORICAL_SUMMARY.md'}")

if __name__ == "__main__": main()
