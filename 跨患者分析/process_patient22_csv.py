import ast
import csv
from pathlib import Path


def compute_success_rate(values):
    """Return count(1) / length for a list."""
    if not values:
        return ""
    return sum(1 for v in values if v == 1) / len(values)


def compute_mean(values):
    """Return arithmetic mean for a list."""
    if not values:
        return ""
    return sum(values) / len(values)


def process_csv(input_path: Path, output_path: Path) -> int:
    rows = []

    with input_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 3:
                continue

            patient = row[0].strip()
            second_col_array = ast.literal_eval(row[1].strip())
            third_col_array = ast.literal_eval(row[2].strip())

            success_rate = compute_success_rate(second_col_array)
            third_col_mean = compute_mean(third_col_array)

            rows.append([patient, success_rate, third_col_mean])

    avg_success_rate = compute_mean([r[1] for r in rows if r[1] != ""])
    avg_third_col_mean = compute_mean([r[2] for r in rows if r[2] != ""])
    rows.append(["overall_avg", avg_success_rate, avg_third_col_mean])

    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["patient", "success_rate", "third_col_mean"])
        writer.writerows(rows)

    return len(rows)


def main():
    input_file = Path(
        r"d:\2025_important_projects\paper-main\跨患者分析\patient23weight0.500000_eegnet_03-30.csv" # patient22weight0.500000_deep4_03-24.csv  patient22weight0.500000_eegnet_03-24.csv
    )
    output_file = input_file.with_name(input_file.stem + "_processed.csv")

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    count = process_csv(input_file, output_file)
    print(f"Processed {count} rows.")
    print(f"Output file: {output_file}")


if __name__ == "__main__":
    main()
