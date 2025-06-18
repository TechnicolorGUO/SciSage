import os
import csv
"""
Traverse a folder containing the results from MapReduce evaluation and compute the average values from all result.csv files.
"""

def find_result_csv_files(root_dir):
    result_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == "result.csv":
                result_files.append(os.path.join(dirpath, filename))
    return result_files

def parse_and_accumulate(csv_file, sums, counts):
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)

        for id,row in enumerate(reader):
            if id < 1:
                continue

            numeric_values = [float(x) for x in row[1:-2]]
            print(numeric_values)
            for i, value in enumerate(numeric_values):
                if i in [0, 1] and value == 0.0:
                    continue
                sums[i] += value
                counts[i] += 1

def compute_averages(sums, counts):
    return [s / c if c > 0 else 0.0 for s, c in zip(sums, counts)]

def main(root_dir):
    result_files = find_result_csv_files(root_dir)
    if not result_files:
        print("No result.csv files found.")
        return

    column_count = 7
    sums = [0.0] * column_count
    counts = [0] * column_count

    for file in result_files:
        parse_and_accumulate(file, sums, counts)

    averages = compute_averages(sums, counts)

    print("\n== Average Scores Across All result.csv Files ==")
    print("Fields:", [
        'language_score (non-zero only)',
        'critical_score (non-zero only)',
        'structure',
        'relevance',
        "outlinse_score",

    ])
    print("Average:", [round(x, 4) for x in averages[:-2]])

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python avg_result_csv.py <root_folder>")
    else:
        main(sys.argv[1])
