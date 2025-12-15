import os
import csv
import sys
from natsort import natsorted

parent_dir = "."

# For vllm 0.5.5
# csv_headers = [
#     "model", "cardnumer", "datatype", "dataset", "batchsize", "request_rate",
#     "prompts", "input_len", "output_len",
#     "Successful requests", "Benchmark duration (s)", "Total input tokens", "Total generated tokens",
#     "Request throughput (req/s)", "Input token throughput (tok/s)", "Output token throughput (tok/s)",
#     "Mean TTFT (ms)", "Median TTFT (ms)", "P99 TTFT (ms)",
#     "Mean TPOT (ms)", "Median TPOT (ms)", "P99 TPOT (ms)",
#     "Mean ITL (ms)", "Median ITL (ms)", "P99 ITL (ms)"
# ]

# For vllm 0.6.4
csv_headers = [
    "model", "cardnumer", "datatype", "dataset",
    "input_len", "output_len", "request_rate", "batchsize", "ratio",
    "Successful requests", "Benchmark duration (s)", "Total input tokens", "Total generated tokens", 
    "Request throughput (req/s)", "Output token throughput (tok/s)", "Total Token throughput (tok/s)", 
    "Mean TPOT (ms)", "Median TPOT (ms)", "P99 TPOT (ms)",
    "Mean TTFT (ms)", "Median TTFT (ms)", "P99 TTFT (ms)",
    "prompts",
    "Mean ITL (ms)", "Median ITL (ms)", "P99 ITL (ms)"
]


def parse_filename(filename):
    parts = filename.replace(".log", "").split("_")
    if len(parts) == 15:
        return {
            "model": parts[2],
            "cardnumer": parts[4],
            "datatype": parts[6],
            "dataset": parts[7],
            "batchsize": parts[9],
            "request_rate": parts[11],
            "prompts": parts[13],
            "input_len": "-",
            "output_len": "-",
            "ratio": "-"
        }
    elif len(parts) == 21:
        return {
            "model": parts[2],
            "cardnumer": parts[4],
            "datatype": parts[6],
            "dataset": parts[7],
            "batchsize": parts[9],
            "input_len": parts[11],
            "output_len": parts[13],
            "ratio": parts[15],
            "request_rate": parts[17],
            "prompts": parts[19]
        }
    else:
        return None


def parse_log_content(parent_dir):
    result_data = {}
    try:
        with open(parent_dir, "r", encoding='utf-8') as f:
            lines = f.readlines()
            if "============ Serving Benchmark Result ============\n" not in lines:
                return None
            start_idx = lines.index("============ Serving Benchmark Result ============\n") + 1
            for line in lines[start_idx:]:
                if line.startswith("=================================================="):
                    break
                key_value = line.split(":")
                if len(key_value) == 2:
                    key = key_value[0].strip()
                    value = key_value[1].strip()
                    result_data[key] = value
    except Exception as e:
        print(f"Error reading file {parent_dir}: {e}")
        return None
    return result_data

def write_to_csv(output_file, csv_headers, data_rows):
    with open(output_file, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_headers)
        writer.writeheader()
        writer.writerows(data_rows)

def main():
    output_csv = "serving_bench.csv"
    if len(sys.argv) == 2:
        output_csv = "serving_bench_" + sys.argv[1] + ".csv"

    data_rows = []

    items = os.listdir(parent_dir)
    sorted_items = natsorted(items)
    for filename in sorted_items:
        if filename.endswith(".log"):
            file_path = os.path.join(parent_dir, filename)

            conditions = parse_filename(filename)
            if not conditions:
                print(f"Invalid file name format: {filename}")
                continue

            results = parse_log_content(file_path)
            if not results:
                print(f"Skipping file {filename}, no valid benchmark results.")
                continue

            data_row = {**conditions, **results}
            data_rows.append(data_row)

    write_to_csv(os.path.join(parent_dir, output_csv), csv_headers, data_rows)
    print(f"Results written to {output_csv}")

if __name__ == "__main__":
    main()
