#!/usr/bin/env python3

"""
Parse benchmark log files and extract metrics to CSV format.

Usage:
    python parse_benchmark_log.py <log_file> [output_csv]

Arguments:
    log_file: Path to the benchmark log file
    output_csv: Optional output CSV file path (default: benchmark_results.csv)
"""

import re
import csv
import sys
import os
from typing import Dict, Any, Optional

def parse_single_namespace(namespace_content: str) -> Dict[str, Any]:
    """Parse a single Namespace content block."""
    args = {}

    # Extract key-value pairs from namespace
    # Handle both quoted and unquoted values
    patterns = [
        r"(\w+)='([^']*)'",  # Single quoted
        r"(\w+)=\"([^\"]*)\"",  # Double quoted
        r"(\w+)=(\d+\.?\d*)",  # Numeric
        r"(\w+)=(None|True|False|inf)",  # Special values
    ]

    def is_numeric_value(val):
        """Check if a string represents a valid numeric value."""
        try:
            float(val)
            return True
        except ValueError:
            return False

    for pattern in patterns:
        matches = re.findall(pattern, namespace_content)
        for key, value in matches:
            # Convert numeric strings to appropriate types
            if is_numeric_value(value):
                # Check if it's a valid number by trying to convert
                try:
                    if '.' in value:
                        args[key] = float(value)
                    else:
                        args[key] = int(value)
                except ValueError:
                    # If conversion fails, keep as string
                    args[key] = value
            elif value in ['None', 'True', 'False', 'inf']:
                if value == 'None':
                    args[key] = None
                elif value == 'True':
                    args[key] = True
                elif value == 'False':
                    args[key] = False
                elif value == 'inf':
                    args[key] = float('inf')
            else:
                args[key] = value

    return args

def find_all_benchmark_runs(log_content: str) -> list:
    """Find all benchmark runs in the log file and return them as a list of dicts."""

    # Split the log into sections based on Namespace occurrences
    namespace_pattern = r'Namespace\((.*?)\)'
    benchmark_result_pattern = r'============ Serving Benchmark Result ============(.*?)=================================================='

    # Find all Namespace sections
    namespace_matches = list(re.finditer(namespace_pattern, log_content, re.DOTALL))

    # Find all benchmark result sections
    benchmark_matches = list(re.finditer(benchmark_result_pattern, log_content, re.DOTALL))

    print(f"Found {len(namespace_matches)} Namespace sections")
    print(f"Found {len(benchmark_matches)} benchmark result sections")

    runs = []
    used_benchmark_indices = set()  # Track which benchmark results have been used

    # Process each benchmark run
    # Note: There might be more Namespace sections than benchmark results (initial tests)
    for i, namespace_match in enumerate(namespace_matches):
        run_data = {
            'namespace': parse_single_namespace(namespace_match.group(1)),
            'benchmark_results': {},
            'additional_info': {}
        }

        # Try to find the corresponding benchmark results
        # Look for benchmark results that come after this namespace
        namespace_end = namespace_match.end()

        for j, benchmark_match in enumerate(benchmark_matches):
            if j not in used_benchmark_indices and benchmark_match.start() > namespace_end:
                # This benchmark result comes after this namespace and hasn't been used
                run_data['benchmark_results'] = parse_benchmark_results(benchmark_match.group(1))

                # Extract additional info from the section between namespace and benchmark results
                section_content = log_content[namespace_end:benchmark_match.start()]
                run_data['additional_info'] = extract_additional_info(section_content)

                # Mark this benchmark result as used
                used_benchmark_indices.add(j)
                break

        runs.append(run_data)

    # Filter out runs that don't have benchmark results (e.g., initial test runs)
    valid_runs = [run for run in runs if run['benchmark_results']]

    print(f"Found {len(valid_runs)} valid benchmark runs with results")
    return valid_runs

def parse_benchmark_results(log_content: str) -> Dict[str, Any]:
    """Parse the benchmark results table from the log."""
    results = {}

    # Extract metrics from the results table
    patterns = {
        'successful_requests': r'Successful requests:\s+(\d+)',
        'benchmark_duration': r'Benchmark duration \(s\):\s+([\d.]+)',
        'total_input_tokens': r'Total input tokens:\s+([\d,]+)',
        'total_generated_tokens': r'Total generated tokens:\s+([\d,]+)',
        'request_throughput': r'Request throughput \(req/s\):\s+([\d.]+)',
        'output_token_throughput': r'Output token throughput \(tok/s\):\s+([\d.]+)',
        'total_token_throughput': r'Total Token throughput \(tok/s\):\s+([\d.]+)',
        'mean_ttft': r'Mean TTFT \(ms\):\s+([\d.]+)',
        'median_ttft': r'Median TTFT \(ms\):\s+([\d.]+)',
        'p99_ttft': r'P99 TTFT \(ms\):\s+([\d.]+)',
        'mean_tpot': r'Mean TPOT \(ms\):\s+([\d.]+)',
        'median_tpot': r'Median TPOT \(ms\):\s+([\d.]+)',
        'p99_tpot': r'P99 TPOT \(ms\):\s+([\d.]+)',
        'mean_itl': r'Mean ITL \(ms\):\s+([\d.]+)',
        'median_itl': r'Median ITL \(ms\):\s+([\d.]+)',
        'p99_itl': r'P99 ITL \(ms\):\s+([\d.]+)',
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, log_content)
        if match:
            value = match.group(1).replace(',', '')  # Remove commas from numbers
            results[key] = float(value) if '.' in value else int(value)

    return results

def extract_additional_info(log_content: str) -> Dict[str, Any]:
    """Extract additional information from log messages."""
    info = {}

    # Extract traffic request rate
    rate_match = re.search(r'Traffic request rate:\s+([\d.]+|inf)', log_content)
    if rate_match:
        rate_str = rate_match.group(1)
        info['traffic_request_rate'] = float('inf') if rate_str == 'inf' else float(rate_str)

    # Extract burstiness
    burstiness_match = re.search(r'Burstiness factor:\s+([\d.]+)', log_content)
    if burstiness_match:
        info['burstiness'] = float(burstiness_match.group(1))

    return info

def create_csv_row(namespace_args: Dict[str, Any],
                   benchmark_results: Dict[str, Any],
                   additional_info: Dict[str, Any]) -> Dict[str, Any]:
    """Create a single CSV row from all parsed data."""

    # Map the data to CSV headers
    row = {
        'Input Len': namespace_args.get('sonnet_input_len') or namespace_args.get('random_input_len'),
        'Output Len': namespace_args.get('sonnet_output_len') or namespace_args.get('random_output_len'),
        'Request Rate': additional_info.get('traffic_request_rate') or namespace_args.get('request_rate'),
        'Max concurrency': namespace_args.get('max_concurrency'),
        'Input Random Range Ratio': namespace_args.get('random_range_ratio'),
        'Dataset': namespace_args.get('dataset_name'),
        'Successful requests': benchmark_results.get('successful_requests'),
        'Benchmark duration': benchmark_results.get('benchmark_duration'),
        'Total input tokens': benchmark_results.get('total_input_tokens'),
        'Total generated tokens': benchmark_results.get('total_generated_tokens'),
        'Request Throughput(QPS)': benchmark_results.get('request_throughput'),
        'Output Token Throughput (TPS)': benchmark_results.get('output_token_throughput'),
        'Total Token Throughput': benchmark_results.get('total_token_throughput'),
        'Mean TTFT (ms)': benchmark_results.get('mean_ttft'),
        'Medium TTFT (ms)': benchmark_results.get('median_ttft'),  # Note: using Medium as in your header
        'P99 TTFT (ms)': benchmark_results.get('p99_ttft'),
        'Mean TPOT (ms)': benchmark_results.get('mean_tpot'),
        'Medium TPOT (ms)': benchmark_results.get('median_tpot'),  # Note: using Medium as in your header
        'P99 TPOT (ms)': benchmark_results.get('p99_tpot'),
        'Mean ITL (ms)': benchmark_results.get('mean_itl'),
        'Median ITL (ms)': benchmark_results.get('median_itl'),
        'P99 ITL (ms)': benchmark_results.get('p99_itl'),
        'Num_Prompts': namespace_args.get('num_prompts'),
    }

    return row

def assign_iterations(csv_rows):
    """Group CSV rows by specified fields and assign iteration numbers."""
    from collections import defaultdict

    # Group by: Num_Prompts, Max concurrency, Request Rate, Input Len, Output Len
    groups = defaultdict(list)

    for row in csv_rows:
        # Create a tuple key for grouping
        group_key = (
            row.get('Num_Prompts'),
            row.get('Max concurrency'),
            row.get('Request Rate'),
            row.get('Input Len'),
            row.get('Output Len')
        )
        groups[group_key].append(row)

    print(f"\nGrouping analysis:")
    print(f"  Found {len(groups)} unique configuration groups")

    # Assign iteration numbers within each group
    result_rows = []
    for group_key, group_rows in groups.items():
        # Sort by Run_Number to maintain chronological order
        group_rows.sort(key=lambda x: x.get('Run_Number', 0))

        print(f"  Group {group_key}: {len(group_rows)} runs -> iterations 1 to {len(group_rows)}")

        for iteration, row in enumerate(group_rows, 1):
            row_copy = row.copy()
            row_copy['iteration'] = iteration
            result_rows.append(row_copy)

    return result_rows

def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_benchmark_log.py <log_file> [output_csv]")
        sys.exit(1)

    log_file = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else 'benchmark_results.csv'

    # Check if log file exists
    if not os.path.exists(log_file):
        print(f"Error: Log file '{log_file}' does not exist")
        sys.exit(1)

    # Read the log file
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            log_content = f.read()
    except Exception as e:
        print(f"Error reading log file: {e}")
        sys.exit(1)

    # Parse all benchmark runs
    benchmark_runs = find_all_benchmark_runs(log_content)

    if not benchmark_runs:
        print("No benchmark runs found in the log file")
        sys.exit(1)

    # Define CSV headers
    headers = [
        'Input Len', 'Output Len', 'Request Rate', 'Max concurrency',
        'Input Random Range Ratio', 'Dataset', 'Successful requests',
        'Benchmark duration', 'Total input tokens', 'Total generated tokens',
        'Request Throughput(QPS)', 'Output Token Throughput (TPS)',
        'Total Token Throughput', 'Mean TTFT (ms)', 'Medium TTFT (ms)',
        'P99 TTFT (ms)', 'Mean TPOT (ms)', 'Medium TPOT (ms)',
        'P99 TPOT (ms)', 'Mean ITL (ms)', 'Median ITL (ms)',
        'P99 ITL (ms)', 'Num_Prompts'
    ]

    # Create CSV rows for all benchmark runs
    csv_rows = []
    for i, run in enumerate(benchmark_runs):
        try:
            csv_row = create_csv_row(run['namespace'], run['benchmark_results'], run['additional_info'])
            csv_row['Run_Number'] = i + 1  # Add run number for reference
            csv_rows.append(csv_row)
            print(f"Processed benchmark run {i + 1}: max_concurrency={run['namespace'].get('max_concurrency', 'N/A')}")
        except Exception as e:
            print(f"Error processing benchmark run {i + 1}: {e}")
            continue

    # Group by specified fields and assign iteration numbers
    if csv_rows:
        csv_rows = assign_iterations(csv_rows)

    # Write to CSV
    try:
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            # Add Run_Number and iteration to headers
            csv_headers = ['Run_Number', 'iteration'] + headers
            writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
            writer.writeheader()

            for row in csv_rows:
                writer.writerow(row)

        print(f"\nSuccessfully parsed {len(csv_rows)} benchmark runs from log file")
        print(f"Saved results to: {output_csv}")
        print(f"Processed log file: {log_file}")

        # Print summary
        print(f"\nSummary:")
        print(f"  Total benchmark runs found: {len(benchmark_runs)}")
        print(f"  Valid runs processed: {len(csv_rows)}")
        print(f"  CSV rows written: {len(csv_rows)}")

        # Show iteration distribution
        if csv_rows:
            iteration_counts = {}
            for row in csv_rows:
                iteration = row.get('iteration', 1)
                iteration_counts[iteration] = iteration_counts.get(iteration, 0) + 1

            print(f"  Iteration distribution: {iteration_counts}")

        if csv_rows:
            print(f"\nSample row (first run):")
            sample_row = csv_rows[0]
            # Show key columns including iteration
            key_headers = ['Run_Number', 'iteration', 'Max concurrency', 'Num_Prompts', 'Request Rate']
            for header in key_headers:
                value = sample_row.get(header, '')
                print(f"  {header}: {value}")
            print("  ... (plus 22 more performance metrics)")

    except Exception as e:
        print(f"Error writing to CSV: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
