#!/usr/bin/env python3

import argparse
import csv
import os
import re
from typing import Dict, Optional, Tuple, List


HEADER = [
    "Input Len",
    "Output Len",
    "Request Rate",
    "Max concurrency",
    "Input Random Range Ratio",
    "Dataset",
    "Successful requests",
    "Mean TPOT (ms)",
    "Medium TPOT (ms)",
    "P99 TPOT (ms)",
]


def parse_namespace(ns_line: str) -> Dict[str, str]:
    """Parse a Python argparse Namespace(...) line into a dict of string values.

    This is a heuristic parser robust to quoted strings and simple tokens.
    """
    # Extract inside Namespace(...)
    m = re.search(r"Namespace\((.*)\)\s*$", ns_line.strip())
    if not m:
        return {}
    payload = m.group(1)

    # Split on ", " at top level (no parentheses in values here)
    parts = []
    current = []
    in_quote = False
    quote_char = ''
    for ch in payload:
        if ch in ('"', "'"):
            if not in_quote:
                in_quote = True
                quote_char = ch
            elif quote_char == ch:
                in_quote = False
        if ch == ',' and not in_quote:
            parts.append(''.join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        parts.append(''.join(current).strip())

    kv: Dict[str, str] = {}
    for p in parts:
        if '=' not in p:
            continue
        k, v = p.split('=', 1)
        k = k.strip()
        v = v.strip()
        # strip matching quotes
        if len(v) >= 2 and ((v[0] == '"' and v[-1] == '"') or (v[0] == "'" and v[-1] == "'")):
            v = v[1:-1]
        kv[k] = v
    return kv


def iter_2nd_round_segments(text: str):
    """Yield slices of the log corresponding to each '2nd ROUND' section."""
    indices = [m.start() for m in re.finditer(r"2nd ROUND", text)]
    if not indices:
        return
    for i, start in enumerate(indices):
        end = indices[i + 1] if i + 1 < len(indices) else len(text)
        yield text[start:end]


def extract_metrics(text: str) -> Optional[Tuple[int, float, float, float]]:
    # Successful requests and TPOT metrics
    sr = re.search(r"Successful requests:\s+(\d+)", text)
    mean = re.search(r"Mean TPOT \(ms\):\s+([0-9.]+)", text)
    med = re.search(r"Median TPOT \(ms\):\s+([0-9.]+)", text)
    p99 = re.search(r"P99 TPOT \(ms\):\s+([0-9.]+)", text)
    if not (sr and mean and med and p99):
        return None
    return int(sr.group(1)), float(mean.group(1)), float(med.group(1)), float(p99.group(1))


def choose_lengths(ns: Dict[str, str]) -> Tuple[Optional[int], Optional[int], str]:
    """Pick Input Len and Output Len based on dataset fields found.

    Returns (input_len, output_len, dataset_name)
    """
    dataset = ns.get('dataset_name') or ns.get('dataset') or ''
    def to_int(key: str) -> Optional[int]:
        v = ns.get(key)
        if v is None or v == 'None':
            return None
        try:
            return int(float(v))
        except Exception:
            return None

    # Prefer explicit dataset-specific lengths
    candidates_in = [
        'sonnet_input_len',
        'random_input_len',
    ]
    candidates_out = [
        'sonnet_output_len',
        'random_output_len',
        'sharegpt_output_len',
        'hf_output_len',
    ]
    in_len: Optional[int] = None
    out_len: Optional[int] = None
    for k in candidates_in:
        in_len = to_int(k)
        if in_len is not None:
            break
    for k in candidates_out:
        out_len = to_int(k)
        if out_len is not None:
            break
    return in_len, out_len, dataset


def extract_rows(text: str) -> List[List[str]]:
    rows: List[List[str]] = []
    found_any = False
    for seg in iter_2nd_round_segments(text):
        found_any = True
        # 1) Namespace line local to this segment
        ns_line_match = re.search(r"^Namespace\(.*\)$", seg, flags=re.M)
        if not ns_line_match:
            continue
        ns = parse_namespace(ns_line_match.group(0))
        # 2) Metrics block within this segment
        metrics_start = seg.find("============ Serving Benchmark Result ============")
        if metrics_start == -1:
            continue
        metrics = extract_metrics(seg[metrics_start:])
        if metrics is None:
            continue
        succ, mean_tpot, med_tpot, p99_tpot = metrics

        in_len, out_len, dataset = choose_lengths(ns)
        req_rate = ns.get('request_rate', '')
        max_conc = ns.get('max_concurrency', '')
        rand_ratio = ns.get('random_range_ratio', '')

        rows.append([
            str(in_len) if in_len is not None else '',
            str(out_len) if out_len is not None else '',
            req_rate,
            str(max_conc),
            str(rand_ratio),
            dataset,
            str(succ),
            f"{mean_tpot:.2f}",
            f"{med_tpot:.2f}",
            f"{p99_tpot:.2f}",
        ])
    # If no explicit 2nd ROUND markers found, try whole text as a single segment (backward compat)
    if not found_any:
        ns_line_match = re.search(r"^Namespace\(.*\)$", text, flags=re.M)
        metrics_start = text.find("============ Serving Benchmark Result ============")
        if ns_line_match and metrics_start != -1:
            ns = parse_namespace(ns_line_match.group(0))
            metrics = extract_metrics(text[metrics_start:])
            if metrics:
                succ, mean_tpot, med_tpot, p99_tpot = metrics
                in_len, out_len, dataset = choose_lengths(ns)
                req_rate = ns.get('request_rate', '')
                max_conc = ns.get('max_concurrency', '')
                rand_ratio = ns.get('random_range_ratio', '')
                rows.append([
                    str(in_len) if in_len is not None else '',
                    str(out_len) if out_len is not None else '',
                    req_rate,
                    str(max_conc),
                    str(rand_ratio),
                    dataset,
                    str(succ),
                    f"{mean_tpot:.2f}",
                    f"{med_tpot:.2f}",
                    f"{p99_tpot:.2f}",
                ])
    return rows


def main():
    ap = argparse.ArgumentParser(description="Extract benchmark metrics from logs and write CSV.")
    ap.add_argument('-i', '--inputs', nargs='+', required=True, help='Input log files')
    ap.add_argument('-o', '--output', required=True, help='Output CSV file path')
    args = ap.parse_args()

    rows: List[List[str]] = []
    for path in args.inputs:
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                txt = f.read()
            file_rows = extract_rows(txt)
            if not file_rows:
                print(f"Warning: Failed to extract metrics from {path}")
            rows.extend(file_rows)
        except Exception as e:
            print(f"Warning: Error reading {path}: {e}")

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == '__main__':
    main()


