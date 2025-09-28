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
    "Benchmark duration",
    "Total input tokens",
    "Total generated tokens",
    "Request Throughput(QPS)",
    "Output Token Throughput (TPS)",
    "Total Token Throughput",
    "Mean TTFT (ms)",
    "Medium TTFT (ms)",
    "P99 TTFT (ms)",
    "Mean TPOT (ms)",
    "Medium TPOT (ms)",
    "P99 TPOT (ms)",
    "Mean ITL (ms)",
    "Median ITL (ms)",
    "P99 ITL (ms)",
    "Num_Prompts",
]


def parse_namespace(ns_line: str) -> Dict[str, str]:
    m = re.search(r"Namespace\((.*)\)\s*$", ns_line.strip())
    if not m:
        return {}
    payload = m.group(1)
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
        if len(v) >= 2 and ((v[0] == '"' and v[-1] == '"') or (v[0] == "'" and v[-1] == "'")):
            v = v[1:-1]
        kv[k] = v
    return kv


def iter_iteration_segments(text: str):
    """Yield segments for each top-level iteration starting line.

    The header line looks like:
    ========  INPUT LEN: 3500 | CONCURRENCY: 64 | NUM_PROMPT: 640  ========
    """
    # Match the iteration banner line, optionally including REQ_RATE field
    # Examples matched:
    # ========  INPUT LEN: 3500 | CONCURRENCY: 64 | NUM_PROMPT: 640  ========
    # ========  INPUT LEN: 3500 | CONCURRENCY: 64 | REQ_RATE: inf | NUM_PROMPT: 640  ========
    banner_pattern = re.compile(
        r"^\s*=*.*INPUT LEN:\s*\d+\s*\|\s*CONCURRENCY:\s*\d+(?:\s*\|\s*REQ_RATE:\s*[^|=\s]+)?\s*\|\s*NUM_PROMPT:\s*\d+.*=*\s*$",
        re.M,
    )
    indices = [m.start() for m in banner_pattern.finditer(text)]
    if not indices:
        return
    for i, start in enumerate(indices):
        end = indices[i + 1] if i + 1 < len(indices) else len(text)
        yield text[start:end]


def parse_iteration_header(seg: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    # Allow optional REQ_RATE field between CONCURRENCY and NUM_PROMPT
    m = re.search(
        r"INPUT LEN:\s*(\d+)\s*\|\s*CONCURRENCY:\s*(\d+)(?:\s*\|\s*REQ_RATE:\s*[^|=\s]+)?\s*\|\s*NUM_PROMPT:\s*(\d+)",
        seg,
    )
    if not m:
        return None, None, None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def extract_metrics_block(text: str) -> Optional[Dict[str, float]]:
    # Find the last occurrence of the Serving Benchmark Result header
    matches = list(re.finditer(r"Serving\s+Benchmark\s+Result", text, flags=re.I))
    if not matches:
        return None
    blk_start = matches[-1].start()
    blk = text[blk_start:]
    def f(regex: str) -> Optional[float]:
        mm = re.search(regex, blk)
        return float(mm.group(1)) if mm else None
    metrics = {
        'successful': f(r"Successful requests:\s+(\d+)"),
        'duration': f(r"Benchmark duration \(s\):\s+([0-9.]+)"),
        'total_input_tokens': f(r"Total input tokens:\s+(\d+)"),
        'total_generated_tokens': f(r"Total generated tokens:\s+(\d+)"),
        'req_qps': f(r"Request throughput \(req/s\):\s+([0-9.]+)"),
        'out_tps': f(r"Output token throughput \(tok/s\):\s+([0-9.]+)"),
        'total_tps': f(r"Total Token throughput \(tok/s\):\s+([0-9.]+)"),
        'mean_ttft': f(r"Mean TTFT \(ms\):\s+([0-9.]+)"),
        'median_ttft': f(r"Median TTFT \(ms\):\s+([0-9.]+)"),
        'p99_ttft': f(r"P99 TTFT \(ms\):\s+([0-9.]+)"),
        'mean_tpot': f(r"Mean TPOT \(ms\):\s+([0-9.]+)"),
        'median_tpot': f(r"Median TPOT \(ms\):\s+([0-9.]+)"),
        'p99_tpot': f(r"P99 TPOT \(ms\):\s+([0-9.]+)"),
        'mean_itl': f(r"Mean ITL \(ms\):\s+([0-9.]+)"),
        'median_itl': f(r"Median ITL \(ms\):\s+([0-9.]+)"),
        'p99_itl': f(r"P99 ITL \(ms\):\s+([0-9.]+)"),
    }
    if any(v is None for v in metrics.values() if v is not metrics['mean_itl']):
        # ITL may not be present in some logs; allow it to be None
        pass
    return metrics


def choose_lengths(ns: Dict[str, str]) -> Tuple[Optional[int], Optional[int], str]:
    dataset = ns.get('dataset_name') or ns.get('dataset') or ''
    def to_int(key: str) -> Optional[int]:
        v = ns.get(key)
        if v is None or v == 'None':
            return None
        try:
            return int(float(v))
        except Exception:
            return None
    for_in = ['sonnet_input_len', 'random_input_len']
    for_out = ['sonnet_output_len', 'random_output_len', 'sharegpt_output_len', 'hf_output_len']
    in_len = next((to_int(k) for k in for_in if to_int(k) is not None), None)
    out_len = next((to_int(k) for k in for_out if to_int(k) is not None), None)
    return in_len, out_len, dataset


def extract_rows(text: str) -> List[List[str]]:
    rows: List[List[str]] = []
    any_iter = False
    for seg in iter_iteration_segments(text):
        any_iter = True
        in_len_hdr, conc_hdr, num_prompts_hdr = parse_iteration_header(seg)
        # Find Namespace in this iteration
        ns_match = re.search(r"^Namespace\(.*\)$", seg, flags=re.M)
        if not ns_match:
            continue
        ns = parse_namespace(ns_match.group(0))
        # Last metrics block within this iteration
        metrics = extract_metrics_block(seg)
        if not metrics:
            continue
        req_rate = ns.get('request_rate', '')
        max_conc = ns.get('max_concurrency', '')
        rand_ratio = ns.get('random_range_ratio', '')
        in_len_ns, out_len_ns, dataset = choose_lengths(ns)
        in_len = in_len_hdr if in_len_hdr is not None else in_len_ns
        out_len = out_len_ns
        num_prompts = num_prompts_hdr if num_prompts_hdr is not None else ns.get('num_prompts', '')

        rows.append([
            str(in_len) if in_len is not None else '',
            str(out_len) if out_len is not None else '',
            str(req_rate),
            str(max_conc),
            str(rand_ratio),
            dataset,
            str(int(metrics['successful'])) if metrics['successful'] is not None else '',
            f"{metrics['duration']:.2f}" if metrics['duration'] is not None else '',
            str(int(metrics['total_input_tokens'])) if metrics['total_input_tokens'] is not None else '',
            str(int(metrics['total_generated_tokens'])) if metrics['total_generated_tokens'] is not None else '',
            f"{metrics['req_qps']:.2f}" if metrics['req_qps'] is not None else '',
            f"{metrics['out_tps']:.2f}" if metrics['out_tps'] is not None else '',
            f"{metrics['total_tps']:.2f}" if metrics['total_tps'] is not None else '',
            f"{metrics['mean_ttft']:.2f}" if metrics['mean_ttft'] is not None else '',
            f"{metrics['median_ttft']:.2f}" if metrics['median_ttft'] is not None else '',
            f"{metrics['p99_ttft']:.2f}" if metrics['p99_ttft'] is not None else '',
            f"{metrics['mean_tpot']:.2f}" if metrics['mean_tpot'] is not None else '',
            f"{metrics['median_tpot']:.2f}" if metrics['median_tpot'] is not None else '',
            f"{metrics['p99_tpot']:.2f}" if metrics['p99_tpot'] is not None else '',
            f"{metrics['mean_itl']:.2f}" if metrics['mean_itl'] is not None else '',
            f"{metrics['median_itl']:.2f}" if metrics['median_itl'] is not None else '',
            f"{metrics['p99_itl']:.2f}" if metrics['p99_itl'] is not None else '',
            str(num_prompts),
        ])
    if not any_iter:
        # Fallback: treat whole file as one iteration
        ns_match = re.search(r"^Namespace\(.*\)$", text, flags=re.M)
        metrics = extract_metrics_block(text)
        if ns_match and metrics:
            ns = parse_namespace(ns_match.group(0))
            in_len_ns, out_len_ns, dataset = choose_lengths(ns)
            req_rate = ns.get('request_rate', '')
            max_conc = ns.get('max_concurrency', '')
            rand_ratio = ns.get('random_range_ratio', '')
            num_prompts = ns.get('num_prompts', '')
            rows.append([
                str(in_len_ns) if in_len_ns is not None else '',
                str(out_len_ns) if out_len_ns is not None else '',
                str(req_rate),
                str(max_conc),
                str(rand_ratio),
                dataset,
                str(int(metrics['successful'])) if metrics['successful'] is not None else '',
                f"{metrics['duration']:.2f}" if metrics['duration'] is not None else '',
                str(int(metrics['total_input_tokens'])) if metrics['total_input_tokens'] is not None else '',
                str(int(metrics['total_generated_tokens'])) if metrics['total_generated_tokens'] is not None else '',
                f"{metrics['req_qps']:.2f}" if metrics['req_qps'] is not None else '',
                f"{metrics['out_tps']:.2f}" if metrics['out_tps'] is not None else '',
                f"{metrics['total_tps']:.2f}" if metrics['total_tps'] is not None else '',
                f"{metrics['mean_ttft']:.2f}" if metrics['mean_ttft'] is not None else '',
                f"{metrics['median_ttft']:.2f}" if metrics['median_ttft'] is not None else '',
                f"{metrics['p99_ttft']:.2f}" if metrics['p99_ttft'] is not None else '',
                f"{metrics['mean_tpot']:.2f}" if metrics['mean_tpot'] is not None else '',
                f"{metrics['median_tpot']:.2f}" if metrics['median_tpot'] is not None else '',
                f"{metrics['p99_tpot']:.2f}" if metrics['p99_tpot'] is not None else '',
                f"{metrics['mean_itl']:.2f}" if metrics['mean_itl'] is not None else '',
                f"{metrics['median_itl']:.2f}" if metrics['median_itl'] is not None else '',
                f"{metrics['p99_itl']:.2f}" if metrics['p99_itl'] is not None else '',
                str(num_prompts),
            ])
    return rows


def main():
    ap = argparse.ArgumentParser(description="Extract end-to-end benchmark metrics from logs and write CSV.")
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


