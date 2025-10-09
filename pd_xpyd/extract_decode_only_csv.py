#!/usr/bin/env python3

import argparse
import csv
import os
import re
from typing import Dict, Optional, Tuple, List, Iterable


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


ROUND_LABEL_RE = re.compile(r"(?P<label>\d+(?:st|nd|rd|th)\s+ROUND)" )


def iter_round_segments(text: str) -> List[Tuple[str, str]]:
    """Split the text by ROUND markers and return labelled segments."""
    matches = list(ROUND_LABEL_RE.finditer(text))
    if not matches:
        return []
    segments: List[Tuple[str, str]] = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        segments.append((match.group('label'), text[start:end]))
    return segments


def iter_2nd_round_segments(text: str) -> Iterable[str]:
    for label, segment in iter_round_segments(text):
        if label.startswith('2nd'):
            yield segment


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


def build_row(ns: Dict[str, str], metrics: Tuple[int, float, float, float]) -> Tuple[List[str], Tuple[str, ...]]:
    succ, mean_tpot, med_tpot, p99_tpot = metrics

    in_len, out_len, dataset = choose_lengths(ns)
    req_rate = ns.get('request_rate', '')
    max_conc = ns.get('max_concurrency', '')
    rand_ratio = ns.get('random_range_ratio', '')

    row = [
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
    ]
    key = (row[0], row[1], row[2], row[3], row[4], row[5])
    return row, key


def parse_segment(seg: str) -> Optional[Tuple[List[str], Tuple[int, float, float, float], str, Tuple[str, ...]]]:
    ns_line_match = re.search(r"^Namespace\(.*\)$", seg, flags=re.M)
    if not ns_line_match:
        return None
    ns = parse_namespace(ns_line_match.group(0))
    metrics_start = seg.find("============ Serving Benchmark Result ============")
    if metrics_start == -1:
        return None
    metrics = extract_metrics(seg[metrics_start:])
    if metrics is None:
        return None
    row, key = build_row(ns, metrics)
    max_conc = ns.get('max_concurrency', '') or ''
    return row, metrics, max_conc, key


def _concurrency_sort_key(conc: str) -> Tuple[int, str]:
    try:
        return (0, f"{float(conc):020.6f}")
    except (TypeError, ValueError):
        return (1, conc)


def extract_rows(text: str, best_of_rounds: bool = False) -> List[List[str]]:
    rows_by_key: Dict[Tuple[str, ...], List[Tuple[List[str], Tuple[int, float, float, float]]]] = {}
    conc_for_key: Dict[Tuple[str, ...], str] = {}
    round_segments = iter_round_segments(text)

    if round_segments:
        for label, seg in round_segments:
            parsed = parse_segment(seg)
            if parsed is None:
                continue
            row, metrics, conc, key = parsed
            if best_of_rounds:
                rows_by_key.setdefault(key, []).append((row, metrics))
                conc_for_key.setdefault(key, conc)
            else:
                if label.startswith('2nd'):
                    rows_by_key.setdefault(key, []).append((row, metrics))
                    conc_for_key.setdefault(key, conc)
        if rows_by_key:
            rows: List[List[str]] = []
            if best_of_rounds:
                for key in sorted(rows_by_key.keys(), key=lambda k: _concurrency_sort_key(conc_for_key.get(k, ''))):
                    candidates = rows_by_key[key]
                    best_row, _ = min(
                        candidates,
                        key=lambda item: (item[1][3], item[1][1], -item[1][0])
                    )
                    rows.append(best_row)
            else:
                for key in sorted(rows_by_key.keys(), key=lambda k: _concurrency_sort_key(conc_for_key.get(k, ''))):
                    candidates = rows_by_key[key]
                    for row, _ in candidates:
                        rows.append(row)
            return rows

    # Fallback when no round markers handled above
    parsed = parse_segment(text)
    if not parsed:
        return []
    row, _, _, _ = parsed
    return [row]


def main():
    ap = argparse.ArgumentParser(description="Extract benchmark metrics from logs and write CSV.")
    ap.add_argument('-i', '--inputs', nargs='+', required=True, help='Input log files')
    ap.add_argument('-o', '--output', required=True, help='Output CSV file path')
    ap.add_argument('--best-of-rounds', action='store_true',
                    help='Select the round with the lowest P99 TPOT (tie-breaking on mean TPOT) from logs that contain multiple rounds.')
    args = ap.parse_args()

    rows: List[List[str]] = []
    for path in args.inputs:
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                txt = f.read()
            file_rows = extract_rows(txt, best_of_rounds=args.best_of_rounds)
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


