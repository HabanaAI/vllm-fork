import argparse
import csv
import re
from typing import List, Tuple


NAMESPACE_RE = re.compile(
    r"^Namespace\(.*?batch_size=(?P<batch_size>\d+),\s*seq_len=(?P<seq_len>\d+),\s*"
    r"num_query_heads=(?P<q_head>\d+),\s*num_kv_heads=(?P<kv_head>\d+),\s*"
    r"head_size=(?P<head_dim>\d+),\s*block_size=(?P<block_size>\d+).*?dtype='(?P<dtype>[^']+)'",
)

KERNEL_TIME_RE = re.compile(r"^Kernel running time:\s*(?P<latency>[0-9.]+)\s*us\b")


def parse_log(lines: List[str]) -> List[Tuple[int, int, int, int, int, int, int, str, float]]:
    """
    Returns list of tuples:
    (block_size, batch_size, q_len, kv_len, head_dim, q_head, kv_head, dtype, latency_us)
    """
    results: List[Tuple[int, int, int, int, int, int, int, str, float]] = []
    pending = None
    for line in lines:
        m = NAMESPACE_RE.search(line)
        if m:
            batch_size = int(m.group("batch_size"))
            seq_len = int(m.group("seq_len"))
            q_head = int(m.group("q_head"))
            kv_head = int(m.group("kv_head"))
            head_dim = int(m.group("head_dim"))
            block_size = int(m.group("block_size"))
            dtype = m.group("dtype")
            # q_len fixed to 1, kv_len = seq_len per user instruction
            pending = (block_size, batch_size, 1, seq_len, head_dim, q_head, kv_head, dtype)
            continue
        if pending is not None:
            t = KERNEL_TIME_RE.search(line)
            if t:
                latency = float(t.group("latency"))
                results.append((*pending, latency))
                pending = None
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_path", help="Path to pa_bench.log")
    parser.add_argument("--out", default="pa_bench.csv", help="Output CSV path")
    args = parser.parse_args()

    with open(args.log_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [l.rstrip("\n") for l in f]

    rows = parse_log(lines)

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "block_size",
            "batch_size",
            "q_len",
            "kv_len",
            "head_dim",
            "q_head",
            "kv_head",
            "dtype",
            "Latency(us)",
        ])
        for r in rows:
            writer.writerow(list(r))


if __name__ == "__main__":
    main()


