import re

def parse_benchmark_report(report: str):
    result = {}

    # Regex pattern to extract key-value pairs
    pattern = r'([A-Za-z0-9\s\(\)\-/]+):\s+([0-9.]+)'

    matches = re.findall(pattern, report)
    for key, value in matches:
        if '.' in value:
            value = float(value)
        else:
            value = int(value)
        result[key.strip()] = value

    return result

def create_csv_line(data: dict):
    # Select specific fields
    fields = [
        'Median E2EL (ms)',
        'Median TTFT (ms)',
        'Median TPOT (ms)',
        'Median ITL (ms)',
        'Total Token throughput (tok/s)',
        'Output token throughput (tok/s)'
    ]
    values = [str(data.get(field, '')) for field in fields]
    return ','.join(values)

def main(args):
    input_file = args.input  # Change this to your filename

    # Read the file
    with open(input_file, 'r') as f:
        report = f.read()

    # Parse and generate CSV
    parsed = parse_benchmark_report(report)
    csv_line = create_csv_line(parsed)
    
    print(csv_line)

if __name__ == '__main__':
    import argparse
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument('--input', type=str, default='benchmark_report.txt')
    args = arg_parse.parse_args()
    main(args)