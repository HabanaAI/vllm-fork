# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tune linear bucket configurations on Gaudi for better throughput and shorter warmup time with Optuna
and benchmark_throughput.py. Currently tested with vllm and vllm-chat backend.
"""

import argparse
import os
import subprocess
import re
import dateutil.parser as date_parser
import optuna
from optuna.storages import RDBStorage
from functools import partial


def parse_log(log_file):
    with open(log_file) as f:
        lines = f.readlines()

    throughput = 0
    warmup_time = calculate_warmup_time(lines)
    if warmup_time < float('inf'):
        for line in reversed(lines):
            if line.startswith("Throughput: "):
                pattern = r"(\d+(\.?\d+)?)\s*" + re.escape("total tokens/s")
                match_throughput = re.search(pattern, line)
                if match_throughput:
                    throughput = float(match_throughput.group(1))
                break
    return throughput, warmup_time


def calculate_warmup_time(lines):
    warmup_start = None
    warmup_end = None

    for line in lines:
        index = line.find('[Warmup][Graph/prompt]')
        if not warmup_start and index != -1:
            warmup_start = line[:index]
            break
    for line in reversed(lines):
        index = line.find('[Warmup][Graph/decode]')
        if not warmup_end and index != -1:
            warmup_end = line[:index]
            break
    print("warmup_start: ", warmup_start)
    print("warmup_end: ", warmup_end)

    if warmup_start and warmup_end:
        try:
            start_time = date_parser.parse(warmup_start, fuzzy=True)
            end_time = date_parser.parse(warmup_end, fuzzy=True)
            warmup_total_sec = (end_time - start_time).total_seconds()
            return warmup_total_sec
        except ValueError as e:
            print(e)
            return float('inf')
    else:
        return float('inf')


def set_trial_value(trial, var_name, search_range, env_var):
    value = trial.suggest_int(var_name,
                              search_range[0],
                              search_range[1],
                              step=search_range[2])
    os.environ[env_var] = str(value)


def objective(trial, args):
    set_trial_value(trial, 'prompt_bs_bucket_min', args.prompt_bs_bucket_min_range,
                    'VLLM_PROMPT_BS_BUCKET_MIN')
    set_trial_value(trial, 'prompt_bs_bucket_step', args.prompt_bs_bucket_step_range,
                    'VLLM_PROMPT_BS_BUCKET_STEP')
    set_trial_value(trial, 'prompt_seq_bucket_min', args.prompt_seq_bucket_min_range,
                    'VLLM_PROMPT_SEQ_BUCKET_MIN')
    set_trial_value(trial, 'prompt_seq_bucket_step', args.prompt_seq_bucket_step_range,
                    'VLLM_PROMPT_SEQ_BUCKET_STEP')
    set_trial_value(trial, 'decode_bs_bucket_min', args.decode_bs_bucket_min_range,
                    'VLLM_DECODE_BS_BUCKET_MIN')
    set_trial_value(trial, 'decode_bs_bucket_step', args.decode_bs_bucket_step_range,
                    'VLLM_DECODE_BS_BUCKET_STEP')
    set_trial_value(trial, 'decode_block_bucket_min', args.decode_block_bucket_min_range,
                    'VLLM_DECODE_BLOCK_BUCKET_MIN')
    set_trial_value(trial, 'decode_block_bucket_step', args.decode_block_bucket_step_range,
                    'VLLM_DECODE_BLOCK_BUCKET_STEP')

    benchmark_cmd = construct_benchmark_cmd(args)
    print(benchmark_cmd)
    log_file = "bucket_tuning_log.log"
    benchmark_cmd = benchmark_cmd + " 2>&1 | tee " + log_file

    throughput = 0
    warmup_time = float('inf')
    try:
        subprocess.run(benchmark_cmd, shell=True, check=True)
        throughput, warmup_time = parse_log(log_file)
    except Exception as e:
        print(f"Exception occurred: {e}")

    return throughput, warmup_time


def construct_benchmark_cmd(args):
    benchmark_cmd = ["python", "benchmark_throughput.py"]
    if hasattr(args, "input_len"):
        benchmark_cmd.extend(["--input-len", str(args.input_len)])
    if hasattr(args, "output_len"):
        benchmark_cmd.extend(["--output-len", str(args.output_len)])
    if hasattr(args, "num_prompts"):
        benchmark_cmd.extend(["--num-prompts", str(args.num_prompts)])
    if hasattr(args, "n"):
        benchmark_cmd.extend(["--n", str(args.n)])
    benchmark_cmd.append(args.benchmark_throughput_args)
    return ' '.join(benchmark_cmd)


def validate_args(args):
    # args.prompt_bs_bucket_min_range max should be less than num-prompts
    if args.prompt_bs_bucket_min_range[1] > args.num_prompts:
        raise ValueError("--prompt_bs_bucket_min_range max should be less than num-prompts")

    # print(args.prompt_bs_bucket_step_range)
    # args.prompt_seq_bucket_min_range max should be less than input-len
    if args.prompt_seq_bucket_min_range[1] > args.input_len:
        raise ValueError("--prompt_seq_bucket_min_range max should be less than input-len")

    # print(args.prompt_seq_bucket_step_range)
    # args.decode_bs_bucket_min_range max should be less than num-prompts * n
    if args.decode_bs_bucket_min_range[1] > args.num_prompts*args.n:
        raise ValueError("--decode_bs_bucket_min_range max should be less than num_prompts*n")

    # print(args.decode_bs_bucket_step_range)
    # args.decode_block_bucket_min_range max should be less than output-len
    if args.decode_block_bucket_min_range[1] > args.output_len:
        raise ValueError("--decode_block_bucket_min_range max should be less than output-len")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bucket Tune")

    parser.add_argument(
        "--task-name",
        type=str,
        default="bucket_tuning",
        help="Study name for Optuna. To continue previous interrupted tuning, re-use the name. "
             "To start tuning from beginning, use a different task name.",
    )
    # Add PROMPT related bucket configs
    parser.add_argument(
        "--prompt-bs-bucket-min-range",
        nargs=3,
        type=int,
        required=True,
        help="Tuning range for VLLM_PROMPT_BS_BUCKET_MIN in format of min max step",
    )
    parser.add_argument(
        "--prompt-bs-bucket-step-range",
        nargs=3,
        type=int,
        required=True,
        help="Tuning range for VLLM_PROMPT_BS_BUCKET_STEP in format of min max step",
    )
    # Note: no need to tune VLLM_PROMPT_BS_BUCKET_MAX since it should be num-prompts for static benchmark

    parser.add_argument(
        "--prompt-seq-bucket-min-range",
        nargs=3,
        type=int,
        required=True,
        help="Tuning range for VLLM_PROMPT_SEQ_BUCKET_MIN in format of min max step",
    )
    parser.add_argument(
        "--prompt-seq-bucket-step-range",
        nargs=3,
        type=int,
        required=True,
        help="Tuning range for VLLM_PROMPT_SEQ_BUCKET_STEP in format of min max step",
    )
    # Note: no need to tune VLLM_PROMPT_SEQ_BUCKET_MAX since it should be input-len for static benchmark

    # Add DECODE related bucket configs
    parser.add_argument(
        "--decode-bs-bucket-min-range",
        nargs=3,
        type=int,
        required=True,
        help="Tuning range for VLLM_DECODE_BS_BUCKET_MIN in format of min max step",
    )
    parser.add_argument(
        "--decode-bs-bucket-step-range",
        nargs=3,
        type=int,
        required=True,
        help="Tuning range for VLLM_DECODE_BS_BUCKET_STEP in format of min max step",
    )
    # Note: no need to tune VLLM_DECODE_BS_BUCKET_MAX since it should be:
    # num-prompts*n (num of seqs generated for each prompt) for static benchmark

    parser.add_argument(
        "--decode-block-bucket-min-range",
        nargs=3,
        type=int,
        required=True,
        help="Tuning range for VLLM_DECODE_BLOCK_BUCKET_MIN in format of min max step",
    )
    parser.add_argument(
        "--decode-block-bucket-step-range",
        nargs=3,
        type=int,
        required=True,
        help="Tuning range for VLLM_DECODE_BLOCK_STEP in format of min max step",
    )
    # Note: no need to tune VLLM_DECODE_BLOCK_BUCKET_MAX since it should be output-len for static benchmark

    # Following args are for benchmark_throughput.py command-line args
    parser.add_argument(
        "--input-len",
        type=int,
        default=1024,
        help="Input prompt length for each request",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=1024,
        help="Output length for each request. Overrides the output length from the dataset.",
    )
    parser.add_argument(
        "--n", type=int, default=1, help="Number of generated sequences per prompt."
    )
    parser.add_argument(
        "--num-prompts", type=int, default=1000, help="Number of prompts to process."
    )
    # other benchmark_throughput cmd-line args
    parser.add_argument(
        "--benchmark-throughput-args",
        type=str,
        required=True,
        help="Other command-line args for benchmark_throughput.py",
    )

    args = parser.parse_args()
    validate_args(args)

    # set env vars that are not tuned
    if os.getenv("VLLM_EXPONENTIAL_BUCKETING", default=True):  # make sure we are tuning linear bucket
        os.environ["VLLM_EXPONENTIAL_BUCKETING"] = "False"
    os.environ["VLLM_PROMPT_BS_BUCKET_MAX"] = str(args.num_prompts)
    os.environ["VLLM_PROMPT_SEQ_BUCKET_MAX"] = str(args.input_len)
    os.environ["VLLM_DECODE_BS_BUCKET_MAX"] = str(args.num_prompts * args.n)
    os.environ["VLLM_DECODE_BLOCK_BUCKET_MAX"] = str(args.output_len)

    # start optuna study
    study_name = args.task_name
    storage_name = f"sqlite:////tmp/{study_name}.db"
    storage = RDBStorage(url=storage_name,
                         engine_kwargs={
                             "pool_size": 100,
                             "connect_args": {"timeout": 30}})  # Increase timeout to 30 seconds
    study = optuna.create_study(study_name=study_name,
                                storage=storage,
                                load_if_exists=True,
                                directions=["maximize", "minimize"])  # max throughput, min warmup time

    objective_with_param = partial(objective, args=args)
    study.optimize(objective_with_param, n_trials=10)

    print("Number of finished trials: ", len(study.trials))

    for i, best_trial in enumerate(study.best_trials):
        print("The {}-th Pareto solution was found at Trial#{}.".format(i, best_trial.number))
        print("  Params: {}".format(best_trial.params))
        best_values = best_trial.values
        print("  [Throughput(tokens/s), Warmup-Time(sec)]: {}".format(best_values))

