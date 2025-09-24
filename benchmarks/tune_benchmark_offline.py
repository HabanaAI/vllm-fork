# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tune linear bucket configurations of VLLM_PROMPT_BS_BUCKET_STEP, VLLM_PROMPT_SEQ_BUCKET_STEP,
VLLM_DECODE_BS_BUCKET_STEP, VLLM_DECODE_BLOCK_BUCKET_STEP and decode batch size for
best throughput and shorter warmup time with Optuna and benchmark_throughput.py.
Currently tested with vllm and vllm-chat backend.
"""

import argparse
import os
import signal
import subprocess
import re
import dateutil.parser as date_parser

import optuna
from optuna.storages import RDBStorage
from functools import partial

from tune_benchmark_common import LogPipe
from tune_benchmark_common import retrieve_block_size_value
from tune_benchmark_common import update_cmd_args
from tune_benchmark_common import parse_log_for_warmup_time


def parse_log(log_file):
    throughput = 0
    warmup_time = parse_log_for_warmup_time(log_file)
    if warmup_time < float("inf"):
        with open(log_file) as f:
            lines = f.readlines()
        for line in reversed(lines):
            if line.startswith("Throughput: "):
                pattern = r"(\d+(\.?\d+)?)\s*" + re.escape("total tokens/s")
                match_throughput = re.search(pattern, line)
                if match_throughput:
                    throughput = float(match_throughput.group(1))
                break
    return throughput, warmup_time


def set_trial_value(prompt_bs_bucket_step, prompt_bs_bucket_max, prompt_seq_step,
                    decode_bs, decode_block_step,
                    block_size, max_num_seqs, args, print_only=False):
    # calculate prompt and decode related configs
    max_model_len = args.input_len + (args.output_len * args.n)
    first_block = int((decode_bs * args.input_len) / block_size)
    last_block = int((decode_bs * max_model_len) / block_size)
    last_block = last_block + decode_block_step  # add 1 more step size as buffer

    if print_only:
        # we don't tune VLLM_PROMPT_BS_BUCKET_MIN, but print out the value here
        print("\t VLLM_PROMPT_BS_BUCKET_MIN=" + str(os.getenv("VLLM_PROMPT_BS_BUCKET_MIN", 1)))  # default:1
        if prompt_bs_bucket_step:
            print(f"\t VLLM_PROMPT_BS_BUCKET_STEP={prompt_bs_bucket_step}")
        else:
            print("\t VLLM_PROMPT_BS_BUCKET_STEP=" + str(os.getenv("VLLM_PROMPT_BS_BUCKET_STEP", 32)))  # default:32
        if prompt_bs_bucket_max:
            print(f"\t VLLM_PROMPT_BS_BUCKET_MAX={prompt_bs_bucket_max}")
        else:
            print("\t VLLM_PROMPT_BS_BUCKET_MAX=" + str(
                os.getenv("VLLM_PROMPT_BS_BUCKET_MAX", max_num_seqs)))  # default:max_num_seqs
        print(f"\t VLLM_PROMPT_SEQ_BUCKET_MIN={args.input_len}")
        if prompt_seq_step:
            print(f"\t VLLM_PROMPT_SEQ_BUCKET_STEP={prompt_seq_step}")
        else:
            print(
                "\t VLLM_PROMPT_SEQ_BUCKET_STEP=" + str(
                    os.getenv("VLLM_PROMPT_SEQ_BUCKET_STEP", block_size)))  # default: block_size
        print(f"\t VLLM_PROMPT_SEQ_BUCKET_MAX={max_model_len}")

        print(f"\t VLLM_DECODE_BS_BUCKET_MIN={decode_bs}")
        print(f"\t VLLM_DECODE_BS_BUCKET_STEP={decode_bs}")
        print(f"\t VLLM_DECODE_BS_BUCKET_MAX={decode_bs}")
        print(f"\t VLLM_DECODE_BLOCK_BUCKET_MIN={first_block}")
        print(f"\t VLLM_DECODE_BLOCK_BUCKET_STEP={decode_block_step}")
        print(f"\t VLLM_DECODE_BLOCK_BUCKET_MAX={last_block}")
        print(f"\t --max-num-seqs {max_num_seqs}")
    else:
        if prompt_bs_bucket_step:
            os.environ["VLLM_PROMPT_BS_BUCKET_STEP"] = str(prompt_bs_bucket_step)
        if prompt_bs_bucket_max:
            os.environ["VLLM_PROMPT_BS_BUCKET_MAX"] = str(prompt_bs_bucket_max)
        os.environ["VLLM_PROMPT_SEQ_BUCKET_MIN"] = str(args.input_len)
        if prompt_seq_step:
            os.environ["VLLM_PROMPT_SEQ_BUCKET_STEP"] = str(prompt_seq_step)
        os.environ["VLLM_PROMPT_SEQ_BUCKET_MAX"] = str(max_model_len)

        os.environ["VLLM_DECODE_BS_BUCKET_MIN"] = str(decode_bs)
        os.environ["VLLM_DECODE_BS_BUCKET_STEP"] = str(decode_bs)
        os.environ["VLLM_DECODE_BS_BUCKET_MAX"] = str(decode_bs)
        os.environ["VLLM_DECODE_BLOCK_BUCKET_MIN"] = str(first_block)
        if decode_block_step:
            os.environ["VLLM_DECODE_BLOCK_BUCKET_STEP"] = str(decode_block_step)
        os.environ["VLLM_DECODE_BLOCK_BUCKET_MAX"] = str(last_block)


def objective(trial, args):
    benchmark_cmd_list = args.benchmark_throughput_cmd.split()
    if args.tune_block_size:
        block_size = trial.suggest_int('block_size', 128, 256, step=128)
    else:
        block_size = retrieve_block_size_value(benchmark_cmd_list)
        trial.set_user_attr("block_size", block_size)

    if args.prompt_bs_bucket_step_range:
        prompt_bs_bucket_step = trial.suggest_int('prompt_bs_bucket_step',
                                                  args.prompt_bs_bucket_step_range[0],
                                                  args.prompt_bs_bucket_step_range[1],
                                                  step=args.prompt_bs_bucket_step_range[2])
    else:
        prompt_bs_bucket_step = None
    if args.prompt_bs_bucket_max_range:
        prompt_bs_bucket_max = trial.suggest_int('prompt_bs_bucket_max',
                                                 args.prompt_bs_bucket_max_range[0],
                                                 args.prompt_bs_bucket_max_range[1],
                                                 step=args.prompt_bs_bucket_max_range[2])
    else:
        prompt_bs_bucket_max = None
    if args.prompt_seq_step_range:
        prompt_seq_step = trial.suggest_int('prompt_seq_step',
                                            args.prompt_seq_step_range[0],
                                            args.prompt_seq_step_range[1],
                                            step=args.prompt_seq_step_range[2])
    else:
        prompt_seq_step = None
    if args.decode_block_bucket_step_range:
        decode_block_step = trial.suggest_int('decode_block_step',
                                              args.decode_block_bucket_step_range[0],
                                              args.decode_block_bucket_step_range[1],
                                              step=args.decode_block_bucket_step_range[2])
    else:
        decode_block_step = int(os.getenv("VLLM_DECODE_BLOCK_BUCKET_STEP", block_size))
        trial.set_user_attr("decode_block_step", decode_block_step)
    decode_bs = trial.suggest_int('decode_bs',
                                  args.decode_bs_range[0],
                                  args.decode_bs_range[1],
                                  step=args.decode_bs_range[2])
    max_num_seqs = max(prompt_bs_bucket_max, decode_bs) if prompt_bs_bucket_max else decode_bs
    trial.set_user_attr("max_num_seqs", max_num_seqs)
    set_trial_value(prompt_bs_bucket_step, prompt_bs_bucket_max,
                    prompt_seq_step, decode_bs, decode_block_step, block_size, max_num_seqs, args)

    benchmark_cmd = construct_benchmark_cmd(args, benchmark_cmd_list, max_num_seqs, block_size)
    print(benchmark_cmd)
    log_file = f"tuning_offline_benchmark_{trial.number}.log"
    benchmark_cmd = benchmark_cmd + " 2>&1 | tee " + log_file

    throughput = 0
    warmup_time = float("inf")
    try:
        logpipe = LogPipe()
        process = subprocess.Popen(benchmark_cmd, shell=True,
                                   stdout=logpipe, stderr=logpipe,
                                   preexec_fn=os.setsid)  # assign a new group id
        process.wait(timeout=args.time_out)
        throughput, warmup_time = parse_log(log_file)
    except subprocess.TimeoutExpired:
        print(f"Command {benchmark_cmd} timed out after {args.time_out} seconds.")
    except subprocess.CalledProcessError as e:
        print(f"Command {benchmark_cmd} failed with error {e.stderr}.")
    except Exception as e:
        print(f"Exception occurred: {e}")
    finally:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)  # kill all processes in the group
        except Exception:
            pass
        logpipe.close()

    return throughput, warmup_time


def construct_benchmark_cmd(args, benchmark_cmd_list, max_num_seqs, block_size):
    if hasattr(args, "input_len"):
        benchmark_cmd_list = \
            update_cmd_args(benchmark_cmd_list, "--input-len", str(args.input_len))
    if hasattr(args, "output_len"):
        benchmark_cmd_list = \
            update_cmd_args(benchmark_cmd_list, "--output-len", str(args.output_len))
    if hasattr(args, "n"):
        benchmark_cmd_list = \
            update_cmd_args(benchmark_cmd_list, "--n", str(args.n))
    # add max_num_seqs
    benchmark_cmd_list = \
        update_cmd_args(benchmark_cmd_list, "--max-num-seqs", str(max_num_seqs))
    # add block_size
    benchmark_cmd_list = \
        update_cmd_args(benchmark_cmd_list, "--block-size", str(block_size))

    return " ".join(benchmark_cmd_list)


def validate_args(args):
    # args.prompt_seq_step_range max should be less than total-model-len
    if args.prompt_seq_step_range and \
            args.prompt_seq_step_range[1] > (args.input_len + (args.output_len * args.n)):
        raise ValueError("--prompt-seq-step-range max should be less than total model context length.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune Offline Benchmark")

    parser.add_argument(
        "--task-name",
        type=str,
        default="tune_benchmark_offline",
        help="Study name for Optuna. To continue previous interrupted tuning, re-use the name. "
             "To start tuning from beginning, use a different task name.",
    )
    parser.add_argument(
        "--prompt-bs-bucket-step-range",
        nargs=3,
        type=int,
        required=False,  # don't tune if not specified. use the default or env value set for VLLM_PROMPT_BS_BUCKET_STEP
        help="Tuning range for VLLM_PROMPT_BS_BUCKET_STEP in format of min max step",
    )
    parser.add_argument(
        "--prompt-bs-bucket-max-range",
        nargs=3,
        type=int,
        required=False,  # don't tune if not specified. use the default or env value
        help="Tuning range for VLLM_PROMPT_BS_BUCKET_MAX in format of min max step. "
    )
    parser.add_argument(
        "--prompt-seq-step-range",
        nargs=3,
        type=int,
        required=False,  # don't tune if not specified. use the default or env value
        help="Tuning range for VLLM_PROMPT_SEQ_BUCKET_STEP in format of min max step. Suggest a factor of 128.",
    )
    parser.add_argument(
        "--decode-block-bucket-step-range",
        nargs=3,
        type=int,
        required=False,
        help="Tuning range for VLLM_DECODE_BLOCK_BUCKET_STEP in format of min max step. Suggest a factor of 128.",
    )
    parser.add_argument(
        "--decode-bs-range",
        nargs=3,
        type=int,
        required=True,
        help="Tuning range for decode-bs-range in format of min max step.",
    )
    parser.add_argument(
        "--tune-block-size",
        action="store_true",
        help="Whether to tune block size.",
    )

    parser.add_argument(
        "--num-trials",
        type=int,
        default=15,  # default 15
        help="Number of trials Optuna runs to tune.",
    )
    parser.add_argument(
        "--time-out",
        type=int,
        default=1200,  # default 20 mins
        help="Terminate the running process after time-out in case the process hangs.",
    )
    parser.add_argument(
        "--database-path",
        type=str,
        default="/tmp/",  # default location is /tmp/
        help="Tuning results will be stored in sqlite database in this location.",
    )

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

    # benchmark_throughput cmd
    parser.add_argument(
        "--benchmark-throughput-cmd",
        type=str,
        required=True,
        help="Full benchmark throughput command starting with 'python benchmark_throughput.py'",
    )

    args = parser.parse_args()
    validate_args(args=args)

    # make sure we are tuning linear bucket
    if os.getenv("VLLM_EXPONENTIAL_BUCKETING", default=True):
        os.environ["VLLM_EXPONENTIAL_BUCKETING"] = "False"

    # start optuna study
    study_name = args.task_name
    storage_name = f"sqlite:///{args.database_path}/{study_name}.db"
    storage = RDBStorage(url=storage_name,
                         engine_kwargs={
                             "pool_size": 100,
                             "connect_args": {"timeout": 30}})  # Increase timeout to 30 seconds
    study = optuna.create_study(study_name=study_name,
                                storage=storage,
                                load_if_exists=True,
                                directions=["maximize", "minimize"])  # max throughput, min warmup time

    objective_with_param = partial(objective, args=args)
    study.optimize(objective_with_param, n_trials=args.num_trials)

    print("=" * 30)
    print("Number of finished trials: ", len(study.trials))

    for i, best_trial in enumerate(study.best_trials):
        print(f"The {i}-th Pareto solution was found at Trial# {best_trial.number}")
        print(f"\t Params: {best_trial.params}, {best_trial.user_attrs}")
        set_trial_value(best_trial.params.get("prompt_bs_bucket_step"),
                        best_trial.params.get("prompt_bs_bucket_max"),
                        best_trial.params.get("prompt_seq_step"),
                        best_trial.params.get("decode_bs"),
                        best_trial.params.get("decode_block_step", best_trial.user_attrs.get("decode_block_step")),
                        best_trial.params.get("block_size", best_trial.user_attrs.get("block_size")),
                        best_trial.params.get("max_num_seqs", best_trial.user_attrs.get("max_num_seqs")),
                        args, print_only=True)
        best_values = best_trial.values
        print(f"\t [Throughput(tokens/s), Warmup-Time(sec)]: {best_values}")

    exit(0)
