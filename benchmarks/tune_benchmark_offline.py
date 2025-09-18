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
import threading
import dateutil.parser as date_parser

import optuna
from optuna.storages import RDBStorage
from functools import partial


class LogPipe(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.daemon = False
        self.fdRead, self.fdWrite = os.pipe()
        self.pipeReader = os.fdopen(self.fdRead)
        self.start()
        self.end = False

    def fileno(self):
        """Return the write file descriptor of the pipe
        """
        return self.fdWrite

    def run(self):
        """Run the thread, logging everything.
        """
        for line in iter(self.pipeReader.readline, b""):
            val = line.strip("\n")
            if val:
                print(val)
            if self.end:
                break
        self.pipeReader.close()

    def close(self):
        """Close the write end of the pipe.
        """
        os.close(self.fdWrite)
        self.end = True


def parse_log(log_file):
    with open(log_file) as f:
        lines = f.readlines()

    throughput = 0
    warmup_time = calculate_warmup_time(lines)
    if warmup_time < float("inf"):
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
        index_start = line.find("INFO")
        index_end = line.find("[Warmup][Graph/prompt]")
        if not warmup_start and index_start != -1 and index_end != -1 and index_start < index_end:
            warmup_start = line[index_start:index_end]
            break
    for line in reversed(lines):
        index_start = line.find("INFO")
        index_end = line.find("[Warmup][Graph/decode]")
        if not warmup_end and index_start != -1 and index_end != -1 and index_start < index_end:
            warmup_end = line[index_start:index_end]
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
            return float("inf")
    else:
        return float("inf")


def set_trial_value(prompt_bs_bucket_step, prompt_bs_bucket_max, prompt_seq_step,
                    decode_bs, decode_block_step,
                    block_size, args, print_only=False):
    # calculate prompt and decode related configs
    max_model_len = args.input_len + (args.output_len * args.n)
    first_block = int((decode_bs * args.input_len) / block_size)
    last_block = int((decode_bs * max_model_len) / block_size)
    last_block = last_block + decode_block_step  # add 1 more step size as buffer

    if print_only:
        if prompt_bs_bucket_step:
            print("  VLLM_PROMPT_BS_BUCKET_STEP = " + str(prompt_bs_bucket_step))
        else:
            print("  VLLM_PROMPT_BS_BUCKET_STEP = " + str(os.getenv("VLLM_PROMPT_BS_BUCKET_STEP", 32)))  # default:32
        if prompt_bs_bucket_max:
            print("  VLLM_PROMPT_BS_BUCKET_MAX = " + str(prompt_bs_bucket_max))
        else:
            print("  VLLM_PROMPT_BS_BUCKET_MAX = " + str(os.getenv("VLLM_PROMPT_BS_BUCKET_MAX", 204)))   # default:204
        os.environ["VLLM_PROMPT_SEQ_BUCKET_MIN"] = str(args.input_len)
        if prompt_seq_step:
            print("  VLLM_PROMPT_SEQ_BUCKET_STEP = " + str(prompt_seq_step))
        else:
            print("  VLLM_PROMPT_SEQ_BUCKET_STEP = " + str(os.getenv("VLLM_PROMPT_SEQ_BUCKET_STEP", 128)))  # default:128
        print("  VLLM_PROMPT_SEQ_BUCKET_MAX = " + str(max_model_len))

        print("  VLLM_DECODE_BS_BUCKET_MIN = " + str(decode_bs))
        print("  VLLM_DECODE_BS_BUCKET_STEP = " + str(decode_bs))
        print("  VLLM_DECODE_BS_BUCKET_MAX = " + str(decode_bs))
        print("  VLLM_DECODE_BLOCK_BUCKET_MIN = " + str(first_block))
        print("  VLLM_DECODE_BLOCK_BUCKET_STEP = " + str(decode_block_step))
        print("  VLLM_DECODE_BLOCK_BUCKET_MAX = " + str(last_block))
        print("  max-num-seqs = " + str(prompt_bs_bucket_max + decode_bs))
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
        decode_block_step = None
    decode_bs = trial.suggest_int('decode_bs',
                              args.decode_bs_range[0],
                              args.decode_bs_range[1],
                              step=args.decode_bs_range[2])
    # currently only fixed block size: 128, 256 are supported
    block_size = trial.suggest_int('block_size', 128, 256, step=128)

    set_trial_value(prompt_bs_bucket_step, prompt_bs_bucket_max,
                    prompt_seq_step, decode_bs, decode_block_step, block_size, args)

    max_num_seqs = prompt_bs_bucket_max + decode_bs
    benchmark_cmd = construct_benchmark_cmd(args, max_num_seqs)
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


def update_cmd_args(benchmark_cmd, throughput_args_list, cmd_arg, cmd_arg_value):
    if cmd_arg in throughput_args_list:  # update cmd_arg value in benchmark_throughput_args
        index = throughput_args_list.index(cmd_arg)
        if index < len(throughput_args_list) - 1:
            throughput_args_list[index + 1] = cmd_arg_value
    else:
        benchmark_cmd.extend([cmd_arg, cmd_arg_value])
    return benchmark_cmd, throughput_args_list


def construct_benchmark_cmd(args, max_num_seqs):
    benchmark_cmd = ["python", "benchmark_throughput.py"]
    throughput_args_list = args.benchmark_throughput_args.split()

    if hasattr(args, "input_len"):
        benchmark_cmd, throughput_args_list = \
            update_cmd_args(benchmark_cmd, throughput_args_list, "--input-len", str(args.input_len))
    if hasattr(args, "output_len"):
        benchmark_cmd, throughput_args_list = \
            update_cmd_args(benchmark_cmd, throughput_args_list, "--output-len", str(args.output_len))
    if hasattr(args, "n"):
        benchmark_cmd, throughput_args_list = \
            update_cmd_args(benchmark_cmd, throughput_args_list, "--n", str(args.n))
    # add max_num_seqs
    benchmark_cmd, throughput_args_list = \
        update_cmd_args(benchmark_cmd, throughput_args_list, "--max-num-seqs", str(max_num_seqs))

    benchmark_cmd.extend(throughput_args_list)
    return " ".join(benchmark_cmd)


def validate_args(args):
    # args.prompt_seq_step_range max should be less than total-model-len (input_len +
    if args.prompt_seq_step_range[1] > (args.input_len + (args.output_len * args.n)):
        raise ValueError("--prompt-seq-step-range max should be less than total model context length.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bucket Tune")

    parser.add_argument(
        "--task-name",
        type=str,
        default="bucket_tuning_offline",
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
        required=True,
        help="Tuning range for VLLM_DECODE_BLOCK_STEP in format of min max step. Suggest a factor of 128.",
    )
    parser.add_argument(
        "--decode-bs-range",
        nargs=3,
        type=int,
        required=True,
        help="Tuning range for decode-bs-range in format of min max step.",
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

    # other benchmark_throughput cmd-line args
    parser.add_argument(
        "--benchmark-throughput-args",
        type=str,
        required=True,
        help="Other command-line args for benchmark_throughput.py",
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
        print("The {}-th Pareto solution was found at Trial#{}.".format(i, best_trial.number))
        print("  Params: {}".format(best_trial.params))
        set_trial_value(best_trial.params.get("prompt_bs_bucket_step"),
                        best_trial.params.get("prompt_bs_bucket_max"),
                        best_trial.params.get("prompt_seq_step"),
                        best_trial.params.get("decode_bs"),
                        best_trial.params.get("decode_block_step"),
                        best_trial.params.get("block_size"), args, print_only=True)
        best_values = best_trial.values
        print("  [Throughput(tokens/s), Warmup-Time(sec)]: {}".format(best_values))

    exit(0)

