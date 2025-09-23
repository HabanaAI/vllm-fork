# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tune linear bucket configurations on Gaudi for best throughput and shorter warmup time with Optuna
and benchmark_serving.py. Currently tested with vllm and vllm-chat backend.
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


def parse_client_log(log_file):
    with open(log_file) as f:
        lines = f.readlines()
    throughput = 0
    for line in lines:
        if line.startswith("Total Token throughput (tok/s):"):
            pattern = r"(\d+(\.?\d+)?)\s*"
            match_throughput = re.search(pattern, line)
            if match_throughput:
                throughput = float(match_throughput.group(1))
            break
    return throughput


def server_started(server_log_file):
    with open(server_log_file) as f:
        lines = f.readlines()

    for line in reversed(lines):
        if 'Application startup complete.' in line:
            return True
    return False


def parse_server_log(log_file):
    with open(log_file) as f:
        lines = f.readlines()

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


def set_trial_value(prompt_bs_bucket_min, prompt_bs_bucket_step, prompt_bs_bucket_max,
                    prompt_seq_bucket_min, prompt_seq_bucket_step, prompt_seq_bucket_max,
                    decode_bs_bucket_min, decode_bs_bucket_step, decode_bs_bucket_max,
                    decode_block_bucket_min, decode_block_bucket_step, decode_block_bucket_max,
                    block_size, print_only=False):
    if print_only:
        # prompt related configurations
        if prompt_bs_bucket_min:
            print(f"  VLLM_PROMPT_BS_BUCKET_MIN={prompt_bs_bucket_min}")
        if prompt_bs_bucket_step:
            print("  VLLM_PROMPT_BS_BUCKET_STEP=" + str(prompt_bs_bucket_step))
        if prompt_bs_bucket_max:
            print("  VLLM_PROMPT_BS_BUCKET_MAX=" + str(prompt_bs_bucket_max))
        if prompt_seq_bucket_min:
            print("  VLLM_PROMPT_SEQ_BUCKET_MIN=" + str(prompt_seq_bucket_min))
        if prompt_seq_bucket_step:
            print("  VLLM_PROMPT_SEQ_BUCKET_STEP=" + str(prompt_seq_bucket_step))
        if prompt_seq_bucket_max:
            print("  VLLM_PROMPT_SEQ_BUCKET_MAX=" + str(prompt_seq_bucket_max))

        # decode related configurations
        if decode_bs_bucket_min:
            print("  VLLM_DECODE_BS_BUCKET_MIN=" + str(decode_bs_bucket_min))
        if decode_bs_bucket_step:
            print("  VLLM_DECODE_BS_BUCKET_STEP=" + str(decode_bs_bucket_step))
        if decode_bs_bucket_max:
            print("  VLLM_DECODE_BS_BUCKET_MAX=" + str(decode_bs_bucket_max))
        if decode_block_bucket_min:
            print("  VLLM_DECODE_BLOCK_BUCKET_MIN=" + str(decode_block_bucket_min))
        if decode_block_bucket_step:
            print("  VLLM_DECODE_BLOCK_BUCKET_STEP=" + str(decode_block_bucket_step))
        if decode_block_bucket_max:
            print("  VLLM_DECODE_BLOCK_BUCKET_MAX=" + str(decode_block_bucket_max))
        print("  Block_Size=", str(block_size))
    else:
        # prompt related configurations
        if prompt_bs_bucket_min:
            os.environ["VLLM_PROMPT_BS_BUCKET_MIN"] = str(prompt_bs_bucket_min)
        if prompt_bs_bucket_step:
            os.environ["VLLM_PROMPT_BS_BUCKET_STEP"] = str(prompt_bs_bucket_step)
        if prompt_bs_bucket_max:
            os.environ["VLLM_PROMPT_BS_BUCKET_MAX"] = str(prompt_bs_bucket_max)
        if prompt_seq_bucket_min:
            os.environ["VLLM_PROMPT_SEQ_BUCKET_MIN"] = str(prompt_seq_bucket_min)
        if prompt_seq_bucket_step:
            os.environ["VLLM_PROMPT_SEQ_BUCKET_STEP"] = str(prompt_seq_bucket_step)
        if prompt_seq_bucket_max:
            os.environ["VLLM_PROMPT_SEQ_BUCKET_MAX"] = str(prompt_seq_bucket_max)

        # decode related configurations
        if decode_bs_bucket_min:
            os.environ["VLLM_DECODE_BS_BUCKET_MIN"] = str(decode_bs_bucket_min)
        if decode_bs_bucket_step:
            os.environ["VLLM_DECODE_BS_BUCKET_STEP"] = str(decode_bs_bucket_step)
        if decode_bs_bucket_max:
            os.environ["VLLM_DECODE_BS_BUCKET_MAX"] = str(decode_bs_bucket_max)
        if decode_block_bucket_min:
            os.environ["VLLM_DECODE_BLOCK_BUCKET_MIN"] = str(decode_block_bucket_min)
        if decode_block_bucket_step:
            os.environ["VLLM_DECODE_BLOCK_BUCKET_STEP"] = str(decode_block_bucket_step)
        if decode_block_bucket_max:
            os.environ["VLLM_DECODE_BLOCK_BUCKET_MAX"] = str(decode_block_bucket_max)


def objective(trial, args):
    if args.prompt_bs_bucket_min_range:
        prompt_bs_bucket_min = trial.suggest_int('prompt_bs_bucket_min',
                                                 args.prompt_bs_bucket_min_range[0],
                                                 args.prompt_bs_bucket_min_range[1],
                                                 step=args.prompt_bs_bucket_min_range[2])
    else:
        prompt_bs_bucket_min = None
    if args.prompt_bs_bucket_step_range:
        prompt_bs_bucket_step = trial.suggest_int('prompt_bs_bucket_step',
                                                  args.prompt_bs_bucket_step_range[0],
                                                  args.prompt_bs_bucket_step_range[1],
                                                  step=args.prompt_bs_bucket_step_range[2])
    else:
        prompt_bs_bucket_step = None
    if args.prompt_bs_bucket_max_range:
        suggested_prompt_bs_max = max(prompt_bs_bucket_min, args.prompt_bs_bucket_max_range[0]) \
            if prompt_bs_bucket_min else args.prompt_bs_bucket_max_range[0]
        prompt_bs_bucket_max = trial.suggest_int('prompt_bs_bucket_max',
                                                 suggested_prompt_bs_max,
                                                 args.prompt_bs_bucket_max_range[1],
                                                 step=args.prompt_bs_bucket_max_range[2])
    else:
        prompt_bs_bucket_max = None

    if args.prompt_seq_bucket_min_range:
        prompt_seq_bucket_min = trial.suggest_int('prompt_seq_bucket_min',
                                                  args.prompt_seq_bucket_min_range[0],
                                                  args.prompt_seq_bucket_min_range[1],
                                                  step=args.prompt_seq_bucket_min_range[2])
    else:
        prompt_seq_bucket_min = None
    if args.prompt_seq_bucket_step_range:
        prompt_seq_bucket_step = trial.suggest_int('prompt_seq_bucket_step',
                                                   args.prompt_seq_bucket_step_range[0],
                                                   args.prompt_seq_bucket_step_range[1],
                                                   step=args.prompt_seq_bucket_step_range[2])
    else:
        prompt_seq_bucket_step = None
    if args.prompt_seq_bucket_max_range:
        suggested_prompt_seq_max = max(prompt_seq_bucket_min, args.prompt_seq_bucket_max_range[0]) \
            if prompt_seq_bucket_min else args.prompt_seq_bucket_max_range[0]
        prompt_seq_bucket_max = trial.suggest_int('prompt_seq_bucket_max',
                                                  suggested_prompt_seq_max,
                                                  args.prompt_seq_bucket_max_range[1],
                                                  step=args.prompt_seq_bucket_max_range[2])
    else:
        prompt_seq_bucket_max = None

    # decode related configurations
    if args.decode_bs_bucket_min_range:
        decode_bs_bucket_min = trial.suggest_int('decode_bs_bucket_min',
                                                 args.decode_bs_bucket_min_range[0],
                                                 args.decode_bs_bucket_min_range[1],
                                                 step=args.decode_bs_bucket_min_range[2])
    else:
        decode_bs_bucket_min = None
    if args.decode_bs_bucket_step_range:
        decode_bs_bucket_step = trial.suggest_int('decode_bs_bucket_step',
                                                  args.decode_bs_bucket_step_range[0],
                                                  args.decode_bs_bucket_step_range[1],
                                                  step=args.decode_bs_bucket_step_range[2])
    else:
        decode_bs_bucket_step = None
    if args.decode_bs_bucket_max_range:
        suggested_decode_bs_max = max(decode_bs_bucket_min, args.decode_bs_bucket_max_range[0]) \
            if decode_bs_bucket_min else args.decode_bs_bucket_max_range[0]
        decode_bs_bucket_max = trial.suggest_int('decode_bs_bucket_max',
                                                 suggested_decode_bs_max,
                                                 args.decode_bs_bucket_max_range[1],
                                                 step=args.decode_bs_bucket_max_range[2])
    else:
        decode_bs_bucket_max = None

    if args.decode_block_bucket_min_range:
        decode_block_bucket_min = trial.suggest_int('decode_block_bucket_min',
                                                    args.decode_block_bucket_min_range[0],
                                                    args.decode_block_bucket_min_range[1],
                                                    step=args.decode_block_bucket_min_range[2])
    else:
        decode_block_bucket_min = None
    if args.decode_block_bucket_step_range:
        decode_block_bucket_step = trial.suggest_int('decode_block_bucket_step',
                                                     args.decode_block_bucket_step_range[0],
                                                     args.decode_block_bucket_step_range[1],
                                                     step=args.decode_block_bucket_step_range[2])
    else:
        decode_block_bucket_step = None
    if args.decode_block_bucket_max_range:
        suggested_decode_block_bucket_max = max(decode_block_bucket_min, args.decode_block_bucket_max_range[0]) \
            if decode_block_bucket_min else args.decode_block_bucket_max_range[0]
        decode_block_bucket_max = trial.suggest_int('decode_block_bucket_max',
                                                    suggested_decode_block_bucket_max,
                                                    args.decode_block_bucket_max_range[1],
                                                    step=args.decode_block_bucket_max_range[2])
    else:
        decode_block_bucket_max = None

    # currently only fixed block size: 128, 256 are supported
    block_size = trial.suggest_int('block_size', 128, 256, step=128)

    set_trial_value(prompt_bs_bucket_min, prompt_bs_bucket_step, prompt_bs_bucket_max,
                    prompt_seq_bucket_min, prompt_seq_bucket_step, prompt_seq_bucket_max,
                    decode_bs_bucket_min, decode_bs_bucket_step, decode_bs_bucket_max,
                    decode_block_bucket_min, decode_block_bucket_step, decode_block_bucket_max,
                    block_size)

    vllm_server_cmd = args.vllm_server_cmd
    if not '--block-size' in vllm_server_cmd:
        vllm_server_cmd = vllm_server_cmd + " --block-size " + str(block_size)

    server_log_file = f"tuning_online_benchmark_server_{trial.number}.log"
    vllm_server_cmd = vllm_server_cmd + " 2>&1 | tee " + server_log_file
    print(vllm_server_cmd)

    benchmark_serving_cmd = args.benchmark_serving_cmd
    client_log_file = f"tuning_online_benchmark_client_{trial.number}.log"
    benchmark_serving_cmd = benchmark_serving_cmd + " 2>&1 | tee " + client_log_file
    print(benchmark_serving_cmd)

    throughput = 0
    warmup_time = float("inf")
    try:
        server_log_pipe = LogPipe()
        server_process = subprocess.Popen(vllm_server_cmd, shell=True,
                                          stdout=server_log_pipe, stderr=server_log_pipe,
                                          preexec_fn=os.setsid)  # assign a new group id
        server_process.wait(timeout=args.time_out)
        # client_process =
        # parse the server log
        warmup_time = parse_server_log(server_log_file)
    except subprocess.TimeoutExpired:
        print(f"Command {vllm_server_cmd} timed out after {args.time_out} seconds.")
        if server_started(server_log_file):
            warmup_time = parse_server_log(server_log_file)
            # start the client benchmark
            client_process = subprocess.Popen(benchmark_serving_cmd, shell=True,
                                              preexec_fn=os.setsid)
            client_process.wait()  # wait for finish
            throughput = parse_client_log(client_log_file)
    except subprocess.CalledProcessError as e:
        print(f"Command {vllm_server_cmd} failed with error {e.stderr}.")
    except Exception as e:
        print(f"Exception occurred: {e}")
    finally:
        try:
            os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)  # kill all processes in the group
            if client_process:
                os.killpg(os.getpgid(client_process.pid), signal.SIGTERM)
        except Exception:
            pass
        server_log_pipe.close()

    return throughput, warmup_time


def validate_args(args):
    # make sure benchmark client command is benchmark_serving
    if not 'benchmark_serving.py' in args.benchmark_serving_cmd:
        raise ValueError("--benchmark-serving-cmd should call benchmark_serving.py")

    # check tuning parameters range
    if args.prompt_bs_bucket_min_range and args.prompt_bs_bucket_max_range:
        if args.prompt_bs_bucket_min_range[1] > args.prompt_bs_bucket_max_range[1]:
            raise ValueError(
                f"--prompt-bs-bucket-min-range max value {args.prompt_bs_bucket_min_range[1]} is greater than "
                f"--prompt-bs-bucket-max-range max value {args.prompt_bs_bucket_max_range[1]}")
        if args.prompt_bs_bucket_min_range[0] > args.prompt_bs_bucket_max_range[0]:
            raise ValueError(
                f"--prompt-bs-bucket-min-range min value {args.prompt_bs_bucket_min_range[0]} is greater than "
                f"--prompt-bs-bucket-max-range min value {args.prompt_bs_bucket_max_range[0]}")
    if args.prompt_seq_bucket_min_range and args.prompt_seq_bucket_max_range:
        if args.prompt_seq_bucket_min_range[1] > args.prompt_seq_bucket_max_range[1]:
            raise ValueError(
                f"--prompt-seq-bucket-min-range max value {args.prompt_seq_bucket_min_range[1]} is greater than "
                f"--prompt-seq-bucket-max-range max value {args.prompt_seq_bucket_max_range[1]}")
        if args.prompt_seq_bucket_min_range[0] > args.prompt_seq_bucket_max_range[0]:
            raise ValueError(
                f"--prompt-seq-bucket-min-range min value {args.prompt_seq_bucket_min_range[0]} is greater than "
                f"--prompt-seq-bucket-max-range min value {args.prompt_seq_bucket_max_range[0]}")
    if args.decode_bs_bucket_min_range and args.decode_bs_bucket_max_range:
        if args.decode_bs_bucket_min_range[1] > args.decode_bs_bucket_max_range[1]:
            raise ValueError(
                f"--decode-bs-bucket-min-range max value {args.decode_bs_bucket_min_range[1]} is greater than "
                f"--decode-bs-bucket-max-range max value {args.decode_bs_bucket_max_range[1]}")
        if args.decode_bs_bucket_min_range[0] > args.decode_bs_bucket_max_range[0]:
            raise ValueError(
                f"--decode-bs-bucket-min-range min value {args.decode_bs_bucket_min_range[0]} is greater than "
                f"--decode-bs-bucket-max-range min value {args.decode_bs_bucket_max_range[0]}")
    if args.decode_block_bucket_min_range and args.decode_block_bucket_max_range:
        if args.decode_block_bucket_min_range[1] > args.decode_block_bucket_max_range[1]:
            raise ValueError(
                f"--decode-block-bucket-min-range max value {args.decode_block_bucket_min_range[1]} is greater than "
                f"--decode-block-bucket-max-range max value {args.decode_block_bucket_max_range[1]}")
        if args.decode_block_bucket_min_range[0] > args.decode_block_bucket_max_range[0]:
            raise ValueError(
                f"--decode-block-bucket-min-range min value {args.decode_block_bucket_min_range[0]} is greater than "
                f"--decode-block-bucket-max-range min value {args.decode_block_bucket_max_range[0]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bucket Tune")

    parser.add_argument(
        "--task-name",
        type=str,
        default="tuning_online_benchmark",
        help="Study name for Optuna. To continue previous interrupted tuning, re-use the name. "
             "To start tuning from beginning, use a different task name.",
    )
    # Add PROMPT related bucket configs
    parser.add_argument(
        "--prompt-bs-bucket-min-range",
        nargs=3,
        type=int,
        required=False,  # if not specified, use value from env var of VLLM_PROMPT_BS_BUCKET_MIN or default
        help="Tuning range for VLLM_PROMPT_BS_BUCKET_MIN in format of min max step",
    )
    parser.add_argument(
        "--prompt-bs-bucket-step-range",
        nargs=3,
        type=int,
        required=False,  # if not specified, use value from env var of VLLM_PROMPT_BS_BUCKET_MIN or default
        help="Tuning range for VLLM_PROMPT_BS_BUCKET_STEP in format of min max step",
    )
    parser.add_argument(
        "--prompt-bs-bucket-max-range",
        nargs=3,
        type=int,
        required=False,  # if not specified, then don't tune, use the default for VLLM_PROMPT_BS_BUCKET_MAX
        help="Tuning range for VLLM_PROMPT_BS_BUCKET_MAX in format of min max step",
    )

    parser.add_argument(
        "--prompt-seq-bucket-min-range",
        nargs=3,
        type=int,
        required=False,
        help="Tuning range for VLLM_PROMPT_SEQ_BUCKET_MIN in format of min max step",
    )
    parser.add_argument(
        "--prompt-seq-bucket-step-range",
        nargs=3,
        type=int,
        required=False,
        help="Tuning range for VLLM_PROMPT_SEQ_BUCKET_STEP in format of min max step. Suggest a factor of 128.",
    )
    parser.add_argument(
        "--prompt-seq-bucket-max-range",
        nargs=3,
        type=int,
        required=False,  # if not specified, then don't tune, use the default for VLLM_PROMPT_SEQ_BUCKET_MAX
        help="Tuning range for VLLM_PROMPT_SEQ_BUCKET_MAX in format of min max step",
    )

    # Add DECODE related bucket configs
    parser.add_argument(
        "--decode-bs-bucket-min-range",
        nargs=3,
        type=int,
        required=False,
        help="Tuning range for VLLM_DECODE_BS_BUCKET_MIN in format of min max step",
    )
    parser.add_argument(
        "--decode-bs-bucket-step-range",
        nargs=3,
        type=int,
        required=False,
        help="Tuning range for VLLM_DECODE_BS_BUCKET_STEP in format of min max step",
    )
    parser.add_argument(
        "--decode-bs-bucket-max-range",
        nargs=3,
        type=int,
        required=False,
        help="Tuning range for VLLM_DECODE_BS_BUCKET_MAX in format of min max step",
    )
    parser.add_argument(
        "--decode-block-bucket-min-range",
        nargs=3,
        type=int,
        required=False,
        help="Tuning range for VLLM_DECODE_BLOCK_BUCKET_MIN in format of min max step",
    )
    parser.add_argument(
        "--decode-block-bucket-step-range",
        nargs=3,
        type=int,
        required=False,
        help="Tuning range for VLLM_DECODE_BLOCK_STEP in format of min max step. Suggest a factor of 128.",
    )
    parser.add_argument(
        "--decode-block-bucket-max-range",
        nargs=3,
        type=int,
        required=False,
        help="Tuning range for VLLM_DECODE_BLOCK_BUCKET_MAX in format of min max step",
    )

    # Add optuna related configurations
    parser.add_argument(
        "--num-trials",
        type=int,
        default=15,
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

    # Add benchmark_serving related commands
    parser.add_argument(
        "--vllm-server-cmd",
        type=str,
        required=True,
        help="Command to start vllm server.",
    )
    parser.add_argument(
        "--benchmark-serving-cmd",
        type=str,
        required=True,
        help="Command to run benchmark-serving.",
    )

    args = parser.parse_args()
    validate_args(args)

    # set env vars that are not tuned
    if os.getenv("VLLM_EXPONENTIAL_BUCKETING", default=True):  # make sure we are tuning linear bucket
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
        set_trial_value(best_trial.params.get("prompt_bs_bucket_min"), best_trial.params.get("prompt_bs_bucket_step"),
                        best_trial.params.get("prompt_bs_bucket_max"), best_trial.params.get("prompt_seq_bucket_min"),
                        best_trial.params.get("prompt_seq_bucket_step"), best_trial.params.get("prompt_seq_bucket_max"),
                        best_trial.params.get("decode_bs_bucket_min"), best_trial.params.get("decode_bs_bucket_step"),
                        best_trial.params.get("decode_bs_bucket_max"), best_trial.params.get("decode_block_bucket_min"),
                        best_trial.params.get("decode_block_bucket_step"),
                        best_trial.params.get("decode_block_bucket_max"),
                        best_trial.params.get("block_size"), print_only=True)
        best_values = best_trial.values
        print("  [Throughput(tokens/s), Warmup-Time(sec)]: {}".format(best_values))

    exit(0)
