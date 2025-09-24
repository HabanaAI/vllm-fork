# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Common functions shared by tune_benchmark_online and tune_benchmark_offline
"""

import os
import threading
import dateutil.parser as date_parser


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
                print(val, flush=True)
            if self.end:
                break
        self.pipeReader.close()

    def close(self):
        """Close the write end of the pipe.
        """
        os.close(self.fdWrite)
        self.end = True


def parse_log_for_warmup_time(log_file):
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


def retrieve_block_size_value(cmd_line_arg_list):
    # retrieve block size value from command line arg list. if not set, return default
    if "--block-size" in cmd_line_arg_list:
        index = cmd_line_arg_list.index("--block-size")
        return int(cmd_line_arg_list[index + 1])
    else:
        return 128  # default block size


def update_cmd_args(benchmark_cmd_list, cmd_arg, cmd_arg_value):
    if cmd_arg in benchmark_cmd_list:
        index = benchmark_cmd_list.index(cmd_arg)
        if index < len(benchmark_cmd_list) - 1:
            benchmark_cmd_list[index + 1] = cmd_arg_value
    else:
        benchmark_cmd_list.extend([cmd_arg, cmd_arg_value])
    return benchmark_cmd_list
