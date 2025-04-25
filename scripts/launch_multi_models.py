# SPDX-License-Identifier: Apache-2.0
# create a python scripts
# goal: launch vllm server
# usage: python launch_multi_models.py --models
# model1 model2 model3 --force-restart --port 8000
# steps:
# 1. parse arguments
# 2. check existing pid in local pid file
# 3. if existing pid exists, check model list from
# server and compare with input model list
# 4. if model list is different, kill existing pid
# 5. launch new server

import argparse
import os
import signal
import subprocess
import time

import requests

# Constants
PID_FILE = "mm_vllm_server.pid"
SERVER_URL_TEMPLATE = "http://localhost:{port}/v1/models"


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Launch vLLM server with multiple models.")
    parser.add_argument("--models", nargs="+", help="List of models to load.")
    parser.add_argument("--force-restart",
                        action="store_true",
                        help="Force restart the server.")
    parser.add_argument("--port",
                        type=int,
                        default=8000,
                        help="Port for the vLLM server.")
    parser.add_argument("--stop-all",
                        action="store_true",
                        help="Stop all running models.")
    return parser.parse_args()


def get_running_models(port):
    """Query the running models from the server."""
    try:
        response = requests.get(SERVER_URL_TEMPLATE.format(port=port))
        if response.status_code == 200:
            ret = response.json().get("data", [])
            models = [model["id"] for model in ret]
            print("running_models", models)
            return models
        else:
            print(f"Failed to query running models. \
                    Status code: {response.status_code}")
            return []
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to server: {e}")
        return []


def kill_existing_pid(pid):
    """Kill the process with the given PID."""
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"Killed existing process with PID: {pid}")
        os.remove(PID_FILE)
    except OSError as e:
        print(f"Error killing PID {pid}: {e}")


def launch_server(models, port, log_file="mm_vllm_server.log", timeout=300):
    """Launch a new vLLM server."""
    mem_ratio = 7 // len(models) / 10
    command = [
        "python3", "-m", "vllm.entrypoints.openai.mm_api_server", "--port",
        str(port), "--device", "hpu", "--dtype", "bfloat16",
        "--gpu-memory-utilization",
        str(mem_ratio), "--use-v2-block-manager", "--max-model-len", "4096",
        "--models"
    ]
    command += models
    # Copy current environment and add custom variables
    env = os.environ.copy()
    additional_env = {
        "VLLM_CONTIGUOUS_PA": "false",
        "VLLM_SKIP_WARMUP": "true"
    }
    env.update(additional_env)  # Add custom environment variables

    try:
        # Open log file for writing
        with open(log_file, "w") as log:
            # Launch the server asynchronously (non-blocking)
            # and redirect output to the log file
            process = subprocess.Popen(command,
                                       env=env,
                                       stdout=log,
                                       stderr=log)
            print(f"Launched new server with PID: {process.pid}")

            # Optionally save the PID to a file
            with open(PID_FILE, "w") as pid_file:
                pid_file.write(str(process.pid))

        # Wait for the server to log "Started server process"
        print(
            f"Waiting for server to start. Monitoring {log_file} for output..."
        )
        server_started = False
        start_time = time.time()

        while time.time() - start_time < timeout or timeout == -1:
            with open(log_file) as log:
                log_content = log.read()
                if "Started server process" in log_content:
                    server_started = True
                    break
                else:
                    print(".", end="", flush=True)
            time.sleep(1)

        if server_started:
            print("Server started successfully! Took {:.2f} seconds.".format(
                time.time() - start_time))
        else:
            print("Server did not start within the timeout period.")

    except Exception as e:
        print(f"Failed to launch server: {e}")


def main():
    args = parse_arguments()

    # Read existing PID if available
    existing_pid = None
    if os.path.exists(PID_FILE):
        with open(PID_FILE) as f:
            try:
                existing_pid = int(f.read().strip())
            except ValueError:
                print("Invalid PID in PID file.")
    if args.stop_all:
        if existing_pid:
            kill_existing_pid(existing_pid)
        return

    # Check existing server
    if existing_pid and not args.force_restart:
        running_models = get_running_models(args.port)
        if set(running_models) == set(args.models):
            print("Server is already running with the \
                    requested models. No action required.")
            return
        else:
            print(
                "Running models do not match the requested \
                    models. Restarting server."
\
    )
            kill_existing_pid(existing_pid)
    if existing_pid and args.force_restart:
        print("Force restart requested. Killing existing server.")
        kill_existing_pid(existing_pid)

    # Launch new server
    launch_server(args.models, args.port)
    get_running_models(args.port)


if __name__ == "__main__":
    main()
