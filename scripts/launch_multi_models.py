# SPDX-License-Identifier: Apache-2.0
# Enhanced launch_multi_models.py with configurable parameters
# New features:
# 1. Configurable max-model-len parameter
# 2. Customizable environment variables via config file or command line
# 3. Environment variable presets (default/performance)
# 4. Better error handling and logging

import argparse
import os
import signal
import subprocess
import time
import json
import sys
from typing import Dict, Any, Optional

import requests

# Constants
PID_FILE = "mm_vllm_server.pid"
SERVER_URL_TEMPLATE = "http://localhost:{port}/v1/models"
LOG_FILE = "mm_vllm_server.log"

# Environment variable presets
DEFAULT_ENV_VARS = {
    "VLLM_CONTIGUOUS_PA": "false",
    "VLLM_SKIP_WARMUP": "true",
    "PT_HPU_LAZY_MODE": "1"
}

PERFORMANCE_ENV_VARS = {
    "VLLM_CONTIGUOUS_PA": "false",
    "VLLM_SKIP_WARMUP": "false",
    "VLLM_PROMPT_BS_BUCKET_MIN": "1",
    "VLLM_PROMPT_BS_BUCKET_STEP": "1",
    "VLLM_PROMPT_BS_BUCKET_MAX": "8",
    "VLLM_PROMPT_SEQ_BUCKET_MIN": "1",
    "VLLM_PROMPT_SEQ_BUCKET_STEP": "512",
    "VLLM_PROMPT_SEQ_BUCKET_MAX": "4096",
    "VLLM_DECODE_BS_BUCKET_MIN": "1",
    "VLLM_DECODE_BS_BUCKET_STEP": "1",
    "VLLM_DECODE_BS_BUCKET_MAX": "2",
    "VLLM_DECODE_BLOCK_BUCKET_MIN": "1",
    "VLLM_DECODE_BLOCK_BUCKET_STEP": "1",
    "VLLM_DECODE_BLOCK_BUCKET_MAX": "2",
    "PT_HPU_LAZY_MODE": "1"
}


def parse_arguments():
    """Parse command-line arguments with enhanced options."""
    parser = argparse.ArgumentParser(
        description="Enhanced vLLM Multi-Model Server Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  python launch_multi_models_new.py --models model1 model2

  # With custom max model length
  python launch_multi_models_new.py --models model1 model2 --max-model-len 4096

  # With performance tuning preset
  python launch_multi_models_new.py --models model1 model2 --env-preset performance

  # With custom environment config file
  python launch_multi_models_new.py --models model1 model2 --env-config custom_env.json

  # With individual environment variables
  python launch_multi_models_new.py --models model1 model2 --env VLLM_SKIP_WARMUP=false --env PT_HPU_LAZY_MODE=1
        """)

    parser.add_argument("--models", nargs="+", required=True,
                        help="List of models to load (required)")
    parser.add_argument("--force-restart", action="store_true",
                        help="Force restart the server")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port for the vLLM server (default: 8000)")
    parser.add_argument("--stop-all", action="store_true",
                        help="Stop all running models")
    parser.add_argument("--max-model-len", type=int, default=512,
                        help="Maximum model length (default: 512)")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float32"],
                        help="Data type (default: bfloat16)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.3,
                        help="GPU memory utilization (0.0-1.0), auto-calculated if not specified")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Server startup timeout in seconds (default: 300)")
    parser.add_argument("--log-file", type=str, default=LOG_FILE,
                        help=f"Log file path (default: {LOG_FILE})")

    # Environment variable options
    env_group = parser.add_mutually_exclusive_group()
    env_group.add_argument("--env-preset", type=str,
                          choices=["default", "performance"],
                          default="default",
                          help="Environment variable preset (default: default)")
    env_group.add_argument("--env-config", type=str,
                          help="Path to JSON file containing environment variables")

    parser.add_argument("--env", action="append", default=[],
                        help="Set individual environment variable (can be used multiple times)")

    parser.add_argument("--list-env-presets", action="store_true",
                        help="List available environment variable presets")

    return parser.parse_args()


def list_env_presets():
    """List available environment variable presets."""
    print("Available environment variable presets:")
    print("\n1. default (conservative settings):")
    for key, value in DEFAULT_ENV_VARS.items():
        print(f"   {key}={value}")

    print("\n2. performance (optimized for throughput):")
    for key, value in PERFORMANCE_ENV_VARS.items():
        print(f"   {key}={value}")
    print("\nNote: performance preset includes bucket configuration for optimal throughput")


def load_env_config(config_path: str) -> Dict[str, str]:
    """Load environment variables from JSON config file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            if not isinstance(config, dict):
                raise Value
            return {str(k): str(v) for k, v in config.items()}
    except FileNotFoundError:
        print(f"Error: Environment config file '{config_path}' not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in config file '{config_path}': {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading config file '{config_path}': {e}")
        sys.exit(1)


def parse_env_vars(env_list: list) -> Dict[str, str]:
    """Parse individual environment variable settings."""
    env_vars = {}
    for env_str in env_list:
        if '=' not in env_str:
            print(f"Error: Invalid environment variable format '{env_str}'. Use KEY=value format")
            sys.exit(1)
        key, value = env_str.split('=', 1)
        env_vars[key.strip()] = value.strip()
    return env_vars


def get_environment_variables(args) -> Dict[str, str]:
    """Get environment variables based on arguments."""
    if args.list_env_presets:
        list_env_presets()
        sys.exit(0)

    # Start with base environment
    env_vars = os.environ.copy()

    # Apply preset or config file
    if args.env_config:
        preset_vars = load_env_config(args.env_config)
        print(f"Loaded environment variables from {args.env_config}")
    elif args.env_preset == "performance":
        preset_vars = PERFORMANCE_ENV_VARS.copy()
        print("Using performance environment preset")
    else:
        preset_vars = DEFAULT_ENV_VARS.copy()
        print("Using default environment preset")

    # Apply individual environment variables (highest priority)
    individual_vars = parse_env_vars(args.env)
    preset_vars.update(individual_vars)

    # Merge with existing environment
    env_vars.update(preset_vars)

    # Print applied environment variables for debugging
    print("Applied environment variables:")
    for key, value in preset_vars.items():
        print(f"  {key}={value}")

    return env_vars


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
            print(f"Failed to query running models. Status code: {response.status_code}")
            return []
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to server: {e}")
        return []


def kill_existing_pid(pid):
    """Kill the process with the given PID."""
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"Killed existing process with PID: {pid}")
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)
    except OSError as e:
        print(f"Error killing PID {pid}: {e}")


def launch_server(models, port, max_model_len, device, dtype, gpu_memory_utilization,
                  env_vars, log_file, timeout):
    """Launch a new vLLM server with enhanced configuration."""

    # Calculate memory ratio if not provided
    if gpu_memory_utilization is None:
        gpu_memory_utilization = 7 // len(models) / 10

    print(f"Launching server with configuration:")
    print(f"  Models: {models}")
    print(f"  Max model length: {max_model_len}")
    print(f"  Device: {device}")
    print(f"  Data type: {dtype}")
    print(f"  GPU memory utilization: {gpu_memory_utilization}")
    print(f"  Port: {port}")

    command = [
        "python3", "-m", "vllm.entrypoints.openai.mm_api_server",
        "--port", str(port),
        "--device", device,
        "--dtype", dtype,
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--use-v2-block-manager",
        "--max-model-len", str(max_model_len),
        "--models"
    ]
    command += models

    try:
        # Open log file for writing
        with open(log_file, "w") as log:
            # Launch the server asynchronously (non-blocking)
            process = subprocess.Popen(command,
                                       env=env_vars,
                                       stdout=log,
                                       stderr=log)
            print(f"Launched new server with PID: {process.pid}")

            # Save the PID to a file
            with open(PID_FILE, "w") as pid_file:
                pid_file.write(str(process.pid))

        # Wait for the server to log "Started server process"
        print(f"Waiting for server to start. Monitoring {log_file} for output...")
        server_started = False
        start_time = time.time()

        while time.time() - start_time < timeout:
            if os.path.exists(log_file):
                with open(log_file) as log:
                    log_content = log.read()
                    if "Application startup complete" in log_content:
                        server_started = True
                        break
            print(".", end="", flush=True)
            time.sleep(1)

        if server_started:
            elapsed_time = time.time() - start_time
            print(f"\nServer started successfully! Took {elapsed_time:.2f} seconds.")
            return True
        else:
            print(f"\nServer did not start within the timeout period ({timeout}s).")
            return False

    except Exception as e:
        print(f"Failed to launch server: {e}")
        return False


def main():
    args = parse_arguments()

    # Get environment variables
    env_vars = get_environment_variables(args)

    # Handle stop-all command
    if args.stop_all:
        if os.path.exists(PID_FILE):
            with open(PID_FILE) as f:
                try:
                    existing_pid = int(f.read().strip())
                    kill_existing_pid(existing_pid)
                except ValueError:
                    print("Invalid PID in PID file.")
        else:
            print("No running server found.")
        return

    # Read existing PID if available
    existing_pid = None
    if os.path.exists(PID_FILE):
        with open(PID_FILE) as f:
            try:
                existing_pid = int(f.read().strip())
            except ValueError:
                print("Invalid PID in PID file.")

    # Check existing server
    if existing_pid and not args.force_restart:
        running_models = get_running_models(args.port)
        if set(running_models) == set(args.models):
            print("Server is already running with the requested models. No action required.")
            return
        else:
            print("Running models do not match the requested models. Restarting server.")
            kill_existing_pid(existing_pid)

    if existing_pid and args.force_restart:
        print("Force restart requested. Killing existing server.")
        kill_existing_pid(existing_pid)

    # Launch new server
    success = launch_server(
        models=args.models,
        port=args.port,
        max_model_len=args.max_model_len,
        device="hpu",
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        env_vars=env_vars,
        log_file=args.log_file,
        timeout=args.timeout
    )

    if success:
        get_running_models(args.port)
    else:
        print("Server launch failed. Check the log file for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
