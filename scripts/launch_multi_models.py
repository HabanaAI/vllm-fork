# SPDX-License-Identifier: Apache-2.0
# Multi-Model vLLM Server Launcher, focusing on embedding and rerank use case
# ENV config are tuned for those models.
#
# NEW FEATURE: Per-Model Configuration Support
# =============================================
# This launcher now supports individual configuration parameters for each model.
# You can specify different vLLM parameters (max_model_len, dtype, \
# gpu_memory_utilization, etc.)
# for each model independently.
#
# Basic Usage:
#    python launch_multi_models.py --models model1 model2
#    python launch_multi_models.py --models model1 model2 --max-model-len 4096
#    python launch_multi_models.py --models model1 model2 \
#                                  --env-preset performance
#
# Per-Model Configuration Usage:
#    # Scenario 1: All models use same global configuration (default behavior)
#    python launch_multi_models.py --models model1 model2 --max-model-len 1024
#
#    # Scenario 2: Two models use global config, one model has specific config
#    python launch_multi_models.py --models model1 model2 --max-model-len 1024 \
#        --model-param model2:max_model_len=2048 \
#        --model-param model2:dtype=float32
#
#    # Scenario 3: Each model has its own specific configuration
#    python launch_multi_models.py --models model1 model2 --max-model-len 512 \
#        --model-param model1:max_model_len=1024 \
#        --model-param model1:enforce_eager=true \
#        --model-param model2:max_model_len=2048 \
#        --model-param model2:dtype=float32 \
#        --model-param model2:gpu_memory_utilization=0.6
#
# Supported Per-Model Parameters:
#    - max_model_len: Maximum model length (integer)
#    - dtype: Data type (string: bfloat16, float16, float32)
#    - gpu_memory_utilization: GPU memory utilization (float: 0.0-1.0)
#    - tensor_parallel_size: Tensor parallel size (integer)
#    - pipeline_parallel_size: Pipeline parallel size (integer)
#    - enforce_eager: Enforce eager mode (boolean: true/false)
#    - trust_remote_code: Trust remote code (boolean: true/false)
#    - disable_log_stats: Disable log stats (boolean: true/false)
#    - max_num_batched_tokens: Max batched tokens (integer)
#    - max_num_seqs: Max sequences (integer)
#    - And many more vLLM parameters...
#
# Parameter Format:
#    --model-param model_name:param_name=value
#
# Examples:
#    --model-param bert-base:max_model_len=512
#    --model-param bert-base:dtype=float32
#    --model-param bert-base:enforce_eager=true
#    --model-param bert-base:gpu_memory_utilization=0.7
#
# Key Parameters:
#    --models: List of models to load (required)
#    --port: Server port (default: 8000)
#    --max-model-len: Maximum model length (default: 512)
#    --dtype: Data type (default: bfloat16)
#    --gpu-memory-utilization: GPU memory utilization (default: 0.3)
#    --env-preset: Environment preset - default/performance (default: default)
#    --env-config: JSON file with environment variables
#    --env: Individual environment variables (KEY=value)
#    --model-param: Per-model parameter (model_name:param_name=value)
#    --model-config: JSON string with per-model configurations
#    --force-restart: Force restart server
#    --stop-all: Stop all running models
#    --timeout: Startup timeout in seconds (default: 300)
#    --log-file: Log file path
#
# Advanced Examples:
# Example 1: Embedding models with different configurations
#    python launch_multi_models.py \
#        --models /data/models/gte-modernbert-base \
#                 /data/models/gte-reranker-modernbert-base \
#        --port 8771 \
#        --model-param gte-modernbert-base:max_model_len=512 \
#        --model-param gte-modernbert-base:dtype=bfloat16 \
#        --model-param gte-reranker-modernbert-base:max_model_len=1024 \
#        --model-param gte-reranker-modernbert-base:dtype=float32
#
# Example 2: Mixed precision and memory allocation
#    python launch_multi_models.py \
#        --models model1 model2 model3 \
#        --gpu-memory-utilization 0.3 \
#        --model-param model1:gpu_memory_utilization=0.5 \
#        --model-param model1:dtype=float32 \
#        --model-param model2:gpu_memory_utilization=0.4 \
#        --model-param model2:enforce_eager=true \
#        --model-param model3:gpu_memory_utilization=0.3
#
# Example 3: Performance tuning per model
#    python launch_multi_models.py \
#        --models bert-large roberta-base albert-xxlarge \
#        --env-preset performance \
#        --model-param bert-large:tensor_parallel_size=2 \
#        --model-param bert-large:max_model_len=2048 \
#        --model-param roberta-base:max_model_len=1024 \
#        --model-param roberta-base:enforce_eager=false \
#        --model-param albert-xxlarge:max_model_len=512 \
#        --model-param albert-xxlarge:disable_log_stats=true

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from typing import Any

import requests

# Constants
PID_FILE = "mm_vllm_server.pid"
SERVER_URL_TEMPLATE = "http://localhost:{port}/v1/models"
LOG_FILE = "mm_vllm_server.log"

# Environment variable presets
DEFAULT_ENV_VARS = {
    "VLLM_CONTIGUOUS_PA": "false",
    "VLLM_SKIP_WARMUP": "true",
    "PT_HPU_LAZY_MODE": "1",
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
    "PT_HPU_LAZY_MODE": "1",
    "VLLM_EXPONENTIAL_BUCKETING": "false",
}


def parse_arguments():
    """Parse command-line arguments with enhanced options."""
    parser = argparse.ArgumentParser(
        description="Enhanced vLLM Multi-Model Server Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  python launch_multi_models.py --models model1 model2

  # With custom max model length
  python launch_multi_models.py --models model1 model2 --max-model-len 4096

  # With performance tuning preset
  python launch_multi_models.py --models model1 model2 --env-preset \
  performance

  # With custom environment config file
  python launch_multi_models.py --models model1 model2 --env-config \
  custom_env.json

  # With individual environment variables
  python launch_multi_models.py --models model1 model2 --env \
  VLLM_SKIP_WARMUP=false --env PT_HPU_LAZY_MODE=1

  # NEW: Per-model configuration - Scenario 1: All models same config
  python launch_multi_models.py --models model1 model2 --max-model-len 1024

  # NEW: Per-model configuration - Scenario 2: Mixed configuration
  python launch_multi_models.py --models model1 model2 --max-model-len 1024 \
      --model-param model3:max_model_len=2048 --model-param model3:dtype=float32

  # NEW: Per-model configuration - Scenario 3: Each model different config
  python launch_multi_models.py --models model1 model2 --max-model-len 512 \
      --model-param model1:max_model_len=1024 \ 
      --model-param model1:enforce_eager=true \
      --model-param model2:max_model_len=2048 \ 
      --model-param model2:dtype=float32 \
      --model-param model2:gpu_memory_utilization=0.6

  # NEW: Complex per-model configuration using JSON
  python launch_multi_models.py --models model1 model2 \
      --model-config '{"model1": {"max_model_len": 1024, "dtype": "float32"}, \
                     "model2": {"max_model_len": 2048, "enforce_eager": true}}'
        """,
    )

    parser.add_argument(
        "--models",
        nargs="+",
        help="List of models to load (required unless --stop-all)",
    )
    parser.add_argument("--force-restart",
                        action="store_true",
                        help="Force restart the server")
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the vLLM server (default: 8000)",
    )
    parser.add_argument("--stop-all",
                        action="store_true",
                        help="Stop all running models")
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=512,
        help="Maximum model length for each model (default: 512). "
        "Set to -1 to use each model's native maximum length.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float32"],
        help="Data type (default: bfloat16)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.3,
        help="GPU memory utilization (0.0-1.0), auto-calculated"
        "if not specified",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Server startup timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=LOG_FILE,
        help=f"Log file path (default: {LOG_FILE})",
    )

    # Environment variable options
    env_group = parser.add_mutually_exclusive_group()
    env_group.add_argument(
        "--env-preset",
        type=str,
        choices=["default", "performance"],
        default="default",
        help="Environment variable preset(default: default)",
    )
    env_group.add_argument(
        "--env-config",
        type=str,
        help="Path to JSON file containing environment variables",
    )

    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="Set individual environment variable (can be used multiple times)",
    )

    parser.add_argument(
        "--list-env-presets",
        action="store_true",
        help="List available environment variable presets",
    )

    # Per-model configuration arguments
    # NEW FEATURE: Allow individual parameter configuration for each model
    parser.add_argument(
        "--model-param",
        action="append",
        type=str,
        help="Per-model parameter in format: model_name:param_name=value. "
        "Can be used multiple times for different models and parameters. "
        "Example: --model-param model1:max_model_len=1024 "
        "--model-param model1:dtype=float32 "
        "--model-param model2:enforce_eager=true",
    )

    parser.add_argument(
        "--model-config",
        type=str,
        help="JSON string containing per-model configurations. "
        'Example: \'{"model1": {"max_model_len": 1024, "dtype": "float32"}, '
        '\'"model2": {"enforce_eager": true}}\' '
        "Alternative to --model-param for complex configurations.",
    )

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
    print("\nNote: performance preset includes bucket configuration for "
          "optimal throughput")


def load_env_config(config_path: str) -> dict[str, str]:
    """Load environment variables from JSON config file."""
    try:
        with open(config_path) as f:
            config = json.load(f)
            if not isinstance(config, dict):
                raise ValueError("Config must be a dictionary")
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


def parse_env_vars(env_list: list) -> dict[str, str]:
    """Parse individual environment variable settings."""
    env_vars = {}
    for env_str in env_list:
        if "=" not in env_str:
            print(f"Error: Invalid environment variable format '{env_str}'. "
                  "Use KEY=value format")
            sys.exit(1)
        key, value = env_str.split("=", 1)
        env_vars[key.strip()] = value.strip()
    return env_vars


def parse_model_params(
    model_param_list: list[str], ) -> dict[str, dict[str, Any]]:
    """Parse model parameters from CLI format to dictionary.

    NEW FEATURE: This function enables per-model configuration by parsing
    command-line parameters in the format 'model_name:param_name=value'.

    Args:
        model_param_list: List of parameter strings in format \ 
        'model:param=value'

    Returns:
        Dictionary mapping model names to their specific configurations

    Example:
        Input: ["model1:max_model_len=1024", "model1:dtype=float32"]
        Output: {"model1": {"max_model_len": 1024, "dtype": "float32"}}
    """
    model_params = {}

    for param_str in model_param_list:
        try:
            # Format: model_name:param_name=value
            if ":" not in param_str or "=" not in param_str:
                print(f"Warning: Invalid model parameter format: {param_str}")
                continue

            model_part, param_part = param_str.split(":", 1)
            param_name, param_value = param_part.split("=", 1)

            model_name = model_part.strip()
            param_name = param_name.strip()
            param_value = param_value.strip()

            # Initialize model config if not exists
            if model_name not in model_params:
                model_params[model_name] = {}

            # Convert parameter value to appropriate type
            converted_value = convert_param_value(param_name, param_value)
            model_params[model_name][param_name] = converted_value

            print(f"Parsed model parameter: \
                    {model_name}:{param_name}={converted_value}")

        except Exception as e:
            print(
                f"Warning: Failed to parse model parameter '{param_str}': {e}")
            continue

    return model_params


def convert_param_value(param_name: str, param_value: str) -> Any:
    """Convert parameter string value to appropriate type.

    NEW FEATURE: This function handles type conversion for per-model parameters.
    Supports boolean, integer, float, and string parameter types.

    Args:
        param_name: Name of the parameter (determines conversion type)
        param_value: String value from command line

    Returns:
        Converted value with appropriate type

    Examples:
        enforce_eager=true -> True
        max_model_len=1024 -> 1024
        gpu_memory_utilization=0.7 -> 0.7
        dtype=float32 -> "float32"
    """
    # Boolean parameters
    if param_name in [
            "enforce_eager",
            "trust_remote_code",
            "disable_log_stats",
            "skip_tokenizer_init",
            "allow_long_max_model_len",
            "disable_logprobs_during_spec_decode",
            "enable_lora",
            "enable_prefix_caching",
            "enable_chunked_prefill",
    ]:
        return param_value.lower() in ("true", "1", "yes", "on")

    # Integer parameters
    elif param_name in [
            "max_model_len",
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "max_num_batched_tokens",
            "max_num_seqs",
            "max_logprobs",
            "block_size",
            "num_scheduler_steps",
            "max_num_on_the_fly",
            "speculative_model_tensor_parallel_size",
            "ngram_prompt_lookup_max",
    ]:
        try:
            return int(param_value)
        except ValueError:
            print(f"Warning: Invalid integer value for \
                    {param_name}: {param_value}")
            return param_value

    # Float parameters
    elif param_name in [
            "gpu_memory_utilization",
            "swap_space",
            "cpu_offload_gb",
    ]:
        try:
            return float(param_value)
        except ValueError:
            print(
                f"Warning: Invalid float value for {param_name}: {param_value}"
            )
            return param_value

    # String parameters (default)
    else:
        return param_value


def parse_model_config_json(config_json: str) -> dict[str, dict[str, Any]]:
    """Parse JSON string containing per-model configurations."""
    try:
        config_data = json.loads(config_json)
        if not isinstance(config_data, dict):
            print("Warning: Model config JSON must be a dictionary")
            return {}
        return config_data
    except json.JSONDecodeError as e:
        print(f"Warning: Invalid JSON in model config: {e}")
        return {}
    except Exception as e:
        print(f"Warning: Failed to parse model config JSON: {e}")
        return {}


def get_environment_variables(args) -> dict[str, str]:
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
            print(f"Failed to query running models. "
                  f"Status code: {response.status_code}")
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


def qwen_model_check(text):
    return all(keyword.lower() in text.lower()
               for keyword in ["qwen", "reranker"])


def launch_server(
    models,
    port,
    max_model_len,
    device,
    dtype,
    gpu_memory_utilization,
    env_vars,
    log_file,
    timeout,
    model_params=None,
):
    """Launch a vLLM server with customized configuration.

    NEW FEATURE: Now supports per-model configuration through model_params.
    Each model can have its own specific vLLM parameters while sharing global
    defaults for unspecified parameters.

    Args:
        models: List of model paths/names to load
        port: Server port number
        max_model_len: Global maximum model length (can be overridden per-model)
        device: Device type (hpu, cuda, etc.)
        dtype: Global data type (can be overridden per-model)
        gpu_memory_utilization: Global GPU memory utilization \
            (can be overridden per-model)
        env_vars: Environment variables dictionary
        log_file: Path to log file
        timeout: Startup timeout in seconds
        model_params: Per-model configuration dictionary (NEW FEATURE)
                     Format: {"model_name": {"param_name": value, ...}}

    Example:
        model_params = {
            "bert-large": {"max_model_len": 2048, "dtype": "float32"},
            "roberta-base": {"max_model_len": 1024, "enforce_eager": True}
        }
        # bert-large will use max_model_len=2048, dtype=float32
        # roberta-base will use max_model_len=1024, enforce_eager=True
        # Other models will use global defaults
    """

    # Calculate memory ratio if not provided
    if gpu_memory_utilization is None:
        gpu_memory_utilization = 7 // len(models) / 10

    print("Launching server with configuration:")
    print(f"  Models: {models}")
    print(f"  Max model length: {max_model_len}")
    print(f"  Device: {device}")
    print(f"  Data type: {dtype}")
    print(f"  GPU memory utilization: {gpu_memory_utilization}")
    print(f"  Port: {port}")

    # Build the base server command
    command = [
        "python3",
        "-m",
        "vllm.entrypoints.openai.mm_api_server",
        "--port",
        str(port),
        "--device",
        device,
        "--dtype",
        dtype,
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--use-v2-block-manager",
        "--models",
    ]
    command += models

    # NEW FEATURE: Add per-model parameters to the command
    # This allows each model to have its own specific configuration
    if model_params:
        print("Adding per-model parameters:")
        for model_name, params in model_params.items():
            for param_name, param_value in params.items():
                command.extend([
                    "--model-param",
                    f"{model_name}:{param_name}={param_value}",
                ])
                print(f"  {model_name}:{param_name}={param_value}")
        print(f"Total per-model parameters added: \
                {sum(len(params) for params in model_params.values())}")

    if max_model_len != -1:
        command += ["--max-model-len", str(max_model_len)]

    qwen_mode = False

    qwen_reranker = ""
    for model in models:
        qwen_mode = qwen_model_check(model)
        if qwen_mode:
            qwen_reranker = model
            break

    if qwen_mode:
        qwen3_config = {
            "architectures": ["Qwen3ForSequenceClassification"],
            "classifier_from_token": ["no", "yes"],
            "is_original_qwen3_reranker": True,
        }

        config_json = json.dumps(qwen3_config)
        command += [
            "--model-param",
            f"{qwen_reranker}:hf_overrides={config_json}",
        ]
        print(f"input : {command}")

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
        print(
            f"Waiting for server to start. Monitoring {log_file} for output..."
        )
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
            print(f"\nServer started successfully! "
                  f"Took {elapsed_time:.2f} seconds.")
            return True
        else:
            print(f"\nServer did not start within the timeout period "
                  f"({timeout}s).")
            return False

    except Exception as e:
        print(f"Failed to launch server: {e}")
        return False


def main():
    args = parse_arguments()

    # Require --models unless --stop-all is specified
    if not args.stop_all and not args.models:
        print("Error: --models is required unless --stop-all is specified.")
        sys.exit(2)

    # Get environment variables
    env_vars = get_environment_variables(args)

    # NEW FEATURE: Parse per-model configurations
    # This enables individual parameter configuration for each model
    model_params = {}

    # Parse --model-param arguments (format: model_name:param_name=value)
    if args.model_param:
        print("Parsing per-model parameters from --model-param arguments...")
        cli_params = parse_model_params(args.model_param)
        model_params.update(cli_params)
        print(f"Parsed {len(cli_params)} models with CLI parameters")

    # Parse --model-config JSON string if provided
    if args.model_config:
        print("Parsing per-model configurations from --model-config JSON...")
        json_params = parse_model_config_json(args.model_config)
        model_params.update(json_params)
        print(f"Parsed {len(json_params)} models from JSON config")

    # Display loaded per-model configurations
    if model_params:
        print("Loaded per-model configurations:")
        for model_name, params in model_params.items():
            print(f"  {model_name}: {params}")
        print(
            f"Total models with specific configurations: {len(model_params)}")
    else:
        print("No per-model configurations specified - using global config")

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
            print("Server is already running with the requested models. "
                  "No action required.")
            return
        else:
            print("Running models do not match the requested models. "
                  "Restarting server.")
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
        timeout=args.timeout,
        model_params=model_params,
    )

    if success:
        get_running_models(args.port)
    else:
        print("Server launch failed. Check the log file for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
