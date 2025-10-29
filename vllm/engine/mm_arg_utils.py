# SPDX-License-Identifier: Apache-2.0
import copy
import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, cast

from vllm.config import VllmConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.logger import init_logger
from vllm.utils import FlexibleArgumentParser

logger = init_logger(__name__)

ALLOWED_DETAILED_TRACE_MODULES = ["model", "worker", "all"]

DEVICE_OPTIONS = [
    "auto",
    "cuda",
    "neuron",
    "cpu",
    "openvino",
    "tpu",
    "xpu",
    "hpu",
]


@dataclass
class MMEngineArgs(EngineArgs):
    """Arguments for Multi Models vLLM engine."""

    models: Optional[List[str]] = None
    model_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        if self.models is None:
            self.models = [self.model]
        else:
            self.model = self.models[0]
        if not self.tokenizer:
            self.tokenizer = self.model
        super().__post_init__()

    @classmethod
    def from_cli_args(cls, args: FlexibleArgumentParser) -> "MMEngineArgs":
        "Create MMEngineArgs from CLI arguments, including per-model param"
        # Get the list of attributes of this dataclass.
        attrs = [attr.name for attr in dataclasses.fields(cls)]

        # Create a dictionary of attributes, handling model_params specially
        kwargs = {}
        for attr in attrs:
            if attr == "model_params":
                # Handle model_params separately - it's derived from model_param
                if hasattr(args, "model_param") and args.model_param:
                    kwargs[attr] = cls._parse_model_params(args.model_param)
                else:
                    kwargs[attr] = {}
            else:
                # For other attributes, get from args if available
                if hasattr(args, attr):
                    kwargs[attr] = getattr(args, attr)

        # Create the MMEngineArgs instance
        return cls(**cast(Dict[str, Any], kwargs))

    @classmethod
    def _parse_model_params(
            cls, model_param_list: List[str]) -> Dict[str, Dict[str, Any]]:
        """Parse model parameters from CLI format to dictionary."""
        model_params: Dict[str, Dict[str, Any]] = {}

        for param_str in model_param_list:
            try:
                # Format: model_name:param_name=value
                if ":" not in param_str or "=" not in param_str:
                    logger.warning("Invalid model parameter format: %s",
                                   param_str)
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
                converted_value = cls._convert_param_value(
                    param_name, param_value)
                model_params[model_name][param_name] = converted_value

                logger.info(
                        "Parsed model parameter: %s:%s=%s", model_name, \
                            param_name, converted_value
                )

            except Exception as e:
                logger.warning("Failed to parse model parameter %s:%s",
                               param_str, e)
                continue

        return model_params

    @staticmethod
    def _convert_param_value(param_name: str, param_value: str) -> Any:
        # Convert parameter string value to appropriate type.
        # parse JSON strings to dictionaries
        if param_name in [
                "hf_overrides",
                "override_neuron_config",
                "override_generation_config",
        ]:
            # These parameters expect JSON strings, parse them to Python objects
            try:
                import json

                return json.loads(param_value)
            except json.JSONDecodeError as e:
                logger.warning(
                    "Failed to parse JSON for %s,%s.Using raw string.", \
                        param_name,e
                )
                return param_value

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
                logger.warning("Invalid integer value for %s: %s", param_name,
                               param_value)
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
                logger.warning("Invalid float value for %s: %s", param_name,
                               param_value)
                return param_value

        # String parameters (default)
        else:
            return param_value

    def _apply_model_config(self, args: EngineArgs,
                            model_name: str) -> EngineArgs:
        """Apply model-specific configuration to engine args."""
        if self.model_params and model_name in self.model_params:
            model_config = self.model_params[model_name]
            logger.info("Applying model-specific config for %s: %s",
                        model_name, model_config)

            # Apply each configuration parameter
            for param_name, param_value in model_config.items():
                if hasattr(args, param_name):
                    setattr(args, param_name, param_value)
                    logger.info("  Set %s =%s", param_name, param_value)
                else:
                    logger.warning("  Unknown parameter %s for model %s",
                                   param_name, param_value)

        return args

    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
        """Shared CLI arguments for vLLM engine."""
        # Model arguments
        parser.add_argument(
            "--models",
            "--names-list",
            nargs="*",
            type=str,
            default=MMEngineArgs.models,
            help="Name or path of the huggingface model to use.",
        )

        # Per-model configuration arguments
        parser.add_argument(
            "--model-param",
            action="append",
            type=str,
            help="Per-model parameter in format: model_name:param_name=value. "
            "Example: --model-param model1:max_model_len=1024 "
            "--model-param model1:dtype=float32",
        )

        return parser

    def create_engine_config(self, *args, **kwargs) -> List[VllmConfig]:
        engine_configs = []

        if self.models is not None:
            for model in self.models:
                tmp_args = copy.deepcopy(self)
                tmp_args.model = model
                tmp_args.tokenizer = model
                tmp_args.models = None

                # Apply model-specific configuration
                tmp_args = cast(MMEngineArgs,
                                self._apply_model_config(tmp_args, model))

                engine_config = tmp_args.create_engine_config(*args, **kwargs)
                engine_configs += engine_config
        else:
            engine_config = super().create_engine_config(*args, **kwargs)
            engine_configs.append(engine_config)
        return engine_configs


@dataclass
class MMAsyncEngineArgs(MMEngineArgs):
    """Arguments for asynchronous vLLM engine."""

    disable_log_requests: bool = False

    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser,
                     async_args_only: bool = False) -> FlexibleArgumentParser:
        if not async_args_only:
            parser = MMEngineArgs.add_cli_args(parser)
        return parser


# These functions are used by sphinx to build the documentation
def _engine_args_parser():
    return MMEngineArgs.add_cli_args(FlexibleArgumentParser())


def _async_engine_args_parser():
    return MMAsyncEngineArgs.add_cli_args(FlexibleArgumentParser(),
                                          async_args_only=True)
