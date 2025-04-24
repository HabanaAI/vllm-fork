# SPDX-License-Identifier: Apache-2.0
import copy
from dataclasses import dataclass
from typing import List, Optional

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

    def __post_init__(self):
        if self.models is None:
            self.models = [self.model]
        else:
            self.model = self.models[0]
        if not self.tokenizer:
            self.tokenizer = self.model
        super().__post_init__()

    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
        """Shared CLI arguments for vLLM engine."""
        # Model arguments
        parser.add_argument(
            '--models',
            '--names-list',
            nargs="*",
            type=str,
            default=MMEngineArgs.models,
            help='Name or path of the huggingface model to use.')
        return parser

    def create_engine_config(self, *args, **kwargs) -> List[VllmConfig]:
        engine_configs = []

        if self.models is not None:
            for model in self.models:
                tmp_args = copy.deepcopy(self)
                tmp_args.model = model
                tmp_args.tokenizer = model
                tmp_args.models = None
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
