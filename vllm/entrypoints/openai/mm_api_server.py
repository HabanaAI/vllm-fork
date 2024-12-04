# SPDX-License-Identifier: Apache-2.0
import uvloop

from vllm.entrypoints.openai.cli_args import (make_mm_arg_parser,
                                              validate_parsed_serve_args)
from vllm.entrypoints.openai.mm_api_server_module import run_server
from vllm.utils import FlexibleArgumentParser

if __name__ == "__main__":
    # NOTE(simon):
    # This section should be in sync with vllm/scripts.py for CLI entrypoints.
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server.")
    parser = make_mm_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)

    uvloop.run(run_server(args))
