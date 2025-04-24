# SPDX-License-Identifier: Apache-2.0
from argparse import Namespace
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Request
from fastapi.responses import JSONResponse

from vllm.engine.mm_arg_utils import MMAsyncEngineArgs
from vllm.engine.multiprocessing.mm_client import MMLLMEngineClient
from vllm.engine.multiprocessing.mm_engine import run_mm_engine
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai import api_server
# yapf conflicts with isort for this block
# yapf: disable
from vllm.entrypoints.openai.protocol import ModelConfigRequest
from vllm.entrypoints.openai.serving_models import BaseModelPath
# yapf: enable
from vllm.logger import init_logger

logger = init_logger('vllm.entrypoints.openai.mm_api_server')


# Dynamic fallback for anything not overridden
def __getattr__(name):
    return getattr(api_server, name)


@asynccontextmanager
async def build_async_engine_client(
        args: Namespace) -> AsyncIterator[EngineClient]:

    # Context manager to handle engine_client lifecycle
    # Ensures everything is shutdown and cleaned up on error/exit
    logger.info("Building mm_api_server async engine client.")
    engine_args = MMAsyncEngineArgs.from_cli_args(args)

    async with api_server.build_async_engine_client_from_engine_args(
            engine_args, args.disable_frontend_multiprocessing) as engine:
        yield engine


router = api_server.router


@router.post("/v1/update_models")
async def update_models(request: ModelConfigRequest, raw_request: Request):
    handler = api_server.base(raw_request)

    response = await handler.update_model_config(request)
    return JSONResponse(content=response)


def additional_preparation(served_model_names, base_model_paths,
                           args: Namespace):
    if args.models is None:
        args.models = [args.model]
        served_model_names = args.models
    else:
        served_model_names = args.models

    base_model_paths = [
        BaseModelPath(name=name, model_path=path)
        for name, path in zip(served_model_names, args.models)
    ]
    return served_model_names, base_model_paths, args


api_server.update_models = update_models  # type: ignore
api_server.additional_preparation = additional_preparation
api_server.build_async_engine_client = build_async_engine_client
api_server.run_mp_engine = run_mm_engine
api_server.MQLLMEngineClient = MMLLMEngineClient
