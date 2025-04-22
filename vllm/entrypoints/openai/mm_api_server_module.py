# SPDX-License-Identifier: Apache-2.0
import asyncio
import atexit
import multiprocessing
import os
import tempfile
from argparse import Namespace
from contextlib import asynccontextmanager
from functools import partial
from typing import AsyncIterator, Optional

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.datastructures import State

import vllm.envs as envs
from vllm.config import ModelConfig
from vllm.engine.mm_arg_utils import MMAsyncEngineArgs
from vllm.engine.multiprocessing.mm_client import MMLLMEngineClient
from vllm.engine.multiprocessing.mm_engine import run_mm_engine
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import load_chat_template
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai import api_server
# yapf conflicts with isort for this block
# yapf: disable
from vllm.entrypoints.openai.protocol import ModelConfigRequest
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_embedding import OpenAIServingEmbedding
from vllm.entrypoints.openai.serving_models import (BaseModelPath,
                                                    OpenAIServingModels)
from vllm.entrypoints.openai.serving_pooling import OpenAIServingPooling
from vllm.entrypoints.openai.serving_rerank import JinaAIServingRerank
from vllm.entrypoints.openai.serving_score import OpenAIServingScores
from vllm.entrypoints.openai.serving_tokenization import (
    OpenAIServingTokenization)
from vllm.entrypoints.openai.serving_transcription import (
    OpenAIServingTranscription)
# yapf: enable
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext
from vllm.utils import get_open_zmq_ipc_path

logger = init_logger('vllm.entrypoints.openai.mm_api_server')

if envs.VLLM_USE_V1:
    from vllm.v1.engine.async_llm import AsyncLLMEngine  # type: ignore
else:
    from vllm.engine.async_llm_engine import AsyncLLMEngine  # type: ignore


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

    async with build_async_engine_client_from_engine_args(
            engine_args, args.disable_frontend_multiprocessing) as engine:
        yield engine


@asynccontextmanager
async def build_async_engine_client_from_engine_args(
    engine_args: MMAsyncEngineArgs,
    disable_frontend_multiprocessing: bool = False,
) -> AsyncIterator[EngineClient]:
    """
    Create EngineClient, either:
        - in-process using the AsyncLLMEngine Directly
        - multiprocess using AsyncLLMEngine RPC

    Returns the Client or None if the creation failed.
    """
    # Fall back
    # TODO: fill out feature matrix.
    logger.info("Building mm_api_server async engine client from engine args")
    if (MMLLMEngineClient.is_unsupported_config(engine_args)  # type: ignore
            or envs.VLLM_USE_V1 or disable_frontend_multiprocessing):

        engine_configs = engine_args.create_engine_configs()
        engine_client: Optional[EngineClient] = None
        try:
            engine_client = AsyncLLMEngine.from_engine_args(
                engine_args=engine_args,
                engine_config=engine_configs,
                usage_context=UsageContext.OPENAI_API_SERVER)
            yield engine_client  # type: ignore[misc]
        finally:
            if engine_client and hasattr(engine_client, "shutdown"):
                engine_client.shutdown()

    # Otherwise, use the multiprocessing AsyncLLMEngine.
    else:
        if "PROMETHEUS_MULTIPROC_DIR" not in os.environ:
            # Make TemporaryDirectory for prometheus multiprocessing
            # Note: global TemporaryDirectory will be automatically
            #   cleaned up upon exit.
            global prometheus_multiproc_dir
            tmp_dir = tempfile.TemporaryDirectory()
            prometheus_multiproc_dir = tmp_dir  # type: ignore
            os.environ[
                "PROMETHEUS_MULTIPROC_DIR"] = \
                    prometheus_multiproc_dir.name # type: ignore
        else:
            logger.warning(
                "Found PROMETHEUS_MULTIPROC_DIR was set by user. "
                "This directory must be wiped between vLLM runs or "
                "you will find inaccurate metrics. Unset the variable "
                "and vLLM will properly handle cleanup.")

        # Select random path for IPC.
        ipc_path = get_open_zmq_ipc_path()
        logger.info("Multiprocessing frontend to use %s for IPC Path.",
                    ipc_path)

        # Start RPCServer in separate process (holds the LLMEngine).
        # the current process might have CUDA context,
        # so we need to spawn a new process
        context = multiprocessing.get_context("spawn")

        # The Process can raise an exception during startup, which may
        # not actually result in an exitcode being reported. As a result
        # we use a shared variable to communicate the information.
        engine_alive = multiprocessing.Value('b', True, lock=False)
        engine_process = context.Process(target=run_mm_engine,
                                         args=(engine_args,
                                               UsageContext.OPENAI_API_SERVER,
                                               ipc_path, engine_alive))
        engine_process.start()
        engine_pid = engine_process.pid
        assert engine_pid is not None, "Engine process failed to start."
        logger.info("Started engine process with PID %d", engine_pid)

        def _cleanup_ipc_path():
            socket_path = ipc_path.replace("ipc://", "")
            if os.path.exists(socket_path):
                os.remove(socket_path)

        # Ensure we clean up the local IPC socket file on exit.
        atexit.register(_cleanup_ipc_path)

        # Build RPCClient, which conforms to EngineClient Protocol.
        engine_configs = engine_args.create_engine_configs()
        build_client = partial(MMLLMEngineClient, ipc_path, engine_configs,
                               engine_pid)
        mq_engine_client = await asyncio.get_running_loop().run_in_executor(
            None, build_client)

        try:
            while True:
                try:
                    await mq_engine_client.setup()
                    break
                except TimeoutError:
                    if (not engine_process.is_alive()
                            or not engine_alive.value):
                        raise RuntimeError(
                            "Engine process failed to start. See stack "
                            "trace for the root cause.") from None

            yield mq_engine_client  # type: ignore[misc]
        finally:
            # Ensure rpc server process was terminated
            engine_process.terminate()

            # Close all open connections to the backend
            mq_engine_client.close()

            # Wait for engine process to join
            engine_process.join(4)
            if engine_process.exitcode is None:
                # Kill if taking longer than 5 seconds to stop
                engine_process.kill()

            # Lazy import for prometheus multiprocessing.
            # We need to set PROMETHEUS_MULTIPROC_DIR environment variable
            # before prometheus_client is imported.
            # See https://prometheus.github.io/client_python/multiprocess/
            from prometheus_client import multiprocess
            multiprocess.mark_process_dead(engine_process.pid)


router = api_server.router


@router.post("/v1/update_models")
async def update_models(request: ModelConfigRequest, raw_request: Request):
    handler = api_server.base(raw_request)

    response = await handler.update_model_config(request)
    return JSONResponse(content=response)


async def init_app_state(
    engine_client: EngineClient,
    model_config: ModelConfig,
    state: State,
    args: Namespace,
) -> None:
    if args.served_model_name is not None:
        served_model_names = args.served_model_name
        args.models = [args.model]
    elif args.models is not None:
        served_model_names = args.models
    else:
        served_model_names = [args.model]
        args.models = [args.model]

    if args.disable_log_requests:
        request_logger = None
    else:
        request_logger = RequestLogger(max_log_len=args.max_log_len)

    base_model_paths = [
        BaseModelPath(name=name, model_path=path)
        for name, path in zip(served_model_names, args.models)
    ]

    state.engine_client = engine_client
    state.log_stats = not args.disable_log_stats

    resolved_chat_template = load_chat_template(args.chat_template)
    logger.info("Using supplied chat template:\n%s", resolved_chat_template)

    state.openai_serving_models = OpenAIServingModels(
        engine_client=engine_client,
        model_config=model_config,
        base_model_paths=base_model_paths,
        lora_modules=args.lora_modules,
        prompt_adapters=args.prompt_adapters,
    )
    await state.openai_serving_models.init_static_loras()
    state.openai_serving_chat = OpenAIServingChat(
        engine_client,
        model_config,
        state.openai_serving_models,
        args.response_role,
        request_logger=request_logger,
        chat_template=resolved_chat_template,
        chat_template_content_format=args.chat_template_content_format,
        return_tokens_as_token_ids=args.return_tokens_as_token_ids,
        enable_auto_tools=args.enable_auto_tool_choice,
        tool_parser=args.tool_call_parser,
        enable_reasoning=args.enable_reasoning,
        reasoning_parser=args.reasoning_parser,
        enable_prompt_tokens_details=args.enable_prompt_tokens_details,
    ) if model_config.runner_type == "generate" else None
    state.openai_serving_completion = OpenAIServingCompletion(
        engine_client,
        model_config,
        state.openai_serving_models,
        request_logger=request_logger,
        return_tokens_as_token_ids=args.return_tokens_as_token_ids,
    ) if model_config.runner_type == "generate" else None
    state.openai_serving_pooling = OpenAIServingPooling(
        engine_client,
        model_config,
        state.openai_serving_models,
        request_logger=request_logger,
        chat_template=resolved_chat_template,
        chat_template_content_format=args.chat_template_content_format,
    ) if model_config.runner_type == "pooling" else None
    state.openai_serving_embedding = OpenAIServingEmbedding(
        engine_client,
        model_config,
        state.openai_serving_models,
        request_logger=request_logger,
        chat_template=resolved_chat_template,
        chat_template_content_format=args.chat_template_content_format,
    ) if model_config.task == "embed" else None
    state.openai_serving_scores = OpenAIServingScores(
        engine_client,
        model_config,
        state.openai_serving_models,
        request_logger=request_logger
    ) if model_config.task == "score" else None
    state.jinaai_serving_reranking = JinaAIServingRerank(
        engine_client,
        model_config,
        state.openai_serving_models,
        request_logger=request_logger
    ) if model_config.task == "score" else None
    state.openai_serving_tokenization = OpenAIServingTokenization(
        engine_client,
        model_config,
        state.openai_serving_models,
        request_logger=request_logger,
        chat_template=resolved_chat_template,
        chat_template_content_format=args.chat_template_content_format,
    )
    state.openai_serving_transcription = OpenAIServingTranscription(
        engine_client,
        model_config,
        state.openai_serving_models,
        request_logger=request_logger,
    ) if model_config.runner_type == "transcription" else None
    state.task = model_config.task


api_server.update_models = update_models  # type: ignore
api_server.init_app_state = init_app_state
api_server.build_async_engine_client = build_async_engine_client
api_server.build_async_engine_client_from_engine_args = (
    build_async_engine_client_from_engine_args)  # type: ignore
