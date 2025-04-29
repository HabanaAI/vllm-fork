# SPDX-License-Identifier: Apache-2.0
import pickle
import signal
import time
import traceback
from dataclasses import dataclass
from typing import List, Optional

import cloudpickle

from vllm import SamplingParams
from vllm.config import VllmConfig
from vllm.engine.llm_engine import LLMEngine
from vllm.engine.mm_arg_utils import MMAsyncEngineArgs
# yapf conflicts with isort for this block
# yapf: disable
from vllm.engine.multiprocessing import (ENGINE_DEAD_ERROR,
                                         VLLM_RPC_SUCCESS_STR, RPCAbortRequest,
                                         RPCError, RPCProcessRequest,
                                         RPCUProfileRequest)
from vllm.engine.multiprocessing.engine import MQLLMEngine
# yapf: enable
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.usage.usage_lib import UsageContext

logger = init_logger(__name__)

POLLING_TIMEOUT_MS = 10000
HEALTHY_RESPONSE = (pickle.dumps(VLLM_RPC_SUCCESS_STR), )


@dataclass
class RPCModelRequest:
    request_id: Optional[str]
    models: List[str]


@dataclass
class RPCModelResponse:
    request_id: Optional[str]
    finished: bool
    text: str
    existing_models: Optional[List[str]] = None
    closed_models: Optional[List[str]] = None
    new_models: Optional[List[str]] = None


class MMLLMEngine(MQLLMEngine):

    def __init__(self,
                 ipc_path: str,
                 use_async_sockets: bool,
                 *args,
                 log_requests: bool = True,
                 **kwargs) -> None:
        kwargs['is_multi_models_engine'] = True
        super().__init__(ipc_path,
                         use_async_sockets,
                         *args,
                         log_requests=log_requests,
                         **kwargs)
        kwargs.pop('is_multi_models_engine', None)
        engine_args = kwargs.pop('engine_args', None)

        self.engine_config = {
            'args': args,
            'kwargs': kwargs,
            'engine_args': engine_args
        }
        # get configs from args and kwargs, determine how many models to load
        original_vllm_config_list = kwargs.get('vllm_config')
        if not isinstance(original_vllm_config_list, List):
            original_vllm_config_list = [original_vllm_config_list]
        self.engines = []

        for i, vllm_config in enumerate(original_vllm_config_list):
            kwargs['vllm_config'] = vllm_config
            self.engines.append(LLMEngine(*args, **kwargs))
        self.engine = self.engines[0]
        self.log_requests = log_requests

        self.use_async_sockets = use_async_sockets
        if self.use_async_sockets:
            for engine in self.engines:
                engine.process_request_outputs_callback = \
                self._async_socket_engine_callback

    @classmethod
    def from_vllm_config(  # type: ignore[override]
            cls, vllm_config_list: List[VllmConfig],
            usage_context: UsageContext, disable_log_requests: bool,
            disable_log_stats: bool, ipc_path: str,
            engine_args: MMAsyncEngineArgs) -> "MMLLMEngine":
        """Creates an MQLLMEngine from the engine arguments."""
        # Setup plugins for each process
        from vllm.plugins import load_general_plugins
        load_general_plugins()

        vllm_config = vllm_config_list[0]

        use_async_sockets = vllm_config.model_config.use_async_output_proc

        return cls(
            vllm_config=vllm_config_list,
            executor_class=LLMEngine._get_executor_cls(vllm_config),
            ipc_path=ipc_path,
            usage_context=usage_context,
            use_async_sockets=use_async_sockets,
            log_requests=(not disable_log_requests),
            log_stats=(not disable_log_stats),
            engine_args=engine_args,
        )

    def cleanup(self):
        """Cleanup zeromq state on shutdown."""
        # Closes all sockets and destroys context.
        self.ctx.destroy(linger=0)
        del self.engines

    def run_engine_loop(self):
        """Core busy loop of the LLMEngine."""

        while True:
            if not any(engine.has_unfinished_requests()
                       for engine in self.engines):
                # Poll until there is work to do.
                while self.input_socket.poll(timeout=POLLING_TIMEOUT_MS) == 0:
                    # When there's no work, check on engine health and send
                    # health status back to client
                    self._health_check()
                    for engine in self.engines:
                        engine.do_log_stats()
                    logger.debug("Waiting for new requests in engine loop.")

            # Handle any input from the client.
            self.handle_new_input()

            # Engine step.
            request_outputs = self.engine_step()

            # Send request outputs (if async, done in engine_step callback).
            if not self.use_async_sockets:
                self._send_outputs(request_outputs)

    def engine_step(self) -> List[RequestOutput]:
        """Engine step wrapper with error handling."""
        try:
            res = []
            for engine in self.engines:
                res += engine.step()
            return res
        except SystemExit:
            raise
        except BaseException as e:
            self._set_errored(e)
            rpc_err = RPCError(request_id=None,
                               is_engine_errored=True,
                               exception=e)
            self._send_outputs(rpc_err)
            raise e

    # FIXME: add model field in RPCProcessRequest,
    # and dispatch to the correct engine
    def handle_new_input(self):
        """Handle new input from the socket"""
        try:
            while self.input_socket.poll(timeout=0) != 0:
                frames = self.input_socket.recv_multipart(copy=False)
                request = pickle.loads(frames[0].buffer)

                logger.info("request type is %s", str(type(request)))
                if isinstance(request, RPCProcessRequest):
                    if len(frames) > 1:
                        # Use cloudpickle for logits processors
                        assert isinstance(request.params, SamplingParams)
                        lprocs = cloudpickle.loads(frames[1].buffer)
                        request.params.logits_processors = lprocs
                    self._handle_process_request(request)
                elif isinstance(request, RPCAbortRequest):
                    self._handle_abort_request(request)
                elif isinstance(request, RPCModelRequest):
                    self._handle_model_request(request)
                elif isinstance(request, RPCUProfileRequest):
                    if request == RPCUProfileRequest.START_PROFILE:
                        self.start_profile()
                    else:
                        self.stop_profile()
                else:
                    raise ValueError("Unknown RPCRequest Type: "
                                     f"{type(request)}")

        except Exception as e:
            self._set_errored(e)
            self._send_unhealthy(e)
            raise e

    def _handle_process_request(self, request: RPCProcessRequest):
        """Handle RPCProcessRequest by adding it to the LLMEngine."""
        request_id = request.request_id

        if self._errored_with is not None:
            rpc_err = RPCError(request_id=request_id,
                               is_engine_errored=True,
                               exception=ENGINE_DEAD_ERROR(self._errored_with))
            self._send_outputs(rpc_err)

        try:
            for engine in self.engines:
                if engine.model_config.model == request.model:
                    engine.add_request(
                        request_id=request_id,
                        prompt=request.prompt,
                        params=request.params,
                        lora_request=request.lora_request,
                        trace_headers=request.trace_headers,
                        prompt_adapter_request=request.prompt_adapter_request,
                        priority=request.priority)

                    if self.log_requests:
                        logger.info("%s - Added request to %s",
                                    str(request.model),
                                    str(request.request_id))

        except Exception as e:
            # We do not set self._errored = True here, since the error
            # is due to an issue adding this request to the engine,
            # rather than an issue with the engine itself.
            is_errored = self._errored_with is not None
            rpc_err = RPCError(request_id=request_id,
                               is_engine_errored=is_errored,
                               exception=e)
            self._send_outputs(rpc_err)

            # Remove request from the engine.
            self.engine.abort_request(request_id)

    def _handle_model_request(self, request: RPCModelRequest):
        logger.info("Get model request - %s", str(request.models))
        """Handle RPCModelRequest by loading the models."""
        try:
            close_engines = []
            models = []
            closed_models = []
            new_models = []
            msg = ""
            for engine in self.engines:
                models.append(engine.model_config.model)
                if engine.model_config.model not in request.models:
                    close_engines.append(engine)
            # del models that are not in close_models
            for engine in close_engines:
                closed_models.append(engine.model_config.model)
                self.engines.remove(engine)
                del engine
            # start the models that are in request.models not in models
            for model in request.models:
                if model not in models:
                    start = time.perf_counter()
                    print('start new model', model)
                    engine_args = self.engine_config['engine_args']
                    engine_args.model = model
                    engine_args.tokenizer = model
                    engine_args.models = None
                    vllm_config = engine_args.create_engine_config()[0]
                    args = self.engine_config['args']
                    kwargs = self.engine_config['kwargs']
                    kwargs['vllm_config'] = vllm_config
                    engine = LLMEngine(*args, **kwargs)
                    if self.use_async_sockets:
                        engine.process_request_outputs_callback = \
                            self._async_socket_engine_callback
                    self.engines.append(engine)
                    new_models.append((
                        model,
                        f"launch time: {time.perf_counter() - start} seconds"))
            msg += f"Existing models: {models}."
            msg += f"Closed Models: {closed_models}."
            msg += f"Starting new model: {new_models}."
            self._send_outputs([
                RPCModelResponse(request_id=request.request_id,
                                 finished=True,
                                 text=msg,
                                 existing_models=models,
                                 closed_models=closed_models,
                                 new_models=[m[0] for m in new_models])
            ])
        except Exception:
            logger.info("error happens: %s", str(traceback.format_exc()))
            rpc_err = RPCError(
                request_id=request.request_id,
                is_engine_errored=False,
                exception=Exception(f"{traceback.format_exc()}"))
            self._send_outputs(rpc_err)

    def _health_check(self):
        # Send unhealthy if engine has already errored
        if self._errored_with is not None:
            self._send_unhealthy(self._errored_with)
        try:
            for engine in self.engines:
                engine.check_health()
            self._send_healthy()
        except Exception as e:
            self._set_errored(e)
            self._send_unhealthy(e)


def signal_handler(*_) -> None:
    raise KeyboardInterrupt("MQLLMEngine terminated")


def run_mm_engine(vllm_config_list: List[VllmConfig],
                  usage_context: UsageContext, ipc_path: str,
                  disable_log_stats: bool, disable_log_requests: bool,
                  engine_alive, engine_args) -> None:
    logger.info("Multi Model LLM Engine starting up...")
    try:
        engine = MMLLMEngine.from_vllm_config(
            vllm_config_list=vllm_config_list,
            usage_context=usage_context,
            disable_log_stats=disable_log_stats,
            disable_log_requests=disable_log_requests,
            ipc_path=ipc_path,
            engine_args=engine_args,
        )

        signal.signal(signal.SIGTERM, signal_handler)

        engine.start()

    except BaseException as e:
        logger.exception(e)
        engine_alive.value = False
        raise e
