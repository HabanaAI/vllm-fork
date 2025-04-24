# SPDX-License-Identifier: Apache-2.0
import asyncio
import copy
import pickle
from typing import AsyncGenerator, List, Mapping, Optional, Union, cast

import cloudpickle

from vllm import PoolingParams
from vllm.config import DecodingConfig, VllmConfig
# yapf conflicts with isort for this block
# yapf: disable
from vllm.engine.async_llm_engine import (
    build_guided_decoding_logits_processor_async)
from vllm.engine.multiprocessing import ENGINE_DEAD_ERROR, RPCProcessRequest
from vllm.engine.multiprocessing.client import MQLLMEngineClient
from vllm.engine.multiprocessing.mm_engine import (RPCModelRequest,
                                                   RPCModelResponse)
# yapf: enable
from vllm.inputs import PromptType
from vllm.inputs.preprocess import InputPreprocessor
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.outputs import EmbeddingRequestOutput, RequestOutput
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.tokenizer_group import init_tokenizer_from_configs
from vllm.utils import deprecate_kwargs

logger = init_logger(__name__)


class MMLLMEngineClient(MQLLMEngineClient):

    def __init__(self, ipc_path: str, engine_configs: List[VllmConfig],
                 engine_pid: int):
        super().__init__(ipc_path, engine_configs, engine_pid)

        # use model_configs.
        engine_config = engine_configs[0]
        self.model_configs = [
            engine_config.model_config for engine_config in engine_configs
        ]
        self.model_config = self.model_configs[0]
        self.engine_configs = engine_configs
        self.decoding_config = engine_config.decoding_config

        # Create the tokenizer group.
        self.tokenizers = []
        self.input_preprocessors = []
        for model_config in self.model_configs:
            self.tokenizers.append(
                init_tokenizer_from_configs(
                    model_config=model_config,
                    scheduler_config=engine_config.scheduler_config,
                    parallel_config=engine_config.parallel_config,
                    lora_config=engine_config.lora_config,
                ))
            self.input_preprocessors.append(
                InputPreprocessor(model_config, self.tokenizers[-1]))

    @staticmethod
    def is_unsupported_config(vllm_config_list: List[VllmConfig]):
        # Pipeline parallel not yet supported
        print(vllm_config_list)
        return vllm_config_list[0].parallel_config.pipeline_parallel_size > 1

    async def get_tokenizer_mm(self,
                               model,
                               lora_request: Optional[LoRARequest] = None):
        for tokenizer in self.tokenizers:
            if tokenizer.tokenizer_id == model:
                return await tokenizer.get_lora_tokenizer_async(lora_request)
        raise ValueError(f"Tokenizer for model {model} not found.")

    @deprecate_kwargs(
        "inputs",
        additional_message="Please use the 'prompt' parameter instead.",
    )
    async def update_model_config(
            self, request_id: str,
            model_list: Optional[List[str]]) -> Union[RPCModelResponse, str]:
        queue: asyncio.Queue[Union[RequestOutput,
                                   BaseException]] = asyncio.Queue()
        self.output_queues[request_id] = queue
        # If already dead, error out.
        if self._errored_with is not None:
            raise ENGINE_DEAD_ERROR(self._errored_with)

        try:
            request_bytes = pickle.dumps(
                RPCModelRequest(request_id=request_id, models=model_list))

            await self.input_socket.send_multipart((request_bytes, ),
                                                   copy=False)

            # add timeout
            request_output = await queue.get()
        finally:
            self.output_queues.pop(request_id)
        logger.info("request_output is %s", str(request_output))
        if isinstance(request_output, BaseException):
            return f"{request_output}"
        else:
            closed_models = request_output.closed_models
            new_models = request_output.new_models

            # update existing model configs
            self._update_model_config(closed_models, new_models)

            return request_output

    def _update_model_config(self, closed_models: Optional[List[str]],
                             new_models: Optional[List[str]]):
        # Create the tokenizer group.
        closed_models = closed_models or []
        new_models = new_models or []
        engine_config = self.engine_configs[0]
        model_config = self.model_configs[0]
        for model_name in new_models:
            new_model_config = copy.deepcopy(model_config)
            new_model_config.model = model_name
            new_model_config.tokenizer = model_name
            self.model_configs.append(new_model_config)
            self.tokenizers.append(
                init_tokenizer_from_configs(
                    model_config=new_model_config,
                    scheduler_config=engine_config.scheduler_config,
                    parallel_config=engine_config.parallel_config,
                    lora_config=engine_config.lora_config,
                ))
            self.input_preprocessors.append(
                InputPreprocessor(new_model_config, self.tokenizers[-1]))

    def generate(
        self,
        prompt: Optional[PromptType] = None,
        sampling_params: Optional[SamplingParams] = None,
        request_id: Optional[str] = None,
        model: Optional[str] = None,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
        *,
        inputs: Optional[PromptType] = None  # DEPRECATED
    ) -> AsyncGenerator[RequestOutput, None]:
        """Generate outputs for a request.

        Generate outputs for a request. This method is a coroutine. It adds the
        request into the waiting queue of the LLMEngine and streams the outputs
        from the LLMEngine to the caller.

        Args:
            prompt: The prompt to the LLM. See :class:`~vllm.inputs.PromptType`
                for more details about the format of each input.
            sampling_params: The sampling parameters of the request.
            request_id: The unique id of the request.
            lora_request: LoRA request to use for generation, if any.
            trace_headers: OpenTelemetry trace headers.
            prompt_adapter_request: Prompt Adapter request to use
                                            for generation, if any.
            priority: Priority of the request (lower means earlier handling). 
                Any priority other than 0 will lead to an error if the 
                scheduling policy is not "priority".
        """
        if inputs is not None:
            prompt = inputs
        assert (prompt is not None and sampling_params is not None
                and request_id is not None)

        return self._process_request(
            prompt,
            sampling_params,
            request_id,
            lora_request,
            trace_headers,
            prompt_adapter_request,
            priority,
            model,
        )

    @deprecate_kwargs(
        "inputs",
        additional_message="Please use the 'prompt' parameter instead.",
    )
    def encode(
        self,
        prompt: Optional[PromptType] = None,
        pooling_params: Optional[PoolingParams] = None,
        request_id: Optional[str] = None,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        priority: int = 0,
        *,
        inputs: Optional[PromptType] = None  # DEPRECATED
    ) -> AsyncGenerator[EmbeddingRequestOutput, None]:
        """Generate outputs for a request from an embedding model.

        Generate outputs for a request. This method is a coroutine. It adds the
        request into the waiting queue of the LLMEngine and streams the outputs
        from the LLMEngine to the caller.

        Args:
            prompt: The prompt to the LLM. See :class:`~vllm.inputs.PromptType`
                for more details about the format of each input.
            pooling_params: The pooling parameters of the request.
            request_id: The unique id of the request.
            lora_request: LoRA request to use for generation, if any.
            trace_headers: OpenTelemetry trace headers.

        Yields:
            The output `EmbeddingRequestOutput` objects from the LLMEngine
            for the request.
        """
        if inputs is not None:
            prompt = inputs
        assert (prompt is not None and pooling_params is not None
                and request_id is not None)

        return cast(
            AsyncGenerator[EmbeddingRequestOutput, None],
            self._process_request(prompt,
                                  pooling_params,
                                  request_id,
                                  lora_request,
                                  trace_headers,
                                  priority=priority))

    async def _process_request(
        self,
        prompt: PromptType,
        params: Union[SamplingParams, PoolingParams],
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
        model: Optional[str] = None,
    ) -> Union[AsyncGenerator[RequestOutput, None], AsyncGenerator[
            EmbeddingRequestOutput, None]]:
        """Send an RPCGenerateRequest to the RPCServer and stream responses."""

        # If already dead, error out.
        if self._errored_with is not None:
            raise ENGINE_DEAD_ERROR(self._errored_with)

        # Constructing guided decoding logits processors is expensive, so we do
        # it here to avoid contending with cpu resources and the GIL on the
        # backend process.
        if isinstance(params, SamplingParams) and \
            params.guided_decoding is not None:
            params = await \
                build_guided_decoding_logits_processor_async(
                    sampling_params=params,
                    tokenizer=await self.get_tokenizer(lora_request),
                    default_guided_backend=(self.decoding_config.guided_decoding_backend
                        if self.decoding_config
                        else DecodingConfig.guided_decoding_backend),
                    model_config=self.model_config,
                    reasoning_backend=self.decoding_config.reasoning_backend,
                )

        # 1) Create output queue for this requests.
        queue: asyncio.Queue[Union[RequestOutput,
                                   BaseException]] = asyncio.Queue()
        self.output_queues[request_id] = queue

        try:
            # 2) Detach logits processors so that they can be pickled
            # separately (may require cloudpickle which is slower)
            if isinstance(params, SamplingParams) and params.logits_processors:
                # Defensive shallow copy
                params = copy.copy(params)
                logits_processors = params.logits_processors
                params.logits_processors = None
                lp_bytes = cloudpickle.dumps(logits_processors)
            else:
                lp_bytes = None

            request_bytes = pickle.dumps(
                RPCProcessRequest(
                    prompt=prompt,
                    params=params,
                    request_id=request_id,
                    model=model,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                    prompt_adapter_request=prompt_adapter_request,
                    priority=priority,
                ))

            # 3) Send the RPCGenerateRequest to the MQLLMEngine.
            parts = (request_bytes,
                     lp_bytes) if lp_bytes else (request_bytes, )
            await self.input_socket.send_multipart(parts, copy=False)

            # 4) Stream the RequestOutputs from the output queue. Note
            # that the output_loop pushes RequestOutput objects to this
            # queue after pulling them from the zmq socket.
            finished = False
            try:
                while not finished:
                    request_output = await queue.get()

                    if isinstance(request_output, BaseException):
                        raise request_output

                    finished = request_output.finished
                    yield request_output
            finally:
                # Request was canceled by the client.
                if not finished and not self.errored:
                    await self.abort(request_id)
        finally:
            self.output_queues.pop(request_id)
