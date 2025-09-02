# SPDX-License-Identifier: Apache-2.0
import argparse
import ipaddress
import itertools
import json
import logging
import os
import sys
import threading
import time
from abc import ABC, abstractmethod
from typing import Callable, Optional

import aiohttp
import requests
import uvicorn
from colorlog.escape_codes import escape_codes
from fastapi import (APIRouter, Depends, FastAPI, Header, HTTPException,
                     Request, status)
from fastapi.responses import JSONResponse, StreamingResponse, PlainTextResponse
from transformers import AutoTokenizer

formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s",
                              "%Y-%m-%d %H:%M:%S")
handler = logging.StreamHandler()
handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False


def log_info_blue(msg):
    logger.info("%s%s%s", escape_codes['cyan'], msg, escape_codes['reset'])


def log_info_green(msg):
    logger.info("%s%s%s", escape_codes['green'], msg, escape_codes['reset'])


def log_info_yellow(msg):
    logger.info("%s%s%s", escape_codes['yellow'], msg, escape_codes['reset'])


def log_info_red(msg):
    logger.info("%s%s%s", escape_codes['red'], msg, escape_codes['reset'])


AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=60 * 60 * 60,
                                        connect=60000,
                                        sock_read=120000,
                                        sock_connect=30000)


async def P_first_token_generator(generator_p,
                                  generator_d,
                                  callback_owner=None,
                                  prefill_instance: str = None,
                                  decode_instance: str = None,
                                  req_len: int = None):
    first_decode = True
    async for chunk in generator_p:
        yield chunk
    if callback_owner and hasattr(callback_owner, "on_done"):
        callback_owner.on_done(prefill_instance=prefill_instance,
                               req_len=req_len)

    async for chunk in generator_d:
        if first_decode:
            first_decode = False
            continue
        yield chunk
    if callback_owner and hasattr(callback_owner, "on_done"):
        callback_owner.on_done(decode_instance=decode_instance,
                               req_len=req_len)


async def D_first_token_generator(generator_p,
                                  generator_d,
                                  callback_owner=None,
                                  prefill_instance: str = None,
                                  decode_instance: str = None,
                                  req_len: int = None):
    async for _ in generator_p:
        continue
    if callback_owner and hasattr(callback_owner, "on_done"):
        callback_owner.on_done(prefill_instance=prefill_instance,
                               req_len=req_len)

    async for chunk in generator_d:
        yield chunk
    if callback_owner and hasattr(callback_owner, "on_done"):
        callback_owner.on_done(decode_instance=decode_instance,
                               req_len=req_len)


class SchedulingPolicy(ABC):

    def __init__(self):
        self.lock = threading.Lock()

    @abstractmethod
    def schedule(self, cycler: itertools.cycle):
        raise NotImplementedError("Scheduling Proxy is not set.")


class Proxy:

    def __init__(self,
                 prefill_instances: list[str],
                 decode_instances: list[str],
                 model: str,
                 scheduling_policy: SchedulingPolicy,
                 custom_create_completion: Optional[Callable[
                     [Request], StreamingResponse]] = None,
                 custom_create_chat_completion: Optional[Callable[
                     [Request], StreamingResponse]] = None,
                 generator_on_p_node: bool = False):
        self.prefill_instances = prefill_instances
        self.decode_instances = decode_instances
        self.prefill_cycler = itertools.cycle(prefill_instances)
        self.decode_cycler = itertools.cycle(decode_instances)
        self.model = model
        self.scheduling_policy = scheduling_policy
        self.custom_create_completion = custom_create_completion
        self.custom_create_chat_completion = custom_create_chat_completion
        self.router = APIRouter()
        self.setup_routes()
        self.generator = (P_first_token_generator
                          if generator_on_p_node else D_first_token_generator)
        self.tokenizer = AutoTokenizer.from_pretrained(model)

    def on_done(self,
                prefill_instance: str = None,
                decode_instance: str = None,
                req_len: int = None):
        self.schedule_completion(prefill_instance,
                                 decode_instance,
                                 req_len=req_len)

    def setup_routes(self):
        self.router.post(
            "/v1/completions",
            dependencies=[
                Depends(self.validate_json_request)
            ])(self.custom_create_completion if self.
               custom_create_completion else self.create_completion)
        self.router.post(
            "/v1/chat/completions",
            dependencies=[
                Depends(self.validate_json_request)
            ])(self.custom_create_chat_completion if self.
               custom_create_chat_completion else self.create_chat_completion)
        self.router.get("/status",
                        response_class=JSONResponse)(self.get_status)
        self.router.post("/instances/add",
                         dependencies=[Depends(self.api_key_authenticate)
                                       ])(self.add_instance_endpoint)
        self.router.get("/health", response_class=PlainTextResponse)(self.get_health)
        self.router.get("/ping", response_class=PlainTextResponse)(self.get_ping)
        self.router.post("/ping", response_class=PlainTextResponse)(self.get_ping)
        self.router.post("/tokenize", response_class=JSONResponse)(self.post_tokenize)
        self.router.post("/detokenize", response_class=JSONResponse)(self.post_detokenize)
        self.router.get("/v1/models", response_class=JSONResponse)(self.get_models)
        self.router.get("/version", response_class=JSONResponse)(self.get_version)
        self.router.post("/v1/embeddings", response_class=JSONResponse)(self.post_embeddings)
        self.router.post("/pooling", response_class=JSONResponse)(self.post_pooling)
        self.router.post("/score", response_class=JSONResponse)(self.post_score)
        self.router.post("/v1/score", response_class=JSONResponse)(self.post_scorev1)
        self.router.post("/rerank", response_class=JSONResponse)(self.post_rerank)
        self.router.post("/v1/rerank", response_class=JSONResponse)(self.post_rerankv1)
        self.router.post("/v2/rerank", response_class=JSONResponse)(self.post_rerankv2)
        self.router.post("/invocations", response_class=JSONResponse)(self.post_invocations)

    async def get_from_instance(self, path: str, is_full_instancelist: int = 0):
        if not self.prefill_instances:
            return JSONResponse(content={"error": "No instances available"}, status_code=500)

        if is_full_instancelist == 0:
            instances = [self.prefill_instances[0]]
        else:
            instances = self.prefill_instances + self.decode_instances

        results = {}
        async with aiohttp.ClientSession() as session:
            for inst in instances:
                url = f"http://{inst}{path}"
                try:
                    async with session.get(url) as resp:
                        try:
                            data = await resp.json()
                            dtype = "json"
                        except aiohttp.ContentTypeError:
                            data = await resp.text()
                            dtype = "text"
                        results[inst] = {
                            "status": resp.status,
                            "type": dtype,
                            "data": data
                        }
                except Exception as e:
                    results[inst] = {
                        "status": 500,
                        "error": str(e)
                    }
                    print(f"Failed to fetch {url}: {e}, continue...")

        return JSONResponse(content=results, status_code=200)

    async def get_version(self):
        return await self.get_from_instance("/version")

    async def get_models(self):
        return await self.get_from_instance("/v1/models")

    async def get_health(self):
        return await self.get_from_instance("/health", is_full_instancelist=1)

    async def get_ping(self):
        return await self.get_from_instance("/ping", is_full_instancelist=1)

    async def post_to_instance(
        self,
        request: Request,
        path: str,
        json_template: dict
    ):
        body = await request.json()

        missing = [k for k in json_template.keys() if k not in body]
        if missing:
            return JSONResponse(
                {"error": f"Missing required fields: {', '.join(missing)}"},
                status_code=400
            )

        payload = json_template.copy()
        payload.update(body)

        url = f"http://{self.prefill_instances[0]}{path}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as resp:
                    try:
                        content = await resp.json()
                    except aiohttp.ContentTypeError:
                        content = {"raw": await resp.text()}
                    return JSONResponse(content, status_code=resp.status)
        except Exception as e:
            return JSONResponse(
                {"error": f"Failed to fetch {url}, reason: {str(e)}"},
                status_code=500
            )

    async def post_detokenize(self, request: Request):
        json_template = {"model": "", "tokens": []}
        return await self.post_to_instance(request, "/detokenize", json_template)

    async def post_tokenize(self, request: Request):
        json_template = {"model": "", "prompt": ""}
        return await self.post_to_instance(request, "/tokenize", json_template)

    async def post_embeddings(self, request: Request):
        json_template = {"model": "", "input": ""}
        return await self.post_to_instance(request, "/v1/embeddings", json_template)

    async def post_pooling(self, request: Request):
        json_template = {"model": "", "messages": ""}
        return await self.post_to_instance(request, "/pooling", json_template)

    async def post_score(self, request: Request):
        json_template = {"model": "", "text_1": "", "text_2": "", "predictions": ""}
        return await self.post_to_instance(request, "/score", json_template)

    async def post_scorev1(self, request: Request):
        json_template = {"model": "", "text_1": "", "text_2": "", "predictions": ""}
        return await self.post_to_instance(request, "/v1/score", json_template)

    async def post_rerank(self, request: Request):
        json_template = {"model": "", "query": "", "documents": ""}
        return await self.post_to_instance(request, "/rerank", json_template)

    async def post_rerankv1(self, request: Request):
        json_template = {"model": "", "query": "", "documents": ""}
        return await self.post_to_instance(request, "/v1/rerank", json_template)

    async def post_rerankv2(self, request: Request):
        json_template = {"model": "", "query": "", "documents": ""}
        return await self.post_to_instance(request, "/v2/rerank", json_template)

    async def post_invocations(self, request: Request):
        json_template = {"model": "", "prompt": ""}
        return await self.post_to_instance(request, "/invocations", json_template)

    async def validate_json_request(self, raw_request: Request):
        content_type = raw_request.headers.get("content-type", "").lower()
        if content_type != "application/json":
            raise HTTPException(
                status_code=415,
                detail=
                "Unsupported Media Type: Only 'application/json' is allowed",
            )

    def api_key_authenticate(self, x_api_key: str = Header(...)):
        expected_api_key = os.environ.get("ADMIN_API_KEY")
        if not expected_api_key:
            logger.error("ADMIN_API_KEY is not set in the environment.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Server configuration error.",
            )
        if x_api_key != expected_api_key:
            logger.warning("Unauthorized access attempt with API Key: %s",
                           x_api_key)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden: Invalid API Key.",
            )

    async def validate_instance(self, instance: str) -> bool:
        url = f"http://{instance}/v1/models"
        try:
            async with aiohttp.ClientSession(
                    timeout=AIOHTTP_TIMEOUT) as client:
                logger.info("Verifying %s ...", instance)
                async with client.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        if "data" in data and len(data["data"]) > 0:
                            model_cur = data["data"][0].get("id", "")
                            if model_cur == self.model:
                                logger.info("Instance: %s could be added.",
                                            instance)
                                return True
                            else:
                                logger.warning("Mismatch model %s : %s != %s",
                                               instance, model_cur, self.model)
                                return False
                        else:
                            return False
                    else:
                        return False
        except aiohttp.ClientError as e:
            logger.error(str(e))
            return False
        except Exception as e:
            logger.error(str(e))
            return False

    async def add_instance_endpoint(self, request: Request):
        try:
            data = await request.json()
            logger.warning(str(data))
            instance_type = data.get("type")
            instance = data.get("instance")
            if instance_type not in ["prefill", "decode"]:
                raise HTTPException(status_code=400,
                                    detail="Invalid instance type.")
            if not instance or ":" not in instance:
                raise HTTPException(status_code=400,
                                    detail="Invalid instance format.")
            host, port_str = instance.split(":")
            try:
                if host != "localhost":
                    ipaddress.ip_address(host)
                port = int(port_str)
                if not (0 < port < 65536):
                    raise HTTPException(status_code=400,
                                        detail="Invalid port number.")
            except Exception as e:
                raise HTTPException(status_code=400,
                                    detail="Invalid instance address.") from e

            is_valid = await self.validate_instance(instance)
            if not is_valid:
                raise HTTPException(status_code=400,
                                    detail="Instance validation failed.")

            if instance_type == "prefill":
                with self.scheduling_policy.lock:
                    if instance not in self.prefill_instances:
                        self.prefill_instances.append(instance)
                        self.prefill_cycler = itertools.cycle(
                            self.prefill_instances)
                    else:
                        raise HTTPException(status_code=400,
                                            detail="Instance already exists.")
            else:
                with self.scheduling_policy.lock:
                    if instance not in self.decode_instances:
                        self.decode_instances.append(instance)
                        self.decode_cycler = itertools.cycle(
                            self.decode_instances)
                    else:
                        raise HTTPException(status_code=400,
                                            detail="Instance already exists.")

            return JSONResponse(content={
                "message":
                f"Added {instance} to {instance_type}_instances."
            })
        except HTTPException as http_exc:
            raise http_exc
        except Exception as e:
            logger.error("Error in add_instance_endpoint: %s", str(e))
            raise HTTPException(status_code=500, detail=str(e)) from e

    async def forward_request(self, url, data, use_chunked=True):
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            headers = {
                "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"
            }
            try:
                async with session.post(url=url, json=data,
                                        headers=headers) as response:
                    if 200 <= response.status < 300 or 400 <= response.status < 500:  # noqa: E501
                        if use_chunked:
                            async for chunk_bytes in response.content.iter_chunked(  # noqa: E501
                                    1024):
                                yield chunk_bytes
                        else:
                            content = await response.read()
                            yield content
                    else:
                        error_content = await response.text()
                        try:
                            error_content = json.loads(error_content)
                        except json.JSONDecodeError:
                            error_content = error_content
                        logger.error("Request failed with status %s: %s",
                                     response.status, error_content)
                        raise HTTPException(
                            status_code=response.status,
                            detail=
                            f"Request failed with status {response.status}: "
                            f"{error_content}",
                        )
            except aiohttp.ClientError as e:
                logger.error("ClientError occurred: %s", str(e))
                raise HTTPException(
                    status_code=502,
                    detail=
                    "Bad Gateway: Error communicating with upstream server.",
                ) from e
            except Exception as e:
                logger.error("Unexpected error: %s", str(e))
                raise HTTPException(status_code=500, detail=str(e)) from e

    def schedule(self,
                 cycler: itertools.cycle,
                 is_prompt: int = None,
                 request_len: Optional[int] = None) -> str:
        return self.scheduling_policy.schedule(cycler, is_prompt, request_len)

    def schedule_completion(self,
                            prefill_instance: str = None,
                            decode_instance: str = None,
                            req_len: int = None):
        self.scheduling_policy.schedule_completion(
            prefill_instance=prefill_instance,
            decode_instance=decode_instance,
            req_len=req_len)

    async def get_status(self):
        status = {
            "prefill_node_count": len(self.prefill_instances),
            "decode_node_count": len(self.decode_instances),
            "prefill_nodes": self.prefill_instances,
            "decode_nodes": self.decode_instances,
        }
        return status

    def get_total_token_length(self, prompt):
        fake_len = 100
        if isinstance(prompt, str):
            return len(self.tokenizer(prompt)["input_ids"])
        elif isinstance(prompt, list):
            if all(isinstance(p, str) for p in prompt):
                return sum(len(self.tokenizer(p)["input_ids"]) for p in prompt)
            elif (all(isinstance(p, list) and 
                all(isinstance(x, int) for x in p) for p in prompt)):
                # Already tokenized
                return sum(len(p) for p in prompt)
            else:
                logger.error(
                    "Unsupported prompt format: %s / nested types. Value: %r",
                    type(prompt), prompt
                )
                return fake_len
        else:
            logger.error("Unsupported prompt type: %s", type(prompt))
            return fake_len

    async def create_completion(self, raw_request: Request):
        try:
            request = await raw_request.json()

            if len(self.prefill_instances) > 0:
                kv_prepare_request = request.copy()
                kv_prepare_request["max_tokens"] = 1

                start_time = time.time()
                prompt = kv_prepare_request.get("prompt")
                total_length = self.get_total_token_length(prompt)
                end_time = time.time()

                log_info_green(
                    f"create_completion -- prompt length: {total_length}, "
                    f"tokenizer took "
                    f"{(end_time - start_time) * 1000:.2f} ms")
                prefill_instance = self.schedule(self.prefill_cycler,
                                                 is_prompt=True,
                                                 request_len=total_length)
                value = b''
                try:
                    async for chunk in self.forward_request(
                            f"http://{prefill_instance}/v1/completions",
                            kv_prepare_request):
                        value += chunk
                except HTTPException as http_exc:
                    self.remove_instance_endpoint("prefill", prefill_instance)
                    raise http_exc

            # Perform kv recv and decoding stage
            decode_instance = self.schedule(self.decode_cycler,
                                            is_prompt=False,
                                            request_len=total_length)
            value = value.strip().decode("utf-8").removesuffix(
                "data: [DONE]").encode("utf-8")

            async def streaming_response(value):
                if value:
                    yield value
                else:
                    yield b""

            generator_p = streaming_response(value)
            try:
                generator_d = self.forward_request(
                    f"http://{decode_instance}/v1/completions", request)
            except HTTPException as http_exc:
                self.remove_instance_endpoint("decode", decode_instance)
                raise http_exc

            if request.get("stream", False):
                generator_class = self.generator
            else:
                # For stream=False request, cannot use P first token
                generator_class = D_first_token_generator
            final_generator = generator_class(generator_p,
                                              generator_d,
                                              self,
                                              prefill_instance,
                                              decode_instance,
                                              req_len=total_length)
            media_type = (
                "text/event-stream"
                if request.get("stream", False)
                else "application/json"
            )
            response = StreamingResponse(final_generator,
                                         media_type=media_type)
            return response
        except Exception:
            import sys

            exc_info = sys.exc_info()
            print("Error occurred in disagg proxy server")
            print(exc_info)

    async def create_chat_completion(self, raw_request: Request):
        try:
            request = await raw_request.json()

            # add params to request
            kv_prepare_request = request.copy()
            kv_prepare_request["max_tokens"] = 1

            start_time = time.time()
            # prefill stage
            total_length = sum(
                self.get_total_token_length(msg['content'])
                for msg in kv_prepare_request['messages'])
            end_time = time.time()
            log_info_green(
                f"create_chat_completion -- prompt length: {total_length}, "
                f"tokenizer took "
                f"{(end_time - start_time) * 1000:.2f} ms")

            prefill_instance = self.schedule(self.prefill_cycler,
                                             is_prompt=True,
                                             request_len=total_length)

            value = b''
            try:
                async for chunk in self.forward_request(
                        f"http://{prefill_instance}/v1/chat/completions",
                        kv_prepare_request):
                    value += chunk
            except HTTPException as http_exc:
                self.remove_instance_endpoint("prefill", prefill_instance)
                raise http_exc
            # Perform kv recv and decoding stage
            decode_instance = self.schedule(self.decode_cycler,
                                            is_prompt=False,
                                            request_len=total_length)
            value = value.strip().decode("utf-8").removesuffix(
                "data: [DONE]").encode("utf-8")

            async def streaming_response(value):
                if value:
                    yield value
                else:
                    yield b""

            generator_p = streaming_response(value)
            try:
                generator_d = self.forward_request(
                    "http://" + decode_instance + "/v1/chat/completions",
                    request)
            except HTTPException as http_exc:
                self.remove_instance_endpoint("decode", decode_instance)
                raise http_exc

            if request.get("stream", False):
                generator_class = self.generator
            else:
                # For stream=False request, cannot use P first token
                generator_class = D_first_token_generator
            final_generator = generator_class(generator_p,
                                              generator_d,
                                              self,
                                              prefill_instance,
                                              decode_instance,
                                              req_len=total_length)
            media_type = (
                "text/event-stream"
                if request.get("stream", False)
                else "application/json"
            )
            response = StreamingResponse(final_generator,
                                         media_type=media_type)
            return response
        except Exception:
            exc_info = sys.exc_info()
            error_messages = [str(e) for e in exc_info if e]
            print("Error occurred in disagg proxy server")
            print(error_messages)
            return StreamingResponse(content=iter(error_messages),
                                     media_type="application/json")

    def remove_instance_endpoint(self, instance_type, instance):
        return

class RoundRobinSchedulingPolicy(SchedulingPolicy):

    def __init__(self):
        print("RoundRobinSchedulingPolicy")
        super().__init__()

    def safe_next(self, cycler: itertools.cycle):
        with self.lock:
            return next(cycler)

    def schedule(self,
                 cycler: itertools.cycle,
                 request: Optional[dict[str, any]] = None) -> str:
        return self.safe_next(cycler)


class LoadBalancedScheduler(SchedulingPolicy):

    def __init__(self, prefill_instances: list[str],
                 decode_instances: list[str]):
        self.prefill_utils_counter = [0] * len(prefill_instances)
        self.prefill_bs_counter = [0] * len(prefill_instances)
        self.decode_kv_utils_counter = [0] * len(
            decode_instances)  #KV cache utils
        self.decode_bs_counter = [0] * len(decode_instances)

        self.prefill_instances = prefill_instances
        self.decode_instances = decode_instances
        print(" LoadBalancedScheduler, prefill/decode instance is = ",
              len(self.prefill_bs_counter), len(self.decode_bs_counter))
        print(" LoadBalancedScheduler, self.prefill_instances =",
              self.prefill_instances)
        print(" LoadBalancedScheduler, self.decode_instances =",
              self.decode_instances)
        self.prefill_schedule_index = 0
        self.prefill_schedule_completion_index = 0
        self.decode_schedule_index = 0
        self.decode_schedule_completion_index = 0

        super().__init__()

    def schedule(self,
                 cycler: itertools.cycle,
                 is_prompt: int = None,
                 request_len: Optional[int] = None) -> str:
        with self.lock:
            if is_prompt:
                min_value = min(self.prefill_utils_counter)
                min_index = self.prefill_utils_counter.index(min_value)
                self.prefill_bs_counter[min_index] += 1
                self.prefill_utils_counter[min_index] += request_len
                self.prefill_schedule_index += 1
                log_info_yellow(
                    f"<schedule prefill {self.prefill_schedule_index}> "
                    f"instance = {min_index}, min_tokens = {min_value}")
                return self.prefill_instances[min_index]
            else:
                min_value = min(self.decode_bs_counter)

                if min_value == 0:
                    min_index = self.decode_bs_counter.index(min_value)
                else:
                    min_indices = [
                        i for i, val in enumerate(self.decode_bs_counter)
                        if val == min_value
                    ]
                    min_index = min(
                        min_indices,
                        key=lambda i: self.decode_kv_utils_counter[i])

                self.decode_bs_counter[min_index] += 1
                self.decode_kv_utils_counter[min_index] += request_len
                self.decode_schedule_index += 1
                log_info_blue(
                    f"<schedule decode {self.decode_schedule_index}> "
                    f"instance = {min_index}, min_batch = {min_value}")
                log_info_blue(f"<schedule decode> "
                              f"decode_bs_counter: {self.decode_bs_counter}")
                log_info_blue(
                    f"<schedule decode> "
                    f"decode_kv_utils_counter: {self.decode_kv_utils_counter}")

                return self.decode_instances[min_index]

    def schedule_completion(self,
                            prefill_instance: str = None,
                            decode_instance: str = None,
                            req_len: int = None):
        with self.lock:
            if prefill_instance:
                index = self.prefill_instances.index(prefill_instance)
                self.prefill_schedule_completion_index += 1
                log_info_yellow(f"<Prefill completed "
                                f"{self.prefill_schedule_completion_index}> "
                                f"instance = {index}, req_len={req_len}")

                self.prefill_bs_counter[index] -= 1
                all_zero = True
                for index, _ in enumerate(self.prefill_instances):
                    if self.prefill_bs_counter[index] != 0:
                        all_zero = False
                        break
                if all_zero:
                    log_info_red("<Prefill in idle state>")
                    for index, _ in enumerate(self.prefill_instances):
                        self.prefill_utils_counter[index] = 0
                else:
                    index = self.prefill_instances.index(prefill_instance)
                    self.prefill_utils_counter[index] -= req_len

            if decode_instance:
                index = self.decode_instances.index(decode_instance)
                self.decode_schedule_completion_index += 1
                log_info_blue(f"<Decode completed "
                              f"{self.decode_schedule_completion_index}> "
                              f"instance = {index}, req_len={req_len}")

                self.decode_bs_counter[index] -= 1
                all_zero = True
                for index, _ in enumerate(self.decode_instances):
                    if self.decode_bs_counter[index] != 0:
                        all_zero = False
                        break
                if all_zero:
                    log_info_red("<Decode in idle state>")
                    self.decode_kv_utils_counter = [0] * len(
                        self.decode_instances)
                else:
                    index = self.decode_instances.index(decode_instance)
                    self.decode_kv_utils_counter[index] -= req_len
                    log_info_blue(
                        f"<schedule_completion decode> "
                        f"decode_bs_counter: {self.decode_bs_counter}")
                    log_info_blue(f"<schedule_completion decode> "
                                  f"decode_kv_utils_counter: "
                                  f"{self.decode_kv_utils_counter}")


class ProxyServer:

    def __init__(
        self,
        args: argparse.Namespace,
        scheduling_policy: Optional[SchedulingPolicy] = None,
        create_completion: Optional[Callable[[Request],
                                             StreamingResponse]] = None,
        create_chat_completion: Optional[Callable[[Request],
                                                  StreamingResponse]] = None,
    ):
        self.validate_parsed_serve_args(args)
        self.port = args.port
        self.proxy_instance = Proxy(
            prefill_instances=[] if args.prefill is None else args.prefill,
            decode_instances=[] if args.decode is None else args.decode,
            model=args.model,
            scheduling_policy=(scheduling_policy(args.prefill, args.decode)
                               if scheduling_policy is not None else
                               RoundRobinSchedulingPolicy()),
            custom_create_completion=create_completion,
            custom_create_chat_completion=create_chat_completion,
            generator_on_p_node=args.generator_on_p_node,
        )

    def validate_parsed_serve_args(self, args: argparse.Namespace):
        # if not args.prefill:
        #     raise ValueError("Please specify at least one prefill node.")
        if not args.decode:
            raise ValueError("Please specify at least one decode node.")
        if args.prefill:
            self.validate_instances(args.prefill)
            self.verify_model_config(args.prefill, args.model)
        self.validate_instances(args.decode)
        self.verify_model_config(args.decode, args.model)

    def validate_instances(self, instances: list):
        for instance in instances:
            if len(instance.split(":")) != 2:
                raise ValueError(f"Invalid instance format: {instance}")
            host, port = instance.split(":")
            try:
                if host != "localhost":
                    ipaddress.ip_address(host)
                port = int(port)
                if not (0 < port < 65536):
                    raise ValueError(
                        f"Invalid port number in instance: {instance}")
            except Exception as e:
                raise ValueError(
                    f"Invalid instance {instance}: {str(e)}") from e

    def verify_model_config(self, instances: list, model: str) -> None:
        for instance in instances:
            try:
                response = requests.get(f"http://{instance}/v1/models")
                if response.status_code == 200:
                    model_cur = response.json()["data"][0]["id"]
                    if model_cur != model:
                        raise ValueError(
                            f"{instance} serves a different model: "
                            f"{model_cur} != {model}")
                else:
                    raise ValueError(f"Cannot get model id from {instance}!")
            except requests.RequestException as e:
                raise ValueError(
                    f"Error communicating with {instance}: {str(e)}") from e

    def run_server(self):
        app = FastAPI()
        app.include_router(self.proxy_instance.router)
        config = uvicorn.Config(app,
                                host="0.0.0.0",
                                port=self.port,
                                loop="uvloop")
        server = uvicorn.Server(config)
        server.run()


if __name__ == "__main__":
    # Todo: allow more config
    parser = argparse.ArgumentParser("vLLM disaggregated proxy server.")
    parser.add_argument("--model",
                        "-m",
                        type=str,
                        required=True,
                        help="Model name")

    parser.add_argument(
        "--prefill",
        "-p",
        type=str,
        nargs="+",
        help="List of prefill node URLs (host:port)",
    )

    parser.add_argument(
        "--decode",
        "-d",
        type=str,
        nargs="+",
        help="List of decode node URLs (host:port)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port number",
    )

    parser.add_argument(
        "--generator_on_p_node",
        action="store_true",
        help="generate first token on P node or D node",
    )

    parser.add_argument(
        "--roundrobin",
        action="store_true",
        help="Use Round Robin scheduling for load balancing",
    )
    args = parser.parse_args()
    if args.roundrobin:
        proxy_server = ProxyServer(args=args)
    else:
        proxy_server = ProxyServer(args=args,
                                   scheduling_policy=LoadBalancedScheduler)
    proxy_server.run_server()
