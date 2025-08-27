# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This file provides a disaggregated prefilling proxy demo to demonstrate an
example usage of XpYd disaggregated prefilling.
We can launch multiple vllm instances (2 for prefill and 2 for decode), and
launch this proxy demo through:
  python3 examples/online_serving/disaggregated_serving/disagg_proxy_demo.py  \
       --model $model_name  \
       --prefill localhost:8100 localhost:8101   \
       --decode localhost:8200 localhost:8201   \
       --port 8000

Note: This demo will be removed once the PDController implemented in PR 15343
(https://github.com/vllm-project/vllm/pull/15343) supports XpYd.
"""

import argparse
import ipaddress
import itertools
import json
import logging
import os
import sys
import threading
from abc import ABC, abstractmethod
from typing import Callable, Optional

import aiohttp
import requests
import uvicorn
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse, PlainTextResponse

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=60 * 60 * 60,
                                        connect=60000,
                                        sock_read=120000,
                                        sock_connect=30000)
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


class SchedulingPolicy(ABC):

    def __init__(self):
        self.lock = threading.Lock()

    @abstractmethod
    def schedule(self, cycler: itertools.cycle):
        raise NotImplementedError("Scheduling Proxy is not set.")


class Proxy:

    def __init__(
        self,
        prefill_instances: list[str],
        decode_instances: list[str],
        model: str,
        scheduling_policy: SchedulingPolicy,
        custom_create_completion: Optional[
            Callable[[Request], StreamingResponse]
        ] = None,
        custom_create_chat_completion: Optional[
            Callable[[Request], StreamingResponse]
        ] = None,
    ):
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

    def setup_routes(self):
        self.router.post(
            "/v1/completions", dependencies=[Depends(self.validate_json_request)]
        )(
            self.custom_create_completion
            if self.custom_create_completion
            else self.create_completion
        )
        self.router.post(
            "/v1/chat/completions", dependencies=[Depends(self.validate_json_request)]
        )(
            self.custom_create_chat_completion
            if self.custom_create_chat_completion
            else self.create_chat_completion
        )
        self.router.get("/status", response_class=JSONResponse)(self.get_status)
        self.router.post(
            "/instances/add", dependencies=[Depends(self.api_key_authenticate)]
        )(self.add_instance_endpoint)
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
                detail="Unsupported Media Type: Only 'application/json' is allowed",
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
            logger.warning("Unauthorized access attempt with API Key: %s", x_api_key)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden: Invalid API Key.",
            )

    async def validate_instance(self, instance: str) -> bool:
        url = f"http://{instance}/v1/models"
        try:
            async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as client:
                logger.info("Verifying %s ...", instance)
                async with client.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        if "data" in data and len(data["data"]) > 0:
                            model_cur = data["data"][0].get("id", "")
                            if model_cur == self.model:
                                logger.info("Instance: %s could be added.", instance)
                                return True
                            else:
                                logger.warning(
                                    "Mismatch model %s : %s != %s",
                                    instance,
                                    model_cur,
                                    self.model,
                                )
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
                raise HTTPException(status_code=400, detail="Invalid instance type.")
            if not instance or ":" not in instance:
                raise HTTPException(status_code=400, detail="Invalid instance format.")
            host, port_str = instance.split(":")
            try:
                if host != "localhost":
                    ipaddress.ip_address(host)
                port = int(port_str)
                if not (0 < port < 65536):
                    raise HTTPException(status_code=400, detail="Invalid port number.")
            except Exception as e:
                raise HTTPException(
                    status_code=400, detail="Invalid instance address."
                ) from e

            is_valid = await self.validate_instance(instance)
            if not is_valid:
                raise HTTPException(
                    status_code=400, detail="Instance validation failed."
                )

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

            return JSONResponse(
                content={"message": f"Added {instance} to {instance_type}_instances."}
            )
        except HTTPException as http_exc:
            raise http_exc
        except Exception as e:
            logger.error("Error in add_instance_endpoint: %s", str(e))
            raise HTTPException(status_code=500, detail=str(e)) from e

    async def forward_request(self, url, data, use_chunked=True):
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}
            try:
                async with session.post(
                    url=url, json=data, headers=headers
                ) as response:
                    if 200 <= response.status < 300 or 400 <= response.status < 500:  # noqa: E501
                        if use_chunked:
                            async for chunk_bytes in response.content.iter_chunked(  # noqa: E501
                                1024
                            ):
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
                        logger.error(
                            "Request failed with status %s: %s",
                            response.status,
                            error_content,
                        )
                        raise HTTPException(
                            status_code=response.status,
                            detail=f"Request failed with status {response.status}: "
                            f"{error_content}",
                        )
            except aiohttp.ClientError as e:
                logger.error("ClientError occurred: %s", str(e))
                raise HTTPException(
                    status_code=502,
                    detail="Bad Gateway: Error communicating with upstream server.",
                ) from e
            except Exception as e:
                logger.error("Unexpected error: %s", str(e))
                raise HTTPException(status_code=500, detail=str(e)) from e

    def schedule(self, cycler: itertools.cycle) -> str:
        return self.scheduling_policy.schedule(cycler)

    async def get_status(self):
        status = {
            "prefill_node_count": len(self.prefill_instances),
            "decode_node_count": len(self.decode_instances),
            "prefill_nodes": self.prefill_instances,
            "decode_nodes": self.decode_instances,
        }
        return status

    async def create_completion(self, raw_request: Request):
        try:
            request = await raw_request.json()

            kv_prepare_request = request.copy()
            kv_prepare_request["max_tokens"] = 1

            prefill_instance = self.schedule(self.prefill_cycler)
            try:
                async for _ in self.forward_request(
                    f"http://{prefill_instance}/v1/completions", kv_prepare_request
                ):
                    continue
            except HTTPException as http_exc:
                self.remove_instance_endpoint("prefill", prefill_instance)
                raise http_exc

            # Perform kv recv and decoding stage
            decode_instance = self.schedule(self.decode_cycler)

            try:
                generator = self.forward_request(
                    f"http://{decode_instance}/v1/completions", request
                )
            except HTTPException as http_exc:
                self.remove_instance_endpoint("decode", decode_instance)
                raise http_exc

            media_type = (
                "text/event-stream"
                if request.get("stream", False)
                else "application/json"
            )
            response = StreamingResponse(generator, media_type=media_type)
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
            if "max_completion_tokens" in kv_prepare_request:
                kv_prepare_request["max_completion_tokens"] = 1

            # prefill stage
            prefill_instance = self.schedule(self.prefill_cycler)
            try:
                async for _ in self.forward_request(
                    f"http://{prefill_instance}/v1/chat/completions", kv_prepare_request
                ):
                    continue
            except HTTPException as http_exc:
                self.remove_instance_endpoint("prefill", prefill_instance)
                raise http_exc
            # Perform kv recv and decoding stage
            decode_instance = self.schedule(self.decode_cycler)

            try:
                generator = self.forward_request(
                    "http://" + decode_instance + "/v1/chat/completions", request
                )
            except HTTPException as http_exc:
                self.remove_instance_endpoint("decode", decode_instance)
                raise http_exc

            media_type = (
                "text/event-stream"
                if request.get("stream", False)
                else "application/json"
            )
            response = StreamingResponse(content=generator, media_type=media_type)
            return response
        except Exception:
            exc_info = sys.exc_info()
            error_messages = [str(e) for e in exc_info if e]
            print("Error occurred in disagg proxy server")
            print(error_messages)
            return StreamingResponse(
                content=iter(error_messages), media_type="text/event-stream"
            )

    def remove_instance_endpoint(self, instance_type, instance):
        with self.scheduling_policy.lock:
            if (instance_type == "decode"
                    and instance in self.decode_instances):
                self.decode_instances.remove(instance)
                self.decode_cycler = itertools.cycle(self.decode_instances)
            if (instance_type == "prefill"
                    and instance in self.prefill_instances):
                self.prefill_instances.remove(instance)
                self.prefill_cycler = itertools.cycle(self.decode_instances)


class RoundRobinSchedulingPolicy(SchedulingPolicy):

    def __init__(self):
        super().__init__()

    def safe_next(self, cycler: itertools.cycle):
        with self.lock:
            return next(cycler)

    def schedule(self, cycler: itertools.cycle) -> str:
        return self.safe_next(cycler)


class ProxyServer:
    def __init__(
        self,
        args: argparse.Namespace,
        scheduling_policy: Optional[SchedulingPolicy] = None,
        create_completion: Optional[Callable[[Request], StreamingResponse]] = None,
        create_chat_completion: Optional[Callable[[Request], StreamingResponse]] = None,
    ):
        self.validate_parsed_serve_args(args)
        self.port = args.port
        self.proxy_instance = Proxy(
            prefill_instances=[] if args.prefill is None else args.prefill,
            decode_instances=[] if args.decode is None else args.decode,
            model=args.model,
            scheduling_policy=(
                scheduling_policy
                if scheduling_policy is not None
                else RoundRobinSchedulingPolicy()
            ),
            custom_create_completion=create_completion,
            custom_create_chat_completion=create_chat_completion,
        )

    def validate_parsed_serve_args(self, args: argparse.Namespace):
        if not args.prefill:
            raise ValueError("Please specify at least one prefill node.")
        if not args.decode:
            raise ValueError("Please specify at least one decode node.")
        self.validate_instances(args.prefill)
        self.validate_instances(args.decode)
        self.verify_model_config(args.prefill, args.model)
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
                    raise ValueError(f"Invalid port number in instance: {instance}")
            except Exception as e:
                raise ValueError(f"Invalid instance {instance}: {str(e)}") from e

    def verify_model_config(self, instances: list, model: str) -> None:
        model_suffix = model.split("/")[-1]
        for instance in instances:
            try:
                response = requests.get(f"http://{instance}/v1/models")
                if response.status_code == 200:
                    model_cur = response.json()["data"][0]["id"]
                    model_cur_suffix = model_cur.split("/")[-1]
                    if model_cur_suffix != model_suffix:
                        raise ValueError(
                            f"{instance} serves a different model: "
                            f"{model_cur} != {model}"
                        )
                else:
                    raise ValueError(f"Cannot get model id from {instance}!")
            except requests.RequestException as e:
                raise ValueError(
                    f"Error communicating with {instance}: {str(e)}"
                ) from e

    def run_server(self):
        app = FastAPI()
        app.include_router(self.proxy_instance.router)
        config = uvicorn.Config(app, host="0.0.0.0", port=self.port, loop="uvloop")
        server = uvicorn.Server(config)
        server.run()

def parse_args():
    # Todo: allow more config
    parser = argparse.ArgumentParser("vLLM disaggregated proxy server.")
    parser.add_argument("--model", "-m", type=str, required=True, help="Model name")

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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    proxy_server = ProxyServer(args=args)
    proxy_server.run_server()

