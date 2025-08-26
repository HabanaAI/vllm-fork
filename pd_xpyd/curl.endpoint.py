import requests

# vLLM API 基础 URL
BASE_URL = "http://10.239.129.9:8868"
MODEL_PATH = "/mnt/disk2/hf_models/DeepSeek-R1-G2/"
# Endpoint 列表
endpoints = [
    ("/openapi.json", ["GET", "HEAD"]),
    ("/docs", ["GET", "HEAD"]),
    ("/docs/oauth2-redirect", ["GET", "HEAD"]),
    ("/redoc", ["GET", "HEAD"]),
    ("/health", ["GET"]),
    ("/ping", ["GET", "POST"]),
    ("/tokenize", ["POST"]),
    ("/detokenize", ["POST"]),
    ("/v1/models", ["GET"]),
    ("/version", ["GET"]),
    ("/v1/chat/completions", ["POST"]),
    ("/v1/completions", ["POST"]),
    ("/v1/embeddings", ["POST"]),
    ("/pooling", ["POST"]),
    ("/score", ["POST"]),
    ("/v1/score", ["POST"]),
#    ("/v1/audio/transcriptions", ["POST"]),
    ("/rerank", ["POST"]),
    ("/v1/rerank", ["POST"]),
    ("/v2/rerank", ["POST"]),
    ("/invocations", ["POST"]),
]

sample_payload = {
    "/ping": {},
    "/tokenize": {"model": MODEL_PATH, "prompt": "Hello world"},
    "/detokenize": {"model": MODEL_PATH, "tokens": [15496, 995]},
    "/v1/chat/completions": {"model": MODEL_PATH, "messages": [{"role": "user", "content": "Hello"}]},
    "/v1/completions": {"model": MODEL_PATH, "prompt": "The capital of France is:", "max_tokens": 50},
    "/v1/embeddings": {"model": MODEL_PATH, "input": "Hello"},
    "/pooling": {"model": MODEL_PATH, "messages": [{"role": "user", "content": "Hello world"}]},
    "/score": {"model": MODEL_PATH, "text_1": "Hello world", "text_2": "Hi there", "predictions": [0.1, 0.9]},
    "/v1/score": {"model": MODEL_PATH, "text_1": "Hello world", "text_2": "Hi there", "predictions": [0.1, 0.9]},
    "/v1/audio/transcriptions": {"audio": "dummy_audio_data"},
    "/rerank": {"model": MODEL_PATH, "query": "hello", "documents": ["world"]},
    "/v1/rerank": {"model": MODEL_PATH, "query": "hello", "documents": ["world"]},
    "/v2/rerank": {"model": MODEL_PATH, "query": "hello", "documents": ["world"]},
    "/invocations": {"model": MODEL_PATH, "prompt": "Hello"}
}

import json

def test_endpoint(route, methods):
    for method in methods:
        url = BASE_URL + route
        try:
            if method == "POST":
                payload = sample_payload.get(route, {})
                resp = requests.post(url, json=payload, timeout=10000)
            else:
                resp = requests.request(method, url, timeout=5)

            print(f"[{method}] {route} -> Status: {resp.status_code}")

            # 仅在 GET / POST 且返回 JSON 时解析
            if method in ["GET", "POST"] and resp.headers.get("content-type", "").startswith("application/json"):
                try:
                    data = resp.json()
                    print("JSON keys:", list(data.keys()))
                    # 打印完整 JSON 内容
                    print(json.dumps(data, indent=2, ensure_ascii=False))
                except Exception as e:
                    print("JSON解析失败:", e)
            else:
                # 非 JSON 返回，打印前 200 个字符
                print(resp.text[:200])
        except Exception as e:
            print(f"[{method}] {route} -> Exception: {e}")

if __name__ == "__main__":
    for route, methods in endpoints:
        test_endpoint(route, methods)
