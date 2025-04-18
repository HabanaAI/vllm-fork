
import requests
import asyncio
import aiohttp
import time

model_path = "/software/data/models/DeepSeek-R1-BF16-w8afp8-static-no-ste-G2/"
# add all dp instance here
dp_server_list = ["10.112.110.51:8801", "10.112.110.51:8802"]
proxy_server_post_url = "http://127.0.0.1:8123/v1/completions"

def gen_metrics_url(ip_port):
    return "http://" + ip_port + "/metrics"
def gen_request_url(ip_port):
    return "http://" + ip_port + "/v1/completions"

full_metrics_url_list = [gen_metrics_url(dp_server) for dp_server in dp_server_list]
full_request_url_list = [gen_request_url(dp_server) for dp_server in dp_server_list]

def get_running_reqs(metrics):
    metrics = metrics.replace("\\n", "\n")
    for line in metrics.splitlines():
        if "vllm:num_requests_running{" in line:
            running_line = line
            break
    if running_line:
        running_value = running_line.split()[-1]
        print(running_value)
        
        running_value = running_value.split('.')[0] 
        running_value_int = int(float(running_value))
        print(running_value_int)
        return running_value_int
    else:
        return 0

def post_dummy_req(url):
    headers = {}
    payload = {
        "model": model_path,
        "prompt": "this is just a test prompt",
        "max_tokens": 128,
        "temperature": 0.0
    }
    response = requests.post(url, headers=headers, json=payload)
    return response

async def post_dummy_req_async(session, url, idx):
    headers = {}
    payload = {
        "model": model_path,
        "prompt": f"rank id {idx}, this is just a test prompt",
        "max_tokens": 128,
        "temperature": 0.0
    }
    async with session.post(url, headers=headers, json=payload) as response:
        return await response.text()

async def post_fun():
    async with aiohttp.ClientSession() as session:
        round = 0
        while True:
            # sleep 50 ms
            time.sleep(0.05)
            round += 1
            # print(f"this is round {round}")
            running_req_nums = []
            # fetch all instance metrics
            for full_metrics_url in full_metrics_url_list:
                response = requests.get(full_metrics_url)

                text = str(response.content)
                running_req_num = get_running_reqs(text)
                running_req_nums.append(running_req_num)
            
            # if all instance don't have running reqs, we skip
            if sum(running_req_nums) == 0:
                continue
            # each instance is running at least one req, don't need post dummy req
            if min(running_req_nums) > 0:
                continue

            # other wise, let's send request to all rank
            tasks = []
            for idx, running_req_num in enumerate(running_req_nums):
                # if running_req_num == 0:
                tasks.append(post_dummy_req_async(session, proxy_server_post_url, idx))
                print(f"posting to instance {idx}")

            responses = await asyncio.gather(*tasks)
            print(f"all task done")
            
    


if __name__ == "__main__":
    asyncio.run(post_fun())