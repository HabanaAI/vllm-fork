import httpx
import time
import json
import random
from threading import Thread, Event
from time import sleep
import sys

HEADERS = {"Content-Type": "application/json"}

ttft_complete = Event()
t_start = time.time()
# Global list to store token arrival times
profiling = []
def get_timestamp_us():
    return time.time() * 1000000.0
def stream_tokens(url, seq_len, i):
    payload = {
        "prompt": "ha" * (seq_len//2 - 1),
        "stream": True,
        "max_tokens": 512,
        "ignore_eos": True,
        "temperature": 0,
        "top_p": 1,
        "model":"/software/data/disk10/models/DeepSeek-R1-BF16-w8afp8-static-no-ste-G2/"
    }
    print("[{:.2f}] Request {} ({}): START".format(time.time() - t_start, i, seq_len))
    start_time = get_timestamp_us()
    with httpx.Client(timeout=None) as client:
        with client.stream("POST", url, json=payload, headers=HEADERS) as response:
            if response.status_code != 200:
                print(f"Failed to connect: {response.status_code}")
                return

            event_name = "TTFT"
            idx = 0
            for line in response.iter_text():
                if not line.strip():
                    continue
                if len(line) < 15:
                    # standalone vllm returns "[DONE]" message which is not a token. In disagg-pd it gets combined with token.
                    continue
                profiling.append({"pid": seq_len, "tid": i, "ph": "X", "name":event_name, "ts": start_time, "dur": get_timestamp_us() - start_time, "args":{"idx":idx}})
                idx+=1
                if event_name == "TTFT":
                    print("[{:.2f}] Request {} ({}): First token received".format(time.time() - t_start, i, seq_len))
                    event_name = "ITL"
                    ttft_complete.set()
                start_time = get_timestamp_us()
                if i == 1 and (idx%100 == 0):
                    print("[{:.2f}] Request {} : {} tokens received".format(time.time() - t_start, i, idx))
            print("[{:.2f}] Request {} ({}): COMPLETED".format(time.time() - t_start, i, seq_len))


threads = []
url = "http://{}/v1/completions".format(sys.argv[1])
filename = "benchmark-{}.json".format(sys.argv[1].replace(":", "-"))
print(url)
print(filename)

seq_lens = [2048, 4096, 8192] * 4
#seq_lens = [512] * 3
#random.Random(1337).shuffle(seq_lens)

for i, seq_len in enumerate(seq_lens, start=1):
    t = Thread(target = stream_tokens, args = (url, seq_len, i, ))
    t.start()
    ttft_complete.wait()
    ttft_complete.clear()
    sleep(1.5)
    threads.append(t)

for t in threads:
    t.join()

with open(filename, 'w') as f:
    json.dump(profiling, f)
