#!/bin/bash


# PT_HPU_LAZY_MODE=1 python3 02_client_updated.py --prompt member -f 50-pages/50-pages/ -o result
# PT_HPU_LAZY_MODE=1 python3 02_client_updated.py --prompt member -f 50-pages/50-pages/ -o result
# PT_HPU_LAZY_MODE=1 python3 02_client_updated.py --prompt member -f 50-pages/50-pages/ -o result
# sleep 10
# PT_HPU_LAZY_MODE=1 python3 02_client_updated.py --prompt validate -f 50-pages/50-pages/ -o result
# PT_HPU_LAZY_MODE=1 python3 02_client_updated.py --prompt validate -f 50-pages/50-pages/ -o result
# curl -X POST http://localhost:12345/start_profile
# PT_HPU_LAZY_MODE=1 python3 02_client_updated.py --prompt validate -f 50-pages/50-pages/ -o result
# curl -X POST http://localhost:12345/stop_profile
# python3 02_client_updated.py --prompt member -f 50-pages/50-pages/ -o result


# Step 1: Encode the local image file to a base64 string
# The `-w 0` flag is crucial to prevent line wrapping
BASE64_IMAGE=$(base64 -w 0 map.jpg)

# Step 2: Create a JSON file with the payload
# Using a 'here document' (cat <<EOF) is a clean way to write multi-line text to a file
cat <<EOF > payload.json
{
    "model": "/software/stanley/benchmark-config/vllm-Internvl/vllm-fork/InternVL3-14B",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What major countries are visible on this map?"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64,${BASE64_IMAGE}"
                    }
                }
            ]
        }
    ],
    "max_tokens": 256,
    "temperature": 0
}
EOF
# curl -X POST http://localhost:12345/start_profile
# Step 3: Send the request using the payload file
# The '-d @payload.json' tells curl to read the POST data from the file
curl http://localhost:12345/v1/chat/completions \
-H "Content-Type: application/json" \
-d @payload.json
# curl -X POST http://localhost:12345/stop_profile