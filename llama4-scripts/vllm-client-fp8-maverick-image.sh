#!/bin/bash
echo "--- Preparing the request ---"

# Step 1: Encode the image
echo "Encoding map.jpg..."
BASE64_IMAGE=$(base64 -w 0 map.jpg)
if [ -z "$BASE64_IMAGE" ]; then
    echo "Error: Failed to encode image or map.jpg is missing. Exiting."
    exit 1
fi
echo "Image encoded successfully."

# Step 2: Create the JSON payload file
echo "Creating payload.json..."
cat <<EOF > payload.json
{
    "model": "/mnt/weka/llm/Llama-4-Maverick-17B-128E-Instruct",
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
echo "payload.json created."

# Step 3: Send the request with VERBOSE output enabled
echo -e "\n--- Sending request to vLLM server ---"
# The -v flag makes curl show all communication details
# curl -X POST http://localhost:12345/start_profile
curl http://localhost:18080/v1/chat/completions \
-H "Content-Type: application/json" \
-d @payload.json
# curl -X POST http://localhost:12345/stop_profile
CURL_EXIT_CODE=$?
echo -e "\n--- Request finished (curl exit code: $CURL_EXIT_CODE) ---"

# Step 4: Clean up
echo "Cleaning up payload.json..."
rm payload.json
echo "--- Script finished ---"
