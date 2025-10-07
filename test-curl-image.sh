#!/bin/bash

# This script sends a multimodal request to the vLLM OpenAI-compatible endpoint.
# It helps to verify if the server can handle list-based content in messages.

# --- Configuration ---
HOST="localhost"
PORT="12345"
API_URL="http://${HOST}:${PORT}/v1/chat/completions"
MODEL_NAME="OpenGVLab/InternVL3-14B"

# --- Payload ---
# A tiny 1x1 red pixel PNG, Base64 encoded. This makes the script self-contained.
BASE64_IMAGE="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/wcAAwAB/epv2AAAAABJRU5ErkJggg=="

echo "------------------------------------"
echo " Sending Multimodal Request to vLLM "
echo "------------------------------------"
echo "Endpoint: ${API_URL}"
echo "Model: ${MODEL_NAME}"
echo ""

# We use a 'here document' (<<EOF) to create the JSON payload cleanly.
# The `curl` command reads this payload from standard input via `-d @-`.
curl "${API_URL}" \
    -X POST \
    -H "Content-Type: application/json" \
    -d @- <<EOF
{
  "model": "${MODEL_NAME}",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "What color is this tiny image?"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/png;base64,${BASE64_IMAGE}"
          }
        }
      ]
    }
  ],
  "max_tokens": 50,
  "temperature": 0.1
}
EOF

# Add a newline at the end for cleaner terminal output
echo ""
echo "------------------------------------"
echo "         Request Complete         "
echo "------------------------------------"
