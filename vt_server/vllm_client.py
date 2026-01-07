import argparse
import base64
import os
import sys
from openai import OpenAI
from PIL import Image


parser = argparse.ArgumentParser(description='vLLM client for image processing')
parser.add_argument('--num_img_to_test', type=int, default=16, help='Number of images to test (default: 16)')
parser.add_argument("--server-url", default="http://localhost:8000", help="Server URL")
args = parser.parse_args()

num_img_to_test = args.num_img_to_test
img_dir = "/root/d/dataset_images/"

# create request
req_content = []

# Read JPG images from directory
if not os.path.exists(img_dir):
    print(f"Error: Directory not found: {img_dir}")
    sys.exit(1)

all_files = os.listdir(img_dir)
jpg_files = sorted([f for f in all_files if f.lower().endswith('.jpg')])

if len(jpg_files) < num_img_to_test:
    print(f"Error: Not enough JPG images in directory. Found {len(jpg_files)}, need {num_img_to_test}")
    sys.exit(1)

# Populate image_paths with the first num_img_to_test images
image_paths = [os.path.join(img_dir, f) for f in jpg_files[:num_img_to_test]]

# create request, add text
req_content.append({
    "type": "text",
    "text": "Describe thouse images."
})

# add images after the text
for path in image_paths:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")
    
    with open(path, "rb") as f:
        encoded_image = base64.b64encode(f.read())
    encoded_image_text = encoded_image.decode("utf-8") 

    req_content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{encoded_image_text}"
        }
    })

# send request to vLLM
print(f"sending {len(image_paths)} images")
openai_api_key = "EMPTY"
openai_api_base = args.server_url

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

chat_response = client.chat.completions.create(
    model="google/gemma-3-12b-it",
    messages=[
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]
        },
        {
            "role": "user",
            "content": req_content
        },
    ],
)

print("Chat response:", chat_response)
