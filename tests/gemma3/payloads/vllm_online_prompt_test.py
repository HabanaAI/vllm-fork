from datasets import load_dataset
import argparse
import json
from argparse import Namespace
from collections.abc import Iterable
from transformers import AutoTokenizer

import requests
infovqa_ckpt = 'LIME-DATA/infovqa'
# vision_arena_ckpt = 'lmarena-ai/vision-arena-bench-v0.1'

TEXT_COL_NAME = 'text'
DATASET_TEXT_IMAGE_COLUMN_MAP = {
    infovqa_ckpt: {'image': 'image', 'text_cols':['description', 'question']},
}

def map_filter_dataset(dataset_ckpt=infovqa_ckpt):
    dataset = load_dataset(dataset_ckpt, split='train')
    text_cols = DATASET_TEXT_IMAGE_COLUMN_MAP[dataset_ckpt]['text_cols']

    def merge_text(row):
        merge_str = ''
        for col in text_cols:
            merge_str += row[col] + '\n'
        return {TEXT_COL_NAME: merge_str}

    dataset = dataset.map(merge_text)

    # num_text_tokens_lower_limit = 100
    # dataset_filtered = dataset.filter(lambda x: len(x[TEXT_COL_NAME].split()) >= num_text_tokens_lower_limit)

    return dataset


def make_payload_dict(dataset, dataset_idx, tokenizer, image_column, model='google/gemma-3-27b-it', min_num_text_tokens=100, n_images=1):
    template_dict = {
        "model": model,
        "messages": [
        {
            "role": "user",
            "content": []
        }
        ]
    }

    prompt_num = dataset_idx

    def add_text(template_dict, text_message):
        text_item = {
            "type": "text",
            "text": text_message
        }

        template_dict['messages'][0]['content'].append(text_item)
        return template_dict

    def add_jpg_images(template_dict, images):
        import base64
        from io import BytesIO

        for image in images:
            img_buffer = BytesIO()
            image.save(img_buffer, format='JPEG')  # Save as JPEG to match the original format
            img_bytes = img_buffer.getvalue()
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')

            image_item = {
                "type": "image_url",
                "image_url": {"url":f"data:image/jpeg;base64,{img_base64}"}
            }

            template_dict['messages'][0]['content'].append(image_item)
        
        return template_dict

    # prepare text prompt to add to payload
    text_prompt = ''
    separator = '  \n\n'
    while min_num_text_tokens > len(tokenizer.tokenize(text_prompt)):
        # breakpoint()
        text_from_dataset = dataset[dataset_idx][TEXT_COL_NAME]
        text_prompt += text_from_dataset + separator
        dataset_idx += 1
    
    # prepare images to add to payload
    images = [dataset[idx][image_column] for idx in range(dataset_idx, dataset_idx + n_images)]

    template_dict = add_text(template_dict, text_prompt)
    template_dict = add_jpg_images(template_dict, images)

    print('*' * 50)
    print(f'Prompt #{prompt_num}: Number of Text tokens={len(tokenizer.tokenize(text_prompt))}, Number of Images={len(images)}')

    return template_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dataset", type=str, default=infovqa_ckpt)
    parser.add_argument("--model", type=str, default="google/gemma-3-4b-it")
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--min_text_tokens", action='extend', nargs='+', type=int, help='Number of minimum text tokens for prompts. Specify multiple values for multiple prompts. Num of values provided must match with \'num_images\'')
    parser.add_argument("--num_images", action='extend', nargs='+', type=int, help='Number of images for prompts. Specify multiple values for multiple prompts. Num of values provided must match with \'min_text_tokens\'')
    parser.add_argument("--stream", action="store_true")

    args = parser.parse_args()
    assert len(args.min_text_tokens) == len(args.num_images), f'The number of min text tokens {len(args.min_text_tokens)} must match with number of num images {len(args.num_images)}.'
    return parser.parse_args()


def post_http_request(
    payload: dict, api_url: str, max_tokens: int, stream: bool = False
) -> requests.Response:
    headers = {"User-Agent": "Test Client"}

    payload['temperature'] = 0.0
    payload['max_tokens'] = max_tokens

    response = requests.post(api_url, headers=headers, json=payload, stream=stream)
    return response

def get_streaming_response(response: requests.Response) -> Iterable[list[str]]:
    for chunk in response.iter_lines(
        chunk_size=8192, decode_unicode=False, delimiter=b"\n"
    ):
        if chunk:
            data = json.loads(chunk.decode("utf-8"))
            output = data["text"]
            yield output


def get_response(response: requests.Response) -> list[str]:
    data = json.loads(response.content)
    # breakpoint()
    # output = data["text"]
    return data


def clear_line(n: int = 1) -> None:
    LINE_UP = "\033[1A"
    LINE_CLEAR = "\x1b[2K"
    for _ in range(n):
        print(LINE_UP, end=LINE_CLEAR, flush=True)


def main(args: Namespace):
    dataset = map_filter_dataset(dataset_ckpt=args.dataset)
    image_column = DATASET_TEXT_IMAGE_COLUMN_MAP[args.dataset]['image']
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    for idx, (min_num_text_token, num_images) in enumerate(zip(args.min_text_tokens, args.num_images)):
        # breakpoint()
        payload_dict = make_payload_dict(dataset, idx, tokenizer, image_column, model=args.model, min_num_text_tokens=min_num_text_token, n_images=num_images)
        
        api_url = f"http://{args.host}:{args.port}/v1/chat/completions"

        response = post_http_request(payload_dict, api_url, args.max_tokens, args.stream)

        print(f'Output from prompt #{idx}')

        if args.stream:
            num_printed_lines = 0
            for h in get_streaming_response(response):
                clear_line(num_printed_lines)
                num_printed_lines = 0
                for i, line in enumerate(h):
                    num_printed_lines += 1
                    print(f"Beam candidate {i}: {line!r}", flush=True)
        else:
            output = get_response(response)
            print(output)
            # for i, line in enumerate(output):
            #     print(f"Beam candidate {i}: {line!r}", flush=True)    

# Sample Usages: uses default dataset LIME-DATA/infovqa
# python vllm_online_prompt_test.py --port 8080  --model google/gemma-3-27b-it  --min_text_tokens 200 500 1000 --num_images 0 0 0
# python vllm_online_prompt_test.py --port 8080  --model google/gemma-3-27b-it  --min_text_tokens 200 500 1000 --num_images 1 1 2

if __name__ == "__main__":
    args = parse_args()
    main(args)
