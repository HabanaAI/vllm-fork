from datasets import load_dataset
from pandas import DataFrame
import argparse
import json
from argparse import Namespace
from collections.abc import Iterable
from transformers import AutoTokenizer

import asyncio
import requests
muirbench_ckpt = 'MUIRBENCH/MUIRBENCH'
# vision_arena_ckpt = 'lmarena-ai/vision-arena-bench-v0.1'

TEXT_COL_NAME = 'text'
DATASET_TEXT_IMAGE_COLUMN_MAP = {
    muirbench_ckpt: {'image': 'image_list', 'text_cols':['question', 'options'], 'answer_col': 'answer'},
}

def map_filter_dataset(dataset_ckpt=muirbench_ckpt, filter_column='task', filter_value='Scene Understanding'):
    dataset = load_dataset(dataset_ckpt, split='test')
    text_cols = DATASET_TEXT_IMAGE_COLUMN_MAP[dataset_ckpt]['text_cols']

    def merge_text(row):
        merge_str = ''
        for col in text_cols:
            if col == 'options':
                options = row[col]
                for i, option in enumerate(options):
                    choice = chr(ord('A') + i)
                    merge_str += choice + '. ' + option +',\n '
            elif col == 'question':
                merge_str += 'question: ' + row[col] + '\n'
        
        merge_str = merge_str.replace('<image>', '<start_of_image>')
        return {TEXT_COL_NAME: merge_str}

    dataset = dataset.map(merge_text)

    # num_text_tokens_lower_limit = 100
    print('#'*50)
    print(f'DATASET FILTER: Filter column ({filter_column}), filter value ({filter_value})')
    dataset_filtered = dataset.filter(lambda x: x[filter_column] == filter_value)
    
    print(dataset_filtered)
    return dataset_filtered


def make_payload_dict(dataset, dataset_idx, tokenizer, image_column, model='google/gemma-3-27b-it', instruction=''):
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

    text_prompt_sufflx = instruction  #'Describe each image as well at the beginning. Provide the only one best answer.' # 'Without any reasoning, just give your answer.'
    # prepare text prompt to add to payload
    text_prompt = dataset[dataset_idx][TEXT_COL_NAME] + text_prompt_sufflx
    answer = dataset[dataset_idx]['answer']
    
    # prepare images to add to payload
    images = [dataset[dataset_idx][image_column][idx] for idx in range(len(dataset[dataset_idx][image_column]))]

    template_dict = add_text(template_dict, text_prompt)
    print('*' * 50)
    print('template_dict w/o image')
    print(template_dict)

    template_dict = add_jpg_images(template_dict, images)

    print('-' * 50)
    print(f'Prompt #{prompt_num}: Number of Text tokens={len(tokenizer.tokenize(text_prompt))}, Number of Images={len(images)}, Answer={answer}')

    return template_dict


def post_http_request(
    payload: dict, api_url: str, max_tokens: int, stream: bool = False
) -> requests.Response:
    headers = {"User-Agent": "Test Client"}

    payload['temperature'] = 0.0
    payload['max_tokens'] = max_tokens

    response = requests.post(api_url, headers=headers, json=payload, stream=stream)
    return response


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dataset", type=str, default=muirbench_ckpt)
    parser.add_argument("--model", type=str, default="google/gemma-3-4b-it")
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--task", type=str, default='Scene Understanding')
    parser.add_argument("--instruction", type=str, default='')
    parser.add_argument("--num_prompts", type=int, default=10)
    parser.add_argument("--starting_prompt_number", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)

    return parser.parse_args()


def get_response(response: requests.Response) -> list[str]:
    data = json.loads(response.content)
    # breakpoint()
    # output = data["text"]
    return data


def get_answer_from_dataset(dataset, idx):
    if type(idx) == list:
        return [dataset[index]['answer'] for index in idx]
    elif type(idx) == int:
        return dataset[idx]['answer']


def get_answer_from_response_json(response: dict):
    import re
    content = response['choices'][0]['message']['content']
    match = re.findall(r'\([A-Z]\)', content)
    if len(match) == 0:
        return 'NA'
    else:
        return match[-1][1]  # match is likely to be e.g. ['(B)']


async def submit_prompt(args, payloads, index):
    api_url = f"http://{args.host}:{args.port}/v1/chat/completions"

    async with semaphore:
        payload_dict = payloads[index]
        try:
            response = post_http_request(payload_dict, api_url, args.max_tokens, False)

            print(f'Output from prompt #{index}')

            output_json = get_response(response)
            answer = get_answer_from_dataset(dataset, args.starting_prompt_number+index)
            response = get_answer_from_response_json(output_json)            
        except Exception:
            print('!'*20)
            print(f'Error while processing prompt #{index}')
            return None, None
        
        print(output_json)

        return answer, response

def main(args: Namespace):
    asyncio.run(process_prompts(args))


async def process_prompts(args: Namespace):
    global dataset
    dataset = map_filter_dataset(dataset_ckpt=args.dataset, filter_value=args.task)
    image_column = DATASET_TEXT_IMAGE_COLUMN_MAP[args.dataset]['image']
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    num_prompts = args.num_prompts

    batch_size = args.batch_size
    global semaphore
    semaphore = asyncio.Semaphore(batch_size)

    answer_response_set = {'task': args.task}  #, 'answer-responses': {}}
    # answer_responses = answer_response_set['answer-responses']
    payloads = []

    for idx in range(args.starting_prompt_number, min(args.starting_prompt_number+num_prompts, len(dataset))):
        payload_dict = make_payload_dict(dataset, idx, tokenizer, image_column, model='google/gemma-3-27b-it', instruction=args.instruction)
        payloads.append(payload_dict)

    tasks = [asyncio.create_task(submit_prompt(args, payloads, i)) for i in range(len(payloads))]
    results = await asyncio.gather(*tasks)

    answer_responses = {i: (answer, response) for i, (answer, response) in enumerate(results)}
    answer_response_set['answer-responses'] = answer_responses

    # answer_responses[idx] = [answer, response]
    answer_responses_valid = {i: (answer, response) for i, (answer, response) in answer_responses.items() if answer != None and response != None}

    num_correct_responses = sum([answer_response[0] == answer_response[1] for answer_response in answer_responses_valid.values()])
    num_responses = len(answer_responses_valid)
    if num_responses == 0:
        print('No valid responses. Exiting...')
        return

    score = num_correct_responses / num_responses
    
    num_incomplete_prompts = len(answer_responses) - len(answer_responses_valid)
    if num_incomplete_prompts > 0:
        print('!'* 50)
        print(f'{len(answer_responses) - len(answer_responses_valid)} Incomplete prompts!!')
        print('!'* 50)

    print('-' * 50)
    print('Answer-Response Set')
    print(answer_response_set)
    df = DataFrame.from_dict(answer_responses, orient='index', columns=['answer', 'response'])
    file_name = f'{args.dataset}_{args.task}_no.{args.starting_prompt_number}-{args.starting_prompt_number+num_prompts-1}.csv'.replace('/', '_').replace(' ', '_')
    csv_path = f'./{file_name}'
    df.to_csv(csv_path)
    print(f'Saved results to {csv_path}')
    print('-' * 50)
    print(f'Accuracy Score: {score:.2f} = {num_correct_responses}/{num_responses}')


# Sample Usages: uses default dataset LIME-DATA/infovqa
# python muirbench_dataset_test.py --port 8080  --model google/gemma-3-27b-it --task "Image-Text Matching" --instruction "Describe each image as well at the beginning. Do not choose a close one when there seems no answer, but choose None instead. Provide the only one best answer at the end with a single letter in a pair of parentheses."  --num_prompts  20  --starting_prompt_number 10  --batch_size 4  2>&1 | tee G3-Gemma3-muirbench-Image-Text_Matching-10-29.txt

if __name__ == "__main__":
    args = parse_args()
    main(args)

