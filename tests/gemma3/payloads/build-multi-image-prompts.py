import json
import copy
import math

with open('safe_multipage_prompt.json', 'r') as f:
    data = json.load(f)  # data is now a Python dict or list


image_urls = []

for i in range(1, len(data['messages'][1]['content'])):
    if 'image_url' in data['messages'][1]['content'][i]['type']:
        image_urls.append(data['messages'][1]['content'][i])


# total_tokens_to_generate = [16384, 32768, 65536, 98304, 128000]
total_tokens_to_generate = range(35000, 65100, 5000)
per_image_tokens = 256
text_tokens = 4313
tokens_in_original_prompt = 6153

tokens_to_generate = [tokens - tokens_in_original_prompt for tokens in total_tokens_to_generate]
num_images_needed = [math.ceil(image_tokens / per_image_tokens) for image_tokens in tokens_to_generate]
# [40, 104, 232, 360, 476]

json_file_prefix = 'safe_multipage_prompt_'

for i, num_images in enumerate(num_images_needed):
    data_copy = copy.deepcopy(data)
    for image_num in range(num_images):
        data_copy['messages'][1]['content'].append(image_urls[image_num % len(image_urls)])
    
    with open(f'{json_file_prefix}{total_tokens_to_generate[i]}_tokens.json', 'w') as f:
        json.dump(data_copy, f, indent=4)

    print(f"Generated {json_file_prefix}{total_tokens_to_generate[i]}.json with {num_images} images.")

