from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image
import requests
import torch
from vllm.multimodal.utils import fetch_image

model_id = "google/gemma-3-4b-it"

model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id, device_map="hpu" #"cpu"
).eval()

processor = AutoProcessor.from_pretrained(model_id)

"""Images are:
1. Medical form
2. Duck
3. Lion
4. Blue Bird
5. Whale
6. Starfish
7. Snail
8. Bee on Purple Flower
9. 2 Dogs
10. Orange Cat
11. Gerbil
12. Rabbit
13. Horse and foal
"""
IMAGE_URLS = [
    #"https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/d/da/2015_Kaczka_krzy%C5%BCowka_w_wodzie_%28samiec%29.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/7/77/002_The_lion_king_Snyggve_in_the_Serengeti_National_Park_Photo_by_Giles_Laurent.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/2/26/Ultramarine_Flycatcher_%28Ficedula_superciliaris%29_Naggar%2C_Himachal_Pradesh%2C_2013_%28cropped%29.JPG",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/Anim1754_-_Flickr_-_NOAA_Photo_Library_%281%29.jpg/2560px-Anim1754_-_Flickr_-_NOAA_Photo_Library_%281%29.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/d/d4/Starfish%2C_Caswell_Bay_-_geograph.org.uk_-_409413.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/6/69/Grapevinesnail_01.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0b/Texas_invasive_Musk_Thistle_1.jpg/1920px-Texas_invasive_Musk_Thistle_1.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7a/Huskiesatrest.jpg/2880px-Huskiesatrest.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/6/68/Orange_tabby_cat_sitting_on_fallen_leaves-Hisashi-01A.jpg/1920px-Orange_tabby_cat_sitting_on_fallen_leaves-Hisashi-01A.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/3/30/George_the_amazing_guinea_pig.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1f/Oryctolagus_cuniculus_Rcdo.jpg/1920px-Oryctolagus_cuniculus_Rcdo.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/9/98/Horse-and-pony.jpg",
]



# image1 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
# image2 = "https://upload.wikimedia.org/wikipedia/commons/d/da/2015_Kaczka_krzy%C5%BCowka_w_wodzie_%28samiec%29.jpg"
# image3 = "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/Anim1754_-_Flickr_-_NOAA_Photo_Library_%281%29.jpg/2560px-Anim1754_-_Flickr_-_NOAA_Photo_Library_%281%29.jpg"

# images = [fetch_image(image1), fetch_image(image2), fetch_image(image3)]
#############
# Experiment 1
# images = [fetch_image(image) for image in IMAGE_URLS]

# text = "Describe all these images in 2-10 words: <start_of_image> <start_of_image> <start_of_image> <start_of_image> <start_of_image> <start_of_image> <start_of_image> <start_of_image> <start_of_image> <start_of_image> <start_of_image> <start_of_image>."


# Convert messages using apply_chat_template
# inputs = processor(images=images, text=text, return_tensors='pt')

# with torch.inference_mode():
#     generation = model.generate(**inputs, max_new_tokens=256, do_sample=False, num_beams=1)
#     generation = generation[0]

# decoded = processor.decode(generation, skip_special_tokens=True)
# print(decoded)

prompts = [
    "Describe the image.",
    "What is happening in the picture?"
]
images = [Image.open(path) for path in IMAGE_URLS[:2]]
inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
outputs = model.generate(**inputs, max_new_tokens=100)
decoded_outputs = processor.batch_decode(outputs, skip_special_tokens=True)

for i, output in enumerate(decoded_outputs):
    print(f"Prompt {i+1}: {output}")


breakpoint()

do_batch = True
#############
# Experiment 2: 3 x 2
if do_batch:
    text = "Describe all these images in 2-10 words: <start_of_image> <start_of_image> <start_of_image>."
    num_images = 3
    batch_size = 2
    image_urls = [IMAGE_URLS[idx % len(IMAGE_URLS)] for idx, i in enumerate(range(num_images * batch_size))]
    import numpy as np
    chunks = np.array_split(image_urls, batch_size)
    # requests = []
    inputs_list = []
    for chunk in chunks:
        print("chunk....", chunk)
        placeholders = [{"type": "image", "image": (url)} for url in chunk]
        messages = [{
            "role":
            "user",
            "content": [
                *placeholders,
                {
                    "type": "text",
                    "text": text
                },
            ],
        }]
        prompt = processor.apply_chat_template(messages,tokenize=True,return_dict=True,return_tensors="pt",add_generation_prompt=True)
        inputs_list.append(prompt)
        #requests.append({"prompt":prompt,"multi_modal_data":{"image":[fetch_image(url) if (type(url)==str) or (type(url)==np.str_) else url for url in chunk]}})

    print('inputs_list[0]', inputs_list[0])
    # Example: stack input_ids and attention_mask
    input_ids = torch.stack([x["input_ids"] for x in inputs_list])
    attention_mask = torch.stack([x["attention_mask"] for x in inputs_list])

    # If images are present, stack pixel_values (assuming same shape)
    if "pixel_values" in inputs_list[0]:
        pixel_values = torch.stack([x["pixel_values"] for x in inputs_list])
        batch_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values
        }
    else:
        batch_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

    batch_inputs = {k: v.to(model.device) for k, v in batch_inputs.items()}

    with torch.inference_mode():
        generation = model.generate(**batch_inputs, max_new_tokens=256, do_sample=False, num_beams=1)
        # generation = model.generate(**requests, max_new_tokens=256, do_sample=False, num_beams=1)

    for i, generation in enumerate(generation):
    # generation = generation[0]
        decoded = processor.decode(generation, skip_special_tokens=True)
        print(decoded)    

# add these for greedy/deterministic decoding: num_beams=1 and do_sample=False
