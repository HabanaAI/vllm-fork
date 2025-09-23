# Enable LLAMA4 on vllm hpu

## deploy
```
docker run -d -it --runtime=habana --name llama4-vllm-1.21 -v /software:/software -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --ipc=host --net=host -e HF_HOME=/data/huggingface vault.habana.ai/gaudi-docker/1.20.1/ubuntu24.04/habanalabs/pytorch-installer-2.6.0:latest /bin
/bash

docker exec -it llama4-vllm-1.21 /bin/bash

cd /data/models; huggingface-cli download --local-dir Llama-4-Scout-17B-16E-Instruct meta-llama/Llama-4-Scout-17B-16E-Instruct --token ${YOUR_TOKEN}

cd /software/users/${YOUR NAME}/;
git clone https://github.com/HabanaAI/vllm-fork -b llama4
pip install -r requirements-hpu.txt; VLLM_TARGET_DEVICE=hpu pip install -e .  --no-build-isolation;

> Note: if you're using 1.20.0, you might need to `pip install numpy==1.26.4" if it's not this version yet

> Note: this is from requirements-hpu.txt, for some reason something is uninstalling our version
pip install git+https://github.com/HabanaAI/vllm-hpu-extension.git@145c63d

# install dependencies for llama4
pip install pydantic msgspec cachetools cloudpickle psutil zmq blake3 py-cpuinfo aiohttp openai uvloop fastapi uvicorn watchfiles partial_json_parser python-multipart gguf llguidance prometheus_client numba compressed_tensors

```

## run example
```
PT_HPU_LAZY_MODE=1 python llama4-scripts/test_vllm.py --model_id /data/models/Llama-4-Scout-17B-16E-Instruct/ 2>&1 | tee llama4-scripts/llama4_vllm.log
```

## FP8 quantization

To get a FP8 Maverick llama4 model, please follow below steps:

```bash
# install specific INC
pip uninstall -y neural-compressor neural-compressor-pt
git clone -b dev/llama4_launch https://github.com/intel/neural-compressor.git
cd neural-compressor
pip install -e .

# calibration
QUANT_CONFIG=measure.json PT_HPU_LAZY_MODE=1 python test_measure.py --model_id meta-llama/Llama-4-Maverick-17B-128E-Instruct

# quantization
QUANT_CONFIG=quant.json PT_HPU_LAZY_MODE=1 python test_vllm_quant.py --model_id meta-llama/Llama-4-Maverick-17B-128E-Instruct
```
# MMLU Pro test

``` bash
cd llama4-scripts/accuracy;
pip install numpy==1.26.4
pip install lm_eval
sh run_mmlupro_acc.sh
```

# Enable Vision Accuracy test using ChartQA Dataset

```bash
pip uninstall lm_eval
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
pip install numpy==1.26.4
cd vllm-fork/llama4-scripts/accuracy
bash run_chartqa_acc.sh
```

---

## Test vLLM on GPU (used as reference output)

### Run following to test Llama-4-Scout-17B-16E-Instruct.

Expected result:


```
-----------------------------------
Prompt: '<|begin_of_text|><|header_start|>user<|header_end|>\n\nwhat is the recipe of mayonnaise?<|header_start|>assistant<|header_end|>\n\n'
Generated text:
 Mayonnaise is a classic condiment made from a mixture of oil, egg yolks, vinegar or lemon juice, and seasonings. Here's a simple recipe to make mayonnaise at home:

**Homemade Mayonnaise Recipe**

**Ingredients:**

* 2 large egg yolks
* 1 tablespoon lemon juice or vinegar (white wine vinegar or apple cider vinegar work well)
* 1/2 teaspoon salt
* 1/4 teaspoon Dijon mustard (optional, but recommended for flavor)
* 1 cup (240 ml) neutral-tasting oil, such as canola, grapeseed, or light olive oil

**Instructions:**

1. **Start with room temperature ingredients**: This is crucial for emulsification. Take the egg yolks, lemon juice, and oil out of the refrigerator about 30 minutes before starting.
2. **Whisk together egg yolks and lemon juice**: In a medium-sized bowl, whisk together the egg yolks and lemon juice until well combined.
3. **Add salt, mustard (if using), and whisk**: Add the salt and mustard (if using) to the bowl and whisk until smooth.
4. **Slowly add oil**: With a hand mixer or whisk, slowly pour the oil into the egg yolk mixture while continuously whisking. Start with a very slow, thin stream and gradually increase the flow as the mixture thickens.
5. **Emulsify**: Continue whisking until the mixture thickens and emulsifies, which should take about 5-7 minutes. You'll know it's working when the mixture becomes smooth, creamy, and doubled in volume.
6. **Taste and adjust**: Stop whisking and taste the mayonnaise. If it's too thick, add a little water. If it's too thin, add a little more oil.
7. **Store**: Transfer the mayonnaise to an airtight container and store it in the refrigerator for up to one week.

**Tips and Variations:**

* Use a neutral-tasting oil to avoid flavoring the mayonnaise.
* If the mayonnaise breaks or separates, don't worry! Simply start again with a new egg yolk and slowly whisk in the broken mayonnaise.
* For a lighter mayonnaise, use more lemon juice or vinegar.
* For a richer mayonnaise, use more oil.
* Add flavorings like minced garlic, chopped herbs, or grated ginger to create different variations.

**Why it works:**

Mayonnaise is an emulsion, meaning it's a mixture of two or more liquids that don't normally mix, like oil and water. The egg yolks contain lecithin, a natural emulsifier that helps combine the oil and water. The acid in the lemon juice or vinegar helps to stabilize the emulsion and adds flavor.

**Common issues and solutions:**

* **Mayonnaise is too thick**: Add a little water or lemon juice.
* **Mayonnaise is too thin**: Add a little more oil.
* **Mayonnaise breaks or separates**: Start again with a new egg yolk and slowly whisk in the broken mayonnaise.

Enjoy your homemade mayonnaise!

-----------------------------------
Prompt: "<|begin_of_text|><|header_start|>system<|header_end|>\n\n[{'type': 'text', 'text': 'Always answer with Haiku'}]<|eot|><|header_start|>user<|header_end|>\n\nI am going to Paris, what should I see?<|header_start|>assistant<|header_end|>\n\n"
Generated text:
 Eiffel Tower high
River Seine's gentle waters flow
Beauty in the night

-----------------------------------
Prompt: "<|begin_of_text|><|header_start|>system<|header_end|>\n\n[{'type': 'text', 'text': 'Always answer with emojis'}]<|eot|><|header_start|>user<|header_end|>\n\nHow to go from Beijing to NY?<|header_start|>assistant<|header_end|>\n\n"
Generated text:
 🗺️✈️🗽️

-----------------------------------
Prompt: "<|begin_of_text|><|header_start|>user<|header_end|>\n\nI am going to Paris, what should I see?<|header_start|>assistant<|header_end|>\n\nParis, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:\n\n1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.\n2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.\n3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.\n\nThese are just a few of the many attractions that Paris has to offer. With so much to see and do, it's no wonder that Paris is one of the most popular tourist destinations in the world.<|header_start|>user<|header_end|>\n\nWhat is so great about #1?<|header_start|>assistant<|header_end|>\n\n"
Generated text:
 The Eiffel Tower! It's an iconic symbol of Paris and one of the most recognizable landmarks in the world. Here are some reasons why it's so great:

1. **Breathtaking Views**: The Eiffel Tower offers stunning 360-degree views of the city from its observation decks, which are located on the first and second floors. On a clear day, you can see for miles in every direction.
2. **Engineering Marvel**: The Eiffel Tower was a groundbreaking feat of engineering when it was built for the 1889 World's Fair. It's an impressive iron lattice structure that was the tallest man-made structure in the world at the time.
3. **Romantic Ambiance**: The Eiffel Tower is often associated with romance and is a popular spot for proposals, weddings, and honeymoons. The tower is especially beautiful at night when it's illuminated with thousands of twinkling lights.
4. **Rich History**: The Eiffel Tower has a fascinating history, having been built for the World's Fair and serving as a radio broadcasting tower, a observation deck, and even a billboard.
5. **Iconic Design**: The Eiffel Tower's unique design has been imitated and parodied countless times, making it one of the most recognizable landmarks in the world.
6. **Cultural Significance**: The Eiffel Tower has been featured in countless films, literature, and art, making it a cultural icon that transcends its functional purpose as a tower.
7. **Accessibility**: The Eiffel Tower is easily accessible by public transportation, and visitors can take a lift or stairs to the top for panoramic views of the city.
8. **Surrounding Park**: The Champ de Mars park, where the Eiffel Tower is located, is a beautiful green space with plenty of street performers, food vendors, and picnic areas.

Some tips to keep in mind when visiting the Eiffel Tower:

* Buy tickets in advance to avoid long lines, especially during peak season.
* Consider visiting at sunset or twilight for a romantic atmosphere.
* Take the stairs instead of the elevator to the top for a more affordable and energetic experience.
* Don't forget your camera to capture the stunning views and memories.

Overall, the Eiffel Tower is a must-visit attraction in Paris, offering a unique blend of history, culture, and romance that's hard to find anywhere else.

Would you like to know more about the Eiffel Tower or Paris in general?


Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:06<00:00,  6.06s/it, est. speed input: 554.06 toks/s, output: 25.09 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:06<00:00,  6.06s/it, est. speed input: 554.06 toks/s, output: 25.09 toks/s]
-----------------------------------
Prompt: '<|begin_of_text|><|begin_of_text|><|header_start|>user<|header_end|>\n\nAnalyze these images and describe their content.<|image_start|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|tile_x_separator|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|tile_x_separator|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|tile_y_separator|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|tile_x_separator|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|tile_x_separator|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|tile_y_separator|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|tile_x_separator|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|tile_x_separator|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|tile_y_separator|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|tile_x_separator|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|tile_x_separator|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|tile_y_separator|><|image|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|image_end|><|image_start|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|tile_x_separator|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|tile_x_separator|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|tile_y_separator|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|tile_x_separator|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|tile_x_separator|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|tile_y_separator|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|tile_x_separator|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|tile_x_separator|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|tile_y_separator|><|image|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|patch|><|image_end|><|header_start|>assistant<|header_end|>\n\n'
Generated text:
 The image on the left shows a large anthropomorphic rabbit standing on a path, with a cottage in the background. The image on the right shows a cat sitting on a stone path, with a cottage in the background.

The images appear to be computer-generated images of anthropomorphic animals, possibly from a children's story or animation. The images feature a rabbit and a cat, both dressed in clothing, and set against a backdrop of rolling hills and countryside. The overall atmosphere of the images is one of whimsy and fantasy, with bright colors and detailed textures that bring the characters to life. 

The images may be intended to evoke a sense of nostalgia or wonder, and could be used in a variety of contexts, such as advertising, illustration, or animation.

```

### Run following test meta-llama/Llama-4-Maverick-17B-128E-Instruct.
```
python test_vllm.py meta-llama/Llama-4-Maverick-17B-128E-Instruct
```

Expected result:

```
-----------------------------------
Prompt: '<|begin_of_text|><|header_start|>user<|header_end|>\n\nwhat is the recipe of mayonnaise?<|header_start|>assistant<|header_end|>\n\n'
Generated text:
 The classic condiment! Mayonnaise is a thick, creamy sauce made from a combination of oil, egg yolks, vinegar or lemon juice, and seasonings. Here's a basic recipe to make mayonnaise from scratch:

**Ingredients:**

* 3 large egg yolks
* 1 tablespoon (15 ml) lemon juice or vinegar (white or apple cider)
* 1/2 cup (120 ml) neutral-tasting oil, such as canola or grapeseed
* 1/2 teaspoon salt
* Optional: 1/4 teaspoon Dijon mustard (for flavor)

**Instructions:**

**Method 1: Using a Hand Whisk or Stand Mixer**

1. In a medium-sized bowl, whisk together the egg yolks, lemon juice or vinegar, and salt until well combined.
2. Slowly pour the oil into the bowl while continuously whisking the mixture. Start with a very slow, thin stream of oil and gradually increase the flow as the mixture thickens.
3. Continue whisking until the mixture emulsifies and becomes thick and creamy, about 5-7 minutes with a hand whisk or 2-3 minutes with a stand mixer.
4. Taste and adjust the seasoning as needed. If the mayonnaise is too thick, add a little water. If it's too thin, add a little more oil.

**Method 2: Using a Blender or Food Processor**

1. In a blender or food processor, combine the egg yolks, lemon juice or vinegar, and salt. Blend on low speed until well combined.
2. With the blender or food processor running, slowly pour the oil into the mixture in a thin, steady stream. Continue blending until the mixture emulsifies and becomes thick and creamy, about 2-3 minutes.
3. Taste and adjust the seasoning as needed.

**Tips and Variations:**

* Use room temperature ingredients for best results.
* If the mayonnaise breaks or separates, don't worry! Simply start again with a new egg yolk and slowly whisk in the broken mayonnaise, adding a little more oil as needed.
* For a lighter mayonnaise, use only 1/4 cup (60 ml) of oil.
* Add flavorings like minced garlic, chopped fresh herbs (e.g., parsley, dill, or tarragon), or grated ginger to create different flavor profiles.
* For a vegan mayonnaise, use a neutral-tasting oil and replace the egg yolks with a mixture of 1 tablespoon aquafaba (liquid from canned chickpeas) and 1 tablespoon lemon juice or vinegar.

**Safety Note:**

When making mayonnaise, it's essential to use safe egg handling practices to minimize the risk of salmonella. Use fresh, clean eggs and keep them refrigerated at 40°F (4°C) or below. If you're concerned about raw eggs, you can use pasteurized eggs or an egg substitute.

Enjoy your homemade mayonnaise!

-----------------------------------
Prompt: "<|begin_of_text|><|header_start|>system<|header_end|>\n\n[{'type': 'text', 'text': 'Always answer with Haiku'}]<|eot|><|header_start|>user<|header_end|>\n\nI am going to Paris, what should I see?<|header_start|>assistant<|header_end|>\n\n"
Generated text:
 Eiffel's iron lace
River Seine's gentle whispers
Art in every bite

-----------------------------------
Prompt: "<|begin_of_text|><|header_start|>system<|header_end|>\n\n[{'type': 'text', 'text': 'Always answer with emojis'}]<|eot|><|header_start|>user<|header_end|>\n\nHow to go from Beijing to NY?<|header_start|>assistant<|header_end|>\n\n"
Generated text:
 🛫️✈️🌎️🗽️

-----------------------------------
Prompt: "<|begin_of_text|><|header_start|>user<|header_end|>\n\nI am going to Paris, what should I see?<|header_start|>assistant<|header_end|>\n\nParis, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:\n\n1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.\n2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.\n3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.\n\nThese are just a few of the many attractions that Paris has to offer. With so much to see and do, it's no wonder that Paris is one of the most popular tourist destinations in the world.<|header_start|>user<|header_end|>\n\nWhat is so great about #1?<|header_start|>assistant<|header_end|>\n\n"
Generated text:
 The Eiffel Tower! It's an iconic symbol of Paris and one of the most recognizable landmarks in the world. Here are some reasons why it's so great:

1. **Engineering marvel**: When it was built for the 1889 World's Fair, it was the tallest structure in the world, standing at 324 meters (1,063 feet). It was a groundbreaking feat of engineering and a testament to French innovation.
2. **Panoramic views**: The Eiffel Tower offers breathtaking 360-degree views of the city from its observation decks on the first and second floors. On a clear day, you can see for miles in every direction.
3. **Romantic atmosphere**: The Eiffel Tower is often associated with romance and love. It's a popular spot for proposals, honeymoons, and romantic dinners.
4. **Iconic architecture**: The Eiffel Tower's lattice-like structure is a masterpiece of ironwork and a symbol of French culture. Its elegant design has been imitated and parodied countless times.
5. **History**: The Eiffel Tower has been a witness to many significant events in history, including the World's Fair, World War I, and World War II.
6. **Illuminations**: At night, the Eiffel Tower is illuminated with thousands of twinkling lights, making it a truly magical sight.
7. **Access to the top**: For the adventurous, there's an elevator to the top of the tower, where you can experience the thrill of being on the highest observation deck in Paris (although be prepared for long lines!).

Whether you're interested in history, architecture, romance, or simply want to take in the views, the Eiffel Tower is an unforgettable experience.

Would you like to know more about visiting the Eiffel Tower or planning a trip to Paris?


Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:02<00:00,  2.78s/it, est. speed input: 1782.16 toks/s, output: 31.31 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:02<00:00,  2.78s/it, est. speed input: 1782.16 toks/s, output: 31.31 toks/s]
-----------------------------------
Generated text:
 The image on the left features a rabbit wearing a blue coat, brown vest, and brown bow tie. The rabbit is standing on a dirt road in front of a stone house with a garden and mountains in the background.

The image on the right features a cat wearing a blue jacket, tan vest, and red bow tie. The cat is sitting on a stone path in front of a garden and a small house in the background.



```

### Test vLLM serve

```
# Only tp8 works for now

export PORT=8000; SAFETENSORS_FAST_GPU=1 vllm serve meta-llama/Llama-4-Scout-17B-16E-Instruct -tp 8 --host :: --port $PORT


# Once server is ready, on another terminal:

curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }'

# ---log looks like this --
#'{"id":"cmpl-8f45d8f2afc64d3ab1b14baab6695831","object":"text_completion","created":1743461366,"model":"meta-llama/Llama-4-Scout-17B-16E-Instruct","choices":[{"index":0,"text":" city known for its vibrant culture,","logprobs":null,"finish_reason":"length","stop_reason":null,"prompt_logprobs":null}],"usage":{"prompt_tokens":5,"total_tokens":12,"completion_tokens":7,"prompt_tokens_details":null}}
```


