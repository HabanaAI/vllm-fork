import torch
from vllm.multimodal.utils import fetch_image
import json
from argparse import Namespace
from vllm.utils import FlexibleArgumentParser
from vllm import LLM, EngineArgs, SamplingParams
import argparse
from dataclasses import asdict
from transformers import AutoProcessor, AutoTokenizer


PROMPTS = [
# 19
"What is the content of each image? Once done, write a story that combines them all.",
# 183
"You are an AI designed to generate extremely long, detailed worldbuilding content. Your goal is to write a fictional encyclopedia with at least 4000 words of content. Do not stop early. Start by describing a fictional planet in detail. Include: \n1. Geography and climate zones (with rich, varied description).\n2. The history of all civilizations, from ancient to modern times.\n3. Cultures, belief systems, and mythologies along with rich detail about where such beliefs came from.\n4. Political structures and conflicts along with their history.\n5. Technology and magic systems (if any) spanning the last 1000 years, highlighting significant discoveries and figures.\n6. Major historical events and characters along with their geneology.\n\n Be descriptive, verbose, and never summarize. Write in a factual tone like an academic encyclopedia. Begin your entry below:",
# 265
'''Here is a short story: It contains some animals as its main characters. Rewrite this story by replacing the animals in this story with any of the animals shown in the images. Do not change the contents of the story, just the characters: One day a shepherd discovered a fat Pig in the meadow where his Sheep were pastured. He very quickly captured the porker, which squealed at the top of its voice the moment the Shepherd laid his hands on it. You would have thought, to hear the loud squealing, that the Pig was being cruelly hurt. But in spite of its squeals and struggles to escape, the Shepherd tucked his prize under his arm and started off to the butcher's in the market place. The Sheep in the pasture were much astonished and amused at the Pig's behavior, and followed the Shepherd and his charge to the pasture gate. "What makes you squeal like that?" asked one of the Sheep. "The Shepherd often catches and carries off one of us. But we should feel very much ashamed to make such a terrible fuss about it like you do." "That is all very well," replied the Pig, with a squeal and a frantic kick. "When he catches you he is only after your wool. But he wants my bacon! gree-ee-ee!"'''
# 611
"Here is a list of creatures. Analyse the images and if there are creatures present in the images and the pick the closest creature from this list for each creature in the images: African Elephant, Bengal Tiger, Arctic Fox, Blue Whale, Brown Bear, Cheetah, Cougar, Dingo, Dolphin, Elk, Flying Fox, Giraffe, Gorilla, Grizzly Bear, Hedgehog, Hippopotamus, Hyena, Indian Elephant, Jaguar, Kangaroo, Koala, Lemur, Leopard, Lion, Lynx, Manatee, Mole, Moose, Mountain Goat, Narwhal, Okapi, Orangutan, Otter, Panda, Platypus, Polar Bear, Porcupine, Possum, Prairie Dog, Puma, Quokka, Rabbit, Raccoon, Red Panda, Reindeer, Rhinoceros, Sea Lion, Seal, Sheep, Skunk, Sloth, Squirrel, Tapir, Tasmanian Devil, Walrus, Weasel, Whale, Wild Boar, Wombat, Yak, Zebra, Albatross, American Robin, Bald Eagle, Barn Owl, Blue Jay, Budgerigar, Canary, Cardinal, Cassowary, Chickadee, Cockatoo, Cormorant, Crane, Crow, Cuckoo, Dove, Duck, Eagle, Egret, Falcon, Finch, Flamingo, Goldfinch, Goose, Great Horned Owl, Gull, Hawk, Heron, Hummingbird, Ibis, Jay, Kestrel, Kingfisher, Kiwi, Lark, Macaw, Magpie, Mockingbird, Nightingale, Nuthatch, Oriole, Ostrich, Owl, Parrot, Partridge, Peacock, Pelican, Penguin, Peregrine Falcon, Pigeon, Puffin, Quail, Raven, Roadrunner, Robin, Rooster, Sparrow, Starling, Stork, Swallow, Swan, Toucan, Turkey, Vulture, Warbler, Woodpecker, Wren, Angelfish, Anglerfish, Barracuda, Betta Fish, Blue Tang, Catfish, Clownfish, Cod, Eel, Flounder, Flying Fish, Goldfish, Grouper, Guppy, Haddock, Halibut, Hammerhead Shark, Herring, Jellyfish, Koi, Lionfish, Lobster, Mackerel, Manta Ray, Marlin, Moray Eel, Octopus, Orca, Piranha, Pufferfish, Rainbow Trout, Salmon, Sardine, Seahorse, Shark, Shrimp, Squid, Starfish, Stingray, Swordfish, Tilapia, Tuna, Walrus, Whale Shark, Zebra Fish, Alligator, Anole, Boa Constrictor, Box Turtle, Chameleon, Cobra, Crocodile, Frog, Gecko, Gila Monster, Green Iguana, Komodo Dragon, Lizard, Monitor Lizard, Newt, Python, Rattlesnake, Salamander, Sea Turtle, Skink, Snake, Toad, Tortoise, Tree Frog, Viper, Ant, Bee, Beetle, Butterfly, Centipede, Cicada, Cricket, Dragonfly, Earthworm, Firefly, Grasshopper, Ladybug, Leech, Millipede, Moth, Praying Mantis, Scorpion, Snail, Spider, Termite, Tick, Wasp"
]


IMAGE_URLS = [
    "https://upload.wikimedia.org/wikipedia/commons/d/da/2015_Kaczka_krzy%C5%BCowka_w_wodzie_%28samiec%29.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/7/77/002_The_lion_king_Snyggve_in_the_Serengeti_National_Park_Photo_by_Giles_Laurent.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/2/26/Ultramarine_Flycatcher_%28Ficedula_superciliaris%29_Naggar%2C_Himachal_Pradesh%2C_2013_%28cropped%29.JPG",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/Anim1754_-_Flickr_-_NOAA_Photo_Library_%281%29.jpg/2560px-Anim1754_-_Flickr_-_NOAA_Photo_Library_%281%29.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/d/d4/Starfish%2C_Caswell_Bay_-_geograph.org.uk_-_409413.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/6/69/Grapevinesnail_01.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0b/Texas_invasive_Musk_Thistle_1.jpg/1920px-Texas_invasive_Musk_Thistle_1.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7a/Huskiesatrest.jpg/2880px-Huskiesatrest.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/6/68/Orange_tabby_cat_sitting_on_fallen_leaves-Hisashi-01A.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/3/30/George_the_amazing_guinea_pig.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/9/98/Horse-and-pony.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/f/fa/Great_Wall_of_China_July_2006.JPG",
    "https://upload.wikimedia.org/wikipedia/commons/a/a6/Brandenburger_Tor_abends.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/4/41/Yellowstone_River_near_Tower_Fall.JPG",
    "https://upload.wikimedia.org/wikipedia/commons/0/0c/GoldenGateBridge-001.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Fronalpstock_big.jpg/1280px-Fronalpstock_big.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/6/6e/Paris_-_Eiffelturm_und_Marsfeld2.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/5/53/Colosseum_in_Rome%2C_Italy_-_April_2007.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/6/6e/Golde33443.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/3/3c/Christ_the_Redeemer%2C_Rio_de_Janeiro.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/f/f6/Great_Sphinx_of_Giza_-_20080716a.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/d/d1/Mount_Everest_as_seen_from_Drukair2_PLW_edit.jpg",
]




def parse_args():
    parser = argparse.ArgumentParser(description='Demo on using vLLM for offline inference with '
        'vision language models that support multi-image input for text '
        'generation')
    parser.add_argument('--model-name',
                        '-m',
                        type=str,
                        default="google/gemma-3-4b-it",
                        choices=['google/gemma-3-4b-it','google/gemma-3-27b-it'],
                        help='Huggingface "model_type".')
    parser.add_argument('--tensor-parallel-size',
                        '-tp',
                        type=int,
                        default=1,
                        help='tensor parallel size.')
    parser.add_argument(
        "--batchconfig",
        "-b",
        type=str,
        required=True,
        help='''
            #Sample json input
            batch0_bs2 = [{"prompt": 1}, {"prompt": 0, "images": [0,1,2]}]
            batch0_bs3 = [{"prompt": 2, "images" : [6,7,8]}, {"prompt": 0, "images": [3,2,1,0]}, {"prompt": 3, "images": [4]}]
            inputs = [batch0_bs2, batch0_bs3]
            so inp json should be:
            [
            [{"prompt": 1}, {"prompt": 0, "images": [0,1,2]}],
            [{"prompt": 2, "images" : [6,7,8]}, {"prompt": 0, "images": [3,2,1,0]}, {"prompt": 3, "images": [4]}]
            ]
            ''')
    parser.add_argument('--max-model-len',
                        '-ml',
                        type=int,
                        default=10240,
                        help='Max-Model-Len.')
    parser.add_argument('--iter',
                        type=int,
                        default=1,
                        help='iteration')

    parser.add_argument("--enforce-eager", action="store_true")

    return parser.parse_args()


def make_model(model_name, max_model_len, tp_size, max_num_seqs, limit_mm_per_prompt, enforce_eager):
    engine_args = EngineArgs(
    model=model_name,
    max_model_len=max_model_len,
    max_num_batched_tokens=max_model_len,
    max_num_seqs=max_num_seqs,
    tensor_parallel_size=tp_size,
    #gpu_memory_utilization=0.9,
    enforce_eager=enforce_eager,
    limit_mm_per_prompt={"image": limit_mm_per_prompt},
    gpu_memory_utilization=0.7,
    trust_remote_code=True,
    )
    engine_args = asdict(engine_args)
    llm = LLM(**engine_args)
    processor = AutoProcessor.from_pretrained(model_name)
    return llm, processor


def create_inp_from_batchconfig(processor, batch):
    requests = []
    for prompt in batch:
        placeholders = [{"type": "image", "image": (url)} for url in prompt.get('images', [])]
        messages = [{
            "role":
            "user",
            "content": [
                *placeholders,
                {
                    "type": "text",
                    "text": PROMPTS[prompt["prompt"]]
                },
            ],
        }]
        final_prompt = processor.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        requests.append({"prompt":final_prompt,"multi_modal_data":{"image":[fetch_image(IMAGE_URLS[urlid]) for urlid in prompt.get('images', [])]}})
        print("request: ", requests)
    return requests


def run_generate(llm, processor, requests, i):
    sampling_params = SamplingParams(temperature=0.0,
                                     max_tokens=500) #8192)

    outputs = llm.generate(requests,
        sampling_params=sampling_params,
        use_tqdm=False
    )
    print("-" * 50)
    for o in outputs:
        generated_text = o.outputs[0].text
        print(len(o.outputs[0].token_ids))
        print(generated_text)
        print("-*." * 50)


def main(args: Namespace):
    with open(args.batchconfig) as f:
        config = json.load(f)

    limit_mm_per_prompt = max(max([len(prompt.get("images", [])) for prompt in batch]) for batch in config)
    max_num_seqs = max(sum([len(prompt.get("images", [])) for prompt in batch]) for batch in config)  # TODO: this one not sure if this is what it means?
    print(f"Make the model: {limit_mm_per_prompt=},{max_num_seqs=}, {args.enforce_eager=}")

    llm, processor = make_model(args.model_name, args.max_model_len, 
        args.tensor_parallel_size, max_num_seqs=max_num_seqs,
        limit_mm_per_prompt=limit_mm_per_prompt, enforce_eager=args.enforce_eager)

    import time
    for i in range(args.iter):
        time_s=time.time()

        for batchidx, batch in enumerate(config):
            requests = create_inp_from_batchconfig(processor, batch)
            time_u=time.time()
            run_generate(llm, processor, requests, i)
            print(f'Done batch {batchidx}, of bs={len(batch)}. config: {batch}, time_taken:{time.time()-time_u}')

    print(f"Total Time iter[{i}] ", time.time()-time_s)

if __name__ == "__main__":
    args = parse_args()
    main(args)
