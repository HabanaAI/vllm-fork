# SPDX-License-Identifier: Apache-2.0
import atexit
import os
from pathlib import Path

import torch
import yaml

from vllm import LLM

TEST_DATA_FILE = os.environ.get(
    "TEST_DATA_FILE", ".jenkins/embedding/configs/e5-mistral-7b-instruct.yaml")

TP_SIZE = int(os.environ.get("TP_SIZE", 1))

# Sample 1k prompts.
prompts = [
    "In the year 2145, humanity stands on the brink of a new era. Technological advancements have transformed society, with artificial intelligence seamlessly integrated into every aspect of life. Autonomous vehicles navigate bustling cities, intelligent personal assistants manage daily tasks, and AI-driven healthcare systems revolutionize medicine. Yet, as AI's influence grows, so do the ethical dilemmas. The line between human and machine intelligence blurs, raising profound questions about consciousness and the essence of humanity. Amidst this backdrop, Dr. Alex Carter, a brilliant young scientist, makes a groundbreaking discovery: an algorithm capable of unlocking AI's true potential. This algorithm promises unprecedented advancements but also poses significant risks. As Dr. Carter grapples with the implications, they must navigate a complex web of corporate interests, government regulations, and moral considerations. The stakes are high, and the future of humanity hangs in the balance. Will Dr. Carter's discovery lead to a utopian future or unleash unforeseen consequences?",
    "In the distant future, Earth has become a hub of technological marvels and interstellar exploration. The year is 2200, and humanity has established colonies on Mars and beyond. Advanced AI systems govern everything from space travel to daily life, creating a seamless blend of human and machine. Amidst this progress, a mysterious signal is detected from a distant galaxy, sparking curiosity and concern. Dr. Elena Ramirez, a renowned astrophysicist, is tasked with deciphering the signal. As she delves deeper, she uncovers a message that hints at an ancient civilization and a potential threat to humanity. The signal's origin is traced to a planet on the edge of the known universe, prompting an urgent mission. Dr. Ramirez joins a diverse team of scientists, engineers, and explorers on a journey to uncover the truth. As they venture into the unknown, they must confront the challenges of deep space, the mysteries of the ancient civilization, and their own fears. The fate of humanity may depend on their success.",
    "In a world where climate change has drastically altered the landscape, humanity has adapted to survive in a new environment. The year is 2085, and rising sea levels have submerged coastal cities, forcing people to relocate to higher ground. Advanced technology has enabled the construction of floating cities and sustainable habitats. Amidst this new way of life, a young environmental scientist named Dr. Maya Patel discovers a hidden ecosystem thriving beneath the ocean's surface. This ecosystem holds the key to reversing some of the damage caused by climate change. However, powerful corporations and political entities have their own agendas, seeking to exploit these resources for profit. Dr. Patel must navigate a treacherous path, balancing scientific integrity with the pressures of a world desperate for solutions. As she uncovers more about this underwater world, she faces ethical dilemmas and dangerous adversaries. The future of the planet depends on her ability to protect this fragile ecosystem and harness its potential for the greater good.",
    "In the year 2075, humanity has achieved remarkable advancements in biotechnology, leading to the creation of enhanced humans known as Neos. These Neos possess extraordinary abilities, from heightened intelligence to superhuman strength. Society is divided between those who embrace these enhancements and those who fear the loss of human identity. Amidst this tension, Dr. Samuel Hayes, a pioneering geneticist, discovers a breakthrough that could bridge the gap between Neos and unenhanced humans. His research reveals a way to safely integrate enhancements without compromising individuality. However, powerful factions oppose his work, fearing it will disrupt the balance of power. As Dr. Hayes races against time to complete his research, he faces threats from both sides. With the help of a diverse team of allies, he must navigate political intrigue, ethical dilemmas, and personal sacrifices. The future of humanity hinges on his ability to unite a divided world and ensure that technological progress benefits all."
]


def fail_on_exit():
    os._exit(1)


def launch_embedding_model(config):
    model_name = config.get('model_name')
    dtype = config.get('dtype', 'bfloat16')
    tensor_parallel_size = TP_SIZE
    llm = LLM(
        model=model_name,
        task="embed",
        dtype=dtype,
        tensor_parallel_size=tensor_parallel_size,
        enforce_eager=False,
    )
    return llm


def get_current_gaudi_platform():

    #Inspired by: https://github.com/HabanaAI/Model-References/blob/a87c21f14f13b70ffc77617b9e80d1ec989a3442/PyTorch/computer_vision/classification/torchvision/utils.py#L274

    import habana_frameworks.torch.utils.experimental as htexp

    device_type = htexp._get_device_type()

    if device_type == htexp.synDeviceType.synDeviceGaudi:
        return "Gaudi1"
    elif device_type == htexp.synDeviceType.synDeviceGaudi2:
        return "Gaudi2"
    elif device_type == htexp.synDeviceType.synDeviceGaudi3:
        return "Gaudi3"
    else:
        raise ValueError(
            f"Unsupported device: the device type is {device_type}.")


def test_embedding_model(record_xml_attribute, record_property):
    try:
        config = yaml.safe_load(
            Path(TEST_DATA_FILE).read_text(encoding="utf-8"))
        # Record JUnitXML test name
        platform = get_current_gaudi_platform()
        testname = (f'test_{Path(TEST_DATA_FILE).stem}_{platform}_'
                    f'tp{TP_SIZE}')
        record_xml_attribute("name", testname)

        llm = launch_embedding_model(config)

        # Generate embedding. The output is a list of EmbeddingRequestOutputs.
        outputs = llm.embed(prompts)
        torch.hpu.synchronize()

        # Print the outputs.
        for i, (prompt, output) in enumerate(zip(prompts, outputs)):
            embeds = output.outputs.embedding
            embeds_trimmed = ((str(embeds[:16])[:-1] +
                               ", ...]") if len(embeds) > 16 else embeds)
            print(f"{i} Prompt: {prompt!r} | "
                  f"Embeddings: {embeds_trimmed} (size={len(embeds)})")
        os._exit(0)

    except Exception as exc:
        atexit.register(fail_on_exit)
        raise exc
