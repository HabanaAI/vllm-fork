import torch
from safetensors import safe_open
from safetensors.torch import save_file
from glob import glob
import os
import pickle
import json

import argparse


def copy_other_files(input_path, output_path):
    import shutil

    for file in os.listdir(input_path):
        if file.endswith(".json") or file.endswith(".py"):
            print(f"copying {file} to {output_path}")
            shutil.copyfile(
                os.path.join(input_path, file),
                os.path.join(output_path, file),
            )

def get_input_scales(pkl_path):
    input_scales = {}
    if pkl_path is not None:
        with open(pkl_path, 'rb') as file:
            input_scales = pickle.load(file)
    return input_scales

def convert_files(input_path, output_path, scales_path):
    all_safetensors = glob(f"{input_path}/*.safetensors")
    # sort by file name
    all_safetensors.sort()

    input_scales = get_input_scales(scales_path)
    print(f"{input_scales.keys()=}")

    tensor_mapping = {}

    for safetensors_path in all_safetensors:
        tensors = {}
        safetensors_name = os.path.basename(safetensors_path)
        print(f"processing {safetensors_path}")
        with safe_open(safetensors_path, framework="pt",
                       device="cpu") as tensor_file:
            for k in tensor_file.keys():
                tensor = tensor_file.get_tensor(k)
                # tensor = tensor.squeeze(-1)
                if "proj" in k:
                    if k.endswith("weight"):
                        tensor = (tensor.float() * 240.0 / 448.0).to(
                            torch.float8_e4m3fn)
                        input_scale_name = k.replace(".weight", ".input_scale")
                        if input_scale_name in input_scales.keys():
                            input_scale = input_scales.pop(input_scale_name)
                            input_scale = input_scale * 448.0 / 240.0
                            tensors[input_scale_name] = input_scale
                            tensor_mapping[input_scale_name] = safetensors_name
                    elif k.endswith("weight_scale_inv") or k.endswith(
                            "input_scale_inv"):
                        # "scale_inv" in deepseek-r1 is actually "scale"
                        tensor = tensor.float() * 448.0 / 240.0
                    else:
                        raise NotImplementedError(f"Cannot convert {k}")
                else:
                    print(f"Skipping conversion for {k}")
                tensors[k] = tensor
                tensor_mapping[k] = safetensors_name
        new_tensor_path = safetensors_path.replace(input_path, output_path)
        print(f"saving to {new_tensor_path}")
        save_file(tensors, new_tensor_path)

    if input_scales.keys():
        print(f"warning: the following input_scales are unused:")
        for k in input_scales.keys():
            print(k)

    tensor_mapping_file = os.path.join(input_path, "model.safetensors.index.json")
    print(f"Saving tensor mapping to {tensor_mapping_file}")
    state_dict_mapping = {
        "metadata": {},
        "weight_map": tensor_mapping,
    }
    with open(tensor_mapping_file, "w") as f:
        json.dump(state_dict_mapping, f, indent=4, sort_keys=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=(
        "Convert FP8 E4M3FN weights to FP8 E4M3FNUZ format for Gaudi2 and "
        "rescale the corresponding weight_scale_inv values."))
    parser.add_argument(
        "-i",
        "--input_path",
        required=True,
        help=("Path to the directory containing the official model weights in "
              ".safetensors format."),
    )
    parser.add_argument(
        "-o",
        "--output_path",
        required=True,
        help=(
            "Path to the directory where the converted weights and other files "
            "will be saved."),
    )
    parser.add_argument(
        "-s",
        "--scales_path",
        help=(
            "Path to the .pkl file containing the input scales used for "
            "static activation quantization."
        ),
    )
    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path
    scales_path = args.scales_path
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path {input_path} does not exist.")

    # create output directory if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    copy_other_files(input_path, output_path)
    convert_files(input_path, output_path, scales_path)
