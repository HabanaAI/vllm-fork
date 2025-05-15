import torch
from safetensors import safe_open
from safetensors.torch import save_file
from glob import glob
import os
import pickle
import json

import argparse


def copy_non_safetensor_files(input_path, output_path):
    import shutil
    shutil.copytree(
        input_path,
        output_path,
        dirs_exist_ok=True,
        ignore=lambda src, names:
        [name for name in names if name.endswith(".safetensors")],
    )


def get_input_scales(pkl_path):
    input_scales = {}
    assert pkl_path is not None and os.path.exists(pkl_path)
    with open(pkl_path, 'rb') as file:
        input_scales = pickle.load(file)
    return input_scales


def update_static_config_json(output_path):
    config_file = os.path.join(output_path, "config.json")
    with open(config_file, "rb") as f:
        config = json.load(f)
    config["quantization_config"]["activation_scheme"] = "static"
    with open(config_file, "w") as f:
        json.dump(config, f, indent=4)


def convert_and_rescale_weights(input_path, output_path, scales_path):

    print("Copying non-safetensor files...")
    copy_non_safetensor_files(input_path, output_path)

    is_static = scales_path is not None

    if is_static:
        print(f"Loading input scales from {scales_path}")
        input_scales = get_input_scales(scales_path)

        print("Updating config.json to set activation_scheme to static...")
        update_static_config_json(output_path)
    else:
        input_scales = {}

    safetensor_files = glob(f"{input_path}/*.safetensors")
    safetensor_files.sort()

    tensor_mapping = {}

    for safetensor_file in safetensor_files:
        tensors = {}
        safetensor_name = os.path.basename(safetensor_file)
        print(f"Processing {safetensor_file}")
        with safe_open(safetensor_file, framework="pt",
                       device="cpu") as tensor_file:
            for tensor_name in tensor_file.keys():
                tensor = tensor_file.get_tensor(tensor_name)
                if "proj" in tensor_name:
                    if tensor_name.endswith("weight"):
                        tensor = (tensor.float() * 240.0 / 448.0).to(
                            torch.float8_e4m3fn)
                        input_scale_key = tensor_name.replace(
                            ".weight", ".input_scale")
                        if input_scale_key in input_scales:
                            input_scale = input_scales.pop(input_scale_key)
                            input_scale = input_scale * 448.0 / 240.0
                            tensors[input_scale_key] = input_scale
                            tensor_mapping[input_scale_key] = safetensor_name
                    elif tensor_name.endswith("weight_scale_inv"):
                        tensor = tensor.float() * 448.0 / 240.0
                    else:
                        raise NotImplementedError(
                            f"Cannot convert {tensor_name}")
                else:
                    print(f"Skipping conversion for {tensor_name}")
                tensors[tensor_name] = tensor
                tensor_mapping[tensor_name] = safetensor_name
        output_tensor_path = safetensor_file.replace(input_path, output_path)
        print(f"Saving to {output_tensor_path}")
        save_file(tensors, output_tensor_path)

    if input_scales:
        print("Warning: the following input_scales are unused:")
        for input_scale_key in input_scales:
            print(input_scale_key)

    if is_static:
        tensor_mapping_file = os.path.join(output_path,
                                           "model.safetensors.index.json")
        print(f"Updating tensor mapping in {tensor_mapping_file}")
        with open(tensor_mapping_file, "rb") as f:
            state_dict_mapping = json.load(f)
        state_dict_mapping["weight_map"] = tensor_mapping
        with open(tensor_mapping_file, "w") as f:
            json.dump(state_dict_mapping, f, indent=4, sort_keys=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Convert FP8 E4M3FN weights to FP8 E4M3FNUZ format for Gaudi2. "
            "Rescale the corresponding weight_scale_inv values. "
            "If --scales_path is provided, merge input_scales from the file "
            "into the output safetensors files and set activation_scheme in "
            "config.json to 'static'."
        )
    )
    parser.add_argument(
        "-i",
        "--input_path",
        required=True,
        help=(
            "Path to the directory containing the original model weights "
            "in .safetensors format."
        ),
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
        default=None,
        help=("Path to the .pkl file containing the input scales used for "
              "static activation quantization."),
    )
    args = parser.parse_args()

    input_path = args.input_path
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path {input_path} does not exist.")

    output_path = args.output_path
    # create output directory if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    scales_path = args.scales_path
    if scales_path and not os.path.exists(scales_path):
        raise FileNotFoundError(f"Scales path {scales_path} does not exist.")

    convert_and_rescale_weights(input_path, output_path, scales_path)
