import torch
from safetensors import safe_open
from safetensors.torch import save_file
from glob import glob
import os

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


def convert_files(input_path, output_path):
    all_safetensors = glob(f"{input_path}/*.safetensors")
    # sort by file name
    all_safetensors.sort()
    for safetensors_path in all_safetensors:
        tensors = {}
        print(f"processing {safetensors_path}")
        with safe_open(safetensors_path, framework="pt",
                       device="cpu") as tensor_file:
            for k in tensor_file:
                tensor = tensor_file.get_tensor(k)
                # tensor = tensor.squeeze(-1)
                if "proj" in k:
                    if k.endswith("weight"):
                        tensor = (tensor.float() * 240.0 / 448.0).to(
                            torch.float8_e4m3fn)
                    elif k.endswith("weight_scale_inv") or k.endswith(
                            "input_scale_inv"):
                        # "scale_inv" in deepseek-r1 is actually "scale"
                        tensor = tensor.float() * 448.0 / 240.0
                    else:
                        raise NotImplementedError(f"Cannot convert {k}")
                else:
                    print(f"skip {k}.")
                tensors[k] = tensor
        new_tensor_path = safetensors_path.replace(input_path, output_path)
        print(f"saving to {new_tensor_path}")
        save_file(tensors, new_tensor_path)


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
    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path

    # create output directory if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    copy_other_files(input_path, output_path)
    convert_files(input_path, output_path)
