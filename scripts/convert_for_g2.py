import torch
from safetensors import safe_open
from safetensors.torch import save_file
from glob import glob
import os


import argparse
import shutil


def copy_other_files(input_path, output_path):
    for root, dirs, files in os.walk(input_path):
        rel_dir = os.path.relpath(root, input_path)
        dest_dir = os.path.join(output_path, rel_dir)
        os.makedirs(dest_dir, exist_ok=True)

        for file in files:
            if file.endswith(".safetensors"):
                continue
            src_file = os.path.join(root, file)
            dst_file = os.path.join(dest_dir, file)
            print(f"Copying file {src_file} to {dst_file}")
            shutil.copyfile(src_file, dst_file)


def convert_files(input_path, output_path):
    all_safetensors = glob(f"{input_path}/**/*.safetensors", recursive=True)  # recursive glob
    # sort by file name
    all_safetensors.sort()
    for safetensors_path in all_safetensors:
        tensors = {}
        print(f"processing {safetensors_path}")
        with safe_open(
            safetensors_path, framework="pt", device="cpu"
        ) as tensor_file:
            for k in tensor_file.keys():
                tensor = tensor_file.get_tensor(k)
                # tensor = tensor.squeeze(-1)
                if "proj" in k:
                    if k.endswith("weight"):
                        tensor = (tensor.float() * 240.0 / 448.0).to(
                            torch.float8_e4m3fn
                        )
                    elif k.endswith("weight_scale_inv") or k.endswith(
                        "input_scale_inv"
                    ):
                        # "scale_inv" in deepseek-r1 is actually "scale"
                        tensor = tensor.float() * 448.0 / 240.0
                    else:
                        raise NotImplementedError(f"Cannot convert {k}")
                else:
                    print(f"skip {k}.")
                tensors[k] = tensor
        rel_path = os.path.relpath(safetensors_path, input_path)
        new_tensor_path = os.path.join(output_path, rel_path)
        os.makedirs(os.path.dirname(new_tensor_path), exist_ok=True)
        print(f"saving to {new_tensor_path}")
        save_file(tensors, new_tensor_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert tensors to float8 format."
    )
    parser.add_argument(
        "-i",
        "--input_path",
        default="/mnt/disk2/hf_models/DeepSeek-R1",
        help="Path to the official model weights.",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        default="/mnt/disk2/hf_models/DeepSeek-R1-G2",
        help="Path to the output directory.",
    )
    args = parser.parse_args()
    input_path = os.path.normpath(args.input_path)
    output_path = os.path.normpath(args.output_path)

    # create output directory if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    copy_other_files(input_path, output_path)
    convert_files(input_path, output_path)
