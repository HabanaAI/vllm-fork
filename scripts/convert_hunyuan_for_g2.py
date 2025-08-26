import torch
from safetensors import safe_open
from safetensors.torch import save_file
from glob import glob
import os
import argparse


FP8_MAX = torch.finfo(torch.float8_e4m3fnuz).max


def calc_maxabs_scale(xmaxabs, fullscale, backoff=1):
    scale = xmaxabs / (fullscale * backoff)
    return scale


def quant_per_tensor(data):
    amax = (torch.abs(data)).max() + 1e-8
    scale = calc_maxabs_scale(amax, FP8_MAX, 1.0)
    scale = scale.to(data.dtype)
    data_fp8 = data / scale
    cliped_qtensor = torch.clamp(data_fp8, -FP8_MAX, FP8_MAX)
    cliped_qtensor_fp8 = cliped_qtensor.to(torch.float8_e4m3fn)
    return cliped_qtensor_fp8, scale.float()


def copy_other_files(input_path, output_path):
    import shutil

    for file in os.listdir(input_path):
        if file.endswith(".json") or file.endswith(".py") or file.endswith(".tiktoken"):
            print(f"copying {file} to {output_path}")
            shutil.copyfile(
                os.path.join(input_path, file),
                os.path.join(output_path, file),
            )


def convert_files(input_path, output_path):
    all_safetensors = glob(f"{input_path}/*.safetensors")
    for safetensors_path in all_safetensors:
        print(f"processing {safetensors_path}")
        tensors = {}
        with safe_open(safetensors_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                if "input_scale" in k:
                    result = (f.get_tensor(k) * 448.0 / 240.0).float()
                    tensors.update({k : result})
                elif "weight_scale" in k:
                    weight_name = k.rstrip("_scale")
                    weight_scale_nv = f.get_tensor(k).float()
                    weight_nv = f.get_tensor(weight_name).float()
                    weight_fp32 = weight_scale_nv * weight_nv
                    weight_fp8, scale = quant_per_tensor(weight_fp32)
                    tensors.update({k: scale})
                    tensors.update({weight_name: weight_fp8})
                elif "proj.weight" in k:
                    continue
                else:
                    result = f.get_tensor(k)
                    tensors.update({k : result})
            new_tensor_path = safetensors_path.replace(input_path, output_path)
            save_file(tensors, new_tensor_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Hunyuan FP8 models with Gaudi2 FP8 format."
    )
    parser.add_argument(
        "-i",
        "--input_path",
        default="/data/Hunyuan-A13B-Instruct-FP8",
        help="Path to the official model weights.",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        default="/data/Hunyuan-A13B-Instruct-FP8",
        help="Path to the output directory.",
    )
    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path

    # create output directory if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    copy_other_files(input_path, output_path)
    convert_files(input_path, output_path)

