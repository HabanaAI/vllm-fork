import torch
from safetensors import safe_open
from safetensors.torch import save_file
from glob import glob
import os

# input_path = "/models/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic"
# output_path = "/models/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic-G2"
input_path = "/models/Llama-4-Maverick-17B-128E-Instruct-FP8"
output_path = "/models/Llama-4-Maverick-17B-128E-Instruct-FP8-G2"

weight_factor = (
    torch.finfo(torch.float8_e4m3fnuz).max / torch.finfo(torch.float8_e4m3fn).max
)
scale_factor = 1.0 / weight_factor
scale_inv_factor = weight_factor

for safetensors_path in glob(f"{input_path}/*.safetensors"):
    tensors = {}
    print(f"processing {safetensors_path}")
    with safe_open(safetensors_path, framework="pt", device="cpu") as tensor_file:
        for k in tensor_file.keys():
            tensor = tensor_file.get_tensor(k)
            # print(f'{k}:{tensor.dtype}')
            if tensor.dtype == torch.float8_e4m3fn:
                tensor = (tensor.float() * weight_factor).to(torch.float8_e4m3fn)
            elif k.endswith("_scale"):
                tensor = tensor.float() * scale_factor
            else:
                print(f"skip {k}.")
            tensors[k] = tensor
    new_tensor_path = safetensors_path.replace(input_path, output_path)
    print(f"saving to {new_tensor_path}")
    save_file(tensors, new_tensor_path)
