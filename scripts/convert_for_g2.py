import torch
from safetensors import safe_open
from safetensors.torch import save_file
from glob import glob
import os
import shutil

# input_path = "/models/DeepSeek-R1-BF16-layer5-w8afp8-static"
# output_path = "/models/DeepSeek-R1-BF16-layer5-w8afp8-static-G2"
input_path = "/models/DeepSeek-R1-BF16-layer5-w8afp8-static-no-ste"
output_path = "/models/DeepSeek-R1-BF16-layer5-w8afp8-static-no-ste-G2"
input_path = "/models/DeepSeek-R1-BF16-w8afp8-static-no-ste"
output_path = "/models/DeepSeek-R1-BF16-w8afp8-static-no-ste-G2"

weight_factor = (
    torch.finfo(torch.float8_e4m3fnuz).max / torch.finfo(torch.float8_e4m3fn).max
)
scale_factor = 1.0 / weight_factor
scale_inv_factor = weight_factor

os.makedirs(output_path, exist_ok=False)

for safetensors_path in glob(f"{input_path}/*.safetensors"):
    tensors = {}
    new_tensor_path = safetensors_path.replace(input_path, output_path)
    if not safetensors_path.endswith(".safetensors"):
        shutil.copy(safetensors_path, new_tensor_path)
        print(f"copying file {safetensors_path}")
        continue

    print(f"processing {safetensors_path}")
    with safe_open(safetensors_path, framework="pt", device="cpu") as tensor_file:
        for k in tensor_file.keys():
            tensor = tensor_file.get_tensor(k)
            # tensor = tensor.squeeze(-1)
            if "proj" in k:
                if k.endswith("weight"):
                    tensor = (tensor.float() * weight_factor).to(torch.float8_e4m3fn)
                elif k.endswith("weight_scale") or k.endswith("input_scale"):
                    tensor = tensor.float() * scale_factor
                elif k.endswith("weight_scale_inv") or k.endswith("input_scale_inv"):
                    # "scale_inv" in deepseek-r1 is actually "scale"
                    tensor = (tensor.float() * scale_factor)  
                else:
                    raise NotImplementedError(f"Cannot covert {k}")
            else:
                print(f"skip {k}.")
            k = k.replace("input_scale_inv", "input_scale")
            tensors[k] = tensor
    print(f"saving to {new_tensor_path}")
    save_file(tensors, new_tensor_path)
