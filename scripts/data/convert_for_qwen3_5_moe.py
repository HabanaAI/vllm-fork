import torch
from safetensors import safe_open
from safetensors.torch import save_file
from glob import glob
import os
import json

import argparse

FP8_MAX = 240.0 #torch.finfo(torch.float8_e4m3fn).max


def calc_maxabs_scale(xmaxabs, fullscale, backoff=1):
    scale = xmaxabs / (fullscale * backoff)
    return scale


def dynamic_quant(data, use_unit_quant=False):
    amax = (torch.abs(data)).max(dim=-1).values + 1e-8
    scale = calc_maxabs_scale(amax, FP8_MAX, 1.0)
    if use_unit_quant:
        scale = torch.ones_like(scale)
    scale = scale.to(data.dtype)
    data_fp8 = data / scale.unsqueeze(-1)
    cliped_qtensor = torch.clamp(data_fp8, -FP8_MAX, FP8_MAX)
    cliped_qtensor_fp8 = cliped_qtensor.to(torch.float8_e4m3fn)
    return cliped_qtensor_fp8, scale.float()


def copy_other_files(input_path, output_path):
    import shutil

    for file in os.listdir(input_path):
        if file.endswith(".json") or \
            file.endswith(".txt") or \
            file.endswith("jinja"):
            print(f"copying {file} to {output_path}")
            shutil.copyfile(
                os.path.join(input_path, file),
                os.path.join(output_path, file),
            )


def add_quant_config(output_path):
    json_file = output_path + "/config.json"
    with open(json_file, 'r') as f:
        config = json.load(f)

    config["quantization_config"] = {
        "activation_scheme": "static",
        "fmt": "e4m3",
        "quant_scheme": "channel",
        "quant_method": "fp8"
    }

    with open(json_file, 'w') as f:
        json.dump(config, f, indent=4)

    return config["text_config"]["num_experts"] 


def convert_files(input_path, output_path, input_scale_path, num_experts, use_unit_quant):
    all_safetensors = glob(f"{input_path}/*.safetensors")
    # sort by file name
    all_safetensors.sort()
    model_list={}

    with safe_open(input_scale_path, framework="pt", device="cpu") as input_scale:
        for safetensors_path in all_safetensors:
            print(f"processing {safetensors_path}")
            tensors = {}
            with safe_open(safetensors_path, framework="pt", device="cpu") as tensor_file:
                for k in tensor_file.keys():
                    tensor = tensor_file.get_tensor(k)
                    if len(tensor.shape) == 3 and "conv1d" not in k:
                        for idx in range(num_experts):
                            if "gate_up_proj" in k:
                                gate_weight_name = k.rstrip("gate_up_proj") + str(idx) + ".gate_proj.weight"
                                up_weight_name = k.rstrip("gate_up_proj") + str(idx) + ".up_proj.weight"
                                gate_up_tensor = tensor[idx].reshape(2, -1, tensor.size(-1))
                                gate_tensor = gate_up_tensor[0]
                                up_tensor = gate_up_tensor[1]
                                gate_weight_fp8, gate_weight_scale = dynamic_quant(gate_tensor, use_unit_quant=use_unit_quant)
                                up_weight_fp8, up_weight_scale = dynamic_quant(up_tensor, use_unit_quant=use_unit_quant)
                                gate_weight_scale_name = gate_weight_name + "_scale"
                                up_weight_scale_name = up_weight_name + "_scale"
                                gate_up_input_scale_name = k + "." + str(idx) + ".input_scale"
                                gate_input_scale_tensor = input_scale.get_tensor(gate_up_input_scale_name).float() * 448.0 / 240.0
                                up_input_scale_tensor = gate_input_scale_tensor.clone()
                                gate_input_scale_name = gate_weight_name.rstrip("weight") + "input_scale"
                                up_input_scale_name = up_weight_name.rstrip("weight") + "input_scale"
                                tensors.update({gate_input_scale_name: gate_input_scale_tensor})
                                tensors.update({gate_weight_name: gate_weight_fp8})
                                tensors.update({gate_weight_scale_name: gate_weight_scale})
                                tensors.update({up_input_scale_name: up_input_scale_tensor})
                                tensors.update({up_weight_name: up_weight_fp8})
                                tensors.update({up_weight_scale_name: up_weight_scale})
                                model_list.update({gate_input_scale_name: safetensors_path.split("/")[-1]})
                                model_list.update({gate_weight_name: safetensors_path.split("/")[-1]})
                                model_list.update({gate_weight_scale_name: safetensors_path.split("/")[-1]})
                                model_list.update({up_input_scale_name: safetensors_path.split("/")[-1]})
                                model_list.update({up_weight_name: safetensors_path.split("/")[-1]})
                                model_list.update({up_weight_scale_name: safetensors_path.split("/")[-1]})
                            else:
                                down_weight_name = k.rstrip("down_proj") + str(idx) + ".down_proj.weight"
                                down_tensor = tensor[idx]
                                down_weight_fp8, down_weight_scale = dynamic_quant(down_tensor, use_unit_quant=use_unit_quant)
                                down_weight_scale_name = down_weight_name + "_scale"
                                down_input_scale_name = k + "." + str(idx) + ".input_scale"
                                down_input_scale_tensor = input_scale.get_tensor(down_input_scale_name).float() * 448.0 / 240.0
                                down_input_scale_name = down_weight_name.rstrip("weight") + "input_scale"
                                tensors.update({down_weight_name: down_weight_fp8})
                                tensors.update({down_weight_scale_name: down_weight_scale})
                                tensors.update({down_input_scale_name: down_input_scale_tensor})
                                model_list.update({down_weight_name: safetensors_path.split("/")[-1]})
                                model_list.update({down_weight_scale_name: safetensors_path.split("/")[-1]})
                                model_list.update({down_input_scale_name: safetensors_path.split("/")[-1]})
                    elif ("down_proj" in k or "gate_proj" in k or "up_proj" in k or "q_proj" in k or "k_proj" in k or "v_proj" in k or "o_proj" in k or "out_proj" in k or "in_proj_qkv" in k or "in_proj_z" in k) and "language_model" in k:
                        weight_name = k
                        weight = tensor_file.get_tensor(weight_name)
                        weight_fp8, weight_scale = dynamic_quant(weight, use_unit_quant=use_unit_quant)
                        weight_scale_name = weight_name + "_scale"
                        input_scale_name = weight_name.rstrip("weight") + "input_scale"
                        input_scale_tensor = input_scale.get_tensor(input_scale_name).float() * 448.0 / 240.0
                        tensors.update({input_scale_name: input_scale_tensor})
                        tensors.update({weight_scale_name: weight_scale})
                        tensors.update({weight_name: weight_fp8})
                        model_list.update({input_scale_name: safetensors_path.split("/")[-1]})
                        model_list.update({weight_name: safetensors_path.split("/")[-1]})
                        model_list.update({weight_scale_name: safetensors_path.split("/")[-1]})
                    else:
                        print(f"skip {k}.")
                        tensors.update({k: tensor})
                        model_list.update({k: safetensors_path.split("/")[-1]})
            new_tensor_path = safetensors_path.replace(input_path, output_path)
            save_file(tensors, new_tensor_path)
            print(f"saving to {new_tensor_path}")

    result = {"weight_map" : model_list, "metadata" : {}}
    out_json_path = output_path + "/model.safetensors.index.json"
    with open(out_json_path, "w") as f:
        json.dump(result, f, indent=2)
    f.close


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert tensors to float8 format."
    )
    parser.add_argument(
        "-i",
        "--input_path",
        default="/data/Qwen3.5-397B-A17B",
        help="Path to the official model weights.",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        default="/data/Qwen3.5-397B-A17B-FP8-G2",
        help="Path to the output directory.",
    )
    parser.add_argument(
        "-s",
        "--input_scale_path",
        default="qwen3.5-397b-moe-input-scale.safetensors",
        help="Path to the output directory.",
    )
    parser.add_argument(
        "-u",
        "--unit_quant",
        action="store_true",
        help="Enable Unit FP8 Quant for the entire model"
    )
    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path
    input_scale_path = args.input_scale_path
    use_unit_quant = args.unit_quant

    # create output directory if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    copy_other_files(input_path, output_path)
    num_experts = add_quant_config(output_path)
    convert_files(input_path, output_path, input_scale_path, num_experts, use_unit_quant)
