import argparse
import glob
import json
import numpy as np
import os
import re

from dataclasses import dataclass
from transformers import AutoConfig
from typing import List, Dict


parser = argparse.ArgumentParser("Unify measurement result script", add_help=False)
parser.add_argument(
    "-m",
    "--model-name-or-path",
    default=None,
    type=str,
    required=True,
    help="path to model  or model name in HF hub",
)
parser.add_argument(
    "-i",
    "--input",
    default=None,
    type=str,
    required=True,
    help="folder path of measurement result",
)
parser.add_argument(
    "-o",
    "--output",
    default=None,
    type=str,
    required=True,
    help="path to save measurement result after unified",
)
parser.add_argument(
    "-r",
    "--rank",
    default=4,
    type=int,
    required=True,
    help="target rank to unify",
)
args = parser.parse_args()

model_info = AutoConfig.from_pretrained(args.model_name_or_path)

@dataclass
class MoeOpInfo:
    num_inputs: int
    num_outputs: int

@dataclass
class Config:
    tp_size: int
    ep_size: int
    num_expert_group: int = 8

    def num_experts_per_rank(self):
        return model_info.num_experts // self.ep_size

    def get_moe_op_info(self):
        num_inputs = 1
        num_outputs = 1 + self.num_experts_per_rank()
        return MoeOpInfo(num_inputs=num_inputs, num_outputs=num_outputs)

@dataclass
class NodeInfo:
    layer_index: int
    rank: int
    group_index: int
    expert_index: int

def filter_moe_nodes(data):
    # end with `MoeOp`
    return {k: v for k, v in data.items() if k.endswith("MoeOp")}

def merge_moe_op(base_data: Dict, target_data: Dict) -> Dict:
    merged_data = {}

    # filter out only MoeOp nodes
    base_data = filter_moe_nodes(base_data)
    target_data = filter_moe_nodes(target_data)

    # assert the keys are same
    assert set(base_data.keys()) == set(
        target_data.keys()
    ), f"Keys are not same"
    all_moe_ops = base_data.keys()
    for moe_op_name in all_moe_ops:
        inputs_0 = base_data.get(moe_op_name, {}).get("inputs", [])
        inputs_1 = target_data.get(moe_op_name, {}).get("inputs", [])

        # Merge inputs: take max element-wise
        merged_inputs = []
        for inp_0, inp_1 in zip(inputs_0, inputs_1):
            merged_inputs.append([[max(a, b) for a, b in zip(inp_0[0], inp_1[0])]])

        # If one input list is longer, use the longer one
        if len(inputs_0) > len(inputs_1):
            merged_inputs.extend(inputs_0[len(inputs_1) :])
        else:
            merged_inputs.extend(inputs_1[len(inputs_0) :])

        # Merge outputs: keep max of the first, retain rest from machine 0, then concatenate all from machine 1
        outputs_0 = base_data.get(moe_op_name, {}).get("outputs", [])
        outputs_1 = target_data.get(moe_op_name, {}).get("outputs", [])

        merged_outputs = []

        if outputs_0 and outputs_1:
            # Take max of the first corresponding output
            merged_outputs.append(
                [[max(a, b) for a, b in zip(outputs_0[0][0], outputs_1[0][0])]]
            )

            # Keep remaining outputs from machine 0
            merged_outputs.extend(outputs_0[1:])

            # Concatenate all outputs from machine 1
            merged_outputs.extend(outputs_1[1:])
        else:
            merged_outputs = outputs_0 or outputs_1

        merged_data[moe_op_name] = {"inputs": merged_inputs, "outputs": merged_outputs}

    return merged_data


def get_moe_op_name(layer_index, expert_index, postfix):
    return f"model.layers.{layer_index}.mlp.experts.hpu_fused_moe.MoeOp.{postfix}.{expert_index}"


def merge_moe_op_experts_list(
    base_data: Dict, merge_data: Dict, is_w13, cur_config, target_config,
) -> Dict:
    merged_data = {}
    if is_w13:
        postfix = "w13_list"
    else:
        postfix = "w2_list"
    for layer_index in range(model_info.num_hidden_layers):
        if (layer_index not in getattr(model_info, "mlp_only_layers", [])) and (
                model_info.num_experts > 0 and
            (layer_index + 1) % model_info.decoder_sparse_step == 0):
            for expert_index_target in range(target_config.num_experts_per_rank()):
                expert_index_cur = cur_config.num_experts_per_rank()
                #print(expert_index_target, expert_index_cur)
                data = merge_data if expert_index_target >= expert_index_cur else base_data
                key1 = get_moe_op_name(
                    layer_index,
                    expert_index_target,
                    postfix,
                )
                key2 = get_moe_op_name(
                    layer_index,
                    expert_index_target % cur_config.num_experts_per_rank(),
                    postfix,
                )
                if key2 in data:
                    merged_data[key1] = data[key2]

    return merged_data

def merge_other(base_data, target_data):

    def merge_entries(entry1, entry2):
        merged_entry = {}

        for field in set(entry1.keys()).union(entry2.keys()):
            if field in entry1 and field in entry2:
                if isinstance(entry1[field], list) and isinstance(entry2[field], list):
                    merged_entry[field] = [
                        max(e1, e2) for e1, e2 in zip(entry1[field], entry2[field])
                    ]
                elif isinstance(entry1[field], dict) and isinstance(
                    entry2[field], dict
                ):
                    merged_entry[field] = merge_entries(entry1[field], entry2[field])
                else:
                    merged_entry[field] = max(entry1[field], entry2[field])
            elif field in entry1:
                merged_entry[field] = entry1[field]
            else:
                merged_entry[field] = entry2[field]

        return merged_entry

    def is_other_keys(key):
        return "MoeOp" not in key

    def merge_dicts(dict1, dict2):
        merged = {}

        assert set(dict1.keys()) == set(dict2.keys()), "Keys are not same"
        filter_keys = list(filter(is_other_keys, dict1.keys()))

        for key in filter_keys:
            if key in dict1 and key in dict2:
                merged[key] = merge_entries(dict1[key], dict2[key])
            elif key in dict1:
                merged[key] = dict1[key]
            else:
                merged[key] = dict2[key]

        return merged

    return merge_dicts(base_data, target_data)

def read_data_from_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data["Nodes"]


def put_together(merged_moe, merged_w13_list, merged_w2_list, merged_other, rank):
    results = {
        "GlobalRank": None,
        "LocalRank": rank,
        "Mode": "DynamicRange",
    }
    merged_json = {}
    merged_json.update(merged_moe)
    merged_json.update(merged_w13_list)
    merged_json.update(merged_w2_list)
    merged_json.update(merged_other)
    # sort result before return
    results["Nodes"] = dict(sorted(merged_json.items()))
    return results


# Function to compare the structure (keys) of two dictionaries (JSON objects)
def compare_structure(json1, json2, path=""):
    # If both are dictionaries, compare the keys
    if isinstance(json1, dict) and isinstance(json2, dict):
        # Compare the keys in both dictionaries
        keys1 = set(json1.keys())
        keys2 = set(json2.keys())

        # Find missing or extra keys
        missing_in_json2 = keys1 - keys2
        missing_in_json1 = keys2 - keys1

        if missing_in_json2:
            print(f"Keys missing in file 2 at {path}: {missing_in_json2}")
        if missing_in_json1:
            print(f"Keys missing in file 1 at {path}: {missing_in_json1}")

        # Recurse for common keys
        for key in keys1.intersection(keys2):
            compare_structure(
                json1[key], json2[key], path=f"{path}.{key}" if path else key
            )

    # If both are lists, compare their lengths
    elif isinstance(json1, list) and isinstance(json2, list):
        if len(json1) != len(json2):
            print(f"List lengths differ at {path}")
        else:
            for i, (item1, item2) in enumerate(zip(json1, json2)):
                compare_structure(item1, item2, path=f"{path}[{i}]")


# Load JSON files
def load_json_file(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


# Main comparison function
def compare_json_structures(file1, file2):
    json_data1 = load_json_file(file1)
    json_data2 = load_json_file(file2)
    compare_structure(json_data1, json_data2)


def dump_all_nodes_name(merged_json):
    mod_list = list(merged_json["Nodes"].keys())
    return sorted(mod_list)


def main():
    measure_result_path = args.input
    measure_files = glob.glob(os.path.join(measure_result_path, "*_mod_list.json"))
    matched = re.match(r"^(\w+)_(\d+)_(\d+)_(\w+)_(\w+)\.json$", os.path.basename(measure_files[0]))
    if matched:
        total_rank = int(matched.group(3))
    else:
        print("File name doesn't match the pattern")
        exit(0)
    
    file_name_pattern = matched.group(1)
    assert total_rank % args.rank == 0
    group_size = total_rank // args.rank
    for i in range(args.rank):
        files_list = [f"{measure_result_path}/{file_name_pattern}_{group_size * i + j}_{total_rank}.json" for j in range(group_size)]
        print(files_list[0])
        base_data = read_data_from_json(files_list[0])
        for file_name in files_list[1:]:
            print("target: ", file_name)
            merge_data = read_data_from_json(file_name)
            merged_moe = merge_moe_op(base_data, merge_data)
            cur_config = Config(tp_size=total_rank, ep_size=total_rank)
            target_config = Config(tp_size=args.rank, ep_size=args.rank)
            merged_w13_list = merge_moe_op_experts_list(
                base_data, merge_data, True, cur_config, target_config,
            )
            merged_w2_list = merge_moe_op_experts_list(
                base_data, merge_data, False, cur_config, target_config,
            )


            merged_other = merge_other(base_data, merge_data)
            base_data = put_together(
                merged_moe, merged_w13_list, merged_w2_list, merged_other, i
            )

    
        merged_json = base_data
        merge_path = args.output
        # create dir
    
        os.makedirs(merge_path, exist_ok=True)
        merged_file = f"{merge_path}/{file_name_pattern}_{i}_{args.rank}.json"
        with open(merged_file, "w") as f:
            json.dump(merged_json, f, indent=4)
        # dump mod list
        mod_list = f"{merge_path}/{file_name_pattern}_{i}_{args.rank}_mod_list.json"
        with open(mod_list, "w") as f:
            json.dump(dump_all_nodes_name(merged_json), f, indent=4)
        print(f"Dumped {merged_file} and {mod_list}")
    
        global_rank = None
        local_rank = i
        mode = ""
        layers = {}
    
        mode = merged_json["Mode"]
        nodes = merged_json["Nodes"]
        output_path = merge_path
        unified_json_name = merged_file
    
        # create unified npz file from the unified json
        unified_npz_path = unified_json_name.replace(".json", ".npz")
        for layer, dlayer in nodes.items():
            layers[layer] = {}
            layers[layer]["inputs"] = [np.array(x) for x in dlayer["inputs"]]
            if dlayer.get("outputs") is not None:
                layers[layer]["outputs"] = [np.array(x) for x in dlayer["outputs"]]
            if (
                dlayer.get("params") is not None
                and dlayer["params"].get("weight") is not None
            ):
                layers[layer]["params"] = {}
                layers[layer]["params"]["weight"] = np.array(dlayer["params"]["weight"])
        df = {
            "GlobalRank": global_rank,
            "LocalRank": local_rank,
            "Mode": mode,
            "Nodes": layers,
        }
        with open(unified_npz_path, "w"):
            np.savez(unified_npz_path, df)
    

if __name__ == "__main__":
    main()
