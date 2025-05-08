import numpy as np
import sys

# Load the original .npz file with allow_pickle=True
file_name = sys.argv[1]  #'inc_output_hooks_maxabs_MAXABS_HW_0_1.npz'
# file_name = "/root/vllm-hpu-extension/calibration/g2/qwen2.5-3b-instruct/g2/inc_output_hooks_maxabs_MAXABS_HW_0_1.npz"
data = np.load(file_name, allow_pickle=True)


# Function to replace keys in the dictionary
def replace_keys(d, old_key, new_key):
    if isinstance(d, dict):
        new_dict = {}
        for k, v in d.items():
            new_k = k.replace(old_key, new_key)
            new_dict[new_k] = replace_keys(v, old_key, new_key)
        return new_dict
    elif isinstance(d, list):
        return [replace_keys(item, old_key, new_key) for item in d]
    else:
        return d


# Replace 'model.' with 'language_model.model.' in the dictionary
data_dict = data["arr_0"].item()
updated_dict = replace_keys(data_dict, "model.", "language_model.model.")

# Save the updated dictionary in a new .npz file
# new_file_name = file_name.split(".")[0] + "_new.npz"
new_file_name = "tmp_new.npz"
np.savez(new_file_name, updated_dict)

print(f"Updated dictionary saved to {new_file_name}")
