from ..utils import compare_two_settings


def test_cpu_offload():
    compare_two_settings("/mnt/weka/data/pytorch/llama2/Llama-2-7b-hf/", [],
                         ["--cpu-offload-gb", "4"])
