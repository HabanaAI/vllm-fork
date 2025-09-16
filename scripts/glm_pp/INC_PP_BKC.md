### Install

- https://github.com/intel/neural-compressor/tree/pp-glm

### Calibration
```
bash run_inc.sh 
```

### Post-Process
Unify and expand measurements for tensor/parallel configurations:
```bash
export VLLM_HPU_EXT_PATH=/mnt/disk3/yiliu4/vllm-hpu-extension
python $VLLM_HPU_EXT_PATH/calibration/step-5-unify_measurements.py --measurements ./air-calibs/w8-tp8  --rank 1 --out ./air-calibs/w1-tp1 --use_expert_paral --skip_unify_scales
python $VLLM_HPU_EXT_PATH/calibration/step-6-expand-measurements.py --measurements ./air-calibs/w1-tp1 --out  ./air-calibs/w1-tp1-expand-w4-tp4 --target_world_size 4
```

### Inference

```bash
tp_size=4
pp_size=2
export INC_ENABLE_TP_RANK_INFO=1
export QUANT_CONFIG="./quant_configs/inc_quant_w4.json"
bash run_pp_glm.sh
```
> [!IMPORTANT]
> For PP scenarios, please set `INC_ENABLE_TP_RANK_INFO=1` to obtain rank information from the TP group instead of the world group.