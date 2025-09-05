## Installation

- Reinstall INC
```bash
sudo pip uninstall neural_compressor_pt
git clone https://github.com/intel/neural-compressor.git inc
cd inc
git checkout -b inc-oss
git log
export INC_PT_ONLY=1
python setup.py pt --user

```

- Install lm-eval
```bash
git clone https://github.com/yiliu30/lm-eval-fork.git lm-eval
cd lm-eval
git checkout oss
pip install -e .
pip install lm_eval[api]
```

### Calibration (Optional)
This step is optional, as we have attached the calibration results in this folder.

### Evaluation Accuray
```bash
# 120b FP8
cd vllm-fork/scripts
bash eval_oss_inc.sh

# 120b BF16
```