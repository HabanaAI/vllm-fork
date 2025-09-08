## Installation

- Reinstall INC
```bash
sudo pip uninstall neural_compressor_pt
git clone https://github.com/intel/neural-compressor.git inc
cd inc
git checkout inc-oss
git log
export INC_PT_ONLY=1
python setup.py pt develop --user

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
This step is not required, as calibration results are already provided in this folder.

### Evaluation Accuray
> [!NOTE]
> Make sure to run the commands under `vllm-fork/scripts`, otherwise the measurement files will not load correctly.

```bash
cd vllm-fork/scripts
# 120b FP8
bash eval_oss_inc.sh

# 120b BF16
bash eval_oss.sh
```
