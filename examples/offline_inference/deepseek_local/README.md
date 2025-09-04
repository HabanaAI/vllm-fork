- QWEN3-30B-3A 1 card
- scalar w/ h2d: hang
- scalar w/o h2d:
- const 


- const
warmup scenario warmup_prompt_bs1_seq8192_ctx0_multimodalF_graphsT took 11.28 seconds
- scalar w/ runtime patching
warmup scenario warmup_prompt_bs1_seq8192_ctx0_multimodalF_graphsT took 293.59 seconds
- scalar w/o runtime patching
warmup scenario warmup_prompt_bs1_seq8192_ctx0_multimodalF_graphsT took 289.47 seconds


- Image: artifactory-kfs.habana-labs.com/docker-local/1.23.0/ubuntu24.04/habanalabs/pytorch-installer-2.8.0:1.23.0-248

# Installation
``` bash
git clone https://github.com/HabanaAI/vllm-fork.git
cd vllm-fork
git checkout ds-example

pip install -r requirements/hpu.txt
VLLM_TARGET_DEVICE=hpu python3 setup.py develop --user
```

# Reproduce CMD
```bash
cd /examples/offline_inference/deepseek
bash reproduce_cmd.sh
```

- const w/o patching
``` bash
[rank0]: Traceback (most recent call last):
[rank0]:   File "/software/users/yiliu4/workdir/vllm-fork/examples/offline_inference/deepseek/deepseek_example.py", line 206, in <module>
[rank0]:     llm = LLM(
[rank0]:   File "/software/users/yiliu4/workdir/vllm-fork/vllm/entrypoints/llm.py", line 244, in __init__
[rank0]:     self.llm_engine = LLMEngine.from_engine_args(
[rank0]:   File "/software/users/yiliu4/workdir/vllm-fork/vllm/engine/llm_engine.py", line 504, in from_engine_args
[rank0]:     return engine_cls.from_vllm_config(
[rank0]:   File "/software/users/yiliu4/workdir/vllm-fork/vllm/engine/llm_engine.py", line 480, in from_vllm_config
[rank0]:     return cls(
[rank0]:   File "/software/users/yiliu4/workdir/vllm-fork/vllm/engine/llm_engine.py", line 274, in __init__
[rank0]:     self.model_executor = executor_class(vllm_config=vllm_config)
[rank0]:   File "/software/users/yiliu4/workdir/vllm-fork/vllm/executor/executor_base.py", line 287, in __init__
[rank0]:     super().__init__(*args, **kwargs)
[rank0]:   File "/software/users/yiliu4/workdir/vllm-fork/vllm/executor/executor_base.py", line 53, in __init__
[rank0]:     self._init_executor()
[rank0]:   File "/software/users/yiliu4/workdir/vllm-fork/vllm/executor/mp_distributed_executor.py", line 126, in _init_executor
[rank0]:     self._run_workers("load_model",
[rank0]:   File "/software/users/yiliu4/workdir/vllm-fork/vllm/executor/mp_distributed_executor.py", line 190, in _run_workers
[rank0]:     driver_worker_output = run_method(self.driver_worker, sent_method,
[rank0]:   File "/software/users/yiliu4/workdir/vllm-fork/vllm/utils.py", line 2784, in run_method
[rank0]:     return func(*args, **kwargs)
[rank0]:   File "/software/users/yiliu4/workdir/vllm-fork/vllm/worker/hpu_worker.py", line 181, in load_model
[rank0]:     self.model_runner.load_model()
[rank0]:   File "/software/users/yiliu4/workdir/vllm-fork/vllm/worker/hpu_model_runner.py", line 1241, in load_model
[rank0]:     htcore.hpu_initialize(self.model,
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/habana_frameworks/torch/core/quantization.py", line 207, in hpu_initialize
[rank0]:     hpu_inference_initialize(model, mark_only_scales_as_const, mark_scales, mark_non_scales, optimizer, args)
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/habana_frameworks/torch/core/quantization.py", line 238, in hpu_inference_initialize
[rank0]:     _check_params_as_const(model=model, mark_scales=mark_only_scales, mark_non_scales=mark_non_scales)
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/habana_frameworks/torch/core/quantization.py", line 140, in _check_params_as_const
[rank0]:     check_constant_mark(param, param_t)
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/habana_frameworks/torch/core/quantization.py", line 132, in check_constant_mark
[rank0]:     is_const = param_t_meta_copy.is_const_tensor
[rank0]: AttributeError: 'NoneType' object has no attribute 'is_const_tensor'
```