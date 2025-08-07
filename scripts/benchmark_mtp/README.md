```
# Baseline
python mtp.py

# Optimized (with delayed speculative decode)
HPU_VLLM_DELAY_SPECDECODE=True python mtp.py
```