export MODEL_PATH=/software/data/models/DeepSeek-R1-BF16-w8afp8-static-no-ste-G2/

python3 examples/online_serving/disagg_examples/dp_proxy_demo.py \
    --model $MODEL_PATH \
    --prefill 10.112.110.50:8100 \
    --decode 10.112.110.50:8801 10.112.110.50:8802 \
    --port 8123