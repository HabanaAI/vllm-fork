export MODEL_PATH=/workspace/Meta-Llama-3.1-8B-Instruct/

python3 ../examples/online_serving/disagg_examples/disagg_proxy_demo.py \
    --model $MODEL_PATH \
    --prefill localhost:8100 \
    --decode localhost:8200 \
    --port 8000
