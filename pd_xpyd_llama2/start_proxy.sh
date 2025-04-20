export MODEL_PATH=/mnt/weka/data/pytorch/llama2/Llama-2-7b-chat-hf/

python3 ../examples/online_serving/disagg_examples/disagg_proxy_demo.py \
    --model $MODEL_PATH \
    --prefill localhost:8100 \
    --decode localhost:8200 \
    --port 8000
