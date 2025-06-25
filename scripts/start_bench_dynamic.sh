export no_proxy=localhost,127.0.0.1
model_path="/mnt/disk3/yiliu4/DeepSeek-R1-G2-INC-424-Converter207/"

# check existence of model path
if [ ! -d "$model_path" ]; then
    echo "Model path $model_path does not exist."
    model_path="/mnt/weka/data/pytorch/DeepSeek-R1/"
    if [ ! -d "$model_path" ]; then
        echo "Model path $model_path does not exist. Exiting."
        exit 1
    else
        echo "Using model path $model_path"
    fi
fi

echo "Using model path: $model_path"

echo "Running benchmark with dynamic quantization and INC fork..."
bash bench_dynamic_quant.sh --model_path $model_path  --tp_size 8 --use_inc 1
sleep 10

echo "Running benchmark with dynamic quantization..."
bash bench_dynamic_quant.sh --model_path $model_path  --tp_size 8