# 2000 input, 1000 output, inf rate
#python benchmark_serving.py --backend vllm --dataset-name sonnet --dataset-path sonnet.txt --sonnet-input-len 2000 --sonnet-output-len 1000 --sonnet-prefix-len 100 --host 10.112.242.154 --port 8868 --max-concurrency 64 --request-rate inf --ignore-eos --model /host/mnt/disk001/HF_Models/DeepSeek-R1 --num-prompt 640
#python benchmark_serving.py --backend vllm --dataset-name sonnet --dataset-path sonnet.txt --sonnet-input-len 2000 --sonnet-output-len 1000 --sonnet-prefix-len 100 --host 10.112.242.154 --port 8868 --max-concurrency 64 --request-rate inf --ignore-eos --model /host/mnt/disk001/HF_Models/DeepSeek-R1 --num-prompt 640
#
#python benchmark_serving.py --backend vllm --dataset-name sonnet --dataset-path sonnet.txt --sonnet-input-len 2000 --sonnet-output-len 1000 --sonnet-prefix-len 100 --host 10.112.242.154 --port 8868 --max-concurrency 96 --request-rate inf --ignore-eos --model /host/mnt/disk001/HF_Models/DeepSeek-R1 --num-prompt 960
#python benchmark_serving.py --backend vllm --dataset-name sonnet --dataset-path sonnet.txt --sonnet-input-len 2000 --sonnet-output-len 1000 --sonnet-prefix-len 100 --host 10.112.242.154 --port 8868 --max-concurrency 96 --request-rate inf --ignore-eos --model /host/mnt/disk001/HF_Models/DeepSeek-R1 --num-prompt 960
#
#python benchmark_serving.py --backend vllm --dataset-name sonnet --dataset-path sonnet.txt --sonnet-input-len 2000 --sonnet-output-len 1000 --sonnet-prefix-len 100 --host 10.112.242.154 --port 8868 --max-concurrency 192 --request-rate inf --ignore-eos --model /host/mnt/disk001/HF_Models/DeepSeek-R1 --num-prompt 1920
#python benchmark_serving.py --backend vllm --dataset-name sonnet --dataset-path sonnet.txt --sonnet-input-len 2000 --sonnet-output-len 1000 --sonnet-prefix-len 100 --host 10.112.242.154 --port 8868 --max-concurrency 192 --request-rate inf --ignore-eos --model /host/mnt/disk001/HF_Models/DeepSeek-R1 --num-prompt 1920
#
#python benchmark_serving.py --backend vllm --dataset-name sonnet --dataset-path sonnet.txt --sonnet-input-len 2000 --sonnet-output-len 1000 --sonnet-prefix-len 100 --host 10.112.242.154 --port 8868 --max-concurrency 208 --request-rate inf --ignore-eos --model /host/mnt/disk001/HF_Models/DeepSeek-R1 --num-prompt 2080
#python benchmark_serving.py --backend vllm --dataset-name sonnet --dataset-path sonnet.txt --sonnet-input-len 2000 --sonnet-output-len 1000 --sonnet-prefix-len 100 --host 10.112.242.154 --port 8868 --max-concurrency 208 --request-rate inf --ignore-eos --model /host/mnt/disk001/HF_Models/DeepSeek-R1 --num-prompt 2080
#
#python benchmark_serving.py --backend vllm --dataset-name sonnet --dataset-path sonnet.txt --sonnet-input-len 2000 --sonnet-output-len 1000 --sonnet-prefix-len 100 --host 10.112.242.154 --port 8868 --max-concurrency 224 --request-rate inf --ignore-eos --model /host/mnt/disk001/HF_Models/DeepSeek-R1 --num-prompt 2240
#python benchmark_serving.py --backend vllm --dataset-name sonnet --dataset-path sonnet.txt --sonnet-input-len 2000 --sonnet-output-len 1000 --sonnet-prefix-len 100 --host 10.112.242.154 --port 8868 --max-concurrency 224 --request-rate inf --ignore-eos --model /host/mnt/disk001/HF_Models/DeepSeek-R1 --num-prompt 2240
#
#python benchmark_serving.py --backend vllm --dataset-name sonnet --dataset-path sonnet.txt --sonnet-input-len 2000 --sonnet-output-len 1000 --sonnet-prefix-len 100 --host 10.112.242.154 --port 8868 --max-concurrency 256 --request-rate inf --ignore-eos --model /host/mnt/disk001/HF_Models/DeepSeek-R1 --num-prompt 2560
#python benchmark_serving.py --backend vllm --dataset-name sonnet --dataset-path sonnet.txt --sonnet-input-len 2000 --sonnet-output-len 1000 --sonnet-prefix-len 100 --host 10.112.242.154 --port 8868 --max-concurrency 256 --request-rate inf --ignore-eos --model /host/mnt/disk001/HF_Models/DeepSeek-R1 --num-prompt 2560
#
#python benchmark_serving.py --backend vllm --dataset-name sonnet --dataset-path sonnet.txt --sonnet-input-len 2000 --sonnet-output-len 1000 --sonnet-prefix-len 100 --host 10.112.242.154 --port 8868 --max-concurrency 608 --request-rate inf --ignore-eos --model /host/mnt/disk001/HF_Models/DeepSeek-R1 --num-prompt 6080
#python benchmark_serving.py --backend vllm --dataset-name sonnet --dataset-path sonnet.txt --sonnet-input-len 2000 --sonnet-output-len 1000 --sonnet-prefix-len 100 --host 10.112.242.154 --port 8868 --max-concurrency 608 --request-rate inf --ignore-eos --model /host/mnt/disk001/HF_Models/DeepSeek-R1 --num-prompt 6080
#
## 2000 input, 1000 output, rate 9 and 15
#python benchmark_serving.py --backend vllm --dataset-name sonnet --dataset-path sonnet.txt --sonnet-input-len 2000 --sonnet-output-len 1000 --sonnet-prefix-len 100 --host 10.112.242.154 --port 8868 --max-concurrency 608 --request-rate 9 --ignore-eos --model /host/mnt/disk001/HF_Models/DeepSeek-R1 --num-prompt 6080
#python benchmark_serving.py --backend vllm --dataset-name sonnet --dataset-path sonnet.txt --sonnet-input-len 2000 --sonnet-output-len 1000 --sonnet-prefix-len 100 --host 10.112.242.154 --port 8868 --max-concurrency 608 --request-rate 9 --ignore-eos --model /host/mnt/disk001/HF_Models/DeepSeek-R1 --num-prompt 6080
#
#python benchmark_serving.py --backend vllm --dataset-name sonnet --dataset-path sonnet.txt --sonnet-input-len 2000 --sonnet-output-len 1000 --sonnet-prefix-len 100 --host 10.112.242.154 --port 8868 --max-concurrency 608 --request-rate 15 --ignore-eos --model /host/mnt/disk001/HF_Models/DeepSeek-R1 --num-prompt 6080
#python benchmark_serving.py --backend vllm --dataset-name sonnet --dataset-path sonnet.txt --sonnet-input-len 2000 --sonnet-output-len 1000 --sonnet-prefix-len 100 --host 10.112.242.154 --port 8868 --max-concurrency 608 --request-rate 15 --ignore-eos --model /host/mnt/disk001/HF_Models/DeepSeek-R1 --num-prompt 6080
#
## 2000 input, 1000 output, inf rate, higher concurrency
#python benchmark_serving.py --backend vllm --dataset-name sonnet --dataset-path sonnet.txt --sonnet-input-len 2000 --sonnet-output-len 1000 --sonnet-prefix-len 100 --host 10.112.242.154 --port 8868 --max-concurrency 640 --request-rate inf --ignore-eos --model /host/mnt/disk001/HF_Models/DeepSeek-R1 --num-prompt 6400
#python benchmark_serving.py --backend vllm --dataset-name sonnet --dataset-path sonnet.txt --sonnet-input-len 2000 --sonnet-output-len 1000 --sonnet-prefix-len 100 --host 10.112.242.154 --port 8868 --max-concurrency 640 --request-rate inf --ignore-eos --model /host/mnt/disk001/HF_Models/DeepSeek-R1 --num-prompt 6400
#
#python benchmark_serving.py --backend vllm --dataset-name sonnet --dataset-path sonnet.txt --sonnet-input-len 2000 --sonnet-output-len 1000 --sonnet-prefix-len 100 --host 10.112.242.154 --port 8868 --max-concurrency 672 --request-rate inf --ignore-eos --model /host/mnt/disk001/HF_Models/DeepSeek-R1 --num-prompt 6720
#python benchmark_serving.py --backend vllm --dataset-name sonnet --dataset-path sonnet.txt --sonnet-input-len 2000 --sonnet-output-len 1000 --sonnet-prefix-len 100 --host 10.112.242.154 --port 8868 --max-concurrency 672 --request-rate inf --ignore-eos --model /host/mnt/disk001/HF_Models/DeepSeek-R1 --num-prompt 6720




# 3500 input, 1000 output, inf rate
python benchmark_serving.py --backend vllm --dataset-name sonnet --dataset-path sonnet.txt --sonnet-input-len 3500 --sonnet-output-len 1000 --sonnet-prefix-len 100 --host 10.112.242.154 --port 8868 --max-concurrency 64 --request-rate inf --ignore-eos --model /host/mnt/disk001/HF_Models/DeepSeek-R1 --num-prompt 640
python benchmark_serving.py --backend vllm --dataset-name sonnet --dataset-path sonnet.txt --sonnet-input-len 3500 --sonnet-output-len 1000 --sonnet-prefix-len 100 --host 10.112.242.154 --port 8868 --max-concurrency 64 --request-rate inf --ignore-eos --model /host/mnt/disk001/HF_Models/DeepSeek-R1 --num-prompt 640
python benchmark_serving.py --backend vllm --dataset-name sonnet --dataset-path sonnet.txt --sonnet-input-len 3500 --sonnet-output-len 1000 --sonnet-prefix-len 100 --host 10.112.242.154 --port 8868 --max-concurrency 64 --request-rate inf --ignore-eos --model /host/mnt/disk001/HF_Models/DeepSeek-R1 --num-prompt 640

python benchmark_serving.py --backend vllm --dataset-name sonnet --dataset-path sonnet.txt --sonnet-input-len 3500 --sonnet-output-len 1000 --sonnet-prefix-len 100 --host 10.112.242.154 --port 8868 --max-concurrency 192 --request-rate inf --ignore-eos --model /host/mnt/disk001/HF_Models/DeepSeek-R1 --num-prompt 1920
python benchmark_serving.py --backend vllm --dataset-name sonnet --dataset-path sonnet.txt --sonnet-input-len 3500 --sonnet-output-len 1000 --sonnet-prefix-len 100 --host 10.112.242.154 --port 8868 --max-concurrency 192 --request-rate inf --ignore-eos --model /host/mnt/disk001/HF_Models/DeepSeek-R1 --num-prompt 1920
python benchmark_serving.py --backend vllm --dataset-name sonnet --dataset-path sonnet.txt --sonnet-input-len 3500 --sonnet-output-len 1000 --sonnet-prefix-len 100 --host 10.112.242.154 --port 8868 --max-concurrency 192 --request-rate inf --ignore-eos --model /host/mnt/disk001/HF_Models/DeepSeek-R1 --num-prompt 1920

python benchmark_serving.py --backend vllm --dataset-name sonnet --dataset-path sonnet.txt --sonnet-input-len 3500 --sonnet-output-len 1000 --sonnet-prefix-len 100 --host 10.112.242.154 --port 8868 --max-concurrency 256 --request-rate inf --ignore-eos --model /host/mnt/disk001/HF_Models/DeepSeek-R1 --num-prompt 2560
python benchmark_serving.py --backend vllm --dataset-name sonnet --dataset-path sonnet.txt --sonnet-input-len 3500 --sonnet-output-len 1000 --sonnet-prefix-len 100 --host 10.112.242.154 --port 8868 --max-concurrency 256 --request-rate inf --ignore-eos --model /host/mnt/disk001/HF_Models/DeepSeek-R1 --num-prompt 2560
python benchmark_serving.py --backend vllm --dataset-name sonnet --dataset-path sonnet.txt --sonnet-input-len 3500 --sonnet-output-len 1000 --sonnet-prefix-len 100 --host 10.112.242.154 --port 8868 --max-concurrency 256 --request-rate inf --ignore-eos --model /host/mnt/disk001/HF_Models/DeepSeek-R1 --num-prompt 2560

# 1000 input, 2000 output, inf rate
#python benchmark_serving.py --backend vllm --dataset-name sonnet --dataset-path sonnet.txt --sonnet-input-len 1000 --sonnet-output-len 2000 --sonnet-prefix-len 100 --host 10.112.242.154 --port 8868 --max-concurrency 544 --request-rate inf --ignore-eos --model /host/mnt/disk001/HF_Models/DeepSeek-R1 --num-prompt 5440
#python benchmark_serving.py --backend vllm --dataset-name sonnet --dataset-path sonnet.txt --sonnet-input-len 1000 --sonnet-output-len 2000 --sonnet-prefix-len 100 --host 10.112.242.154 --port 8868 --max-concurrency 544 --request-rate inf --ignore-eos --model /host/mnt/disk001/HF_Models/DeepSeek-R1 --num-prompt 5440
#
#python benchmark_serving.py --backend vllm --dataset-name sonnet --dataset-path sonnet.txt --sonnet-input-len 1000 --sonnet-output-len 2000 --sonnet-prefix-len 100 --host 10.112.242.154 --port 8868 --max-concurrency 560 --request-rate inf --ignore-eos --model /host/mnt/disk001/HF_Models/DeepSeek-R1 --num-prompt 5600
#python benchmark_serving.py --backend vllm --dataset-name sonnet --dataset-path sonnet.txt --sonnet-input-len 1000 --sonnet-output-len 2000 --sonnet-prefix-len 100 --host 10.112.242.154 --port 8868 --max-concurrency 560 --request-rate inf --ignore-eos --model /host/mnt/disk001/HF_Models/DeepSeek-R1 --num-prompt 5600
#
#python benchmark_serving.py --backend vllm --dataset-name sonnet --dataset-path sonnet.txt --sonnet-input-len 1000 --sonnet-output-len 2000 --sonnet-prefix-len 100 --host 10.112.242.154 --port 8868 --max-concurrency 576 --request-rate inf --ignore-eos --model /host/mnt/disk001/HF_Models/DeepSeek-R1 --num-prompt 5760
#python benchmark_serving.py --backend vllm --dataset-name sonnet --dataset-path sonnet.txt --sonnet-input-len 1000 --sonnet-output-len 2000 --sonnet-prefix-len 100 --host 10.112.242.154 --port 8868 --max-concurrency 576 --request-rate inf --ignore-eos --model /host/mnt/disk001/HF_Models/DeepSeek-R1 --num-prompt 5760
#
#python benchmark_serving.py --backend vllm --dataset-name sonnet --dataset-path sonnet.txt --sonnet-input-len 1000 --sonnet-output-len 2000 --sonnet-prefix-len 100 --host 10.112.242.154 --port 8868 --max-concurrency 592 --request-rate inf --ignore-eos --model /host/mnt/disk001/HF_Models/DeepSeek-R1 --num-prompt 5920
#python benchmark_serving.py --backend vllm --dataset-name sonnet --dataset-path sonnet.txt --sonnet-input-len 1000 --sonnet-output-len 2000 --sonnet-prefix-len 100 --host 10.112.242.154 --port 8868 --max-concurrency 592 --request-rate inf --ignore-eos --model /host/mnt/disk001/HF_Models/DeepSeek-R1 --num-prompt 5920
#
#python benchmark_serving.py --backend vllm --dataset-name sonnet --dataset-path sonnet.txt --sonnet-input-len 1000 --sonnet-output-len 2000 --sonnet-prefix-len 100 --host 10.112.242.154 --port 8868 --max-concurrency 608 --request-rate inf --ignore-eos --model /host/mnt/disk001/HF_Models/DeepSeek-R1 --num-prompt 6080
#python benchmark_serving.py --backend vllm --dataset-name sonnet --dataset-path sonnet.txt --sonnet-input-len 1000 --sonnet-output-len 2000 --sonnet-prefix-len 100 --host 10.112.242.154 --port 8868 --max-concurrency 608 --request-rate inf --ignore-eos --model /host/mnt/disk001/HF_Models/DeepSeek-R1 --num-prompt 6080
#
#python benchmark_serving.py --backend vllm --dataset-name sonnet --dataset-path sonnet.txt --sonnet-input-len 1000 --sonnet-output-len 2000 --sonnet-prefix-len 100 --host 10.112.242.154 --port 8868 --max-concurrency 624 --request-rate inf --ignore-eos --model /host/mnt/disk001/HF_Models/DeepSeek-R1 --num-prompt 6240
#python benchmark_serving.py --backend vllm --dataset-name sonnet --dataset-path sonnet.txt --sonnet-input-len 1000 --sonnet-output-len 2000 --sonnet-prefix-len 100 --host 10.112.242.154 --port 8868 --max-concurrency 624 --request-rate inf --ignore-eos --model /host/mnt/disk001/HF_Models/DeepSeek-R1 --num-prompt 6240
#
#python benchmark_serving.py --backend vllm --dataset-name sonnet --dataset-path sonnet.txt --sonnet-input-len 1000 --sonnet-output-len 2000 --sonnet-prefix-len 100 --host 10.112.242.154 --port 8868 --max-concurrency 640 --request-rate inf --ignore-eos --model /host/mnt/disk001/HF_Models/DeepSeek-R1 --num-prompt 6400
#python benchmark_serving.py --backend vllm --dataset-name sonnet --dataset-path sonnet.txt --sonnet-input-len 1000 --sonnet-output-len 2000 --sonnet-prefix-len 100 --host 10.112.242.154 --port 8868 --max-concurrency 640 --request-rate inf --ignore-eos --model /host/mnt/disk001/HF_Models/DeepSeek-R1 --num-prompt 6400
#
#python benchmark_serving.py --backend vllm --dataset-name sonnet --dataset-path sonnet.txt --sonnet-input-len 1000 --sonnet-output-len 2000 --sonnet-prefix-len 100 --host 10.112.242.154 --port 8868 --max-concurrency 672 --request-rate inf --ignore-eos --model /host/mnt/disk001/HF_Models/DeepSeek-R1 --num-prompt 6720
#python benchmark_serving.py --backend vllm --dataset-name sonnet --dataset-path sonnet.txt --sonnet-input-len 1000 --sonnet-output-len 2000 --sonnet-prefix-len 100 --host 10.112.242.154 --port 8868 --max-concurrency 672 --request-rate inf --ignore-eos --model /host/mnt/disk001/HF_Models/DeepSeek-R1 --num-prompt 6720
#
#python benchmark_serving.py --backend vllm --dataset-name sonnet --dataset-path sonnet.txt --sonnet-input-len 1000 --sonnet-output-len 2000 --sonnet-prefix-len 100 --host 10.112.242.154 --port 8868 --max-concurrency 704 --request-rate inf --ignore-eos --model /host/mnt/disk001/HF_Models/DeepSeek-R1 --num-prompt 7040
#python benchmark_serving.py --backend vllm --dataset-name sonnet --dataset-path sonnet.txt --sonnet-input-len 1000 --sonnet-output-len 2000 --sonnet-prefix-len 100 --host 10.112.242.154 --port 8868 --max-concurrency 704 --request-rate inf --ignore-eos --model /host/mnt/disk001/HF_Models/DeepSeek-R1 --num-prompt 7040
#
