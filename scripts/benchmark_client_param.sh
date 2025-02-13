#!/bin/bash
set -x

# Usage: source benchmark_client_param.sh 
#        test_benchmark_serving INPUT_LEN OUTPUT_LEN MAX_CONCURRENCY NUM_PROMPTS [LEN_RATIO] [HOST] [PORT] [MODEL_PATH] [EVALS]
# Defaults: LEN_RATIO=1.0, HOST=127.0.0.1, PORT=8688, MODEL_PATH=${MODEL_PATH:-/root/.cache/huggingface/DeepSeek-R1-BF16-w8afp8-dynamic-no-ste-G2}, EVALS=""

test_benchmark_client_serving() {
  export PT_HPU_LAZY_MODE=1
  INPUT_LEN=$1
  OUTPUT_LEN=$2
  MAX_CONCURRENCY=$3
  NUM_PROMPTS=$4
  LEN_RATIO=${5:-1.0}
  HOST=${6:-127.0.0.1}
  PORT=${7:-8688}
  MODEL_PATH=${8:-${MODEL_PATH:-/root/.cache/huggingface/DeepSeek-R1-BF16-w8afp8-dynamic-no-ste-G2}}
  RESULTS_DIR=${9:-logs/test-results}
  mkdir -p "$RESULTS_DIR"

  export no_proxy=localhost,${HOST},10.239.129.9

  # Run serving benchmark
  echo "Running serving benchmark: input=${INPUT_LEN}, output=${OUTPUT_LEN}, concurrency=${MAX_CONCURRENCY}, prompts=${NUM_PROMPTS}, ratio=${LEN_RATIO}"
  TIMESTAMP=$(TZ='Asia/Kolkata' date +%F-%H-%M-%S)
  LOG_BASE="benchmark_${NUM_PROMPTS}prompts_${MAX_CONCURRENCY}bs_in${INPUT_LEN}_out${OUTPUT_LEN}_ratio${LEN_RATIO}_${TIMESTAMP}"

  python3 ../benchmarks/benchmark_serving.py \
      --backend vllm \
      --model "${MODEL_PATH}" \
      --trust-remote-code \
      --host "${HOST}" \
      --port "${PORT}" \
      --dataset-name random \
      --random-input-len "${INPUT_LEN}" \
      --random-output-len "${OUTPUT_LEN}" \
      --random-range-ratio "${LEN_RATIO}" \
      --max-concurrency "${MAX_CONCURRENCY}" \
      --num-prompts "${NUM_PROMPTS}" \
      --request-rate inf \
      --seed 0 \
      --ignore-eos \
      --save-result \
      --result-filename "${RESULTS_DIR}/${LOG_BASE}.json"
}

test_benchmark_client_accuracy() {
  export PT_HPU_LAZY_MODE=1
  MAX_CONCURRENCY=$1
  HOST=${2:-127.0.0.1}
  PORT=${3:-8688}
  MODEL_PATH=${4:-${MODEL_PATH:-/root/.cache/huggingface/DeepSeek-R1-BF16-w8afp8-dynamic-no-ste-G2}}
  EVALS=${5:-}
  RESULTS_DIR=${6:-logs/test-results}
  mkdir -p "$RESULTS_DIR"

  export no_proxy=localhost,${HOST},10.239.129.9

  # Parse and run optional LM-Eval tasks
  if [[ -n "$EVALS" && "$EVALS" != "none" ]]; then
    # Set up proxies and install lm_eval
    export no_proxy=localhost,127.0.0.1,10.239.129.9
    export http_proxy=http://proxy.ims.intel.com:911
    export https_proxy=http://proxy.ims.intel.com:911
    pip install lm_eval[api]
    export HF_ENDPOINT=https://huggingface.co
    export HF_ALLOW_CODE_EVAL=1

    IFS=',' read -ra TASKS <<< "$EVALS"
    for task_spec in "${TASKS[@]}"; do
      # task_spec format: name[:samples]
      IFS=':' read -r TASK_NAME SAMPLES <<< "$task_spec"
      CMD=(lm_eval --model local-completions --tasks "$TASK_NAME" \
        --model_args model="${MODEL_PATH}",base_url="http://${HOST}:${PORT}/v1/completions",num_concurrent="${MAX_CONCURRENCY}" \
        --batch_size 1 --confirm_run_unsafe_code --log_samples \
        --output_path "${RESULTS_DIR}/lm_eval_output_${TASK_NAME}_${TIMESTAMP}.json")
      if [[ -n "$SAMPLES" ]]; then
        CMD+=(--limit "$SAMPLES")
      fi
      echo "Running LM-Eval for task ${TASK_NAME}${SAMPLES:+ with limit ${SAMPLES}}"
      "${CMD[@]}"
    done
  fi
}