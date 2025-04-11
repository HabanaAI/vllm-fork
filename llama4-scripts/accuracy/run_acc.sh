model_path="/mnt/weka/llm/Llama-4-Scout-17B-16E-Instruct"
log_name="accuracy_run"
bash 01-gaudi-vllm-serve.sh ${log_name} ${model_path}

until [[ $ready == true ]]; do
    n=$((n+1))
    if grep -q "Started server process" benchmark_logs/${log_name}_serving.log; then
        break
    fi
    sleep 5s
done
sleep 10s

pid=`cat accuracy_server.pid`
echo "============== server pid is ${pid} =============="
bash 02-gaudi-vllm-eval.sh $model_path
sleep 5
kill -9 $pid