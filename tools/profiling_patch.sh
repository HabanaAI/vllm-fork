#!/usr/bin/env bash

export RUNNER_ID=$1
export PROFILING_DEST_PATH=$2
export RUNNER_PATH=/software/users/jbyczkowski/dev-jenkins/jenkins-vllm-benchmarks-runner-$RUNNER_ID
export RUN_FILE=$RUNNER_PATH/run.sh
export DRIVER_FILE=$RUNNER_PATH/repositories/vllm-benchmarks/benchmarks/ibm/driver

# add profiling configuration
if ! grep -q "hl-prof-config" $RUN_FILE; then
    sed -i "/^run_vllm_benchmarks/i hl-prof-config --use-template profile_api_with_nics --fuser on --trace-analyzer on" $RUN_FILE
    sed -i "/^run_vllm_benchmarks/i export HABANA_PROFILE=1" $RUN_FILE
    sed -i "/^run_vllm_benchmarks/i export VLLM_PROFILER_ENABLED=full" $RUN_FILE
    sed -i "/^run_vllm_benchmarks/i mkdir $PROFILING_DEST_PATH" $RUN_FILE
    sed -i "/^run_vllm_benchmarks/i export VLLM_TORCH_PROFILER_DIR=$PROFILING_DEST_PATH" $RUN_FILE
    sed -i "/^run_vllm_benchmarks/i #" $RUN_FILE
fi

# patch in profiling tweaks
if ! grep -q "stvar.llm.start_profile" $DRIVER_FILE; then
    sed -i "/^def llm/i \ \ \ \ par.reps = 1" $DRIVER_FILE
    sed -i "/^def llm/i #" $DRIVER_FILE

    sed -i "s/= run(input_size, output_size, batch_size)/= run(input_size, 16, batch_size)/" $DRIVER_FILE

    sed -i "/for rep in range(par.reps)/i \ \ \ \ var.llm.start_profile()" $DRIVER_FILE
    sed -i "/return ret/i \ \ \ \ var.llm.stop_profile()" $DRIVER_FILE
fi

hlctl create containers \
	-f $RUNNER_PATH/pod.yml \
	--flavor g3.l \
	--watch \
	--mount-sshkeys \
    --namespace framework
