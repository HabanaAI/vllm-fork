#!/usr/bin/env bash

export FOLDER_PREFIX="/software/users/jbyczkowski/dev-jenkins/jenkins-vllm-benchmarks-runner-"
export ALT_QUANT_CONFIG="/software/users/jbyczkowski/maxabs_quant_scale_method_unit_scale_g3.json"

function add_line_to_file {
    local filename="$1"
    local pattern="$2"
    local newline="$3"

    sed -i "/${pattern}/i ${newline}" "$filename"
}

function add_before_run {
    local run_id="$1"
    local newline="$2"
    filename="$FOLDER_PREFIX$run_id"
    add_line_to_file "$filename" "run_vllm_benchmarks" "$newline"
}

function set_gc_multiplier {
    local run_id="$1"
    local newline="export VLLM_GC_THR_MULTIPLIER=$2"
    add_before_run "$run_id" "$newline"
}

function set_strided_groups {
    local run_id="$1"
    local newline="export PT_HPU_DISABLE_pass_batch_as_strided_groups=True"
    add_before_run "$run_id" "$newline"
}

function set_quant_config {
    local run_id="$1"
    local newline="export QUANT_CONFIG=$ALT_QUANT_CONFIG"
    add_before_run "$run_id" "$newline"
}

run_id=
output_file=
gc_multiplier=
do_strided=

while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        -i|--run_id)
            run_id="$2"
            shift 2
        ;;
        -o|--output_file)
            output_file="$2"
            shift 2
        ;;
        -m|--gc_multiplier)
            gc_multiplier="$2"
            shift 2
        ;;
        -s|--strided_group)
            do_strided=1
            shift 1
        ;;
        -s|--quant_config)
            quant_config=1
            shift 1
        ;;
        *)
            echo "Unknown option: $1"
            exit 1
        ;;
    esac
done

if [[ -z "$run_id" ]]; then
  echo "need to set run_id"
  exit 1
fi

if [[ -z "$output_file" ]]; then
  echo "need to set output_file"
  exit 1
fi

if [[ -n "$gc_multiplier" ]]; then
    set_gc_multiplier $run_id $gc_multiplier
fi

if [[ -n "$strided_group" ]]; then
    set_strided_groups $run_id
fi

if [[ -n "$quant_config" ]]; then
    set_quant_config $run_id
fi

yaml_file="$FOLDER_PREFIX$run_id/pod.yml"

hlctl create containers \
    -f $yaml_file \
    --flavor g3.l \
    --watch \
    --mount-sshkeys \
    --namespace framework \
2>&1 | tee $output_file
