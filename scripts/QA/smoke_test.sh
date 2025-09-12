#!/bin/bash

# Please configure the model path here, leave empty to skip test
declare -A MODEL_PATHS=(
    ["DeepSeek-R1-Distill-Qwen-7B"]="/data/hf_models/DeepSeek-R1-Distill-Qwen-7B"
    ["Qwen3-30B-A3B"]="" 
    ["Qwen3-32B"]=""
)

declare -A MODEL_CARDS=(
    ["DeepSeek-R1-Distill-Qwen-7B"]="1"
    ["Qwen3-30B-A3B"]="2"
    ["Qwen3-32B"]="1"
)

# Get the absolute path of the script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
SERVER_SCRIPT="$PARENT_DIR/start_gaudi_vllm_server.sh"

# Check if the server script exists
if [ ! -f "$SERVER_SCRIPT" ]; then
    echo "Error: start_gaudi_vllm_server.sh not found in parent directory: $PARENT_DIR"
    echo "Please ensure this script is placed in a subdirectory of the directory containing start_gaudi_vllm_server.sh"
    exit 1
fi

TARGET_STRING="paris"
TIMEOUT=180
INTERVAL=5
PORT="8688"

# Results array
declare -A TEST_RESULTS
declare -A ERROR_MESSAGES
declare -A FAILURE_REASONS
declare -A RESPONSE_CONTENTS

# Function to kill server and all related processes
kill_server() {
    local port=$1
    local server_pid=$2
    
    # Kill the main server process
    if [ -n "$server_pid" ] && ps -p $server_pid > /dev/null 2>&1; then
        kill -TERM $server_pid 2>/dev/null
        sleep 2
        # Force kill if still running
        if ps -p $server_pid > /dev/null 2>&1; then
            kill -KILL $server_pid 2>/dev/null
        fi
    fi
    
    # Kill any processes using the port
    local port_processes=$(lsof -ti:$port 2>/dev/null)
    if [ -n "$port_processes" ]; then
        echo "$port_processes" | xargs kill -TERM 2>/dev/null
        sleep 1
        # Force kill any remaining processes
        local remaining=$(lsof -ti:$port 2>/dev/null)
        if [ -n "$remaining" ]; then
            echo "$remaining" | xargs kill -KILL 2>/dev/null
        fi
    fi
}

# Function to test a single model
test_model() {
    local model_name=$1
    local model_path=$2
    local num_cards=$3
    
    local error_message failure_reason response http_code server_pid

    if [ -z "$model_path" ]; then
        return 0
    fi

    echo "========================================"
    echo "Testing model: $model_name"
    echo "Model path: $model_path"
    echo "Number of cards: $num_cards"
    echo "========================================"
    
    local server_host="127.0.0.1:$PORT"
    
    echo "Starting server on: $server_host"
    
    # Clean up any existing processes on this port
    kill_server "$PORT" ""
    
    # Start vLLM server
    bash "$SERVER_SCRIPT" -w "$model_path" -a "$server_host" -t "$num_cards" -s > /dev/null 2>&1 &
    server_pid=$!

    # Wait for server to start
    local elapsed=0
    local server_started=false
    
    while [ $elapsed -lt $TIMEOUT ]; do
        if curl -s http://$server_host/v1/models > /dev/null 2>&1; then
            server_started=true
            break
        fi
        
        if ! ps -p $server_pid > /dev/null 2>&1; then
            error_message="Server process terminated unexpectedly"
            failure_reason="SERVER_PROCESS_TERMINATED"
            break
        fi
        
        sleep $INTERVAL
        elapsed=$((elapsed + INTERVAL))
    done
    
    if [ "$server_started" = false ]; then
        if [ -z "$error_message" ]; then
            error_message="Server startup timeout (${TIMEOUT}s)"
            failure_reason="STARTUP_TIMEOUT"
        fi
    fi
    
    if [ -n "$error_message" ]; then
        ERROR_MESSAGES[$model_name]="$error_message"
        FAILURE_REASONS[$model_name]="$failure_reason"
        TEST_RESULTS[$model_name]="FAIL"
        kill_server "$PORT" "$server_pid"
        return 1
    fi
    
    echo "Server started successfully, waiting for stabilization..."
    sleep 10
    
    # Send test request
    echo "Sending test request..."
    response=$(curl -s -w "\nHTTP_CODE:%{http_code}" http://$server_host/v1/chat/completions \
        -X POST \
        -d "{\"model\": \"$model_path\", \"messages\": [{\"role\": \"user\", \"content\": \"The capital of France is?\"}], \"max_tokens\": 128, \"temperature\": 0.1}" \
        -H 'Content-Type: application/json' 2>&1)
    
    # Extract HTTP status code
    http_code=$(echo "$response" | grep "HTTP_CODE:" | cut -d':' -f2)
    response=$(echo "$response" | sed '/HTTP_CODE:/d')


    if command -v jq > /dev/null 2>&1; then
        response=$(echo "$response" | jq -r '.choices[0].message.content' 2>/dev/null)
    fi
    
    # Store the response content for reporting
    RESPONSE_CONTENTS[$model_name]="$response"
    
    # Kill server process
    kill_server "$PORT" "$server_pid"
    
    # Check for curl errors
    if echo "$response" | grep -q "curl:"; then
        ERROR_MESSAGES[$model_name]="Curl error: $(echo "$response" | grep "curl:")"
        FAILURE_REASONS[$model_name]="CURL_EXECUTION_ERROR"
        TEST_RESULTS[$model_name]="FAIL"
        return 1
    fi
    
    if [ -n "$http_code" ] && [ "$http_code" -ne 200 ]; then
        ERROR_MESSAGES[$model_name]="HTTP $http_code response: $response"
        FAILURE_REASONS[$model_name]="HTTP_ERROR_$http_code"
        TEST_RESULTS[$model_name]="FAIL"
        return 1
    fi
    
    if [ -z "$response" ] || [ "$response" = "null" ]; then
        ERROR_MESSAGES[$model_name]="Empty response"
        FAILURE_REASONS[$model_name]="EMPTY_RESPONSE"
        TEST_RESULTS[$model_name]="FAIL"
        return 1
    fi
    
    # Check for target string
    if echo "$response" | grep -qi "$TARGET_STRING"; then
        TEST_RESULTS[$model_name]="PASS"
        echo "Test PASSED for $model_name"
    else
        ERROR_MESSAGES[$model_name]="Response missing target string '$TARGET_STRING'"
        FAILURE_REASONS[$model_name]="CONTENT_VALIDATION_FAILED"
        TEST_RESULTS[$model_name]="FAIL"
        echo "Test FAILED for $model_name"
        echo "Response received: $response"
    fi
    
    return 0
}

# Function to generate report
generate_report() {
    echo ""
    echo "========================================"
    echo "       SMOKE TEST REPORT"
    echo "========================================"
    echo "Test Time:    $(date)"
    echo "Target Text:  $TARGET_STRING"
    echo "Test Query:   The capital of France is?"
    echo "========================================"
    
    local total_models=0
    local passed_models=0
    local failed_models=0
    
    for model_name in "${!MODEL_PATHS[@]}"; do
        total_models=$((total_models + 1))
        
        local model_path="${MODEL_PATHS[$model_name]}"
        local num_cards="${MODEL_CARDS[$model_name]}"
        
        echo ""
        echo "Model:        $model_name"
        echo "Model Path:   $model_path"
        echo "HPU Cards:    $num_cards"
        echo "Result:       ${TEST_RESULTS[$model_name]:-NOT_TESTED}"
        
        if [ "${TEST_RESULTS[$model_name]}" = "PASS" ]; then
            passed_models=$((passed_models + 1))
            echo "Status:       SUCCESS"
            echo "Response:     ${RESPONSE_CONTENTS[$model_name]}"
        elif [ "${TEST_RESULTS[$model_name]}" = "FAIL" ]; then
            failed_models=$((failed_models + 1))
            echo "Status:       FAILED"
            echo "Failure:      ${FAILURE_REASONS[$model_name]}"
            echo "Error:        ${ERROR_MESSAGES[$model_name]}"
            echo "Response:     ${RESPONSE_CONTENTS[$model_name]:-No response received}"
        else
            echo "Status:       NOT TESTED"
        fi
        
        echo "----------------------------------------"
    done
    
    echo ""
    echo "SUMMARY:"
    echo "Total Models: $total_models"
    echo "Passed:       $passed_models"
    echo "Failed:       $failed_models"
    
    echo "========================================"
    
    # Return appropriate exit code
    if [ $failed_models -gt 0 ]; then
        return 1
    else
        return 0
    fi
}

# Trap to ensure cleanup on script exit
cleanup() {
    echo "Cleaning up..."
    kill_server "$PORT" ""
    exit 0
}

trap cleanup EXIT INT TERM

# Main execution
echo "Starting smoke test"
echo ""

# Display configuration
echo "Model Configuration:"
for model_name in "${!MODEL_PATHS[@]}"; do
    echo "  - $model_name: ${MODEL_PATHS[$model_name]} (${MODEL_CARDS[$model_name]} card(s))"
done
echo ""

# Test each model
for model_name in "${!MODEL_PATHS[@]}"; do
    model_path="${MODEL_PATHS[$model_name]}"
    num_cards="${MODEL_CARDS[$model_name]}"
    
    test_model "$model_name" "$model_path" "$num_cards"
    echo ""
    sleep 10
done

# Generate final report
generate_report
exit $?