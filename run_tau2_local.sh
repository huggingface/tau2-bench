#!/bin/bash

"""
TAU2 Benchmark Evaluation Script

This script runs TAU2 benchmark evaluation with configurable models and parameters.

Usage Examples:
  # Run with default values
  ./run_tau2_local.sh

  # Run with custom model
  ./run_tau2_local.sh --model_id HuggingFaceH4/Qwen3-4B-Instruct-Agentic --model_revision v01.05-step-000000790

  # Run with different domain and trials
  ./run_tau2_local.sh --domain airline --num-trials 1

Default Values:
  --model_id: Qwen/Qwen3-4B-Instruct-2507
  --model_revision: main
  --domain: retail
  --num-trials: 4

Available domains: airline, retail, telecom
"""

# Needed for vLLM / LiteLLM
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCH_COMPILE_DISABLE=1
export HOSTED_VLLM_API_BASE="http://localhost:8000/v1"

# Record start time for timing the evaluation
SCRIPT_START_TIME=$(date +%s)
echo "Evaluation started at: $(date)"

# Server PIDs for cleanup
AGENT_LLM_PID=""
USER_LLM_PID=""

# Function to shut down both servers
function shutdown_servers {
    echo "Shutting down servers..."
    if [[ -n "$AGENT_LLM_PID" && "$AGENT_LLM_PID" != "0" ]]; then
        echo "Stopping agent LLM server (PID: $AGENT_LLM_PID)..."
        kill $AGENT_LLM_PID 2>/dev/null || true
    fi
    if [[ -n "$USER_LLM_PID" && "$USER_LLM_PID" != "0" ]]; then
        echo "Stopping user LLM server (PID: $USER_LLM_PID)..."
        kill $USER_LLM_PID 2>/dev/null || true
    fi
    if [[ -n "$AGENT_LLM_PID" ]]; then
        wait $AGENT_LLM_PID 2>/dev/null || true
    fi
    if [[ -n "$USER_LLM_PID" ]]; then
        wait $USER_LLM_PID 2>/dev/null || true
    fi
    echo "Servers shut down."
    exit 0
}

# Function to determine tool call parser based on model ID
function get_tool_parser {
    local model_id="$1"
    case "$model_id" in
        *deepseek*|*DeepSeek*) echo "deepseek_v3" ;;
        *kimi*|*Kimi*) echo "kimi_k2" ;;
        *minimax*|*MiniMax*) echo "minimax_m1" ;;
        *hunyuan*|*Hunyuan*) echo "hunyuan_a13b" ;;
        *granite*|*Granite*) echo "granite" ;;
        *xlam*|*xLAM*) echo "xlam" ;;
        *jamba*|*Jamba*) echo "jamba" ;;
        *internlm*|*InternLM*) echo "internlm" ;;
        *mistral*|*Mistral*) echo "mistral" ;;
        *llama*|*Llama*) echo "llama3_json" ;;
        *qwen*|*Qwen*|*hermes*|*Hermes*|*nous-hermes*) echo "hermes" ;;
        *)
            echo "ERROR: No tool call parser found for model: $model_id" >&2
            echo "Supported patterns: deepseek, kimi, minimax, hunyuan, granite, xlam, jamba, internlm, mistral, llama, qwen, hermes" >&2
            exit 1 ;;
    esac
}

function get_reasoning_parser {
    local model_id="$1"
    case "$model_id" in
        *deepseek*|*DeepSeek*) echo "deepseek_r1" ;;
        *Qwen3*Thinking*|*qwen3*thinking*) echo "deepseek_r1" ;;
        *gpt-oss*|*GPT-OSS*) echo "GptOss" ;;
        *glm-4.5*|*GLM-4.5*) echo "glm45" ;;
        *hunyuan*|*Hunyuan*) echo "hunyuan_a13b" ;;
        *granite*|*Granite*) echo "granite" ;;
        *mistral*|*Mistral*) echo "mistral" ;;
        *step3*|*Step3*) echo "step3" ;;
        *qwen3*|*Qwen3*) echo "qwen3" ;;
        *smollm3*|*SmolLM3*) echo "qwen3" ;;
        *)
            echo "ERROR: No reasoning parser found for model: $model_id" >&2
            echo "Supported patterns: deepseek, gpt-oss, glm-4.5, hunyuan, granite, mistral, step3, qwen3" >&2
            exit 1 ;;
    esac
}

# Default values
TASK_NAME="tau2_bench"
DOMAIN="retail" # "airline" "retail" "telecom"
MODEL_ID="Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION="main"
NUM_TRIALS=4
USER_MODEL_ID=Qwen/Qwen3-30B-A3B-Instruct-2507

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_id)
            MODEL_ID="$2"
            shift 2
            ;;
        --model_revision)
            MODEL_REVISION="$2"
            shift 2
            ;;
        --domain)
            DOMAIN="$2"
            shift 2
            ;;
        --num-trials)
            NUM_TRIALS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option $1"
            echo "Usage: $0 [--model_id <model_id>] [--model_revision <model_revision>] [--domain <domain>] [--num-trials <num_trials>]"
            echo "Defaults: --model_id Qwen/Qwen3-4B-Instruct-2507 --model_revision main --domain retail --num-trials 4"
            echo "Available domains: airline, retail, telecom"
            exit 1
            ;;
    esac
done

TIMESTAMP=$(date +"%Y-%m-%dT%H-%M-%S")
OUTPUT_DIR="eval_results/$MODEL_ID/$MODEL_REVISION/$TASK_NAME/$TIMESTAMP"
# We need this flag since we run this script from training jobs that use DeepSpeed and the env vars get progated which causes errors during evaluation
ACCELERATE_USE_DEEPSPEED=false

NUM_GPUS=$(nvidia-smi -L | wc -l)

# Download benchmark data
export TAU2_DATA_DIR=$OUTPUT_DIR
hf download HuggingFaceH4/tau2-bench-data --repo-type dataset --local-dir $TAU2_DATA_DIR/tau2/

# Determine appropriate tool call parser for this model
TOOL_PARSER=$(get_tool_parser "$MODEL_ID")
echo "Using tool call parser: $TOOL_PARSER for model: $MODEL_ID"

# Determine reasoning parser for this model
REASONING_PARSER=$(get_reasoning_parser "$MODEL_ID")
echo "Using reasoning parser: $REASONING_PARSER for model: $MODEL_ID"

# Trap interruption signals and call the shutdown function
trap shutdown_servers SIGINT SIGTERM

# Function to check if server is up by checking /health endpoint
function check_server {
    local port=$1
    curl -i http://0.0.0.0:$port/health 2>/dev/null | head -n 1 | grep "200 OK"
}

# Function to wait for server with timeout
function wait_for_server {
    local port=$1
    local name=$2
    echo "Waiting for $name server to start on port $port..."
    local attempt=0
    local max_attempts=120  # 10 minutes total (120 * 5 seconds)

    while ! check_server $port; do
        if [ $attempt -ge $max_attempts ]; then
            echo "ERROR: $name server failed to start after $((max_attempts * 5)) seconds"
            exit 1
        fi
        echo "$name server is not yet available. Checking again in 5 seconds... ($((attempt + 1))/$max_attempts)"
        sleep 5
        attempt=$((attempt + 1))
    done

    echo "$name server is up and running."
}

# Check if user model is local or external
if [[ "$USER_MODEL_ID" == "gpt-4.1" || "$USER_MODEL_ID" == claude-* ]]; then
    # External model - use all GPUs for agent model
    echo "Starting agent vLLM server with $NUM_GPUS GPUs..."
    nohup vllm serve $MODEL_ID --revision $MODEL_REVISION \
        --tensor-parallel-size $NUM_GPUS \
        --trust-remote-code \
        --enable-auto-tool-choice \
        --tool-call-parser $TOOL_PARSER \
        --reasoning-parser $REASONING_PARSER \
        --host 0.0.0.0 --port 8000 \
        >"$OUTPUT_DIR/llm_agent_8000.log" 2>&1 &
    
    AGENT_LLM_PID=$!
    wait_for_server 8000 "Agent"
    
    USER_LLM_CONFIG="$USER_MODEL_ID"
    USER_LLM_ARGS='{"temperature": 0}'
else
    # Local model - split GPUs (4 for user, 4 for agent)
    echo "Starting user vLLM server with 4 GPUs on port 8001..."
    CUDA_VISIBLE_DEVICES=0,1,2,3 nohup vllm serve $USER_MODEL_ID \
        --tensor-parallel-size 4 \
        --trust-remote-code \
        --enable-auto-tool-choice \
        --tool-call-parser hermes \
        --host 0.0.0.0 --port 8001 \
        >"$OUTPUT_DIR/user_agent_8001.log" 2>&1 &

    USER_LLM_PID=$!
    
    echo "Starting agent vLLM server with 4 GPUs on port 8000..."
    CUDA_VISIBLE_DEVICES=4,5,6,7 nohup vllm serve $MODEL_ID --revision $MODEL_REVISION \
        --tensor-parallel-size 4 \
        --trust-remote-code \
        --enable-auto-tool-choice \
        --tool-call-parser $TOOL_PARSER \
        --reasoning-parser $REASONING_PARSER \
        --host 0.0.0.0 --port 8000 \
        >"$OUTPUT_DIR/llm_agent_8000.log" 2>&1 &

    AGENT_LLM_PID=$!
    
    wait_for_server 8001 "User"
    wait_for_server 8000 "Agent"
    
    USER_LLM_CONFIG="hosted_vllm/$USER_MODEL_ID"
    # Recommended sampling params from: https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507
    USER_LLM_ARGS='{"api_base": "http://localhost:8001/v1", "temperature": 0.7, "top_p": 0.8, "top_k": 20, "timeout": 300}'
fi

echo "Running tau2-bench evaluation ..."
echo "Trajectories results will be saved to $OUTPUT_DIR"
tau2 run \
    --domain "$DOMAIN" \
    --agent-llm hosted_vllm/$MODEL_ID \
    --agent-llm-args '{"temperature": 0.6, "top_p": 0.95, "timeout": 300}' \
    --user-llm "$USER_LLM_CONFIG" \
    --user-llm-args "$USER_LLM_ARGS" \
    --num-trials $NUM_TRIALS \
    --max-concurrency 8

shutdown_servers

# Calculate and display total runtime
SCRIPT_END_TIME=$(date +%s)
TOTAL_SECONDS=$((SCRIPT_END_TIME - SCRIPT_START_TIME))
HOURS=$((TOTAL_SECONDS / 3600))
MINUTES=$(((TOTAL_SECONDS % 3600) / 60))

echo "Evaluation completed at: $(date)"
echo "Total runtime: ${HOURS} hours and ${MINUTES} minutes (${TOTAL_SECONDS} seconds)"

echo "Done!"