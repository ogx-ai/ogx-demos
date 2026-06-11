#!/usr/bin/env bash
set -uo pipefail
trap 'echo -e "\n\nInterrupted."; exit 130' INT

HOST="${1:-localhost}"
PORT="${2:-8321}"
TIMEOUT="${3:-120}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
    echo "Error: .venv/bin/python not found. Run 'uv sync' first."
    exit 1
fi

echo "Syncing dependencies..."
(cd "$REPO_ROOT" && uv sync --quiet)
echo "Done."
echo ""

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'

if [ -f .env ]; then
    set -a; source .env; set +a
fi

pass=0 fail=0 skip=0 timeout_count=0
results=()

PREVIEW_LINES="${PREVIEW_LINES:-3}"

run_demo() {
    local module="$1"
    local label
    label=$(echo "$module" | sed 's/demos\.//')

    printf "  %-55s " "$label"
    output=$(timeout "$TIMEOUT" "$PYTHON" -m "$module" "$HOST" "$PORT" 2>&1) && rc=$? || rc=$?

    local status
    if [ $rc -eq 124 ]; then
        status="timeout"
        printf "${YELLOW}TIMEOUT${NC}\n"
        results+=("TMOUT $label")
        ((timeout_count++)) || true
    elif [ $rc -ne 0 ]; then
        status="fail"
        printf "${RED}FAIL${NC} (exit $rc)\n"
        results+=("FAIL  $label  |  $(echo "$output" | tail -1)")
        ((fail++)) || true
    elif echo "$output" | grep -qE 'Traceback|ModuleNotFoundError|No available models|No available chat-capable models|No available embedding models|ImportError|SyntaxError'; then
        status="fail"
        printf "${RED}FAIL${NC} (error in output)\n"
        local err_line
        err_line=$(echo "$output" | grep -E 'Traceback|ModuleNotFoundError|No available|ImportError|SyntaxError' | head -1)
        results+=("FAIL  $label  |  $err_line")
        ((fail++)) || true
    else
        status="pass"
        printf "${GREEN}PASS${NC}\n"
        results+=("PASS  $label")
        ((pass++)) || true
    fi

    if [ -n "$output" ]; then
        echo "$output" | tail -"$PREVIEW_LINES" | sed 's/^/    | /'
        echo ""
    fi
}

skip_demo() {
    local label="$1"
    local reason="$2"
    printf "  %-55s ${YELLOW}SKIP${NC}  (%s)\n" "$label" "$reason"
    results+=("SKIP  $label  ($reason)")
    ((skip++))
}

echo "=== OGX Demos Test Suite ==="
echo "Server: $HOST:$PORT  Timeout: ${TIMEOUT}s"
echo ""

# 00 Setup
echo "--- 00_setup ---"
run_demo demos.00_setup.01_list_providers

# 01 Foundations
echo "--- 01_foundations ---"
run_demo demos.01_foundations.01_client_setup
run_demo demos.01_foundations.02_chat_completion
run_demo demos.01_foundations.03_system_prompts
run_demo demos.01_foundations.04_vector_db_basics
run_demo demos.01_foundations.05_insert_documents
run_demo demos.01_foundations.06_search_vectors
run_demo demos.01_foundations.07_tool_registration
skip_demo "01_foundations.08_mcp_tools" "requires external MCP server"

# 02 Responses Basics
echo "--- 02_responses_basics ---"
run_demo demos.02_responses_basics.01_simple_response
run_demo demos.02_responses_basics.02_tool_calling
run_demo demos.02_responses_basics.03_conversation_turns
run_demo demos.02_responses_basics.04_streaming_responses
run_demo demos.02_responses_basics.05_response_formats

# 03 RAG
echo "--- 03_rag ---"
run_demo demos.03_rag.01_simple_rag
run_demo demos.03_rag.02_multi_source_rag
run_demo demos.03_rag.03_rag_with_metadata
run_demo demos.03_rag.04_chunking_strategies
run_demo demos.03_rag.05_hybrid_search

# 04 Agents
echo "--- 04_agents ---"
if [ -n "${TAVILY_SEARCH_API_KEY:-}" ]; then
    run_demo demos.04_agents.01_simple_agent_chat
else
    skip_demo "04_agents.01_simple_agent_chat" "TAVILY_SEARCH_API_KEY not set"
fi
skip_demo "04_agents.02_chat_multimodal" "requires vision model"
run_demo demos.04_agents.03_chat_with_documents
run_demo demos.04_agents.04_agent_with_tools
run_demo demos.04_agents.05_rag_agent
if [ -n "${TAVILY_SEARCH_API_KEY:-}" ]; then
    run_demo demos.04_agents.06_react_agent
else
    skip_demo "04_agents.06_react_agent" "TAVILY_SEARCH_API_KEY not set"
fi
run_demo demos.04_agents.07_agent_routing

# 06 OpenAI Compatibility
echo "--- 06_openai_compatibility ---"
run_demo demos.06_openai_compatibility.01_chat_completion
run_demo demos.06_openai_compatibility.02_tool_calling
run_demo demos.06_openai_compatibility.03_responses_api
run_demo demos.06_openai_compatibility.04_responses_max_output_tokens
run_demo demos.06_openai_compatibility.05_responses_top_p
run_demo demos.06_openai_compatibility.06_responses_truncation
run_demo demos.06_openai_compatibility.07_responses_streaming
run_demo demos.06_openai_compatibility.08_responses_parallel_tool_calls
run_demo demos.06_openai_compatibility.09_responses_service_tier
run_demo demos.06_openai_compatibility.10_responses_logprobs
run_demo demos.06_openai_compatibility.11_responses_reasoning
run_demo demos.06_openai_compatibility.12_responses_temperature
run_demo demos.06_openai_compatibility.13_responses_combined

# Summary
echo ""
echo "=== Summary ==="
total=$((pass + fail + skip + timeout_count))
printf "  Total: %d  ${GREEN}Pass: %d${NC}  ${RED}Fail: %d${NC}  ${YELLOW}Skip: %d  Timeout: %d${NC}\n" \
    "$total" "$pass" "$fail" "$skip" "$timeout_count"

if [ $fail -gt 0 ]; then
    echo ""
    echo "Failed demos:"
    for r in "${results[@]}"; do
        if [[ "$r" == FAIL* ]]; then
            echo "  $r"
        fi
    done
    exit 1
fi
