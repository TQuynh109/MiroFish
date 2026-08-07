# Config serving script for project.

vllm serve Qwen/Qwen3.6-27B \
    --host 0.0.0.0 \
    --port 9999 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9 \
    --tensor-parallel-size 8 \
    --dtype bfloat16 \
    --reasoning-parser qwen3 \
    --trust-remote-code \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder