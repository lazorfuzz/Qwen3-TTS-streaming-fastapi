#!/bin/bash
set -e

BASE_PORT=8001

# --- MIG detection ---
MIG_ENABLED=false
MIG_UUIDS=()

if command -v nvidia-smi &>/dev/null; then
    while IFS= read -r uuid; do
        MIG_UUIDS+=("$uuid")
    done < <(nvidia-smi -L 2>/dev/null | grep -o 'MIG-[a-f0-9-]*' || true)

    if [ ${#MIG_UUIDS[@]} -gt 0 ]; then
        MIG_ENABLED=true
        echo "[entrypoint] MIG enabled — detected ${#MIG_UUIDS[@]} MIG device(s)"
    fi
fi

# --- Resolve NUM_WORKERS ---
RAW_WORKERS=${TTS_NUM_WORKERS:-1}

if [ "$RAW_WORKERS" = "auto" ]; then
    if [ "$MIG_ENABLED" = true ]; then
        NUM_WORKERS=${#MIG_UUIDS[@]}
        echo "[entrypoint] TTS_NUM_WORKERS=auto → ${NUM_WORKERS} (from MIG device count)"
    else
        NUM_WORKERS=1
        echo "[entrypoint] TTS_NUM_WORKERS=auto → 1 (no MIG detected)"
    fi
else
    NUM_WORKERS=$RAW_WORKERS
fi

echo "[entrypoint] Starting with ${NUM_WORKERS} worker(s)..."

# Single worker: run uvicorn directly on port 8000 (no nginx needed)
if [ "$NUM_WORKERS" -eq 1 ]; then
    echo "[entrypoint] Single worker mode — binding uvicorn directly to :8000"
    if [ "$MIG_ENABLED" = true ] && [ ${#MIG_UUIDS[@]} -gt 0 ]; then
        export CUDA_VISIBLE_DEVICES="${MIG_UUIDS[0]}"
        echo "[entrypoint] Using MIG device: ${CUDA_VISIBLE_DEVICES}"
    fi
    exec uvicorn fastapi_tts_server:app --host 0.0.0.0 --port 8000
fi

# Multi-worker: use nginx + supervisord
echo "[entrypoint] Multi-worker mode — using nginx load balancer"

# Generate nginx upstream servers
UPSTREAM=""
for i in $(seq 0 $((NUM_WORKERS - 1))); do
    UPSTREAM="${UPSTREAM}        server 127.0.0.1:$((BASE_PORT + i));"$'\n'
done

# Write nginx.conf directly (no sed/template dependency)
cat > /etc/nginx/nginx.conf << NGINX_EOF
events {
    worker_connections 64;
}

http {
    upstream tts_workers {
        least_conn;
${UPSTREAM}    }

    server {
        listen 8000;

        location / {
            proxy_pass http://tts_workers;
            proxy_http_version 1.1;
            proxy_set_header Connection "";
            proxy_buffering off;
            proxy_cache off;
            chunked_transfer_encoding on;
            proxy_connect_timeout 10s;
            proxy_read_timeout 600s;
            proxy_send_timeout 600s;
        }
    }
}
NGINX_EOF

# Generate supervisord worker sections
WORKERS=""
for i in $(seq 0 $((NUM_WORKERS - 1))); do
    PORT=$((BASE_PORT + i))

    # Assign MIG device to each worker if available
    ENV_LINE=""
    if [ "$MIG_ENABLED" = true ] && [ $i -lt ${#MIG_UUIDS[@]} ]; then
        ENV_LINE="environment=CUDA_VISIBLE_DEVICES=\"${MIG_UUIDS[$i]}\""
        echo "[entrypoint] Worker ${i} → MIG device ${MIG_UUIDS[$i]}"
    fi

    WORKERS="${WORKERS}
[program:tts-worker-${i}]
command=uvicorn fastapi_tts_server:app --host 127.0.0.1 --port ${PORT}
directory=/app
${ENV_LINE}
autorestart=true
startretries=999
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0
"
done

# Write final supervisord.conf
cat > /etc/supervisord.conf << EOF
[supervisord]
nodaemon=true
logfile=/dev/stdout
logfile_maxbytes=0

[program:nginx]
command=nginx -g "daemon off;"
autorestart=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0

${WORKERS}
EOF

# MIG provides hardware isolation — skip MPS when MIG is active
if [ "$MIG_ENABLED" = true ]; then
    echo "[entrypoint] MIG active — skipping MPS setup (hardware isolation provided by MIG)"
else
    export CUDA_VISIBLE_DEVICES=0
    export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
    export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log

    mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY

    nvidia-cuda-mps-control -d
fi

echo "[entrypoint] Generated nginx config with ${NUM_WORKERS} upstream(s)"
echo "[entrypoint] Starting supervisord..."

exec supervisord -c /etc/supervisord.conf
