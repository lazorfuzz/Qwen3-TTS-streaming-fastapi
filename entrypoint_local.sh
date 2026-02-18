#!/bin/bash
set -e

BASE_PORT=8001
WORK_DIR="$(pwd)"
LOCAL_RUN_DIR="${WORK_DIR}/.run"
export TTS_VOICE_META_DIR="${TTS_VOICE_META_DIR:-${WORK_DIR}/voices}"

# --- MIG detection ---
MIG_ENABLED=false
MIG_UUIDS=()

if command -v nvidia-smi &>/dev/null; then
    while IFS= read -r uuid; do
        MIG_UUIDS+=("$uuid")
    done < <(nvidia-smi -L 2>/dev/null | grep -o 'MIG-[a-f0-9-]*' || true)

    if [ ${#MIG_UUIDS[@]} -gt 0 ]; then
        MIG_ENABLED=true
        echo "[entrypoint_local] MIG enabled — detected ${#MIG_UUIDS[@]} MIG device(s)"
    fi
fi

# --- Resolve NUM_WORKERS ---
RAW_WORKERS=${TTS_NUM_WORKERS:-1}

if [ "$RAW_WORKERS" = "auto" ]; then
    if [ "$MIG_ENABLED" = true ]; then
        NUM_WORKERS=${#MIG_UUIDS[@]}
        echo "[entrypoint_local] TTS_NUM_WORKERS=auto → ${NUM_WORKERS} (from MIG device count)"
    else
        NUM_WORKERS=1
        echo "[entrypoint_local] TTS_NUM_WORKERS=auto → 1 (no MIG detected)"
    fi
else
    NUM_WORKERS=$RAW_WORKERS
fi

echo "[entrypoint_local] Working directory: ${WORK_DIR}"
echo "[entrypoint_local] Starting with ${NUM_WORKERS} worker(s)..."

# Single worker: run uvicorn directly on port 8000 (no nginx needed)
if [ "$NUM_WORKERS" -eq 1 ]; then
    echo "[entrypoint_local] Single worker mode — binding uvicorn directly to :8000"
    if [ "$MIG_ENABLED" = true ] && [ ${#MIG_UUIDS[@]} -gt 0 ]; then
        export CUDA_VISIBLE_DEVICES="${MIG_UUIDS[0]}"
        echo "[entrypoint_local] Using MIG device: ${CUDA_VISIBLE_DEVICES}"
    fi
    exec uvicorn fastapi_tts_server:app --host 0.0.0.0 --port 8000
fi

# Multi-worker: use nginx + supervisord
echo "[entrypoint_local] Multi-worker mode — using nginx load balancer"

# Create local runtime dirs for nginx (avoids writing to /var or /etc)
mkdir -p "${LOCAL_RUN_DIR}/nginx" "${LOCAL_RUN_DIR}/logs"

# Generate nginx upstream servers
UPSTREAM=""
for i in $(seq 0 $((NUM_WORKERS - 1))); do
    UPSTREAM="${UPSTREAM}        server 127.0.0.1:$((BASE_PORT + i));"$'\n'
done

# Write nginx.conf to working directory
cat > "${LOCAL_RUN_DIR}/nginx.conf" << NGINX_EOF
# Run as current user — no 'user' directive, no root needed
worker_processes 1;
pid ${LOCAL_RUN_DIR}/nginx/nginx.pid;
error_log ${LOCAL_RUN_DIR}/logs/nginx_error.log;

events {
    worker_connections 64;
}

http {
    access_log ${LOCAL_RUN_DIR}/logs/nginx_access.log;

    # Temp file paths inside local .run dir
    client_body_temp_path ${LOCAL_RUN_DIR}/nginx/client_body;
    proxy_temp_path ${LOCAL_RUN_DIR}/nginx/proxy;
    fastcgi_temp_path ${LOCAL_RUN_DIR}/nginx/fastcgi;
    uwsgi_temp_path ${LOCAL_RUN_DIR}/nginx/uwsgi;
    scgi_temp_path ${LOCAL_RUN_DIR}/nginx/scgi;

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
        echo "[entrypoint_local] Worker ${i} → MIG device ${MIG_UUIDS[$i]}"
    fi

    WORKERS="${WORKERS}
[program:tts-worker-${i}]
command=uvicorn fastapi_tts_server:app --host 127.0.0.1 --port ${PORT}
directory=${WORK_DIR}
${ENV_LINE}
autorestart=true
startretries=999
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0
"
done

# Write supervisord.conf to working directory
cat > "${LOCAL_RUN_DIR}/supervisord.conf" << EOF
[supervisord]
nodaemon=true
logfile=${LOCAL_RUN_DIR}/logs/supervisord.log
pidfile=${LOCAL_RUN_DIR}/supervisord.pid

[program:nginx]
command=nginx -c ${LOCAL_RUN_DIR}/nginx.conf -g "daemon off;"
autorestart=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0

${WORKERS}
EOF

# MIG provides hardware isolation — skip MPS when MIG is active
if [ "$MIG_ENABLED" = true ]; then
    echo "[entrypoint_local] MIG active — skipping MPS setup (hardware isolation provided by MIG)"
else
    export CUDA_VISIBLE_DEVICES=0
    export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
    export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log

    mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY

    if ! nvidia-cuda-mps-control -d 2>/dev/null; then
        echo "[entrypoint_local] MPS daemon already running or unavailable — skipping"
    fi
fi

echo "[entrypoint_local] Generated configs in ${LOCAL_RUN_DIR}/"
echo "[entrypoint_local] Starting supervisord..."

exec supervisord -c "${LOCAL_RUN_DIR}/supervisord.conf"
