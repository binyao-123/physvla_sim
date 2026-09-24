#!/bin/bash
# 保活 A100 policy server 的 SSH 隧道：掉了就重建。
# 客户端那层已经能重连重试，但它重连的是本地端口；本地端口是这条隧道提供的，
# 所以两层都要有：这里负责端口一直在，客户端负责断开期间不崩。
#
#   nohup bash scripts/keep_tunnel.sh > /tmp/keep_tunnel.log 2>&1 &

set -u
PORT="${PORT:-10093}"
HOST_ALIAS="${HOST_ALIAS:-huoshan-a100}"
CHECK_INTERVAL="${CHECK_INTERVAL:-10}"

echo "[keep_tunnel] port=$PORT host=$HOST_ALIAS interval=${CHECK_INTERVAL}s"
while true; do
  if ! ss -ltn | grep -q ":$PORT "; then
    echo "[keep_tunnel] $(date '+%F %T') port $PORT down, reconnecting..."
    # 清掉可能残留的僵死 ssh：它占着进程但不再监听端口。
    pkill -f "ssh -f -N .*-L $PORT:" 2>/dev/null
    ssh -f -N \
      -o ExitOnForwardFailure=yes \
      -o ServerAliveInterval=15 \
      -o ServerAliveCountMax=3 \
      -o ConnectTimeout=15 \
      -o StrictHostKeyChecking=accept-new \
      -L "$PORT:127.0.0.1:$PORT" "$HOST_ALIAS"
    sleep 2
    if ss -ltn | grep -q ":$PORT "; then
      echo "[keep_tunnel] $(date '+%F %T') reconnected"
    else
      echo "[keep_tunnel] $(date '+%F %T') reconnect failed, will retry"
    fi
  fi
  sleep "$CHECK_INTERVAL"
done
