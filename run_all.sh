#!/bin/bash
# Starts all three HuDex services — local (pipenv) or Docker mode.
#
# Usage:
#   ./run_hudex_all.sh           # local pipenv (default)
#   ./run_hudex_all.sh --docker  # Docker (all three via docker-compose)

TDA_DIR="$HOME/git/arlequin_ai_hudex_tda"
DEMO_DIR="$HOME/git/arlequin_ai_hudex_demo_prototype"

echo "╔══════════════════════════════════════════════╗"
echo "║           HuDex — all services               ║"
echo "║  8001  hudex-demo    (FastAPI + ML engine)   ║"
echo "║  8002  hudex-tda     (TDA + Neo4j engine)    ║"
echo "║  8003  hudex-proto   (FastAPI prototype)     ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# ── Docker mode ────────────────────────────────────────────────────────────
if [ "$1" = "--docker" ]; then
    echo "→ Mode: Docker"
    echo ""

    trap 'echo ""; echo "Stopping Docker services..."; \
          docker compose -f "$DEMO_DIR/docker-compose.yml" down; \
          docker compose -f "$TDA_DIR/docker-compose.yml" down' SIGINT SIGTERM

    echo "→ [8001 + 8003] Starting hudex-demo + hudex-prototype..."
    docker compose -f "$DEMO_DIR/docker-compose.yml" up --build -d

    echo "→ [8002] Starting hudex-tda..."
    docker compose -f "$TDA_DIR/docker-compose.yml" up --build -d

    echo ""
    echo "All services running. Logs:"
    echo "  docker compose -f $DEMO_DIR/docker-compose.yml logs -f"
    echo "  docker compose -f $TDA_DIR/docker-compose.yml logs -f"
    echo ""
    echo "Press Ctrl-C to stop all."

    # Follow logs from both compose stacks
    docker compose -f "$DEMO_DIR/docker-compose.yml" logs -f &
    docker compose -f "$TDA_DIR/docker-compose.yml" logs -f &
    wait

# ── Local pipenv mode ───────────────────────────────────────────────────────
else
    echo "→ Mode: local pipenv  (use --docker for Docker)"
    echo ""

    trap 'echo ""; echo "Stopping all services..."; kill 0' SIGINT SIGTERM

    echo "→ [8001] Starting hudex-demo..."
    bash "$DEMO_DIR/run.sh" > /tmp/hudex_demo.log 2>&1 &
    PID_DEMO=$!

    echo "→ [8002] Starting hudex-tda..."
    bash "$TDA_DIR/run.sh" > /tmp/hudex_tda.log 2>&1 &
    PID_TDA=$!

    echo "→ [8003] Starting hudex-prototype..."
    bash "$DEMO_DIR/run_prototype.sh" > /tmp/hudex_proto.log 2>&1 &
    PID_PROTO=$!

    echo ""
    echo "All services starting. Logs:"
    echo "  tail -f /tmp/hudex_demo.log"
    echo "  tail -f /tmp/hudex_tda.log"
    echo "  tail -f /tmp/hudex_proto.log"
    echo ""
    echo "Press Ctrl-C to stop all."

    wait $PID_DEMO $PID_TDA $PID_PROTO
fi
