#!/bin/bash
# Full MoE test pipeline: rebuild → deploy → copy model → benchmark
# Usage: bash tests/run_moe_test.sh [model_dir]
#
# Example:
#   bash tests/run_moe_test.sh /path/to/model/Q4_K_M

set -e

MODEL_DIR="${1:?usage: tests/run_moe_test.sh <model_dir>}"
ENGINE_URL="http://localhost:8090"
COMPOSE="docker compose -f compose.yaml -f compose.engine.yaml"

echo "=== MoE Expert Offload Test Pipeline ==="
echo "Model dir: $MODEL_DIR"
echo ""

# Step 1: Verify model files exist
FIRST_SHARD=$(ls "$MODEL_DIR"/*-00001-of-*.gguf 2>/dev/null | head -1)
if [ -z "$FIRST_SHARD" ]; then
    # Try single file
    FIRST_SHARD=$(ls "$MODEL_DIR"/*.gguf 2>/dev/null | head -1)
fi

if [ -z "$FIRST_SHARD" ]; then
    echo "ERROR: No GGUF files found in $MODEL_DIR"
    exit 1
fi

MODEL_NAME=$(basename "$FIRST_SHARD")
TOTAL_SIZE=$(du -sh "$MODEL_DIR" | awk '{print $1}')
SHARD_COUNT=$(ls "$MODEL_DIR"/*.gguf 2>/dev/null | wc -l)
echo "Model: $MODEL_NAME ($SHARD_COUNT shards, $TOTAL_SIZE total)"

# Step 2: Rebuild engine
echo ""
echo "=== Step 1: Rebuild Engine ==="
$COMPOSE build engine 2>&1 | tail -5

# Step 3: Stop existing engine
echo ""
echo "=== Step 2: Stop Existing Engine ==="
docker stop augmentum-engine-1 2>/dev/null || true
docker rm augmentum-engine-1 2>/dev/null || true

# Step 4: Copy model to Docker volume
echo ""
echo "=== Step 3: Copy Model to Docker Volume ==="
for f in "$MODEL_DIR"/*.gguf; do
    fname=$(basename "$f")
    echo "  Copying $fname..."
    docker run --rm -v engine_models:/data -v "$MODEL_DIR:/src:ro" \
        alpine cp "/src/$fname" "/data/$fname"
done
echo "  Done. Models in volume:"
docker run --rm -v engine_models:/data alpine ls -lh /data/

# Step 5: Start engine with MoE offload
echo ""
echo "=== Step 4: Start Engine ==="
$COMPOSE run -d --name augmentum-engine-1 \
    -p 8090:8090 \
    -e ENGINE_DEFAULT_MODEL="" \
    -e ENGINE_MOE_EXPERT_OFFLOAD=auto \
    engine

# Wait for startup
echo "Waiting for engine startup..."
for i in $(seq 1 30); do
    if curl -s "$ENGINE_URL/health" > /dev/null 2>&1; then
        echo "Engine ready!"
        break
    fi
    sleep 2
done

# Step 6: Run benchmark
echo ""
echo "=== Step 5: Run Benchmark ==="
python tests/bench_moe.py --url "$ENGINE_URL" --model "$MODEL_NAME"

echo ""
echo "=== Test Complete ==="
