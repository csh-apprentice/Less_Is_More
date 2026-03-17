#!/usr/bin/env bash
# =============================================================================
# smoke_test.sh  —  Less Is More: portability & inference sanity checks
#
# Usage (from repo root):
#   bash smoke_test.sh              # run Tier 0 only (no GPU needed)
#   bash smoke_test.sh --tier1      # run Tier 0 + Tier 1 (GPU + backbone required)
#
# Tier 0: import chain, TOML validation, checkpoint key structure. No GPU.
# Tier 1: end-to-end inference with 4 frames / 4 steps. Requires ~36 GB VRAM.
#         Set BACKBONE_PATH and CHECKPOINT_PATH below before running.
# =============================================================================

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'
pass() { echo -e "${GREEN}[PASS]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; exit 1; }
info() { echo -e "${YELLOW}[INFO]${NC} $1"; }

# ---------------------------------------------------------------------------
# USER CONFIG — set these before running Tier 1
# ---------------------------------------------------------------------------
BACKBONE_PATH="${BACKBONE_PATH:-/path/to/Wan2.1-T2V-14B}"  # override: export BACKBONE_PATH=/your/path
CHECKPOINT_DIR="checkpoints/20251023_00-19-38/epoch1000"   # temperature checkpoint
TOML_CONFIG="configs/train_temperature.toml"
PROMPT_FILE="metric/high_quality_prompts_96.txt"
INFERENCE_PORT=29500
# ---------------------------------------------------------------------------

RUN_TIER1=false
for arg in "$@"; do [[ "$arg" == "--tier1" ]] && RUN_TIER1=true; done

echo "========================================================"
echo "  Less Is More — Smoke Test"
echo "  Repo: $REPO_ROOT"
echo "  Tier 1: $RUN_TIER1"
echo "========================================================"

# ============================================================
# TIER 0 — No GPU required
# ============================================================
info "--- TIER 0: portability checks (no GPU) ---"

# T0.1: Import chain
info "T0.1  Import chain"
python - <<'EOF'
import sys
from inference.inference import load_prompts_from_file, load_prompts_from_directory
from inference.inference import (
    load_pipeline_from_toml, get_fps_block_indices,
    load_adapter_weights_selective, apply_checkpoint
)
from models.wan.wan import WanPipeline
from utils.common import DTYPE_MAP
from inference_utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
print("All imports resolved OK")
EOF
pass "T0.1  Import chain"

# T0.2: TOML validation
info "T0.2  TOML config parsing"
python - <<'EOF'
import toml, os
configs = [
    "configs/train_shutter.toml",
    "configs/train_temperature.toml",
    "configs/train_bokeh.toml",
    "configs/dataset_shutter.toml",
    "configs/dataset_temperature.toml",
    "configs/dataset_bokeh.toml",
]
for f in configs:
    c = toml.load(f)
    if "model" in c:
        ckpt = c["model"]["ckpt_path"]
        assert "/root/" not in ckpt, f"Hardcoded /root/ path still in {f}: {ckpt}"
    # check no absolute /root/ paths in dataset entries
    for section in c.get("directory", []):
        path = section.get("path", "")
        assert "/root/" not in path, f"Hardcoded /root/ path in {f}: {path}"
    print(f"  OK  {f}")
print("All TOMLs parsed and path-clean")
EOF
pass "T0.2  TOML config parsing"

# T0.3: Checkpoint key structure
info "T0.3  Checkpoint adapter key structure"
python - <<'EOF'
import safetensors.torch as st
ckpt = "checkpoints/20251023_00-19-38/epoch1000/adapter_model.safetensors"
d = st.load_file(ckpt)
fps_keys  = [k for k in d if "fps_adapter" in k]
lora_keys = [k for k in d if "lora" in k]
assert fps_keys,  f"No fps_adapter keys found in {ckpt}"
assert lora_keys, f"No LoRA keys found in {ckpt}"
print(f"  Total: {len(d)} keys  |  LoRA: {len(lora_keys)}  |  FPS adapter: {len(fps_keys)}")
# sanity: fps adapters should only be in deepest third (blocks 21-31 for 32-block model)
block_ids = set()
for k in fps_keys:
    parts = k.split(".")
    if "blocks" in parts:
        idx = parts.index("blocks")
        block_ids.add(int(parts[idx + 1]))
print(f"  FPS adapter block indices: {sorted(block_ids)}")
assert min(block_ids) >= 21, f"FPS adapter in unexpected block {min(block_ids)} (expected >= 21)"
print("Checkpoint key structure valid")
EOF
pass "T0.3  Checkpoint adapter key structure"

# T0.4: PEFT adapter config
info "T0.4  PEFT adapter_config.json"
python - <<'EOF'
import json
with open("checkpoints/20251023_00-19-38/epoch1000/adapter_config.json") as f:
    cfg = json.load(f)
assert cfg["peft_type"] == "LORA", "Expected LORA peft_type"
assert cfg["r"] == 32, f"Expected rank 32, got {cfg['r']}"
expected_targets = {"q", "k", "v", "o", "ffn.0", "ffn.2"}
actual_targets   = set(cfg["target_modules"])
assert actual_targets == expected_targets, f"Unexpected target_modules: {actual_targets}"
print(f"  peft_type=LORA  rank={cfg['r']}  targets={sorted(actual_targets)}")
print("PEFT config valid")
EOF
pass "T0.4  PEFT adapter_config.json"

echo ""
pass "TIER 0 complete — all portability checks passed"

# ============================================================
# TIER 1 — GPU required (~36 GB VRAM, A6000 or larger)
# ============================================================
if [[ "$RUN_TIER1" == false ]]; then
    echo ""
    info "Skipping Tier 1 (pass --tier1 to enable)"
    info "Ensure BACKBONE_PATH is set correctly in this script before running Tier 1."
    exit 0
fi

echo ""
info "--- TIER 1: end-to-end inference (GPU required) ---"

# Check backbone path
if [[ "$BACKBONE_PATH" == "/path/to/Wan2.1-T2V-14B" ]]; then
    fail "Set BACKBONE_PATH at the top of smoke_test.sh before running Tier 1."
fi
if [[ ! -d "$BACKBONE_PATH" ]]; then
    fail "BACKBONE_PATH does not exist: $BACKBONE_PATH"
fi
pass "T1.0  Backbone path exists: $BACKBONE_PATH"

# Patch ckpt_path in the temp TOML for this run only (write to a temp file)
TEMP_TOML=$(mktemp /tmp/smoke_train_XXXXXX.toml)
sed "s|ckpt_path = '.*'|ckpt_path = '$BACKBONE_PATH'|" "$TOML_CONFIG" > "$TEMP_TOML"
info "  Using temp TOML: $TEMP_TOML"

mkdir -p output/smoke_test/backbone_only output/smoke_test/decoupled

# T1.1: Backbone-only — no adapters, 1 prompt, 4 frames, 4 steps
info "T1.1  Backbone-only inference (4 frames, 4 steps)"
PYTHONPATH=. python inference/inference.py \
    --config "$TEMP_TOML" \
    --condition_values 0.5 \
    --steps 4 \
    --frames 4 \
    --output_dir output/smoke_test/backbone_only \
    --prompt_file "$PROMPT_FILE" \
    --seed 42 \
    --width 512 --height 512 \
    --backbone_only \
    --port "$INFERENCE_PORT" \
    2>&1 | tail -20

BACKBONE_VIDS=$(find output/smoke_test/backbone_only -name "*.mp4" | wc -l)
[[ "$BACKBONE_VIDS" -gt 0 ]] || fail "T1.1  No output videos found in output/smoke_test/backbone_only"
pass "T1.1  Backbone-only — $BACKBONE_VIDS video(s) generated"

# T1.2: Decoupled mode — LoRA + FPS adapters on deepest third (paper method)
info "T1.2  Decoupled inference (4 frames, 4 steps, 3 conditioning values)"
PYTHONPATH=. python inference/inference.py \
    --config "$TEMP_TOML" \
    --checkpoint "$CHECKPOINT_DIR" \
    --condition_values 0.3 0.5 0.8 \
    --steps 4 \
    --frames 4 \
    --output_dir output/smoke_test/decoupled \
    --prompt_file "$PROMPT_FILE" \
    --seed 42 \
    --width 512 --height 512 \
    --decoupled \
    --port "$INFERENCE_PORT" \
    2>&1 | tail -20

DECOUPLED_VIDS=$(find output/smoke_test/decoupled -name "*.mp4" | wc -l)
[[ "$DECOUPLED_VIDS" -gt 0 ]] || fail "T1.2  No output videos found in output/smoke_test/decoupled"
pass "T1.2  Decoupled mode — $DECOUPLED_VIDS video(s) generated"

# T1.3: Check decoupled produced 3 distinct outputs (one per conditioning value)
[[ "$DECOUPLED_VIDS" -ge 3 ]] || fail "T1.3  Expected >= 3 videos (one per condition value), got $DECOUPLED_VIDS"
pass "T1.3  Distinct conditioning values produced distinct outputs"

rm -f "$TEMP_TOML"

echo ""
pass "TIER 1 complete — end-to-end inference verified"
echo ""
echo "========================================================"
echo "  All smoke tests passed."
echo "  Output videos: output/smoke_test/{backbone_only,decoupled}/"
echo "========================================================"
