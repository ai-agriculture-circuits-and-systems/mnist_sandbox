#!/bin/bash
# Regression suite: run all DL models across loss/optimizer combos with NAS-style
# hyperparameter search and early stopping. Produces REPORT.md with ranked results.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "====================================================="
echo -e "${GREEN}MNIST Regression Suite (NAS-style)${NC}"
echo "====================================================="

mkdir -p outputs/regression

# Defaults (override via flags)
QUICK_TEST=true
FULL=false
MAX_EPOCHS=200
PATIENCE=5
MIN_DELTA=0.1
NAS_TRIALS=20
MAX_BATCH_SIZE=0
OUTPUT_DIR="outputs/regression"
REPORT_PATH="REPORT.md"
DEVICE="auto"
MODELS=""
SEED=42
WORKERS=8
PARALLEL=false

show_help() {
    echo -e "${BLUE}Usage:${NC} $0 [options]"
    echo ""
    echo "Runs all (or selected) models with every valid loss × optimizer combination."
    echo "For each combination, NAS-style search samples hyperparameters (lr, batch_size,"
    echo "weight_decay) and keeps the best trial. Training uses early stopping when the"
    echo "validation metric does not improve significantly for PATIENCE epochs."
    echo ""
    echo -e "${BLUE}Options:${NC}"
    echo "  -h, --help              Show this help"
    echo "  -q, --quick-test        Use 100-image subset (default: on)"
    echo "  -f, --full              Full MNIST dataset (disables quick-test)"
    echo "  -e, --max-epochs N      Max epochs per trial (default: 200)"
    echo "  -p, --patience N        Early-stop patience (default: 5)"
    echo "  -d, --min-delta X       Min improvement to reset patience (default: 0.1)"
    echo "  -n, --nas-trials N      Random hyperparameter trials per config (default: 20)"
    echo "  -o, --output-dir DIR    Output directory (default: outputs/regression)"
    echo "  -r, --report PATH       Report markdown path (default: REPORT.md)"
    echo "  --device DEVICE         auto | cuda | cpu (default: auto)"
    echo "  -m, --models LIST       Comma-separated model subset"
    echo "  --seed N                Random seed (default: 42)"
    echo "  --max-batch-size N      Cap training batch size (0=auto: 16 parallel / 32 sequential)"
    echo "  -j, --workers N         Parallel workers (default: 8; capped by GPU count)"
    echo "  --parallel              Auto workers (up to 8 GPUs or 8 CPU cores)"
    echo "  -j 1                    Sequential (disable parallelism)"
    echo ""
    echo -e "${BLUE}Examples:${NC}"
    echo "  $0 --quick-test                    # fast smoke regression"
    echo "  $0 --full --max-epochs 50          # full dataset, longer runs"
    echo "  $0 -m alexnet,simple_cnn,mlp -q    # subset of models, quick"
    echo "  $0 -q                              # quick-test with 8 workers (default)"
    echo "  $0 -j 1 -q                         # sequential quick-test"
    echo "  $0 -j 4 -m mlp,lenet,resnet -q     # 4 workers on selected models"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help) show_help ;;
        -q|--quick-test) QUICK_TEST=true; shift ;;
        -f|--full) FULL=true; QUICK_TEST=false; shift ;;
        -e|--max-epochs) MAX_EPOCHS="$2"; shift 2 ;;
        -p|--patience) PATIENCE="$2"; shift 2 ;;
        -d|--min-delta) MIN_DELTA="$2"; shift 2 ;;
        -n|--nas-trials) NAS_TRIALS="$2"; shift 2 ;;
        -o|--output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        -r|--report) REPORT_PATH="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        -m|--models) MODELS="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --max-batch-size) MAX_BATCH_SIZE="$2"; shift 2 ;;
        -j|--workers) WORKERS="$2"; shift 2 ;;
        --parallel) PARALLEL=true; shift ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            ;;
    esac
done

if [ ! -f "data/MNISTtrain.mat" ] && [ "$QUICK_TEST" = false ]; then
    echo -e "${RED}Full mode requires data/MNISTtrain.mat and data/MNISTtest.mat${NC}"
    exit 1
fi

# Reduce CUDA fragmentation when many models run in parallel
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [ "$QUICK_TEST" = true ] && [ ! -f "data/test_data/test_images.npy" ]; then
    echo -e "${YELLOW}Quick-test data missing; creating test subset...${NC}"
    python3 utils/create_test_data.py || true
fi

CMD="python3 -m utils.regression --output-dir $OUTPUT_DIR --report-path $REPORT_PATH"
CMD="$CMD --max-epochs $MAX_EPOCHS --patience $PATIENCE --min-delta $MIN_DELTA"
CMD="$CMD --nas-trials $NAS_TRIALS --device $DEVICE --seed $SEED --workers $WORKERS"
if [ "$MAX_BATCH_SIZE" -gt 0 ]; then
    CMD="$CMD --max-batch-size $MAX_BATCH_SIZE"
fi
if [ "$PARALLEL" = true ]; then
    CMD="$CMD --parallel"
fi

if [ "$QUICK_TEST" = true ]; then
    CMD="$CMD --quick-test"
else
    CMD="$CMD --full"
fi

if [ -n "$MODELS" ]; then
    CMD="$CMD --models $MODELS"
fi

echo -e "${YELLOW}Configuration:${NC}"
echo "  Quick test: $QUICK_TEST"
echo "  Max epochs: $MAX_EPOCHS"
echo "  Patience: $PATIENCE"
echo "  Min delta: $MIN_DELTA"
echo "  NAS trials: $NAS_TRIALS"
echo "  Output: $OUTPUT_DIR"
echo "  Report: $REPORT_PATH"
[ -n "$MODELS" ] && echo "  Models: $MODELS"
echo "  Workers: $WORKERS (parallel=$PARALLEL)"
[ "$MAX_BATCH_SIZE" -gt 0 ] && echo "  Max batch size: $MAX_BATCH_SIZE"
echo ""
echo -e "${YELLOW}Running:${NC} $CMD"
echo ""

$CMD

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Regression complete.${NC} See $REPORT_PATH"
else
    echo -e "${RED}Regression failed.${NC}"
    exit 1
fi
