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
echo -e "${GREEN}Model Regression Suite (NAS-style)${NC}"
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
DATASET="mnist"
DATA_ROOT=""

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
    echo "  -f, --full              Full dataset (disables quick-test)"
    echo "  --dataset NAME          mnist | strawberry | plant_village_raspberry | plant_village_orange"
    echo "                          | pistachio | acfr_multifruit (aliases: raspberry, orange, acfr)"
    echo "  --data-root DIR         Override dataset root (see utils/dataset_config.py)"
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
    echo "  $0 --quick-test                                    # MNIST smoke test"
    echo "  $0 --dataset strawberry -q -m resnet,simple_cnn    # strawberry quick run"
    echo "  $0 --dataset plant_village_raspberry -q -m resnet,lenet"
    echo "  $0 --dataset plant_village_orange -q -m resnet,lenet"
    echo "  $0 --dataset pistachio -q -m resnet,lenet"
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
        --dataset) DATASET="$2"; shift 2 ;;
        --data-root) DATA_ROOT="$2"; shift 2 ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            ;;
    esac
done

validate_dataset() {
    case "$DATASET" in
        mnist)
            if [ "$QUICK_TEST" = false ] && [ ! -f "data/MNISTtrain.mat" ]; then
                echo -e "${RED}Full MNIST mode requires data/MNISTtrain.mat and data/MNISTtest.mat${NC}"
                exit 1
            fi
            ;;
        strawberry)
            STRAW_ROOT="${DATA_ROOT:-data/Strawberry/strawberries}"
            if [ ! -d "$STRAW_ROOT" ]; then
                echo -e "${RED}Strawberry dataset not found at: $STRAW_ROOT${NC}"
                exit 1
            fi
            for cls in early-turning green late-turning red turning white; do
                if [ ! -f "$STRAW_ROOT/$cls/sets/train.txt" ]; then
                    echo -e "${RED}Missing split file: $STRAW_ROOT/$cls/sets/train.txt${NC}"
                    exit 1
                fi
            done
            ;;
        plant_village_raspberry|raspberry)
            RASP_ROOT="${DATA_ROOT:-data/Plant_Village_Raspberry/raspberries}"
            if [ ! -d "$RASP_ROOT" ]; then
                echo -e "${RED}Plant Village Raspberry dataset not found at: $RASP_ROOT${NC}"
                exit 1
            fi
            if [ ! -f "$RASP_ROOT/healthy/color/sets/train.txt" ]; then
                echo -e "${RED}Missing: $RASP_ROOT/healthy/color/sets/train.txt${NC}"
                exit 1
            fi
            if [ ! -f "$RASP_ROOT/background_without_leaves/without_augmentation/sets/train.txt" ]; then
                echo -e "${RED}Missing: $RASP_ROOT/background_without_leaves/without_augmentation/sets/train.txt${NC}"
                exit 1
            fi
            ;;
        plant_village_orange|orange)
            ORANGE_ROOT="${DATA_ROOT:-data/Plant_Village_Orange/oranges}"
            if [ ! -d "$ORANGE_ROOT" ]; then
                echo -e "${RED}Plant Village Orange dataset not found at: $ORANGE_ROOT${NC}"
                exit 1
            fi
            if [ ! -f "$ORANGE_ROOT/huanglongbing_citrus_greening/color/sets/train.txt" ]; then
                echo -e "${RED}Missing: $ORANGE_ROOT/huanglongbing_citrus_greening/color/sets/train.txt${NC}"
                exit 1
            fi
            if [ ! -f "$ORANGE_ROOT/background_without_leaves/without_augmentation/sets/train.txt" ]; then
                echo -e "${RED}Missing: $ORANGE_ROOT/background_without_leaves/without_augmentation/sets/train.txt${NC}"
                exit 1
            fi
            ;;
        *)
            DATA_ROOT_ARG="${DATA_ROOT:-}"
            if ! python3 -c "
from utils.dataset_config import get_dataset_spec
spec = get_dataset_spec('${DATASET}', '${DATA_ROOT_ARG}' or None)
spec.validate()
print(f'OK: {spec.name} ({spec.num_classes} classes) at {spec.data_root}')
" 2>&1; then
                echo -e "${RED}Unknown or invalid dataset: $DATASET${NC}"
                echo "  Use: mnist, strawberry, plant_village_raspberry, plant_village_orange, pistachio, acfr_multifruit"
                exit 1
            fi
            ;;
    esac
}

validate_dataset

# Reduce CUDA fragmentation when many models run in parallel
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [ "$DATASET" = "mnist" ] && [ "$QUICK_TEST" = true ] && [ ! -f "data/test_data/test_images.npy" ]; then
    echo -e "${YELLOW}Quick-test MNIST data missing; creating test subset...${NC}"
    python3 utils/create_test_data.py || true
fi

CMD="python3 -m utils.regression --output-dir $OUTPUT_DIR --report-path $REPORT_PATH"
CMD="$CMD --max-epochs $MAX_EPOCHS --patience $PATIENCE --min-delta $MIN_DELTA"
CMD="$CMD --nas-trials $NAS_TRIALS --device $DEVICE --seed $SEED --workers $WORKERS"
CMD="$CMD --dataset $DATASET"
if [ -n "$DATA_ROOT" ]; then
    CMD="$CMD --data-root $DATA_ROOT"
fi
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
echo "  Dataset: $DATASET"
[ -n "$DATA_ROOT" ] && echo "  Data root: $DATA_ROOT"
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
