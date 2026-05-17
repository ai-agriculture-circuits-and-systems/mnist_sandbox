#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Create or clear the log file
echo "Starting test run at $(date)" > test.log

# All models from model_factory (single source of truth)
mapfile -t models < <(python -c "from models.model_factory import ModelFactory; print('\n'.join(ModelFactory.get_available_models()))")

echo "Testing ${#models[@]} models..." | tee -a test.log

# Loop through each model and run it
for model in "${models[@]}"; do
    echo "Running model: $model" | tee -a test.log
    "$SCRIPT_DIR/run.sh" -m "$model" --quick-test 2>&1 | tee -a test.log
    echo "Finished running $model" | tee -a test.log
    echo "----------------------------------------" | tee -a test.log
done

echo "Test run completed at $(date)" | tee -a test.log
