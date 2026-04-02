#!/usr/bin/env sh

set -eu

SCRIPT_DIR=$(
    CDPATH= cd -- "$(dirname -- "$0")"
    pwd
)
MODELS_FILE="${MODELS_FILE:-$SCRIPT_DIR/models.txt}"
PYTHON_BIN="${PYTHON_BIN:-python3.13}"
BENCHMARKS_DIR="${BENCHMARKS_DIR:-$SCRIPT_DIR/benchmarks}"
WHEEL_DIR="${WHEEL_DIR:-$SCRIPT_DIR/../../target/wheels}"

mkdir -p "$BENCHMARKS_DIR"
wheel_path=$(find "$WHEEL_DIR" -maxdepth 1 -name 'ese_py-*.whl' | sort | tail -n 1)
[ -n "$wheel_path" ] || {
    printf '%s\n' "error: no ese_py wheel found under $WHEEL_DIR" >&2
    exit 1
}

(
    cd "$SCRIPT_DIR"
    while IFS= read -r model_name; do
        [ -n "$model_name" ] || continue
        BENCHMARKS_DIR="$BENCHMARKS_DIR" \
            uv run --python "$PYTHON_BIN" --with "$wheel_path" python nanobeir.py "$model_name"
    done <"$MODELS_FILE"
)
