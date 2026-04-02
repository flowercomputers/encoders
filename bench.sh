#!/usr/bin/env sh

set -eu

ROOT_DIR=$(
    CDPATH= cd -- "$(dirname -- "$0")"
    pwd
)
API_PY_DIR="$ROOT_DIR/api-py"
PY_BENCH_DIR="$ROOT_DIR/benches/py"
RESULTS_DIR="${RESULTS_DIR:-$ROOT_DIR/bench_out}"
MODELS_FILE="${MODELS_FILE:-$PY_BENCH_DIR/models.txt}"
PYTHON_BIN="${PYTHON_BIN:-python3.13}"
SMOKE_MODEL="${SMOKE_MODEL:-ours}"
ARCHIVE_PATH="${ARCHIVE_PATH:-$ROOT_DIR/bench_results.tar.gz}"

export RUSTFLAGS="${RUSTFLAGS:--Ctarget-cpu=native}"

usage() {
    cat <<EOF
Usage: ./bench.sh <command> [options]

Commands:
  verify           Run the lightweight Rust verification path.
  rust             Run the Rust Criterion benchmarks and collect results.
  python [model]   Run the Python GooAQ benchmark for one model.
  python --all     Run the Python GooAQ benchmark for every model in MODELS_FILE.
  nanobeir [model] Run the NanoBEIR retrieval evaluation for one model.
  nanobeir --all   Run the NanoBEIR retrieval evaluation for every model in MODELS_FILE.
  all              Run verify, rust, python, and nanobeir with smoke defaults.
  help             Show this message.

Environment:
  PYTHON_BIN       Python interpreter for Python benchmarks. Default: python3.13
  RESULTS_DIR      Where collected benchmark outputs are stored. Default: bench_out
  MODELS_FILE      Model list used by --all. Default: benches/py/models.txt
  SMOKE_MODEL      Model used by python/nanobeir without arguments. Default: ours
  BENCH_TIME_SECONDS
                   Optional override for benches/py/bench.py. Default: 10

Notes:
  - Rust paths require only cargo and rustc.
  - Python paths require uv, maturin, and the configured Python interpreter.
  - The first run downloads model assets into target/ese-cache. Benchmark runs may
    also download datasets and third-party model weights into local caches.
EOF
}

die() {
    printf '%s\n' "error: $*" >&2
    exit 1
}

need_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

ensure_file() {
    [ -f "$1" ] || die "missing required file: $1"
}

ensure_dir() {
    mkdir -p "$1"
}

reset_dir() {
    rm -rf "$1"
    mkdir -p "$1"
}

archive_results() {
    ensure_dir "$RESULTS_DIR"
    tar -czf "$ARCHIVE_PATH" -C "$(dirname "$RESULTS_DIR")" "$(basename "$RESULTS_DIR")"
    printf '%s\n' "Archived results to $ARCHIVE_PATH"
}

copy_if_exists() {
    src=$1
    dest=$2
    [ -e "$src" ] || return 0
    rm -rf "$dest"
    mkdir -p "$(dirname "$dest")"
    cp -R "$src" "$dest"
}

find_latest_wheel() {
    wheel_dir=$ROOT_DIR/target/wheels
    [ -d "$wheel_dir" ] || die "wheel directory not found: $wheel_dir"
    wheel=$(find "$wheel_dir" -maxdepth 1 -name 'ese_py-*.whl' | sort | tail -n 1)
    [ -n "$wheel" ] || die "no ese_py wheel found under $wheel_dir"
    printf '%s\n' "$wheel"
}

sync_python_env() {
    (
        cd "$PY_BENCH_DIR"
        uv sync --python "$PYTHON_BIN"
    )
}

build_python_wheel() {
    (
        cd "$API_PY_DIR"
        maturin build --release --interpreter "$PYTHON_BIN"
    )
    find_latest_wheel
}

run_python_script() {
    script_name=$1
    model_name=$2
    wheel_path=$3
    output_dir=$4

    (
        cd "$PY_BENCH_DIR"
        BENCHMARKS_DIR="$output_dir" \
            uv run --python "$PYTHON_BIN" --with "$wheel_path" python "$script_name" "$model_name"
    )
}

read_models() {
    ensure_file "$MODELS_FILE"
    while IFS= read -r model_name; do
        [ -n "$model_name" ] || continue
        printf '%s\n' "$model_name"
    done <"$MODELS_FILE"
}

verify() {
    need_cmd cargo
    ensure_dir "$RESULTS_DIR"

    cargo test -p ese --features tests
    cargo build -p ese-c
    cargo bench --bench bench_encode --no-run
    cargo bench --bench gooaq --no-run
}

run_rust_benches() {
    mode_name=$1
    criterion_dest=$2
    shift 2

    rm -rf "$ROOT_DIR/target/criterion"
    cargo bench "$@"
    copy_if_exists "$ROOT_DIR/target/criterion" "$criterion_dest"
    printf '%s\n' "Collected $mode_name results in $criterion_dest"
}

rust_benchmarks() {
    need_cmd cargo
    rust_results_dir="$RESULTS_DIR/rust"
    ensure_dir "$rust_results_dir"

    run_rust_benches \
        "sequential Criterion" \
        "$rust_results_dir/criterion_sequential" \
        --bench bench_encode --bench gooaq
    run_rust_benches \
        "Rayon Criterion" \
        "$rust_results_dir/criterion_rayon" \
        --features rayon --bench bench_encode --bench gooaq

    archive_results
}

python_dependencies() {
    need_cmd cargo
    need_cmd "$PYTHON_BIN"
    need_cmd uv
    need_cmd maturin
}

python_benchmarks() {
    mode=${1:-smoke}
    model_arg=${2:-}

    python_dependencies
    ensure_dir "$RESULTS_DIR/python"
    output_dir="$RESULTS_DIR/python/gooaq"
    reset_dir "$output_dir"
    sync_python_env
    wheel_path=$(build_python_wheel)

    if [ "$mode" = "all" ]; then
        read_models | while IFS= read -r model_name; do
            printf '%s\n' "Running GooAQ benchmark for $model_name"
            run_python_script "bench.py" "$model_name" "$wheel_path" "$output_dir"
        done
    else
        model_name=${model_arg:-$SMOKE_MODEL}
        printf '%s\n' "Running GooAQ benchmark for $model_name"
        run_python_script "bench.py" "$model_name" "$wheel_path" "$output_dir"
    fi

    archive_results
}

nanobeir_benchmarks() {
    mode=${1:-smoke}
    model_arg=${2:-}

    python_dependencies
    ensure_dir "$RESULTS_DIR/python"
    output_dir="$RESULTS_DIR/python/nanobeir"
    reset_dir "$output_dir"
    sync_python_env
    wheel_path=$(build_python_wheel)

    if [ "$mode" = "all" ]; then
        read_models | while IFS= read -r model_name; do
            printf '%s\n' "Running NanoBEIR evaluation for $model_name"
            run_python_script "nanobeir.py" "$model_name" "$wheel_path" "$output_dir"
        done
    else
        model_name=${model_arg:-$SMOKE_MODEL}
        printf '%s\n' "Running NanoBEIR evaluation for $model_name"
        run_python_script "nanobeir.py" "$model_name" "$wheel_path" "$output_dir"
    fi

    archive_results
}

command_name=${1:-help}

case "$command_name" in
    help|-h|--help)
        usage
        ;;
    verify)
        verify
        ;;
    rust)
        rust_benchmarks
        ;;
    python)
        case "${2:-}" in
            --all)
                python_benchmarks all
                ;;
            "")
                python_benchmarks smoke "$SMOKE_MODEL"
                ;;
            *)
                python_benchmarks smoke "$2"
                ;;
        esac
        ;;
    nanobeir)
        case "${2:-}" in
            --all)
                nanobeir_benchmarks all
                ;;
            "")
                nanobeir_benchmarks smoke "$SMOKE_MODEL"
                ;;
            *)
                nanobeir_benchmarks smoke "$2"
                ;;
        esac
        ;;
    all)
        verify
        rust_benchmarks
        python_benchmarks smoke "$SMOKE_MODEL"
        nanobeir_benchmarks smoke "$SMOKE_MODEL"
        ;;
    *)
        usage >&2
        die "unknown command: $command_name"
        ;;
esac
