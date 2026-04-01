#!/usr/bin/env sh

set -eux

export RUSTFLAGS="-Ctarget-cpu=native"
# mkdir -p bench_out
# cargo clean
# cargo fetch
# mkdir -p target
# mkdir -p target/ese-cache
# curl -Lo target/ese-cache/model.safetensors \
# "https://huggingface.co/sentence-transformers/static-retrieval-mrl-en-v1/resolve/main/0_StaticEmbedding/model.safetensors"
# curl -Lo target/ese-cache/tokenizer.json \
# "https://huggingface.co/sentence-transformers/static-retrieval-mrl-en-v1/resolve/main/0_StaticEmbedding/tokenizer.json"
# { time cargo build --release --features=dim-1024; } 2> bench_out/compile_time.txt

# # our gooaq
# rm -rf target/criterion
# cargo bench
# cp -r target/criterion bench_out/criterion_sequential

# # our gooaq parallel
# rm -rf target/criterion
# cargo bench --features=rayon
# cp -r target/criterion bench_out/criterion_parallel

# python benchmarks
cd api-py
maturin build --release --interpreter python3.13
cd ../py-bench
mkdir -p benchmarks
uv sync
uv add --find-links ../target/wheels ese_py
uv run bench_ours.py
uv run bench_bm25.py
while IFS= read -r model; do
  uv run bench.py "$model"
done < models.txt
cd ..
cp -r py-bench/benchmarks bench_out/py_benchmarks

tar -czf bench_results.tar.gz bench_out
