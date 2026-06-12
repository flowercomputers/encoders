# greedy-fidelity spike

One-off measurement backing Appendix A of `docs/rfc-generalization.md`. Results
are frozen in `results.json` and are not re-generated as part of the build or
test workflow. This directory is not maintained.

To re-run: `uv run python measure_fidelity.py` (requires network access to
Hugging Face on first run to download the reference model).
