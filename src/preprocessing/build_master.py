"""Build / refresh the preprocessed master dataset.

The master JSON (Q*_corrected.json) was produced from data/raw/*.xlsx via the fixed
prompt in prompts/preprocessing_prompt.txt (model: claude-opus-4-8[1m]). This module
documents that step and can re-run it through an API for reproducibility.
"""
# TODO: xlsx_to_input_batches(), run_preprocessing_api(), validate_schema()
