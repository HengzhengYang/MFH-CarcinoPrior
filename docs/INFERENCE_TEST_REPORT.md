# Standalone inference smoke tests

Date: 29 October 2026. Entry point: `predict.py`.

## Environment actually used

- Linux/WSL Ubuntu, Python 3.10.20.
- PyTorch 2.11.0+cu128; CUDA available.
- RDKit 2025.03.6, Transformers 4.44.2.
- Existing project environment; no new training or package installation.

## Executed tests

`MFH_REFERENCE_TEST=1 python -B -m unittest discover -s tests -v`

Five test methods passed:

1. Original CUDA helper comparison: unchanged graph `build_model`/`predict_raw` and MolFormer `extract_or_load_embeddings`, plus independently evaluated RF/calibration/head, versus the new CLI computation. Each member and ensemble agreed within absolute tolerance 1e-7 on ethanol, benzene and aspirin. CPU versus CUDA agreed within 1e-6.
2. CLI rejection of an existing output, nonpositive batch size and missing CSV column; existing content preserved.
3. Real CPU prediction from CSV and direct SMILES, batch sizes 32 and 1, from a temporary working directory outside the project. IDs retained; all member probabilities finite and within [0,1]; ensemble and threshold logic checked.
4. Mixed valid/invalid/empty/duplicate inputs: four rows retained, error rows blank, duplicate predictions equal, exit code 2.
5. Canonicalization and row preservation.

A separate CUDA CLI run on `examples/smiles.csv` exited 0 and wrote `examples/predictions_cuda.csv` (three rows); the reference test also checks this saved example output. No network downloads or historical production runners were used. Temporary test files were removed by the test harness. Deprecation warnings from SWIG and expected RDKit errors for deliberately invalid SMILES did not fail tests.

The existing `tools/check_release.py` also passed: 60 copied files checked, and all 16 model/backbone assets matched their existing references. The new CLI help command exited 0.

This verifies a small inference path, not generalization performance, a retraining reproduction or every possible chemical input. CPU execution was tested in the same WSL environment, not on a separate Windows installation. Historical model code, checkpoints and datasets were not edited.
