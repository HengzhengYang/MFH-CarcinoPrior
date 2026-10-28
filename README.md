# MFH-CarcinoPrior

MFH-CarcinoPrior links molecular hazard predictions to traceable medicinal and food homologous (MFH) source associations. This repository contains source–structure data, training-record provenance, model code, E005 weights and a standalone prediction script.

## Contents and data

- `data/source_structure/`: source–structure associations and provenance.
- `data/training_provenance/`: training records and model-unit tables.
- `data/chemical_eligibility/`: element-only exclusions used in reporting.
- `code/original_project/`: original model and screening code.
- `weights/e005/`: three models and their Platt calibrators.
- `weights/molformer_backbone/`: local MolFormer encoder and tokenizer.
- `predict.py`, `examples/`, `tests/`: standalone inference and smoke tests.

The association table contains 11,703 pairs linking 106 MFH sources to 5,003 structures: 11,689 HERB-derived associations and 14 manual supplements. The molecular reporting set excludes 26 element-only records. The primary training set contains 1,494 model units, with provenance for 1,668 original records. Source associations are not measurements of occurrence or exposure.

## Predicting from SMILES

E005 averages calibrated probabilities from RandomForest-MACCS, MPNN and a MolFormer embedding-based logistic model. Its registered classification threshold is 0.42534557978312176. Predictions support computational hazard prioritization, not direct estimates of human cancer risk.

Use Python 3.10 and install a platform-appropriate PyTorch build, followed by:

```bash
python -m pip install -r requirements-inference.txt
python -B predict.py --smiles "CCO" "c1ccccc1" --output outputs/predictions.csv
python -B predict.py --input examples/smiles.csv --id-column id --output outputs/batch_predictions.csv
```

The output includes each member's calibrated probability, `p_E005`, the threshold comparison and SMILES status. `--device auto` selects CUDA when available, otherwise CPU; use `--device cpu` or `--device cuda` to choose explicitly. Existing output files are not overwritten. Invalid SMILES remain in the output with blank probabilities; exit code 2 indicates one or more invalid rows. The script canonicalizes input but does not remove salts, neutralize charges or select tautomers. MolFormer input beyond 256 tokens is truncated and flagged.

## Verification and release

```bash
python -B tools/check_release.py
python -B -m unittest discover -s tests -v
```

The CLI was smoke-tested on three example molecules in the existing Python 3.10.20 WSL environment, on both CPU and CUDA. CUDA results agreed with the original graph and MolFormer inference helpers within an absolute tolerance of 1e-7; CPU/CUDA results agreed within 1e-6. See `docs/INFERENCE_TEST_REPORT.md`. These checks are not a new model-performance evaluation. Historical production runners retain project-specific paths and are not standalone entrypoints.

Model files use Python deserialization and the local MolFormer implementation uses `trust_remote_code=True`; load only trusted files. The encoder is about 187 MB and requires Git LFS for GitHub distribution. The bundled MolFormer backbone is Apache-2.0; see its licence in `weights/molformer_backbone/`.
