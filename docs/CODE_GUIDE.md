# Model source guide and reproducibility

The files in `code/original_project/` retain their original project-relative layout and contents.

## Entry points and modules

Paths below are relative to `code/original_project/revision_management/`.

| Path | Purpose |
| --- | --- |
| `02_model_rebuild/src/split_protocol.py` | Structure/group splitting logic. |
| `02_model_rebuild/src/hybrid_models.py` | Conventional model components, including RF and SVM. |
| `02_model_rebuild/src/graph_models.py` | Graph construction and graph models, including MPNN. |
| `02_model_rebuild/src/gpu_models.py` | GPU model components. |
| `02_model_rebuild/src/lm_embeddings.py` | Molecular language-model representations. |
| `02_model_rebuild/src/runtime_guards.py` | Original runtime checks. |
| `02_model_rebuild/scripts/run_full_training_e001.py` | Historical base-model training orchestration, not a separate training run for the E005 selection. |
| `02_model_rebuild/scripts/generate_split_manifests.py` | Split-file generation. |
| `02_model_rebuild/scripts/validate_full_training_e001.py` | Historical training-result validation. |
| `04_mfh_rescreening/v3_1/src/task04_v3_1_core.py` | Three-member screening functions, including `canonical_equal_member_probability`. |
| `04_mfh_rescreening/v3_2/runners/task04_mfh_screening_runner_v3_2_candidate.py` | Original three-member screening, calibration and inference orchestration. |

## E005 composition

The fixed member order is `RandomForest_MACCS`, `MPNN`, `MolFormer_CUDA_Logistic`. Their calibrated probabilities are averaged with equal weight; the classification threshold is `0.42534557978312176`. Do not average uncalibrated raw scores or confuse E005 with the historical 14-member E001 set.

Historical identifiers in candidate and protocol filenames are retained for provenance, not as instructions to restart historical runs. The model-selection reference is an existing E005 record; this package does not make a new model-selection or performance claim.

## Environment and inference

`02_model_rebuild/environment/full_training_environment_e001.json` records Python 3.10.20, NumPy 2.2.6, pandas 2.3.3, RDKit 2025.03.6, scikit-learn 1.7.2, PyTorch 2.11.0+cu128, XGBoost 3.2.0, LightGBM 4.6.0 and psutil 7.2.2. Refer to each stage's record for its environment. Other GPU, hybrid, observed and screening environment records are included; these historical YAML/JSON files are not guaranteed to be directly installable Conda or pip specifications.

The package includes three trained E005 models, three Platt calibrators and the specified MolFormer backbone, tokenizer and implementation. The GPU environment itself is not bundled; non-PyTorch dependencies are listed separately and PyTorch must be installed for the target platform. Historical run authorizations, plans, outputs, external evaluation data, and other upstream data and auxiliary implementations are not included.

Original runners retain their project-specific paths, run plans and checks. `predict.py` instead loads packaged weights using package-relative paths, reuses the original graph features and MPNN class, and handles device and batching independently. See the [README](../README.md) for SMILES and CSV commands. `tests/test_predict.py` covers CPU inference and an optional comparison against the original CUDA helpers; see [INFERENCE_TEST_REPORT.md](INFERENCE_TEST_REPORT.md). Small CPU/CUDA smoke tests were run in the existing WSL environment; the environment was not reinstalled.

`tools/assemble_local_package.py` copies a fixed list of original project files and checks consistency and formats without training. It expects the original project to be the package's parent directory; external users do not need to run it.
