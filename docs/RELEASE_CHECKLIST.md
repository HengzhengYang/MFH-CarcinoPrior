# Before uploading to GitHub

This is a release-preparation checklist for the repository.

1. Document the sources of the association and training-data tables.
2. Provide appropriate citations for the sources used in the study.
3. Retain the MolFormer Apache 2.0 licence with the model files.
4. Confirm authors, affiliations, citation format and contact information. Do not invent a DOI or repository URL.
5. `FILE_ORIGINS.csv` contains original paths for provenance. Unmodified code and reference records also retain local paths and historical management fields. Review these before publication; prepare a separate public copy if they need to be removed, without changing historical source files.
6. E005 weights, calibrators and MolFormer assets are included. The complete runtime environment and historical run data are not bundled, and independent-machine end-to-end reproduction has not been established.

Upload this directory, not the entire project's historical run trees, other model caches or account configuration. `.gitignore` includes exceptions for the six selected model and calibrator files while excluding common caches, other checkpoints and credentials. The approximately 187 MB MolFormer backbone requires Git LFS; it cannot be uploaded through the ordinary GitHub web interface.
