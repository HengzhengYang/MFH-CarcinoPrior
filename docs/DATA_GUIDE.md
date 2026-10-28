# Data and field guide

## 1. Source–structure associations

Main table: `data/source_structure/mapping_pair_provenance_frozen.csv`.

| Field | Meaning and use |
| --- | --- |
| `mapping_pair_id` | Association-pair identifier; do not substitute a row number. |
| `mfh_id`, `official_name` | Source identifier and name. |
| `standardized_inchikey`, `standardized_smiles` | Standardized structure identifier and representation. |
| `source_csv_row_numbers` | Input association-row lineage. |
| `herb_ids`, `ingredient_ids`, `pubchem_cids` | Upstream database identifiers; some fields aggregate multiple values. |
| `source_database`, `source_batch`, `source_url` | Source database, batch and original link. |
| `herb_or_manual_evidence_urls`, `ingredient_structure_urls` | Evidence links for source associations and structures. |
| `manual_supplement_flag`, `shared_herb_association_flag` | Manual-supplement and shared-association flags. |
| `association_measurement_caveat` | Distinction between database association and measurement. |

The source-row disposition table contains 15,774 rows; the deduplicated association-pair table contains 11,703. The different counts do not imply missing data. The two supplementary tables in the same directory describe count reconciliation and manual-supplement sensitivity. Upstream links are retained as recorded; relative URLs are not complete external addresses.

The original data were not restandardized or reassigned during packaging. Retain original identifiers and transformation records if applying additional structure transformations. The prediction CLI separately canonicalizes its own input SMILES, as described in the README.

## 2. Training-record provenance

Each row of `training_record_disposition_frozen.csv` corresponds to an original training record. Key fields are:

| Field | Meaning and use |
| --- | --- |
| `training_record_id`, `train_row_id`, `source_csv_row_number` | Stable record identifiers and original row index. |
| `chemical_name`, `casrn`, `original_smiles` | Original chemical identity. |
| `evidence_sources`, `evidence_details`, `balanced_source_file` | Training-label sources, evidence and source file. |
| `integrated_label` | Original integrated label, not necessarily the final label for every scenario. |
| `record_level_conflict`, `statement_level_conflict` | Label-evidence conflict flags. |
| `has_core_bioassay_evidence`, `core_evidence_rederived_label` | Core evidence and rederived label. |
| `structure_decision`, `structure_exclusion_reason` | Structure-processing decision and exclusion reason. |
| `final_isomeric_smiles`, `final_nonisomeric_smiles`, `final_inchikey` | Processed structure fields. |
| `final_model_unit_id`, `final_cv_group_id` | Final model-unit and cross-validation group identifiers. |

Use `scenario_record_disposition_frozen.csv`, `scenario_record_to_model_unit.csv` and the corresponding `model_units_*.csv` together to interpret scenario-specific inclusion and labels. The selected scenario's model-unit `final_label` is the training label; do not substitute the original integrated label.

For the primary analysis, 1,628 of 1,668 original records passed label eligibility, 1,538 passed structure eligibility, and deduplication yielded 1,494 model units. The other four model-unit tables represent sensitivity scenarios, not additional independent training data. The five scenarios contain 1,494, 1,526, 1,522, 1,216 and 1,208 model units, respectively; see `frozen_label_scenario_counts.csv`.

## 3. Element-only exclusions and reporting

`data/chemical_eligibility/` retains the existing definition, 26 excluded records and count reconciliation. The original association table covers 5,003 structures and 11,703 pairs; the adjusted molecular analysis set covers 4,977 structures and 11,642 pairs, with 106 sources unchanged.

Records can be matched against the exclusion registry using `standardized_inchikey`. The reporting projection did not recalculate model probabilities. Distinguish all molecules from the `STRUCTURE_UNSEEN` subset: at the 0.75 threshold, the adjusted counts are 125 overall and 111 unseen; at 0.80, they are 46 overall and 39 unseen. The copied count JSON provides the denominators and threshold definitions.

## 4. File provenance

`FILE_ORIGINS.csv` records package paths, original project paths, sizes, CSV data-row counts and copy comparisons. CSV rows were counted with a CSV parser, not by counting line breaks. Historical records, data and model files were not regenerated or overwritten.
