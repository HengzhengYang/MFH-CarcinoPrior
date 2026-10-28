"""Copy an explicit project allowlist; never run training or alter source files.

Run only from this package's original location in the MFH project.
Existing destination files are accepted only when byte-identical.
"""
from pathlib import Path
import ast
import csv
import json
import shutil
import os
import hashlib

PACKAGE = Path(__file__).resolve().parents[1]
PROJECT = PACKAGE.parent
TASK01 = 'revision_management/01_data_audit/frozen/'
TASK02 = 'revision_management/02_model_rebuild/'
TASK04 = 'revision_management/04_mfh_rescreening/'


def plan():
    items = []
    def add(source, destination):
        items.append((source, destination))
    for name in (
        'mapping_pair_provenance_frozen.csv',
        'mapping_source_row_disposition_frozen.csv',
        'mapping_count_reconciliation_frozen.csv',
        'mapping_manual_supplement_sensitivity.csv',
    ):
        add(TASK01 + name, 'data/source_structure/' + name)
    for name in (
        'training_record_disposition_frozen.csv',
        'scenario_record_disposition_frozen.csv',
        'scenario_record_to_model_unit.csv',
        'frozen_label_scenario_counts.csv',
        'model_units_primary_conflict_excluded.csv',
        'model_units_sensitivity_conflict_positive.csv',
        'model_units_sensitivity_conflict_negative.csv',
        'model_units_strict_statement_conflict_excluded.csv',
        'model_units_core_bioassay_only_relabelled.csv',
    ):
        add(TASK01 + name, 'data/training_provenance/' + name)
    raw = 'carcinogenicity_training_positive_anchor_deduplicated_1to1_v1.csv'
    # Preserve the original workspace directory name using Unicode escapes.
    add('\u96c6\u6210\u6a21\u578b\u8bad\u7ec3/\u8bad\u7ec3\u96c6/' + raw, 'data/training_provenance/original/' + raw)
    for name in (
        'TASK04_D234_CHEMICAL_ELIGIBILITY_DEFINITION.md',
        'TASK04_D234_SINGLE_ELEMENT_EXCLUSION_REGISTRY.csv',
        'TASK04_D234_CHEMICAL_ELIGIBILITY_COUNT_CLOSURE.json',
    ):
        add(TASK04 + 'eligibility_projection_d234/' + name,
            'data/chemical_eligibility/' + name)
    modules = ('gpu_models.py', 'graph_models.py', 'hybrid_models.py',
               'lm_embeddings.py', 'runtime_guards.py', 'split_protocol.py')
    scripts = (
        'generate_and_validate_full_execution_plan.py',
        'generate_split_manifests.py',
        'preflight_validate_locked_inputs.py',
        'revalidate_split_manifests_p002.py',
        'run_full_training_e001.py',
        'run_limited_gpu_smoke.py',
        'run_limited_rf_svm_smoke.py',
        'validate_full_training_e001.py',
        'verify_gpu_runtime.py',
    )
    code_paths = [TASK02 + 'src/' + name for name in modules]
    code_paths += [TASK02 + 'scripts/' + name for name in scripts]
    code_paths += [TASK02 + 'configs/' + name for name in (
        'registered_protocol_v3_gpu_only.json',
        'registered_protocol_v4_hybrid_restricted.json')]
    code_paths += [TASK02 + 'environment/' + name for name in (
        'full_training_environment_e001.json', 'gpu_environment.yml',
        'hybrid_environment.yml', 'observed_environment.yml')]
    code_paths += [
        TASK04 + 'v3_1/src/task04_v3_1_core.py',
        TASK04 + 'v3_2/runners/task04_mfh_screening_runner_v3_2_candidate.py',
        TASK04 + 'v3/configs/PROTOCOL_TASK04_MFH_SCREENING_V3_CANDIDATE.json',
        TASK04 + 'v3/environment/TASK04_RUNTIME_ENVIRONMENT_LOCK_V3_CANDIDATE.json',
    ]
    for path in code_paths:
        add(path, 'code/original_project/' + path)
    add(TASK02 + 'e003_route/MODEL_LOCK_E005.json',
        'docs/existing_model_reference/MODEL_LOCK_E005.json')
    lock = json.loads((PROJECT / (TASK02 + 'e003_route/MODEL_LOCK_E005.json')).read_text(encoding='utf-8'))
    for member in lock['selected_member_artifacts']:
        for role in ('base_model', 'calibrator'):
            source = member[role + '_path']
            add(source, 'weights/e005/' + Path(source).name)
    asset_manifest = TASK04 + 'v3/assets/MOLFORMER_LOCAL_ASSET_MANIFEST_V3_CANDIDATE.csv'
    add(asset_manifest, 'docs/existing_model_reference/MOLFORMER_LOCAL_ASSET_MANIFEST_V3_CANDIDATE.csv')
    with (PROJECT / asset_manifest).open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            # Copy the actual blob, not a platform-dependent Hugging Face symlink.
            add(row['resolved_blob_path'], 'weights/molformer_backbone/' + row['filename'])
    return items


def source_path(source):
    if source.startswith('/home/') and os.name == 'nt':
        return Path('//wsl.localhost/Ubuntu' + source)
    return PROJECT / source


def main():
    items = plan()
    for source, destination in items:
        src, dst = source_path(source), PACKAGE / destination
        if not src.is_file() or src.is_symlink():
            raise RuntimeError(f'Missing or symbolic-link source: {source}')
        if dst.exists() and src.read_bytes() != dst.read_bytes():
            raise RuntimeError(f'Preserving non-identical destination: {destination}')
    rows = []
    for source, destination in items:
        src, dst = source_path(source), PACKAGE / destination
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copyfile(src, dst)
        data = dst.read_bytes()
        if data != src.read_bytes():
            raise RuntimeError(f'Copy differs: {destination}')
        record_count = ''
        if dst.suffix == '.csv':
            with dst.open(encoding='utf-8-sig', newline='') as stream:
                reader = csv.reader(stream)
                next(reader)
                record_count = sum(1 for _ in reader)
        elif dst.suffix == '.json':
            json.loads(data.decode('utf-8-sig'))
        elif dst.suffix == '.py':
            ast.parse(data.decode('utf-8-sig'), filename=destination)
        rows.append(dict(package_path=destination, original_project_path=source,
                         bytes=len(data), csv_data_rows=record_count,
                         copy_check='byte-identical'))
    # Reuse the existing model/asset identifiers; do not create a new model lock.
    lock = json.loads((PACKAGE / 'docs/existing_model_reference/MODEL_LOCK_E005.json').read_text(encoding='utf-8'))
    expected = []
    for member in lock['selected_member_artifacts']:
        for role in ('base_model', 'calibrator'):
            expected.append((PACKAGE / 'weights/e005' / Path(member[role + '_path']).name,
                             member[role + '_size_bytes'], member[role + '_sha256']))
    with (PACKAGE / 'docs/existing_model_reference/MOLFORMER_LOCAL_ASSET_MANIFEST_V3_CANDIDATE.csv').open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            expected.append((PACKAGE / 'weights/molformer_backbone' / row['filename'], int(row['size_bytes']), row['sha256']))
    for path, size, digest in expected:
        data = path.read_bytes()
        if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
            raise RuntimeError(f'Existing model reference mismatch: {path.name}')
    with (PACKAGE / 'docs/FILE_ORIGINS.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(dict(copied_source_files=len(rows),
                          copied_bytes=sum(row['bytes'] for row in rows),
                          copy_checks='all byte-identical',
                          csv_rows={r['package_path']: r['csv_data_rows'] for r in rows if r['csv_data_rows'] != ''},
                          scientific_execution=False), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
