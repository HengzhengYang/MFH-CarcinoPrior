"""Check packaged assets without importing ML libraries or unpickling weights."""
from pathlib import Path
import ast
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def main():
    errors = []
    with (ROOT / 'docs/FILE_ORIGINS.csv').open(encoding='utf-8-sig', newline='') as stream:
        origins = list(csv.DictReader(stream))
    for row in origins:
        path = ROOT / row['package_path']
        if not path.is_file() or path.stat().st_size != int(row['bytes']):
            errors.append(f"Missing file or size mismatch: {row['package_path']}")
            continue
        if path.suffix == '.json':
            json.loads(path.read_text(encoding='utf-8-sig'))
        elif path.suffix == '.py':
            ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
        elif path.suffix == '.csv' and row['csv_data_rows']:
            with path.open(encoding='utf-8-sig', newline='') as stream:
                reader = csv.reader(stream)
                next(reader)
                count = sum(1 for _ in reader)
            if count != int(row['csv_data_rows']):
                errors.append(f'CSV row count mismatch: {path.name}')
    reference = json.loads((ROOT / 'docs/existing_model_reference/MODEL_LOCK_E005.json').read_text(encoding='utf-8'))
    assets = []
    for member in reference['selected_member_artifacts']:
        for role in ('base_model', 'calibrator'):
            assets.append((ROOT / 'weights/e005' / Path(member[role + '_path']).name,
                           member[role + '_size_bytes'], member[role + '_sha256']))
    with (ROOT / 'docs/existing_model_reference/MOLFORMER_LOCAL_ASSET_MANIFEST_V3_CANDIDATE.csv').open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            assets.append((ROOT / 'weights/molformer_backbone' / row['filename'], int(row['size_bytes']), row['sha256']))
    for path, size, digest in assets:
        if not path.is_file():
            errors.append(f'Missing asset: {path.name}')
            continue
        data = path.read_bytes()
        if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
            errors.append(f'Asset differs from original reference: {path.name}; if cloned, run git lfs pull')
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f'PASS: {len(origins)} copied files; {len(assets)} model/backbone assets match existing references.')
    print('No model deserialization, training, feature extraction or prediction performed.')


if __name__ == '__main__':
    main()
