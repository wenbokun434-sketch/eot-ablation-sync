"""Copy restricted synchronization artifacts into an ignored local directory."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--source', required=True, type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
destination = root / 'private-sync'
names = [f'data/{folder}/fold_{fold}.json' for folder in ('splits', 'time_stats') for fold in range(5)]
missing = [name for name in names if not (args.source / name).is_file()]
if missing:
    raise FileNotFoundError(f'Missing synchronization files: {missing}')
existing = [name for name in names if (destination / name).exists()]
if existing or (destination / 'manifest.json').exists():
    raise FileExistsError('private-sync already contains artifacts; move it aside before exporting again.')
records = []
for name in names:
    source = args.source / name
    target = destination / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    records.append({'path': name, 'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
(destination / 'manifest.json').write_text(json.dumps({'private_only': True, 'files': records}, indent=2) + '\n', encoding='utf-8')
print('Exported 10 files to ignored private-sync/. Share only with authorized teammates.')
