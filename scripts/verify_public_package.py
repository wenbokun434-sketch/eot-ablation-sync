"""Verify copied sources and the exact staged public Git file inventory."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / 'docs/source_manifest.json').read_text(encoding='utf-8'))
for record in manifest['files']:
    path = root / record['path']
    if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
        raise ValueError(f'Original source checksum mismatch: {record["path"]}')
result = subprocess.run(['git', 'ls-files', '-z'], cwd=root, capture_output=True, check=True)
names = [name for name in result.stdout.decode('utf-8').split('\0') if name]
if not names:
    raise ValueError('No staged/tracked files; run git add with the approved code inventory first.')
for name in names:
    path = Path(name)
    if path.parts[0] in {'data', 'outputs', 'private-sync', 'models', 'notebooks'}:
        raise ValueError(f'Restricted directory in Git inventory: {name}')
    if path.suffix.lower() not in {'.py', '.yaml', '.md', '.toml', '.txt', '.json'} and name not in {'.gitignore', '.gitattributes', '.env.example'}:
        raise ValueError(f'Unexpected public file type: {name}')
    text = (root / name).read_text(encoding='utf-8')
    if re.search(r'github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----', text):
        raise ValueError(f'Possible credential in {name}')
    if re.search(r'[A-Za-z]:[/\\]Users[/\\]', text, re.IGNORECASE):
        raise ValueError(f'Personal absolute path in {name}')
print(f'PASS: {len(manifest["files"])} original files match; {len(names)} Git files pass the publication inventory check.')
