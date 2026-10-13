"""Hash-checked inner-H readers. No C/T data reader is implemented for V02."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eot.optimization.evaluation import (FINAL_COLUMNS, PREFIX_COLUMNS, PredictionBundle,
                                         _validated_frame, _check_pair, frame_digest)
from eot.optimization.splits import digest

BASE = 'outputs/optimization/v2/protocol'


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def bundle(final, prefix, expected_ids, provenance=None):
    # Reject null IDs, fractional positions/lengths and all-prefix label mismatches
    # before the original validators cast integer columns.
    for frame, columns in ((final, FINAL_COLUMNS), (prefix, PREFIX_COLUMNS)):
        if set(frame.columns) != set(columns) or frame.empty or frame.isna().any().any():
            raise ValueError('Missing columns, empty input or null prediction value')
        if any(not isinstance(x, str) or not x.strip() for x in frame.user_id):
            raise ValueError('Invalid user ID')
        for name in ('length', 'position'):
            if name in frame:
                values = frame[name].to_numpy(dtype=float)
                if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
                    raise ValueError('Non-integral sequence length or position')
                if (values < 1).any() or (values > 256).any():
                    raise ValueError('Sequence position/length out of frozen bounds')
    final = _validated_frame(final, FINAL_COLUMNS, 'v2 final')
    prefix = _validated_frame(prefix, PREFIX_COLUMNS, 'v2 prefix')
    if len(expected_ids) != len(set(expected_ids)) or final.user_id.tolist() != sorted(expected_ids):
        raise ValueError('Missing or unauthorized cohort users')
    if set(final.label) != {0, 1}:
        raise ValueError('Both classes required')
    if (prefix.groupby('user_id').label.nunique() != 1).any():
        raise ValueError('Inconsistent labels within user prefixes')
    _check_pair(final, prefix, 'v2 pair')
    return PredictionBundle(json.dumps(provenance or {}, sort_keys=True), final, prefix)


def subset(parent, ids):
    return bundle(parent.final[parent.final.user_id.isin(ids)],
                  parent.prefix[parent.prefix.user_id.isin(ids)], ids)


class Sources:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.lock = read(self.root / BASE / 'protocol_lock.json')
        for relative, expected in self.lock['files'].items():
            if sha(self.root / relative) != expected:
                raise ValueError(f'Protocol hash differs: {relative}')
        self.lock_hash = sha(self.root / BASE / 'protocol_lock.json')
        acceptance = read(self.root / BASE / 'v01_acceptance.json')
        if acceptance['status'] != 'pass' or acceptance['lock_sha256'] != self.lock_hash:
            raise ValueError('V01 acceptance or protocol identity differs')
        self.manifest = read(self.root / BASE / 'sources.json')
        self.rules = read(self.root / BASE / 'rules.json')
        self.used = {}

    def checked(self, relative):
        relative = Path(relative).as_posix()
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root) or relative not in self.manifest['project_files']:
            raise ValueError('Unlisted/out-of-project input')
        expected = self.manifest['project_files'][relative]
        actual = sha(path)
        if actual != expected['sha256'] or path.stat().st_size != expected['bytes']:
            raise ValueError(f'Frozen input changed: {relative}')
        self.used[relative] = actual
        return path

    def json(self, relative):
        return read(self.checked(relative))

    def inner(self, f, j, *, role='inner_holdout'):
        if role != 'inner_holdout' or type(f) is not int or type(j) is not int or f not in range(5) or j not in range(3):
            raise ValueError('V02 reader only permits inner holdout scopes')
        group = next(g for g in self.manifest['reusable_prediction_groups']
                     if g['role'] == role and g['outer_fold'] == f and g['inner_fold'] == j)
        paths = {Path(p).name: self.checked(p) for p in group['files']}
        meta = read(paths['ensemble.json'])
        prov = meta['provenance']
        for key, value in {'outer_fold': f, 'inner_fold': j, 'role': role,
                           'candidate_id': 'A', 'seeds': [42, 43, 44], 'stage': 'stability',
                           'aggregation': 'arithmetic_mean_probabilities_across_seeds_within_fold'}.items():
            if prov[key] != value:
                raise ValueError(f'Ensemble role/model mismatch: {key}')
        split = self.json(f'data/optimization/v1/splits/outer_{f}.json')
        ids = sorted(split['inner'][j]['roles']['holdout'])
        for role_name in ('train', 'early_stop'):
            if set(ids) & set(split['inner'][j]['roles'][role_name]):
                raise ValueError('Inner role overlap')
        for role_name in ('calibration', 'test'):
            if set(ids) & set(split['outer_roles'][role_name]):
                raise ValueError('Outer role overlap')
        for name in ('final', 'prefix'):
            if sha(paths[f'{name}.parquet']) != meta[f'{name}_sha256']:
                raise ValueError('Ensemble file manifest differs')
        result = bundle(pd.read_parquet(paths['final.parquet']), pd.read_parquet(paths['prefix.parquet']), ids, prov)
        for name, columns in (('final', FINAL_COLUMNS), ('prefix', PREFIX_COLUMNS)):
            if frame_digest(getattr(result, name), columns) != prov[f'{name}_sha256']:
                raise ValueError('Ensemble frame content hash differs')
        if digest(ids) != prov['cohort_user_ids_sha256'] or digest([[r.user_id, int(r.label)] for r in result.final.itertuples()]) != prov['labels_sha256']:
            raise ValueError('Ensemble users/labels hash differs')
        # Verify all three frozen seed predictions and average them without loading a model.
        seed_bundles = []
        for seed in (42, 43, 44):
            stage = 'screening' if seed == 42 else 'stability'
            directory = f'outputs/optimization/v1/{stage}/evaluation/outer_{f}/inner_{j}/A/seed_{seed}'
            record = self.json(directory + '/prediction_manifest.json')
            link = next(r for r in prov['seed_runs'] if r['seed'] == seed)
            for key, value in {'seed': seed, 'outer_fold': f, 'inner_fold': j, 'candidate_id': 'A',
                               'run_id': link['run_id'], 'checkpoint_sha256': link['checkpoint_sha256'],
                               'run_manifest_sha256': link['run_sha256'], 'labels_sha256': prov['labels_sha256']}.items():
                if record[key] != value:
                    raise ValueError(f'Seed source differs: {key}')
            # O10 includes a checkpoint path; O11 records only its hash. Both
            # resolve through the frozen run identity and its committed slot.
            run_dir = f'outputs/optimization/v1/{stage}/outer_{f}/inner_{j}/A/seed_{seed}/{record["run_id"]}'
            run_path = self.checked(run_dir + '/run.json')
            run = read(run_path)
            checkpoint_relative = run_dir + '/' + run['checkpoint']
            checkpoint = self.checked(checkpoint_relative)
            if record.get('checkpoint', checkpoint_relative) != checkpoint_relative:
                raise ValueError('Prediction checkpoint path differs from committed run')
            if sha(checkpoint) != record['checkpoint_sha256'] or sha(run_path) != record['run_manifest_sha256']:
                raise ValueError('Checkpoint/run hash mismatch')
            if run['checkpoint_sha256'] != record['checkpoint_sha256'] or run['binding']['run_id'] != record['run_id']:
                raise ValueError('Committed run source differs from prediction')
            if run['status'] != 'complete' or run['best_epoch'] != record['best_epoch']:
                raise ValueError('Incomplete or mismatched frozen training run')
            frames = []
            for name in ('final', 'prefix'):
                if record[name] != directory + f'/{name}.parquet':
                    raise ValueError('Unexpected seed prediction path')
                path = self.checked(record[name])
                if sha(path) != record[name + '_sha256']:
                    raise ValueError('Seed prediction checksum differs')
                frames.append(pd.read_parquet(path))
            seed_bundles.append(bundle(*frames, ids))
        for name in ('final', 'prefix'):
            baseline = getattr(result, name)
            for seed_bundle in seed_bundles:
                pd.testing.assert_frame_equal(getattr(seed_bundle, name).drop(columns='probability'),
                                              baseline.drop(columns='probability'))
            mean = np.mean([getattr(b, name).probability.to_numpy() for b in seed_bundles], axis=0)
            if not np.array_equal(mean, baseline.probability.to_numpy()):
                raise ValueError('Saved ensemble is not exact seed mean')
        return result, split


def verify_protected(root):
    """File bytes only: never deserialize C/T predictions during V02."""
    source = Sources(root)
    for relative in source.manifest['project_files']:
        source.checked(relative)
    raw = Path(source.manifest['raw_root'])
    for record in source.manifest['original_protection']:
        path = (source.root if record['anchor'] == 'project' else raw) / record['path']
        if sha(path) != record['sha256'] or path.stat().st_size != record['bytes']:
            raise ValueError(f'Protected file changed: {record["path"]}')
    actual = sorted(p.relative_to(source.root).as_posix() for p in (source.root / 'outputs/optimization/v1').rglob('*')
                    if p.is_file() and '__pycache__' not in p.parts)
    if set(actual) != set(source.manifest['v1_members']):
        raise ValueError('v1 directory membership changed')
    return {'project_files': len(source.manifest['project_files']),
            'original_protected_files': len(source.manifest['original_protection']),
            'protocol_lock_sha256': source.lock_hash}
