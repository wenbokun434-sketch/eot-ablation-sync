"""Pure fit-only threshold policy. Does not receive evaluation records."""
from fractions import Fraction

import numpy as np

from eot.metrics import DEFAULT_THRESHOLD_GRID, RiskSequence, select_threshold
from eot.optimization.decisions import _final_search
from eot.optimization.evaluation import confusion_counts
from eot.optimization.splits import digest


def checked_sequences(sequences, expected_ids):
    sequences = list(sequences)
    expected_ids = list(expected_ids)
    if not sequences or not expected_ids:
        raise ValueError('Empty selection cohort')
    if any(not isinstance(s, RiskSequence) for s in sequences):
        raise ValueError('Expected RiskSequence records')
    ids = [s.user_id for s in sequences]
    if any(not isinstance(x, str) or not x.strip() for x in ids + expected_ids):
        raise ValueError('User IDs must be non-empty strings')
    if len(set(ids)) != len(ids) or len(set(expected_ids)) != len(expected_ids):
        raise ValueError('Duplicate users')
    if set(ids) != set(expected_ids):
        raise ValueError('Missing or unauthorized users')
    if {s.label for s in sequences} != {0, 1}:
        raise ValueError('Both binary classes are required')
    for s in sequences:
        p = np.asarray(s.probabilities)
        if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError('Invalid probability sequence')
    return sorted(sequences, key=lambda s: s.user_id)


def select_accuracy_first(sequences, reference_threshold, *, expected_ids):
    sequences = checked_sequences(sequences, expected_ids)
    if isinstance(reference_threshold, bool) or not isinstance(reference_threshold, (int, float)):
        raise ValueError('Reference threshold must be numeric')
    if not np.isfinite(reference_threshold) or not np.any(DEFAULT_THRESHOLD_GRID == reference_threshold):
        raise ValueError('Reference threshold is not on the frozen grid')
    labels = np.array([s.label for s in sequences])
    probabilities = np.array([s.probabilities[-1] for s in sequences])
    reference = confusion_counts(labels, probabilities >= reference_threshold)
    rows = []
    for threshold in DEFAULT_THRESHOLD_GRID:
        counts = confusion_counts(labels, probabilities >= threshold)
        rows.append({'threshold': float(threshold), **counts,
                     'precision': counts['tp'] / (counts['tp'] + counts['fp']) if counts['tp'] + counts['fp'] else 0.0,
                     'feasible': counts['correct'] >= reference['correct'] and counts['tp'] >= reference['tp']})
    feasible = [row for row in rows if row['feasible']]
    if not feasible:
        raise AssertionError('Reference should be feasible; refusing to invent a threshold')
    best = max(feasible, key=lambda r: (r['correct'], r['tp'],
               Fraction(r['tp'], r['tp'] + r['fp']) if r['tp'] + r['fp'] else Fraction(0), r['threshold']))
    unchanged = best['correct'] == reference['correct'] and best['tp'] == reference['tp']
    threshold = float(reference_threshold) if unchanged else best['threshold']
    return {'threshold_final': threshold, 'threshold_early': float(reference_threshold),
            'status': 'unchanged_reference' if unchanged else 'selected',
            'reference': reference, 'counts': confusion_counts(labels, probabilities >= threshold),
            'best_grid_threshold': best['threshold'], 'search': rows}


def fit_policies(fit_sequences, *, expected_fit_ids):
    """Only Q is accepted; P probabilities and labels cannot enter this call."""
    fit = checked_sequences(fit_sequences, expected_fit_ids)
    early = select_threshold(fit, consecutive=2, candidates=DEFAULT_THRESHOLD_GRID)
    reference = float(early['threshold'])
    new = select_accuracy_first(fit, reference, expected_ids=expected_fit_ids)
    old = _final_search(np.array([s.label for s in fit]),
                        np.array([s.probabilities[-1] for s in fit]),
                        new['reference']['correct'], DEFAULT_THRESHOLD_GRID)
    if old['threshold_final'] is None:
        raise AssertionError('Old reference policy unexpectedly infeasible')
    return {'fit_user_ids': [s.user_id for s in fit],
            'fit_labels_sha256': digest([[s.user_id, int(s.label)] for s in fit]),
            'fit_predictions_sha256': digest([[s.user_id, int(s.label), s.probabilities.tolist()] for s in fit]),
            'early_selection': early,
            'policies': {'R1': {'threshold_final': reference, 'threshold_early': reference,
                                'counts': new['reference']},
                         'R2': {**old, 'threshold_early': reference}, 'ACC_V2': new}}
