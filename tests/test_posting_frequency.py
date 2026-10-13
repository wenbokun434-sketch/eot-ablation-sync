"""Small synthetic checks for subgroup definitions and metric semantics."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, confusion_matrix

spec = importlib.util.spec_from_file_location('posting_frequency', Path(__file__).resolve().parents[1] / 'scripts/evaluate_posting_frequency.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PostingFrequencyTests(unittest.TestCase):
    def users(self, frequencies):
        start = pd.Timestamp('2020-01-01', tz='UTC')
        return pd.DataFrame({'user_id': [str(i) for i in range(len(frequencies))],
                             'sequence_length': [11]*len(frequencies),
                             'first_created_utc': [start]*len(frequencies),
                             'last_created_utc': [start+pd.Timedelta(days=10/f) for f in frequencies]})

    def test_ties_are_not_split_and_input_unchanged(self):
        users = self.users([1, 1, 1, 2, 3, 4, 5, 6, 7])
        original = users.copy(deep=True)
        grouped, _ = module.group_users(users)
        self.assertEqual(grouped.loc[grouped.posts_per_day == 1, 'frequency_group'].nunique(), 1)
        self.assertEqual(len(grouped), len(users))
        self.assertFalse(grouped.frequency_group.isna().any())
        pd.testing.assert_frame_equal(users, original)

    def test_zero_span_fails_instead_of_silent_exclusion(self):
        users = self.users([1, 2, 3])
        users.loc[0, 'last_created_utc'] = users.loc[0, 'first_created_utc']
        with self.assertRaisesRegex(ValueError, 'Undefined frequency'):
            module.group_users(users)

    def test_single_post_fails(self):
        users = self.users([1, 2, 3])
        users.loc[0, 'sequence_length'] = 1
        with self.assertRaises(ValueError):
            module.group_users(users)

    def test_indistinguishable_frequencies_fail(self):
        with self.assertRaisesRegex(ValueError, 'Tied quantiles'):
            module.group_users(self.users([1]*9))

    def test_metrics_use_frozen_decisions_not_default_threshold(self):
        y = np.array([0, 0, 0, 1, 1, 1])
        p = np.array([.05, .1, .2, .3, .4, .45])
        d = np.array([0, 0, 1, 0, 1, 1])
        result = module.metrics(y, p, d)
        tn, fp, fn, tp = confusion_matrix(y, d).ravel()
        self.assertEqual([result[k] for k in ('tn', 'fp', 'fn', 'tp')], [tn, fp, fn, tp])
        self.assertAlmostEqual(result['balanced_accuracy'], balanced_accuracy_score(y, d))
        self.assertAlmostEqual(result['pr_auc'], average_precision_score(y, p))
        self.assertAlmostEqual(result['recall'], 2/3)

    def test_bootstrap_reproducible_and_accounts_for_skips(self):
        f = pd.DataFrame({'label': [0, 1], 'probability': [.1, .9], 'final_prediction': [0, 1]})
        a, valid = module.bootstrap(f, 100, 42)
        self.assertEqual((a, valid), module.bootstrap(f, 100, 42))
        self.assertTrue(0 < valid < 100)
        self.assertEqual(a['recall'], {'lower': 1., 'upper': 1.})


if __name__ == '__main__':
    unittest.main()
