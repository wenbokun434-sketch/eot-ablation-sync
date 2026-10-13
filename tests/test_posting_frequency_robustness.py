"""Independent small-data checks of count-weighted user bootstrap."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, balanced_accuracy_score, f1_score

spec = importlib.util.spec_from_file_location('frequency_robustness', Path(__file__).resolve().parents[1]/'scripts/evaluate_posting_frequency_robustness.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class RobustnessTests(unittest.TestCase):
    def test_weighted_counts_equal_explicit_user_repetition(self):
        y = np.array([0, 0, 1, 1, 1])
        d = np.array([[0, 1], [1, 0], [1, 0], [0, 1], [1, 1]])
        weights = np.array([2, 1, 3, 1, 2])
        counts = np.einsum('n,nsc->sc', weights, m.indicators(y, d))
        actual = m.from_counts(counts)
        idx = np.repeat(np.arange(len(y)), weights)
        for strategy in range(2):
            expected = [recall_score(y[idx], d[idx, strategy]),
                        balanced_accuracy_score(y[idx], d[idx, strategy]),
                        f1_score(y[idx], d[idx, strategy])]
            np.testing.assert_allclose(actual[strategy], expected)

    def test_single_class_draws_remain_undefined(self):
        a = m.from_counts(np.array([[0, 5, 1, 0], [3, 0, 0, 1]]))
        self.assertTrue(np.isnan(a[:, 1]).all())
        self.assertTrue(np.isnan(a[0, 0]))

    def test_reproducible_and_identical_strategies_share_draws(self):
        y = np.array([0, 0, 1, 1])
        d = np.array([[0, 0], [1, 1], [1, 1], [0, 0]])
        p1, a = m.resample_metrics(y, d, 100, 42)
        p2, b = m.resample_metrics(y, d, 100, 42)
        np.testing.assert_array_equal(p1, p2)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a[:, 0], a[:, 1])

    def test_sensitivity_rules_preserve_group_membership(self):
        u = pd.DataFrame({'sequence_length': [256, 256, 20, 20],
                          'span_days': [.5, 1., 2., .5], 'frequency_group': ['low', 'high', 'medium', 'low']})
        original = u.copy(deep=True)
        np.testing.assert_array_equal(m.scenario_mask(u, 'length_256_span_ge_1_day'), [False, True, False, False])
        np.testing.assert_array_equal(m.scenario_mask(u, 'span_ge_1_day'), [False, True, True, False])
        pd.testing.assert_frame_equal(u, original)

    def test_high_minus_low_direction(self):
        high = m.from_counts(np.array([10, 10, 0, 0]))
        low = m.from_counts(np.array([0, 10, 0, 10]))
        np.testing.assert_allclose(high-low, [1., .5, 1.])


if __name__ == '__main__':
    unittest.main()
