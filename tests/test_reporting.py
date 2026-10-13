import unittest
import pandas as pd

from eot.reporting import prefix_performance, select_cases


class ReportingTests(unittest.TestCase):
    def test_prefix_curve_keeps_short_users_and_frozen_thresholds(self):
        final = pd.DataFrame({"user_id": ["a", "b"], "label": [0, 1], "length": [1, 3],
                              "fold": [0, 0], "threshold": [0.95, 0.95]})
        prefix = pd.DataFrame({"user_id": ["a", "b", "b", "b"], "position": [1, 1, 2, 3],
                               "probability": [0.1, 0.6, 0.9, 0.99]})
        results = prefix_performance(final, prefix, [1, 2, 3])
        pooled = results.loc[results.scope == "oof"].set_index("checkpoint")
        self.assertEqual(pooled.users.tolist(), [2, 2, 2])
        self.assertEqual(pooled.users_with_at_least_k_posts.tolist(), [2, 1, 1])
        self.assertEqual(pooled.recall.tolist(), [0.0, 0.0, 1.0])

    def test_missing_prefix_is_rejected_instead_of_dropping_user(self):
        final = pd.DataFrame({"user_id": ["a", "b"], "label": [0, 1], "length": [1, 3],
                              "fold": [0, 0], "threshold": [0.5, 0.5]})
        prefix = pd.DataFrame({"user_id": ["a", "b"], "position": [1, 1], "probability": [0.1, 0.9]})
        with self.assertRaises(ValueError):
            prefix_performance(final, prefix, [3])

    def test_case_selection_is_order_independent_and_explains_alert_prefix(self):
        final = pd.DataFrame({"user_id": ["z", "a", "b"], "label": [1, 1, 0],
                              "length": [10, 10, 20], "alert_prediction": [1, 1, 0],
                              "alert_position": [2.0, 4.0, float("nan")]})
        a = select_cases(final)
        b = select_cases(final.iloc[::-1])
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(a.iloc[0].user_id, "a")
        self.assertEqual(a.iloc[0].explanation_position, 4)
        self.assertEqual(a.iloc[1].explanation_position, 20)


if __name__ == "__main__":
    unittest.main()
