import unittest

import numpy as np
import pandas as pd

from eot.time_features import fit_time_normalizer


class TimeFeatureTests(unittest.TestCase):
    def test_time_normalizer_is_train_only_and_bounded(self):
        posts = pd.DataFrame(
            {
                "user_id": ["train", "train", "train", "test", "test"],
                "sequence_index": [0, 1, 2, 0, 1],
                "delta_days": [0.0, 1.0, 9.0, 0.0, 1e12],
            }
        )
        normalizer, details = fit_time_normalizer(posts, ["train"], 0.99)
        transformed = normalizer.transform(np.array([0.0, 1.0, 9.0, 1e12]))
        self.assertEqual(details["fit_non_initial_gap_count"], 2)
        self.assertEqual(transformed[0], 0.0)
        self.assertTrue(np.all((0.0 <= transformed) & (transformed <= 1.0)))
        self.assertEqual(transformed[-1], 1.0)


if __name__ == "__main__":
    unittest.main()
