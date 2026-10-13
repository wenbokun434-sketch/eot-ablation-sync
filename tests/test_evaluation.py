import unittest

import numpy as np
import pandas as pd

from eot.evaluation import metrics_from_final_predictions


class EvaluationTests(unittest.TestCase):
    def test_saved_final_predictions_are_sufficient_to_recompute_metrics(self):
        frame = pd.DataFrame(
            {
                "label": [0, 0, 1, 1],
                "probability": [0.1, 0.2, 0.8, 0.9],
                "final_prediction": [0, 0, 1, 1],
                "alert_prediction": [0, 0, 1, 1],
                "alert_position": [np.nan, np.nan, 2.0, 3.0],
            }
        )
        metrics = metrics_from_final_predictions(frame)
        self.assertAlmostEqual(metrics["classification"]["pr_auc"], 1.0)
        self.assertAlmostEqual(metrics["classification"]["mcc"], 1.0)
        self.assertAlmostEqual(metrics["early_detection"]["early_f1"], 1.0)

    def test_inconsistent_saved_alert_columns_are_rejected(self):
        frame = pd.DataFrame(
            {
                "label": [0, 1],
                "probability": [0.1, 0.9],
                "final_prediction": [0, 1],
                "alert_prediction": [0, 0],
                "alert_position": [np.nan, 2.0],
            }
        )
        with self.assertRaises(AssertionError):
            metrics_from_final_predictions(frame)


if __name__ == "__main__":
    unittest.main()
