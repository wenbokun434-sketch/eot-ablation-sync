import math
import unittest

import numpy as np

from eot.metrics import (
    RiskSequence,
    alert_positions,
    bootstrap_metric_intervals,
    classification_metrics,
    early_detection_metrics,
    erde_latency_cost,
    first_alert_position,
    latency_penalty,
    select_threshold,
)


class MetricTests(unittest.TestCase):
    def test_consecutive_alert_position_is_one_based(self):
        probabilities = np.array([0.8, 0.2, 0.7, 0.9, 0.95])
        self.assertEqual(first_alert_position(probabilities, 0.7, consecutive=2), 4)
        self.assertIsNone(first_alert_position([0.9], 0.7, consecutive=2))
        self.assertEqual(first_alert_position([0.9], 0.7, consecutive=1), 1)

    def test_official_latency_functions(self):
        self.assertAlmostEqual(erde_latency_cost(5, 5), 0.5)
        self.assertAlmostEqual(erde_latency_cost(50, 50), 0.5)
        self.assertAlmostEqual(latency_penalty(1), 0.0)
        self.assertGreater(latency_penalty(100), latency_penalty(10))

    def test_speed_uses_median_of_individual_penalties(self):
        labels = np.array([1, 1], dtype=np.int8)
        positions = np.array([1.0, 101.0])
        metrics = early_detection_metrics(labels, positions)
        expected = 1.0 - np.median([latency_penalty(1), latency_penalty(101)])
        self.assertAlmostEqual(metrics["median_latency"], 51.0)
        self.assertAlmostEqual(metrics["speed"], expected)

    def test_erde_and_f_latency_known_example(self):
        labels = np.array([1, 1, 0, 0], dtype=np.int8)
        positions = np.array([1.0, np.nan, 1.0, np.nan])
        metrics = early_detection_metrics(labels, positions)
        expected_erde_5 = (erde_latency_cost(1, 5) + 1.0 + 0.5 + 0.0) / 4.0
        self.assertAlmostEqual(metrics["erde_5"], expected_erde_5)
        self.assertEqual(metrics["median_latency"], 1.0)
        self.assertAlmostEqual(metrics["speed"], 1.0)
        self.assertAlmostEqual(metrics["early_f1"], 0.5)
        self.assertAlmostEqual(metrics["f_latency"], 0.5)

    def test_threshold_is_selected_only_from_supplied_sequences(self):
        sequences = [
            RiskSequence("positive", 1, np.array([0.2, 0.8, 0.9])),
            RiskSequence("negative", 0, np.array([0.1, 0.2, 0.3])),
        ]
        selected = select_threshold(sequences, consecutive=2, candidates=[0.25, 0.85])
        self.assertEqual(selected["threshold"], 0.25)
        positions = alert_positions(sequences, selected["threshold"], consecutive=2)
        np.testing.assert_equal(positions, np.array([3.0, np.nan]))

    def test_classification_and_bootstrap_are_deterministic(self):
        labels = np.array([0, 0, 1, 1], dtype=np.int8)
        probabilities = np.array([0.1, 0.2, 0.8, 0.9])
        predictions = np.array([0, 0, 1, 1], dtype=np.int8)
        metrics = classification_metrics(labels, probabilities, predictions=predictions)
        self.assertAlmostEqual(metrics["f1"], 1.0)
        positions = np.array([np.nan, np.nan, 2.0, 3.0])
        intervals_a, records_a = bootstrap_metric_intervals(
            labels,
            probabilities,
            predictions,
            positions,
            iterations=10,
            level=0.8,
            seed=7,
        )
        intervals_b, records_b = bootstrap_metric_intervals(
            labels,
            probabilities,
            predictions,
            positions,
            iterations=10,
            level=0.8,
            seed=7,
        )
        self.assertEqual(len(records_a), len(records_b))
        for record_a, record_b in zip(records_a, records_b, strict=True):
            self.assertEqual(record_a.keys(), record_b.keys())
            for name in record_a:
                if isinstance(record_a[name], float) and math.isnan(record_a[name]):
                    self.assertTrue(math.isnan(record_b[name]))
                else:
                    self.assertEqual(record_a[name], record_b[name])
        self.assertEqual(intervals_a, intervals_b)
        self.assertTrue(math.isfinite(intervals_a["f_latency"]["lower"]))


if __name__ == "__main__":
    unittest.main()
