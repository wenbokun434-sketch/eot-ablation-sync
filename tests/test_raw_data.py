import unittest

from eot.raw_data import RawEvent, build_event_text, parse_created_utc


class RawDataTests(unittest.TestCase):
    def test_submission_cleaning_keeps_title_when_body_removed(self):
        event = RawEvent(
            "subject_x.json",
            "subject_x",
            "submission",
            {"title": " A title ", "body": "[removed]"},
        )
        self.assertEqual(build_event_text(event, {"[removed]", "[deleted]"}), "A title")

    def test_comment_normalization_and_timestamp(self):
        event = RawEvent(
            "subject_x.json",
            "subject_x",
            "comment",
            {"body": "hello u/person https://example.com"},
        )
        self.assertEqual(
            build_event_text(event, {"[removed]", "[deleted]"}), "hello <USER> <URL>"
        )
        parsed = parse_created_utc("2024-09-06 16:36:22 UTC")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.utcoffset().total_seconds(), 0)


if __name__ == "__main__":
    unittest.main()
