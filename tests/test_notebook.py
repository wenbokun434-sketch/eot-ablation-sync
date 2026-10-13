import ast
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from eot.config import load_config
from eot.notebook_document import make_notebook
from eot.notebook_support import notebook_config, require_files
from eot.reproducibility import write_json


class NotebookTests(unittest.TestCase):
    def test_generated_code_compiles_and_training_defaults_off(self):
        nb = make_notebook()
        code = [cell.source for cell in nb.cells if cell.cell_type == "code"]
        for source in code:
            tree = ast.parse(source)
            self.assertFalse(any(isinstance(node, (ast.FunctionDef, ast.ClassDef)) for node in tree.body))
        assignment = ast.parse(code[0]).body[0]
        self.assertEqual(assignment.targets[0].id, "RUN_TRAINING")
        self.assertIs(assignment.value.value, False)
        self.assertEqual(sum("display(Image(" in source for source in code), 10)

    def test_missing_artifact_errors_include_specific_recovery_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError) as caught:
                require_files(Path(directory), {
                    "embedding.npy": "python scripts/extract_embeddings.py --device cuda",
                    "best.pt": "python scripts/train.py --fold 0 --seed 42 --device cuda",
                    "oof.parquet": "python scripts/evaluate.py --device cuda",
                })
            for fragment in ("embedding.npy", "extract_embeddings.py", "--fold 0", "evaluate.py"):
                self.assertIn(fragment, str(caught.exception))

    def test_result_mode_restores_hash_without_raw_directory_or_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "configs").mkdir()
            source = root / "configs/eot_main.yaml"
            source.write_text('paths:\n  project_root: ..\n  raw_data_root: "${ERISK_TASK2_ROOT}"\n', encoding="utf-8")
            missing_raw = str(root / "not_mounted")
            with patch.dict(os.environ, {"ERISK_TASK2_ROOT": missing_raw}):
                original = load_config(source)
            write_json({"dataset": {"root": missing_raw}, "config_hash": original.config_hash},
                       root / "outputs/reports/raw_data_audit.json")
            with patch.dict(os.environ, {}, clear=True):
                restored = notebook_config(root)
                self.assertEqual(restored.config_hash, original.config_hash)
                self.assertNotIn("ERISK_TASK2_ROOT", os.environ)
                self.assertFalse(restored.path("raw_data_root").exists())
                with self.assertRaisesRegex(ValueError, "ERISK_TASK2_ROOT"):
                    notebook_config(root, run_training=True)


if __name__ == "__main__":
    unittest.main()
