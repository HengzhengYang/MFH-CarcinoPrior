"""Small model smoke tests, not a training or performance evaluation.

python -B -m unittest discover -s tests -v
MFH_REFERENCE_TEST=1 additionally checks original CUDA helpers in their recorded environment.
"""
import csv
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import predict


class PredictionTests(unittest.TestCase):
    def run_cli(self, output, *arguments):
        return subprocess.run([sys.executable, "-B", str(ROOT / "predict.py"),
                               *arguments, "--output", str(output)],
                              cwd=output.parent, text=True, capture_output=True, timeout=180)

    def test_parse_preserves_rows(self):
        rows = predict.prepare_rows([("same", "OCC"), ("same", ""), ("x", "invalid")])
        self.assertEqual([r["row_id"] for r in rows], ["same", "same", "x"])
        self.assertEqual(rows[0]["canonical_smiles"], "CCO")
        self.assertEqual([r["status"] for r in rows], ["pending", "invalid_smiles", "invalid_smiles"])

    def test_csv_vs_direct_cpu_and_ensemble(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self.run_cli(root / "first.csv", "--input", str(ROOT / "examples/smiles.csv"),
                                 "--id-column", "id", "--device", "cpu")
            self.assertEqual(first.returncode, 0, first.stderr)
            second = self.run_cli(root / "second.csv", "--smiles", "CCO", "c1ccccc1",
                                  "CC(=O)Oc1ccccc1C(=O)O", "--device", "cpu", "--batch-size", "1")
            self.assertEqual(second.returncode, 0, second.stderr)
            with (root / "first.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            with (root / "second.csv").open() as stream:
                others = list(csv.DictReader(stream))
            self.assertEqual([r["row_id"] for r in rows], ["ethanol", "benzene", "aspirin"])
            for row, other in zip(rows, others):
                p = [float(row["p_" + name]) for name in predict.MEMBERS]
                self.assertTrue(all(0 <= value <= 1 for value in p))
                self.assertEqual(float(row["p_E005"]), float(np.mean(p, dtype=np.float64)))
                self.assertEqual(int(row["signal_positive"]), int(float(row["p_E005"]) >= predict.THRESHOLD))
                for name in (*predict.MEMBERS, "E005"):
                    self.assertAlmostEqual(float(row["p_" + name]), float(other["p_" + name]), delta=1e-6)

    def test_mixed_invalid_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "mixed.csv"
            result = self.run_cli(output, "--smiles", "CCO", "invalid", "", "CCO", "--device", "cpu")
            self.assertEqual(result.returncode, 2, result.stderr)
            with output.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 4)
            self.assertEqual([r["status"] for r in rows], ["ok", "invalid_smiles", "invalid_smiles", "ok"])
            self.assertEqual(rows[1]["p_E005"], "")
            self.assertEqual(rows[0]["p_E005"], rows[3]["p_E005"])

    def test_cli_rejections(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "existing.csv"
            output.write_text("keep me", encoding="utf-8")
            self.assertEqual(self.run_cli(output, "--smiles", "CCO").returncode, 1)
            self.assertEqual(output.read_text(), "keep me")
            self.assertEqual(self.run_cli(root / "new.csv", "--smiles", "CCO", "--batch-size", "0").returncode, 1)
            self.assertEqual(self.run_cli(root / "new.csv", "--input", str(ROOT / "examples/smiles.csv"),
                                         "--smiles-column", "missing").returncode, 1)
            self.assertFalse((root / "new.csv").exists())

    @unittest.skipUnless(os.environ.get("MFH_REFERENCE_TEST") == "1", "Original CUDA environment required")
    def test_against_original_cuda_helpers(self):
        import joblib
        import numpy as np
        import pandas as pd
        import torch
        from rdkit import Chem, DataStructs
        from rdkit.Chem import MACCSkeys
        smiles = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"]
        actual, lengths = predict.predict_probabilities(smiles, "cuda", 32)
        with (ROOT / "examples/predictions_cuda.csv").open() as stream:
            saved_examples = list(csv.DictReader(stream))
        for name in (*predict.MEMBERS, "E005"):
            np.testing.assert_allclose(actual[name], [float(r["p_" + name]) for r in saved_examples],
                                       atol=1e-7, rtol=0)
        from graph_models import build_model, build_graph_records, predict_raw
        from lm_embeddings import extract_or_load_embeddings
        weights = ROOT / "weights/e005"
        frame = pd.DataFrame({"model_unit_id": ["a", "b", "c"], "cv_group_id": "example",
                              "final_label": -1, "canonical_isomeric_smiles": smiles})
        mpnn = build_model("MPNN", 256, 0.30)
        mpnn.load_state_dict(torch.load(weights / "MPNN.pt", weights_only=False)["state_dict"])
        raw_mpnn = predict_raw(mpnn, build_graph_records(frame), np.arange(3), 128)
        with tempfile.TemporaryDirectory() as directory:
            embedding, _ = extract_or_load_embeddings(
                frame, {"name": "MolFormer", "model_id": str(ROOT / "weights/molformer_backbone"),
                        "revision": "7b12d946c181a37f6012b9dc3b002275de070314",
                        "smoke_batch_size": 32, "single_oom_reduced_batch_size": 16},
                "synthetic-smoke", "synthetic-smoke", Path(directory), 256, 20260805)
        head_data = torch.load(weights / "MolFormer_CUDA_Logistic.pt", map_location="cuda:0", weights_only=False)
        head = torch.nn.Linear(embedding.shape[1], 1, device="cuda:0")
        head.load_state_dict(head_data["state_dict"])
        with torch.no_grad():
            raw_lm = head((torch.tensor(embedding, device="cuda:0") - head_data["mean"]) /
                          head_data["scale"]).squeeze(1).double().cpu().numpy()
        features = []
        for text in smiles:
            bits = np.zeros(167, dtype=np.int8)
            DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(text)), bits)
            features.append(bits.astype(np.float32))
        raw_rf = joblib.load(weights / "RandomForest_MACCS.joblib")["model"].predict_proba(features)[:, 1]
        reference = []
        for name, raw in zip(predict.MEMBERS, (raw_rf, raw_mpnn, raw_lm)):
            saved = torch.load(weights / (name + "_platt.pt"), map_location="cuda:0", weights_only=False)
            layer = torch.nn.Linear(1, 1, device="cuda:0")
            layer.load_state_dict(saved["state_dict"])
            with torch.no_grad():
                values = torch.sigmoid(layer(torch.tensor(np.asarray(raw, dtype=np.float32).reshape(-1, 1),
                                                          device="cuda:0"))).flatten().double().cpu().numpy()
            np.testing.assert_allclose(actual[name], values, atol=1e-7, rtol=0)
            reference.append(values)
        np.testing.assert_allclose(actual["E005"], np.mean(reference, axis=0), atol=1e-7, rtol=0)
        cpu, _ = predict.predict_probabilities(smiles, "cpu", 32)
        for name in (*predict.MEMBERS, "E005"):
            np.testing.assert_allclose(actual[name], cpu[name], atol=1e-6, rtol=0)


if __name__ == "__main__":
    unittest.main()
