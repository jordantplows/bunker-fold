"""Contract tests for structure model adapters without large weights."""

import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from bunker.models.boltz_model import BoltzWrapper
from bunker.models.esmfold_model import ESMFoldWrapper, load_esmfold


class FakeTensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def __mul__(self, factor):
        return FakeTensor(self.values * factor)

    def __getitem__(self, index):
        return FakeTensor(self.values[index])

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class FakeFoldModel:
    def __init__(self):
        self.trunk = SimpleNamespace(set_chunk_size=lambda size: None)

    def to(self, device):
        return self

    def eval(self):
        return self

    def infer(self, sequence):
        length = len(sequence)
        return {
            "plddt": FakeTensor(np.full((1, length, 37), 0.8)),
            "positions": FakeTensor(np.zeros((1, 1, length, 14, 3))),
            "aatype": FakeTensor(np.zeros((1, length))),
            "atom37_atom_exists": FakeTensor(np.ones((1, length, 37))),
            "residue_index": FakeTensor(np.arange(length)[None, :]),
        }


def test_esmfold_residue_confidence_and_loader(tmp_path):
    model = FakeFoldModel()
    transformers = ModuleType("transformers")

    class FakeEsmForProteinFolding:
        @classmethod
        def from_pretrained(cls, model_id, cache_dir):
            assert model_id == "facebook/esmfold_v1"
            assert cache_dir == str(tmp_path)
            return model

    transformers.EsmForProteinFolding = FakeEsmForProteinFolding
    transformers.__path__ = []
    models_package = ModuleType("transformers.models")
    models_package.__path__ = []
    esm_package = ModuleType("transformers.models.esm")
    esm_package.__path__ = []
    openfold_utils = ModuleType("transformers.models.esm.openfold_utils")
    openfold_utils.atom14_to_atom37 = lambda positions, output: FakeTensor(
        np.zeros((1, 3, 37, 3))
    )
    openfold_utils.OFProtein = lambda **kwargs: SimpleNamespace(**kwargs)

    def to_pdb(protein):
        assert protein.b_factors.max() == pytest.approx(80)
        return "MODEL\n"

    openfold_utils.to_pdb = to_pdb
    torch = ModuleType("torch")
    torch.no_grad = nullcontext
    with patch.dict(
        sys.modules,
        {
            "transformers": transformers,
            "transformers.models": models_package,
            "transformers.models.esm": esm_package,
            "transformers.models.esm.openfold_utils": openfold_utils,
            "torch": torch,
        },
    ):
        wrapper = load_esmfold("esmfold", "cpu", None, tmp_path)
        result = wrapper.predict("MKT")
    assert isinstance(wrapper, ESMFoldWrapper)
    assert result[0]["pdb"] == "MODEL\n"
    assert result[0]["plddt"] == pytest.approx([80, 80, 80])
    assert result[0]["mean_plddt"] == pytest.approx(80)


def test_boltz_cli_adapter_uses_supported_command_and_reads_pdb(tmp_path):
    wrapper = BoltzWrapper("/fake/bin/boltz", tmp_path / "cache", "cpu")
    pdb_line = (
        "ATOM      1  CA  ALA A   1      11.104  13.207   8.678"
        "  1.00 85.50           C\n"
    )

    def fake_run(command, **kwargs):
        assert command[:2] == ["/fake/bin/boltz", "predict"]
        assert command[command.index("--model") + 1] == "boltz1"
        assert command[command.index("--accelerator") + 1] == "cpu"
        assert "--no_kernels" in command
        input_file = Path(command[2])
        data = json.loads(input_file.read_text())
        assert data["sequences"][0]["protein"] == {
            "id": "A",
            "sequence": "MKT",
            "msa": "empty",
        }
        result_file = (
            Path(command[command.index("--out_dir") + 1])
            / "boltz_results_input"
            / "predictions"
            / "input"
            / "input_model_0.pdb"
        )
        result_file.parent.mkdir(parents=True)
        result_file.write_text(pdb_line)

    with patch("bunker.models.boltz_model.subprocess.run", side_effect=fake_run):
        result = wrapper.predict("MKT")
    assert result == [{"pdb": pdb_line, "plddt": [85.5], "mean_plddt": 85.5}]
