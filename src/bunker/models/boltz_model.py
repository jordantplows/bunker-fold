"""Boltz-1 structure prediction through its supported command-line interface."""

import json
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from bunker.registry import register_model


class BoltzWrapper:
    """Run Boltz-1 predictions in isolated temporary directories."""

    def __init__(self, executable: str, cache_dir: Path, device: str):
        self.executable = executable
        self.cache_dir = cache_dir
        self.device = device

    def predict(
        self, sequences: str | list[str], temperature: float = 1.0
    ) -> list[dict[str, Any]]:
        """Predict structures with Boltz-1's single-sequence mode."""
        if isinstance(sequences, str):
            sequences = [sequences]
        accelerator = "gpu" if self.device == "cuda" else "cpu"
        if self.device not in {"cuda", "cpu"}:
            raise ValueError(f"Boltz does not support device '{self.device}'")

        results = []
        for sequence in sequences:
            with TemporaryDirectory(prefix="bunker-boltz-") as temporary:
                directory = Path(temporary)
                input_file = directory / "input.yaml"
                # JSON is valid YAML. An explicit empty MSA avoids an implicit
                # external sequence-search request from the Boltz CLI.
                input_file.write_text(
                    json.dumps(
                        {
                            "version": 1,
                            "sequences": [
                                {
                                    "protein": {
                                        "id": "A",
                                        "sequence": sequence,
                                        "msa": "empty",
                                    }
                                }
                            ],
                        }
                    )
                )
                command = [
                    self.executable,
                    "predict",
                    str(input_file),
                    "--out_dir",
                    str(directory),
                    "--cache",
                    str(self.cache_dir),
                    "--model",
                    "boltz1",
                    "--output_format",
                    "pdb",
                    "--accelerator",
                    accelerator,
                    "--num_workers",
                    "0",
                    "--no_kernels",
                ]
                try:
                    subprocess.run(
                        command,
                        check=True,
                        timeout=600,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                except subprocess.CalledProcessError as error:
                    raise RuntimeError(
                        f"Boltz prediction failed with exit code {error.returncode}"
                    ) from error
                except subprocess.TimeoutExpired as error:
                    raise RuntimeError(
                        "Boltz prediction exceeded 10 minutes"
                    ) from error

                pdb_file = (
                    directory
                    / "boltz_results_input"
                    / "predictions"
                    / "input"
                    / "input_model_0.pdb"
                )
                if not pdb_file.is_file():
                    raise RuntimeError("Boltz did not produce a PDB structure")
                pdb = pdb_file.read_text()
                plddt = [
                    float(line[60:66])
                    for line in pdb.splitlines()
                    if line.startswith("ATOM") and line[12:16].strip() == "CA"
                ]
                results.append(
                    {
                        "pdb": pdb,
                        "plddt": plddt or None,
                        "mean_plddt": sum(plddt) / len(plddt) if plddt else None,
                    }
                )
        return results


def load_boltz(
    name: str,
    device: str,
    dtype: str | None,
    cache_dir: Any,
    **kwargs: Any,
) -> BoltzWrapper:
    """Find the CLI supplied by the Boltz package."""
    executable = Path(sys.executable).with_name("boltz")
    if not executable.is_file():
        found = shutil.which("boltz")
        if not found:
            raise ImportError("Install the 'boltz' extra to use Boltz-1")
        executable = Path(found)
    return BoltzWrapper(str(executable), Path(cache_dir), device)


register_model(
    name="boltz_1",
    description="Boltz-1 unified biomolecular structure prediction",
    task="structure",
    extra="boltz",
    loader=load_boltz,
    weights_url="boltz-1",
    license="MIT",
    paper_url="https://arxiv.org/abs/2408.14820",
    memory_gb=20.0,
)
