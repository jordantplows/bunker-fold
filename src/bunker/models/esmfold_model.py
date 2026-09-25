"""ESMFold structure prediction model integration."""

from typing import Any

import numpy as np

from bunker.registry import register_model


class ESMFoldWrapper:
    """Wrapper for ESMFold providing consistent interface."""

    def __init__(self, model: Any, device: str):
        """Initialize ESMFold wrapper.

        Args:
            model: ESMFold model instance
            device: Device string
        """
        self.model = model
        self.device = device

    def predict(
        self, sequences: str | list[str], temperature: float = 1.0
    ) -> list[dict[str, Any]]:
        """Predict protein structures from sequences.

        Args:
            sequences: Single sequence or list of sequences
            temperature: Sampling temperature (unused for ESMFold)

        Returns:
            List of structure dictionaries with PDB and confidence scores
        """
        import torch

        # Normalize to list
        if isinstance(sequences, str):
            sequences = [sequences]

        results = []
        for seq in sequences:
            with torch.no_grad():
                output = self.model.infer(seq)

            # Transformers returns confidence in [0, 1]; expose conventional
            # pLDDT scores in [0, 100], including PDB B factors.
            output["plddt"] = output["plddt"] * 100
            from transformers.models.esm.openfold_utils import (
                OFProtein,
                atom14_to_atom37,
                to_pdb,
            )

            positions = atom14_to_atom37(output["positions"][-1], output)[0]
            protein = OFProtein(
                aatype=output["aatype"][0].cpu().numpy(),
                atom_positions=positions.cpu().numpy(),
                atom_mask=output["atom37_atom_exists"][0].cpu().numpy(),
                residue_index=output["residue_index"][0].cpu().numpy() + 1,
                b_factors=output["plddt"][0].cpu().numpy(),
            )
            pdb_string = to_pdb(protein)

            # Atom index 1 is CA. The API returns one confidence per residue.
            plddt = output["plddt"][0, :, 1].cpu().numpy()
            mean_plddt = float(np.mean(plddt))

            results.append(
                {
                    "pdb": pdb_string,
                    "plddt": plddt.tolist(),
                    "mean_plddt": mean_plddt,
                }
            )

        return results


def load_esmfold(
    name: str,
    device: str,
    dtype: str | None,
    cache_dir: Any,
    **kwargs: Any,
) -> ESMFoldWrapper:
    """Load ESMFold model.

    Args:
        name: Model name
        device: Device to load on
        dtype: Data type
        cache_dir: Cache directory
        **kwargs: Additional arguments

    Returns:
        ESMFoldWrapper instance
    """
    from transformers import EsmForProteinFolding

    model = EsmForProteinFolding.from_pretrained(
        "facebook/esmfold_v1", cache_dir=str(cache_dir)
    )
    model = model.to(device)
    model.eval()

    model.trunk.set_chunk_size(128)

    return ESMFoldWrapper(model, device)


# Register ESMFold
register_model(
    name="esmfold",
    description="ESMFold structure prediction model",
    task="structure",
    extra="esmfold",
    loader=load_esmfold,
    weights_url="facebook/esmfold_v1",
    license="MIT",
    paper_url="https://www.science.org/doi/10.1126/science.ade2574",
    memory_gb=16.0,
)
