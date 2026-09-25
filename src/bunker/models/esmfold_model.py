"""ESMFold structure prediction model integration."""

from typing import Any, Dict, List, Optional, Union
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
        self, sequences: Union[str, List[str]], temperature: float = 1.0
    ) -> List[Dict[str, Any]]:
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

            # Extract PDB string
            pdb_string = self.model.output_to_pdb(output)[0]

            # Extract pLDDT scores
            plddt = output["plddt"][0].cpu().numpy()
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
    dtype: Optional[str],
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
    import torch
    import esm

    # Load ESMFold model
    model = esm.pretrained.esmfold_v1()
    model = model.to(device)
    model.eval()

    # Set trunk to inference mode
    model.set_chunk_size(128)

    return ESMFoldWrapper(model, device)


# Register ESMFold
register_model(
    name="esmfold",
    description="ESMFold structure prediction model",
    task="structure",
    extra="esm",
    loader=load_esmfold,
    weights_url="facebook/esmfold_v1",
    license="MIT",
    paper_url="https://www.science.org/doi/10.1126/science.ade2574",
    memory_gb=16.0,
)
