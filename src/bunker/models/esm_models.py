"""ESM2 embedding model integrations."""

from typing import Any, List, Optional, Union
import numpy as np

from bunker.registry import register_model


class ESM2Wrapper:
    """Wrapper for ESM2 models providing consistent interface."""

    def __init__(self, model: Any, alphabet: Any, device: str):
        """Initialize ESM2 wrapper.

        Args:
            model: ESM2 model instance
            alphabet: ESM2 alphabet
            device: Device string
        """
        self.model = model
        self.alphabet = alphabet
        self.device = device
        self.batch_converter = alphabet.get_batch_converter()

    def predict(self, sequences: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for protein sequences.

        Args:
            sequences: Single sequence or list of sequences

        Returns:
            Numpy array of embeddings (shape: [batch_size, embedding_dim])
        """
        import torch

        # Normalize to list
        if isinstance(sequences, str):
            sequences = [sequences]
            single_input = True
        else:
            single_input = False

        # Prepare batch
        data = [
            (f"seq_{i}", seq) for i, seq in enumerate(sequences)
        ]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)

        # Generate embeddings
        with torch.no_grad():
            results = self.model(
                batch_tokens, repr_layers=[self.model.num_layers], return_contacts=False
            )

        # Extract per-sequence embeddings (mean over sequence length)
        token_representations = results["representations"][self.model.num_layers]

        # Average over sequence length (excluding BOS/EOS tokens)
        embeddings = []
        for i, (_, seq) in enumerate(data):
            seq_len = len(seq)
            # tokens: [BOS, seq..., EOS, padding]
            # We take mean over seq tokens (indices 1:seq_len+1)
            seq_emb = token_representations[i, 1 : seq_len + 1].mean(0)
            embeddings.append(seq_emb.cpu().numpy())

        embeddings_array = np.array(embeddings)

        if single_input:
            return embeddings_array[0]
        return embeddings_array


def load_esm2(
    name: str,
    device: str,
    dtype: Optional[str],
    cache_dir: Any,
    **kwargs: Any,
) -> ESM2Wrapper:
    """Load an ESM2 model.

    Args:
        name: Model name
        device: Device to load on
        dtype: Data type (unused for ESM2)
        cache_dir: Cache directory
        **kwargs: Additional arguments

    Returns:
        ESM2Wrapper instance
    """
    import torch
    import esm

    # Map model names to ESM2 model names
    model_map = {
        "esm2_t6_8M": "esm2_t6_8M_UR50D",
        "esm2_t12_35M": "esm2_t12_35M_UR50D",
        "esm2_t30_150M": "esm2_t30_150M_UR50D",
        "esm2_t33_650M": "esm2_t33_650M_UR50D",
        "esm2_t36_3B": "esm2_t36_3B_UR50D",
        "esm2_t48_15B": "esm2_t48_15B_UR50D",
    }

    esm_name = model_map.get(name, name)

    # Load model
    model, alphabet = esm.pretrained.load_model_and_alphabet(esm_name)
    model = model.to(device)
    model.eval()

    return ESM2Wrapper(model, alphabet, device)


# Register ESM2 models
register_model(
    name="esm2_t6_8M",
    description="ESM-2 8M parameter model",
    task="embedding",
    extra="esm",
    loader=load_esm2,
    weights_url="facebook/esm2_t6_8M_UR50D",
    license="MIT",
    paper_url="https://www.biorxiv.org/content/10.1101/2022.07.20.500902v1",
    memory_gb=0.5,
)

register_model(
    name="esm2_t12_35M",
    description="ESM-2 35M parameter model",
    task="embedding",
    extra="esm",
    loader=load_esm2,
    weights_url="facebook/esm2_t12_35M_UR50D",
    license="MIT",
    paper_url="https://www.biorxiv.org/content/10.1101/2022.07.20.500902v1",
    memory_gb=1.0,
)

register_model(
    name="esm2_t30_150M",
    description="ESM-2 150M parameter model",
    task="embedding",
    extra="esm",
    loader=load_esm2,
    weights_url="facebook/esm2_t30_150M_UR50D",
    license="MIT",
    paper_url="https://www.biorxiv.org/content/10.1101/2022.07.20.500902v1",
    memory_gb=2.0,
)

register_model(
    name="esm2_t33_650M",
    description="ESM-2 650M parameter model",
    task="embedding",
    extra="esm",
    loader=load_esm2,
    weights_url="facebook/esm2_t33_650M_UR50D",
    license="MIT",
    paper_url="https://www.biorxiv.org/content/10.1101/2022.07.20.500902v1",
    memory_gb=4.0,
)

register_model(
    name="esm2_t36_3B",
    description="ESM-2 3B parameter model",
    task="embedding",
    extra="esm",
    loader=load_esm2,
    weights_url="facebook/esm2_t36_3B_UR50D",
    license="MIT",
    paper_url="https://www.biorxiv.org/content/10.1101/2022.07.20.500902v1",
    memory_gb=12.0,
)

register_model(
    name="esm2_t48_15B",
    description="ESM-2 15B parameter model",
    task="embedding",
    extra="esm",
    loader=load_esm2,
    weights_url="facebook/esm2_t48_15B_UR50D",
    license="MIT",
    paper_url="https://www.biorxiv.org/content/10.1101/2022.07.20.500902v1",
    memory_gb=60.0,
)
