# Bunker API

**OpenAI-compatible API for biological foundation models**

Bunker provides a unified REST API for running protein embedding, structure prediction, and sequence design models with OpenAI-compatible endpoints. Deploy locally or in the cloud.

[![Tests](https://github.com/yourusername/bunker-fold/workflows/Tests/badge.svg)](https://github.com/yourusername/bunker-fold/actions)
[![PyPI](https://img.shields.io/pypi/v/bunker-fold)](https://pypi.org/project/bunker-fold/)
[![Python](https://img.shields.io/pypi/pyversions/bunker-fold)](https://pypi.org/project/bunker-fold/)
[![License](https://img.shields.io/github/license/yourusername/bunker-fold)](LICENSE)

## Features

- 🧬 **Multiple Models**: ESM-2, ESMFold, Boltz-1, ProteinMPNN
- 🔌 **OpenAI Compatible**: Drop-in replacement for OpenAI API
- 🚀 **Production Ready**: FastAPI + Docker + GPU support
- 📖 **OpenAPI Docs**: Interactive API documentation
- 🎯 **Type Safe**: Full type hints and Pydantic validation
- 💾 **Auto Caching**: Model weights cached automatically

## Quick Start

### Installation

```bash
# Core API (no models)
pip install bunker-fold

# With specific models
pip install "bunker-fold[esm]"      # ESM-2 + ESMFold
pip install "bunker-fold[boltz]"    # Boltz-1
pip install "bunker-fold[all]"      # All models
```

### Run Server

```bash
# Start the API server
bunker serve --host 0.0.0.0 --port 8000

# Or with Docker
docker-compose up
```

The API will be available at `http://localhost:8000` with interactive docs at `/docs`.

## API Usage

### Python Client

```python
from openai import OpenAI

# Point to your Bunker instance
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # Bunker doesn't require auth by default
)

# Generate protein embeddings
response = client.embeddings.create(
    model="esm2_t33_650M",
    input="MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK"
)
embedding = response.data[0].embedding

# Predict protein structure
response = client.completions.create(
    model="esmfold",
    prompt="MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK"
)
pdb_string = response.data[0].pdb
```

### cURL

```bash
# List available models
curl http://localhost:8000/v1/models

# Generate embeddings
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "esm2_t33_650M",
    "input": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK"
  }'

# Predict structure
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "esmfold",
    "prompt": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK"
  }'
```

## Supported Models

| Model | Task | Extra | GPU Memory | Status |
|-------|------|-------|------------|--------|
| `esm2_t6_8M` | Embedding | `esm` | 0.5 GB | ✅ |
| `esm2_t12_35M` | Embedding | `esm` | 1 GB | ✅ |
| `esm2_t30_150M` | Embedding | `esm` | 2 GB | ✅ |
| `esm2_t33_650M` | Embedding | `esm` | 4 GB | ✅ |
| `esm2_t36_3B` | Embedding | `esm` | 12 GB | ✅ |
| `esm2_t48_15B` | Embedding | `esm` | 60 GB | ✅ |
| `esmfold` | Structure | `esm` | 16 GB | ✅ |
| `boltz_1` | Structure | `boltz` | 20 GB | ⚠️ Experimental |
| `proteinmpnn` | Design | `proteinmpnn` | 4 GB | 🚧 Coming Soon |

## API Endpoints

### `POST /v1/embeddings`

Generate embeddings for protein sequences.

**Request:**
```json
{
  "model": "esm2_t33_650M",
  "input": "MKTAYIAKQRQISFVK",
  "encoding_format": "float"
}
```

**Response:**
```json
{
  "object": "list",
  "data": [{
    "object": "embedding",
    "embedding": [0.1, 0.2, ...],
    "index": 0
  }],
  "model": "esm2_t33_650M",
  "usage": {
    "prompt_tokens": 16,
    "completion_tokens": 0,
    "total_tokens": 16
  }
}
```

### `POST /v1/completions`

Predict protein structures from sequences.

**Request:**
```json
{
  "model": "esmfold",
  "prompt": "MKTAYIAKQRQISFVK",
  "temperature": 1.0
}
```

**Response:**
```json
{
  "object": "structure",
  "model": "esmfold",
  "data": [{
    "pdb": "ATOM   1  N   MET A   1...",
    "plddt": [85.2, 87.1, ...],
    "mean_plddt": 86.5
  }],
  "usage": {
    "prompt_tokens": 16,
    "completion_tokens": 16,
    "total_tokens": 32
  }
}
```

### `GET /v1/models`

List all available models.

**Response:**
```json
{
  "object": "list",
  "data": [{
    "id": "esm2_t33_650M",
    "object": "model",
    "created": 1632492800,
    "owned_by": "bunker"
  }]
}
```

## Deployment

### Docker

```bash
# Build image
docker build -t bunker-api .

# Run with GPU
docker run --gpus all -p 8000:8000 bunker-api

# Or use docker-compose
docker-compose up -d
```

### Cloud Deployment

The Bunker API can be deployed to any cloud provider that supports Docker and GPUs:

- **AWS**: ECS with GPU instances
- **Google Cloud**: Cloud Run with GPU
- **Azure**: Container Instances with GPU
- **Modal**: Serverless GPU inference

Example Modal deployment:

```python
import modal

app = modal.App("bunker-api")

@app.function(
    image=modal.Image.from_dockerfile("Dockerfile"),
    gpu="A10G",
    container_idle_timeout=300,
)
@modal.asgi_app()
def web():
    from bunker.api.app import app
    return app
```

## Configuration

### Environment Variables

- `BUNKER_CACHE`: Cache directory for model weights (default: `~/.cache/bunker`)
- `HOST`: Server host (default: `0.0.0.0`)
- `PORT`: Server port (default: `8000`)

### CLI Commands

```bash
# Start server
bunker serve --host 0.0.0.0 --port 8000 --workers 4

# List available models
bunker models

# Development mode (auto-reload)
bunker serve --reload
```

## Development

### Setup

```bash
git clone https://github.com/yourusername/bunker-fold.git
cd bunker-fold
pip install -e ".[dev]"
```

### Run Tests

```bash
# Unit tests only
pytest -v -m "not slow"

# All tests (requires model dependencies)
pytest -v

# With coverage
pytest --cov=bunker --cov-report=html
```

### Linting

```bash
ruff check src/ tests/
black src/ tests/
mypy src/
```

## Model Licenses

This software integrates with third-party models. Users must comply with each model's license:

- **ESM-2 & ESMFold**: MIT License ([Meta AI](https://github.com/facebookresearch/esm))
- **Boltz-1**: MIT License
- **ProteinMPNN**: MIT License

See [LICENSE](LICENSE) for details.

## Citation

If you use Bunker in your research, please cite the underlying models:

**ESM-2:**
```bibtex
@article{lin2022language,
  title={Language models of protein sequences at the scale of evolution enable accurate structure prediction},
  author={Lin, Zeming and Akin, Halil and others},
  journal={bioRxiv},
  year={2022}
}
```

**ESMFold:**
```bibtex
@article{lin2023evolutionary,
  title={Evolutionary-scale prediction of atomic-level protein structure with a language model},
  author={Lin, Zeming and Akin, Halil and others},
  journal={Science},
  year={2023}
}
```

## Contributing

Contributions welcome! Please open an issue or PR.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing`)
5. Open a Pull Request

## License

MIT License - see [LICENSE](LICENSE) for details.

## Support

- 📖 [Documentation](https://github.com/yourusername/bunker-fold#readme)
- 🐛 [Issue Tracker](https://github.com/yourusername/bunker-fold/issues)
- 💬 [Discussions](https://github.com/yourusername/bunker-fold/discussions)
