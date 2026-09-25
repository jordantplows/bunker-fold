# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-09-25

### Added
- Initial release of Bunker API
- OpenAI-compatible REST API for biological foundation models
- `/v1/embeddings` endpoint for protein sequence embeddings
- `/v1/completions` endpoint for protein structure prediction
- `/v1/models` endpoint for listing available models
- ESM-2 model family support (8M to 15B parameters)
- ESMFold structure prediction support
- Boltz-1 structure prediction support (experimental)
- ProteinMPNN sequence design support (coming soon)
- FastAPI-based server with OpenAPI/Swagger documentation
- Automatic model caching and lazy loading
- Device auto-detection (CUDA, MPS, CPU)
- Health check and monitoring endpoints
- CORS support for web clients
- Docker support with GPU acceleration
- GitHub Actions CI/CD pipeline
- Comprehensive test suite

### Technical Details
- Python 3.10+ support
- Modular architecture with pluggable model backends
- Environment-based configuration
- Type hints throughout
- OpenAPI 3.0 compliant schemas

### Model Support Matrix

| Model | Task | Extra | GPU Memory | Status |
|-------|------|-------|------------|--------|
| esm2_t6_8M | Embedding | esm | 0.5 GB | ✅ Ready |
| esm2_t12_35M | Embedding | esm | 1 GB | ✅ Ready |
| esm2_t30_150M | Embedding | esm | 2 GB | ✅ Ready |
| esm2_t33_650M | Embedding | esm | 4 GB | ✅ Ready |
| esm2_t36_3B | Embedding | esm | 12 GB | ✅ Ready |
| esm2_t48_15B | Embedding | esm | 60 GB | ✅ Ready |
| esmfold | Structure | esm | 16 GB | ✅ Ready |
| boltz_1 | Structure | boltz | 20 GB | ⚠️ Experimental |
| proteinmpnn | Design | proteinmpnn | 4 GB | 🚧 Coming Soon |

[0.1.0]: https://github.com/yourusername/bunker-fold/releases/tag/v0.1.0
