# QuantaSlice

QuantaSlice is a hackathon project for 5G Network Slice Allocation. It combines AI-based emergency detection with quantum optimization (QUBO + QAOA) to dynamically allocate radio resources under changing network conditions.

## Project Structure

```text
quantaslice/
├── src/
│   └── quantaslice/
│       ├── ai/
│       ├── quantum/
│       ├── orchestrator/
│       ├── pipeline/
│       ├── simulation/
│       └── core/
├── tests/
├── configs/
├── examples/
└── docs/
```

## Architecture

The project follows a layered modular architecture.

- **Core** – shared interfaces, types, configuration, registry
- **AI** – preprocessing, feature engineering, prediction providers
- **Quantum** – QUBO construction, Hamiltonian generation, optimization solvers
- **Pipeline** – application layer coordinating the workflow
- **Orchestrator** – resource allocation execution
- **Simulation** – telemetry and dataset streaming

The AI and Quantum modules are completely decoupled and communicate only through shared interfaces defined in `quantaslice.core`.

## Installation

```bash
git clone https://github.com/dolong226/quantaslice
cd quantaslice

python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate

pip install -e ".[ai,quantum,dev]"
```

## Run tests:

```bash
pytest
```

## Running

The project is currently under active development. Example scripts and CLI commands will be added in future releases.

## Documentation

Detailed design documents will be added to the `docs/` directory.

- Architecture
- Module design
- Interfaces
- Development roadmap

## License

This project is released under the MIT License.