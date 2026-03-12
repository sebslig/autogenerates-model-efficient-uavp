# NAS-Toolkit

A minimal Neural Architecture Search (NAS) toolkit designed to auto-generate efficient model topologies.

## Features

*   **Genetic Algorithm Based Search**: Explores the architecture space efficiently.
*   **Simple Search Space**: Focuses on common CNN operations.
*   **Evaluation Placeholder**: Ready for integration with your model training and evaluation.

## Getting Started

### Prerequisites

*   Python 3.8+
*   PyTorch

### Installation

```bash
pip install torch
```

### Usage

Run the search:

```bash
python nas_search.py
```

The `output/best_architecture.json` will contain the best-found architecture.
