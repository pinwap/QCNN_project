# QCNN Project: Evolutionary Architecture Search & Training

This project implements a framework for **Quantum Convolutional Neural Networks (QCNNs)**. It supports standard training of fixed architectures and **Evolutionary Architecture Search (EAS)** to automatically discover optimal QCNN structures. It works in a valid **Hybrid (PyTorch + Qiskit)** mode and a pure **Qiskit** mode.

## 🚀 Features
- **Evolutionary Search**: Automatically find the best QCNN circuit structure for a given dataset using Genetic Algorithms.
- **Hybrid Engine**: Uses PyTorch for classical optimization (Backprop) and Qiskit for Quantum execution (fastest/production).
- **Qiskit Engine**: Uses Qiskit Machine Learning for pure quantum optimization.
- **Hydra Configuration**: Full flexibility to swap datasets, engines, strategies, and hyperparameters via CLI or YAML.
- **Robust Logging**: Automatically saves plots, history, configuration, and trained models for every run.

---

## 🛠️ Installation

Ensure you have Python 3.10+ and `uv` installed.

```bash
# Clone the repository
git clone <repo_url>
cd QCNN_Project_Fork

# Install dependencies using uv
uv sync
```

---

## 🏃‍♂️ How to Run

The entry point is `main.py`. You monitor and control everything using **Hydra**.

### 1. Standard Training
Train a standard QCNN model (fixed structure) on a dataset.

```bash
# Debug run (fast, small dataset)
uv run main.py preset=debug task=train

# Production run (full dataset, more epochs)
uv run main.py preset=production task=train

# Override specific parameters
uv run main.py preset=debug task=train engine.epochs=10 engine.lr=0.05
```

### 2. Evolutionary Search
Run a genetic algorithm to find the best circuit structure.

```bash
# Debug search (2 generations, pop size 2)
uv run main.py preset=debug task=evolution

# Custom search query
uv run main.py preset=debug task=evolution evolution.n_gen=5 evolution.n_pop=10
```

### 3. Running Sweeps (Multi-Run)
You can run multiple combinations at once using Hydra's `-m` (multirun) flag.

```bash
# Run ALL combinations of engines and strategies
uv run main.py -m task=train,evolution engine=hybrid,qiskit strategy=hybrid,qiskit preset=debug
```

---

## 📂 Project Structure

```text
├── config/                 # Hydra Configuration
│   ├── config.yaml         # Main entry point
│   ├── preset/             # Presets (debug, production)
│   ├── task/               # Task definitions (train, evolution)
│   ├── dataset/            # Dataset configs (mnist, cifar10)
│   ├── engine/             # Training engine settings
│   └── strategy/           # Evaluation strategy settings
├── src/
│   ├── training/           # Core logic
│   │   ├── engines/        # Optimization loops (Hybrid, Qiskit)
│   │   ├── strategies/     # Evaluation strategies
│   │   ├── evolution.py    # Genetic Algorithm implementation
│   │   └── pipeline.py     # Standard training pipeline
│   ├── models/             # QCNN Circuit definitions
│   ├── data/               # Data loading & preprocessing
│   └── utils/              # Logging & Plotting
├── outputs/                # Artifacts (Logs, Plots, Models)
└── main.py                 # Entry Point
```

---

## ⚙️ Configuration Guide

### Key Concepts
*   **Engine**: The "Optimizer".
    *   `hybrid`: PyTorch-based (Supports true epochs, recommended).
    *   `qiskit`: Qiskit VQC-based (Uses `max_iter` instead of epochs).
*   **Strategy**: The "Evaluator" (used only during Evolution).
    *   `hybrid`: Evaluates candidates using the Hybrid Engine.
    *   `qiskit`: Evaluates candidates using the Qiskit Engine.

### Presets
*   **`preset=debug`**: Uses tiny subsets of data (10 samples) and minimal epochs. Great for checking code logic.
*   **`preset=production`**: Uses full datasets and realistic training schedules.

### Overriding Epochs
*   **`engine.epochs`**: Controls training duration for the **Train** task.
*   **`strategy.epochs`**: Controls training duration for **Evolution** candidates (usually kept low for speed).

---

### 4. LINE Notifications (Optional)
Receive a notification on your phone when long sweeps finish.

1.  Get a **Channel Access Token** from the [LINE Developers Console](https://developers.line.biz/console/).
2.  Create a `.env` file in the project root:
    ```bash
    LINE_CHANNEL_ACCESS_TOKEN="your_long_token_here"
    ```
3.  Run jobs with `notifications.enabled=true`:
    ```bash
    uv run main.py -m task=train notifications.enabled=true ...
    ```

---

## 👨‍💻 Contributing

Before Committing, please run standard linting and formatting checks:

```bash
# Check for lint errors (and automatically fix some)
uv run ruff check . --fix

# Format code to standard style
uv run ruff format
```

---

## ⚠️ Known Issues / Limitations

1.  **Qiskit Engine History**:
    *   The `qiskit` engine uses `NeuralNetworkClassifier` which does not easily expose loss history per epoch.
    *   **Symptom**: In plots/logs, Qiskit runs will show only a single data point (the final result) instead of a learning curve.
    *   **Workaround**: Use `engine=hybrid` if you need detailed loss curves.

2.  **Genetic Limits on Big Qubits**:
    *   If running evolution on large qubit counts (e.g., 16+), ensure `evolution.n_gates` is high (e.g., >100).
    *   **Fix**: The `debug` preset has been updated to `n_gates: 200` to prevent `StopIteration` errors during circuit construction.

3.  **MPS/GPU Support**:
    *   The code supports `mps` (Apple Silicon) via PyTorch in the Hybrid, but Qiskit Aer GPU acceleration requires specific `qiskit-aer-gpu` installation on Linux/CUDA.

