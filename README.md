# Grokking

An implementation of the OpenAI 'Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets' paper in PyTorch. Originally forked and modified from:

https://github.com/danielmamay/grokking

---

<img src="figures/grokking_no_reg.png" height="300">

**Figure**: Grokking Example for modular division without any regularization (No weight decay, no dropout)

## Extension

This fork significantly extends the original in repository in the following ways:

1. Model architecture is changed slightly
2. Training logging has been extended significantly
3. Additional novel algorithmic tasks have been added
4. A new task representation was added: The algorithmic tasks can now be generated as Seq2Seq tasks **in arbitrary number bases**, for example Base 2. The goal of the model is then not just to predict one number, but to predict multiple tokens autoregressively. This necessitated major overhaul of the data generation module and the training routine.
5. Multitask Grokking was added, enabling training on multiple distinct parts simultaneously.

## Motivation

TODO: Rationale for the changes

## Installation

* Clone the repo and cd into it:
    ```bash
    git clone https://github.com/lakaschus/grokking.git
    cd grokking
    ```
* Use Python 3.9 or later:
    ```bash
    conda create -n grokking python=3.9
    conda activate grokking
    pip install -r requirements.txt
    ```

## Structure

The project is organized into the following main components:

### Core Components

- **Training Module** (`grokking/training.py`): Handles the main training loop, model initialization, and evaluation routines.

- **Data Generation** (`grokking/data.py`, `grokking/data_encoder.py`): 
  - Implements various algorithmic tasks (modular arithmetic, binary operations, etc.)
  - Supports two task types:
    1. Classification: One-token prediction
    2. Sequence-to-sequence: Binary number representation with autoregressive prediction

- **Model Architecture** (`grokking/model.py`): Standard transformer-decoder architecture with configurable layers, dimensions, and heads.

### Experiment Infrastructure

- **CLI Interface** (`grokking/cli.py`): Command-line interface for running experiments with configurable hyperparameters.

- **Experiment Tracking**:
  - Integration with Weights & Biases for experiment logging
  - Support for both online and offline tracking modes

- **Results Analysis** (`results/classification_vs_seq2seq/`):
  - Jupyter notebooks for analyzing experimental results
  - Comparison scripts between classification and sequence tasks

- **Hyperparameter Management**:
  - Support for grid search (`sweep.yaml`)
  - Configurable model architecture (layers, dimensions, heads)
  - Training parameters (learning rate, batch size, weight decay)
  - Task-specific settings (bit length, training fraction)


## Usage


### Experiment Tracking

The project uses [Weights & Biases](https://wandb.ai/site) to keep track of experiments. Run `wandb login` to use the online dashboard, or `wandb offline` to store the data on your local machine.


### Running Experiments

Basic command structure:
```bash
python grokking/cli.py [options]
```

**Parameters:**

*Training & Model Parameters*

- `--num_layers`: Number of transformer layers (default: 2)
- `--dim_model`: Model dimension (default: 128)
- `--num_heads`: Number of attention heads (default: 4)
- `--dropout`: Dropout rate (default: 0.1)
- `--batch_size`: Batch size (default: 512)
- `--learning_rate`: Learning rate (default: 1e-3)
- `--weight_decay`: Weight decay for regularization (default: 1)
- `--device`: Training device ("cuda" or "cpu")
- `--num_steps`: Total number of training steps

*Task Configuration*

- `--operation`: Type of operation (e.g., "x/y_mod", "x+y_mod", "x^2+y^2_mod")
- `--task_type`: "**classification**" or "**sequence**"
- `--multitask`: Enables multitask learning
- `--base`: Number base for operations for Seq2Seq task representation (default: 2)
- `--fixed_sequence_length`: Use fixed sequence length operands for seq2seq tasks (E.g. "001 + 011 = 100" vs. "1 + 11 = 100")
 
*Data Generation*

- `--training_fraction`: Fraction of data used for training (default: 0.5)
- `--max_bit_length_train`: Maximum bit length of training example (This will determine the size of your training set, which will be 2^max_bit_length_train)
- `--max_bit_length_val_out`: Needs to be larger than `max_bit_length_train` in order to measure out-of-domain accuracy
- `--wandb_tracking`: Weights & Biases tracking mode ("maximal", "minimal", "disabled")

**--> Check out `.vscode/launch.json` for various example configurations!**