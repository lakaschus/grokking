# Grokking

An implementation of the OpenAI 'Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets' paper in PyTorch.

<img src="figures/Figure_1_left_accuracy.png" height="300">

## Installation

* Clone the repo and cd into it:
    ```bash
    git clone https://github.com/danielmamay/grokking.git
    cd grokking
    ```
* Use Python 3.9 or later:
    ```bash
    conda create -n grokking python=3.9
    conda activate grokking
    pip install -r requirements.txt
    ```

## Usage

The project uses [Weights & Biases](https://wandb.ai/site) to keep track of experiments. Run `wandb login` to use the online dashboard, or `wandb offline` to store the data on your local machine.

* To run a single experiment using the [CLI](grokking/cli.py):
    ```bash
    wandb login
    python grokking/cli.py
    ```

* To run a grid search using W&B Sweeps:
    ```bash
    wandb sweep sweep.yaml
    wandb agent {entity}/grokking/{sweep_id}
    ```

## TODOS

- Hyperparmeter Sweep
- Out of distribution validation
- Optimizer benchmark! Maybe it only works with specific ones
- Come up with explanation why grokking works -> Study optimizer first!
- With sequence tasks it seems that validation more closely follows training acc. Think about why that's the case and what's different to one-token classification.
- Mirror binary data. Reasoning: A model cannot compute step by step, if it has to predict the largest token first. Example: 19 + 12 = 31. So the model needs to predict "3" first before it can predict "1". That means it cannot use an addition rule with carry over method.