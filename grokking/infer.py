#!/usr/bin/env python3
"""
Inference Script for Transformer Model

This script loads a saved Transformer model and performs inference based on user input.
Usage:
    python infer.py --model_path path/to/best_model.pt
    OR
    python infer.py --model_path path/to/best_model.pt --input "your input here"
"""

import argparse
import torch
import json
import os
from model import Transformer  # Ensure this path is correct
import wandb  # Optional, only if needed for token mappings


def load_model(checkpoint_path, device):
    """
    Loads the model checkpoint and returns the model instance along with its configuration.

    Args:
        checkpoint_path (str): Path to the saved model checkpoint.
        device (torch.device): Device to load the model on.

    Returns:
        model (Transformer): The loaded Transformer model.
        model_config (dict): Configuration parameters used to initialize the model.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Model file not found at {checkpoint_path}")

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract model configuration from checkpoint
    model_config = checkpoint.get("model_config", None)
    if model_config is None:
        raise ValueError("The checkpoint does not contain 'model_config'.")

    # Initialize the model using the extracted configuration
    model = Transformer(
        num_layers=model_config["num_layers"],
        dim_model=model_config["dim_model"],
        num_heads=model_config["num_heads"],
        num_tokens=model_config["num_tokens"],
        seq_len=model_config["seq_len"],
    ).to(device)

    # Load the state_dict
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()  # Set to evaluation mode
    return model, model_config


def main():
    parser = argparse.ArgumentParser(
        description="Inference Script for Transformer Model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the saved model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--input", type=str, default=None, help="Input example for inference"
    )
    parser.add_argument(
        "--use_gpu", action="store_true", help="Use GPU for inference if available"
    )
    args = parser.parse_args()

    # Set device
    device = torch.device(
        "cuda" if args.use_gpu and torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Load the model and configuration
    try:
        model, model_config = load_model(args.model_path, device)
        print(f"Model loaded successfully from {args.model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Extract tokens from model_config
    op_token = model_config.get("op_token", 0)
    eq_token = model_config.get("eq_token", 0)

    while True:
        input_x = torch.tensor(int(input("Enter x: ")))
        input_y = torch.tensor(int(input("Enter y: ")))
        seq = [input_x, op_token, input_y, eq_token]

        input_tokens = torch.tensor([seq]).to(device)

        # Perform inference
        try:
            with torch.no_grad():
                output = model(input_tokens)[-1, :, :]
                prediction = str(torch.argmax(output).item())
        except Exception as e:
            print(f"Error during inference: {e}")
            return

        # Display the prediction
        print(f"Input: {input_tokens}")
        print(f"Prediction: {prediction}")


if __name__ == "__main__":
    main()
