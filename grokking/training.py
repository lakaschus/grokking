from math import ceil
import torch
from tqdm import tqdm
import wandb
from wandb import Table
import json

from data import get_data
from model import Transformer


def export_dataloader_data(dataloader, filename):
    """
    Extracts all inputs and labels from a DataLoader.

    Args:
        dataloader (DataLoader): The DataLoader to extract data from.
        device (torch.device): The device to which tensors are moved.

    Returns:
        dict: A dictionary containing lists of inputs and labels.
    """
    all_inputs = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Extracting data"):
        # Move data to CPU and convert to NumPy for serialization
        batch = tuple(t.to("cpu") for t in batch)
        inputs, labels = batch
        all_inputs.extend(inputs.numpy().tolist())
        all_labels.extend(labels.numpy().tolist())

    data = {"inputs": all_inputs, "labels": all_labels}
    with open(filename, "w") as f:
        json.dump(data, f)


def log_model_parameters_wandb(model):
    for name, param in model.named_parameters():
        # Convert the parameter tensor to a NumPy array
        param_np = param.detach().cpu().numpy()

        # Log as a histogram to wandb
        wandb.log({f"parameters/{name}": wandb.Histogram(param_np)})


def main(args: dict):
    wandb.init(project="grokking", config=args)
    config = wandb.config
    device = torch.device(config.device)

    # Define time scales
    wandb.define_metric("step")
    wandb.define_metric("epoch")

    # Define metrics
    wandb.define_metric("training/accuracy", step_metric="step")
    wandb.define_metric("training/loss", step_metric="step")
    wandb.define_metric("validation/accuracy", step_metric="epoch")
    wandb.define_metric("validation/loss", step_metric="epoch")

    train_loader, val_loader, op_token, eq_token = get_data(
        config.operation, config.prime, config.training_fraction, config.batch_size
    )
    export_dataloader_data(train_loader, "train_data.json")
    export_dataloader_data(val_loader, "val_data.json")

    model = Transformer(
        num_layers=config.num_layers,
        dim_model=config.dim_model,
        num_heads=config.num_heads,
        num_tokens=eq_token + 1,
        seq_len=5,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.98),
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=9
    )

    num_epochs = ceil(config.num_steps / len(train_loader))

    best_val_acc = 0.0

    for epoch in tqdm(range(num_epochs)):
        train(model, train_loader, optimizer, scheduler, device, config.num_steps)
        val_acc, val_loss = evaluate(model, val_loader, device, epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_filename = f"models/best_model_{config.operation}.pt"
            model_config = {
                "num_layers": config.num_layers,
                "dim_model": config.dim_model,
                "num_heads": config.num_heads,
                "num_tokens": eq_token + 1,
                "seq_len": 5,
                "op_token": op_token,
                "eq_token": eq_token,
            }
            checkpoint = {
                "epoch": epoch,
                "model_config": model_config,  # Embed configuration here
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "validation_accuracy": val_acc,
                "validation_loss": val_loss,
            }
            torch.save(checkpoint, model_filename)


def train(model, train_loader, optimizer, scheduler, device, num_steps):
    # Set model to training mode
    model.train()
    criterion = torch.nn.CrossEntropyLoss()

    # Loop over each batch from the training set
    for batch in train_loader:

        # Copy data to device if needed
        batch = tuple(t.to(device) for t in batch)

        # Unpack the batch from the loader
        inputs, labels = batch

        # Zero gradient buffers
        optimizer.zero_grad()

        # Forward pass
        output = model(inputs)[-1, :, :]
        loss = criterion(output, labels)
        acc = (torch.argmax(output, dim=1) == labels).sum() / len(labels)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()
        scheduler.step()

        metrics = {
            "training/accuracy": acc,
            "training/loss": loss,
            "step": wandb.run.step,
        }
        wandb.log(metrics)

        # Finish training at maximum gradient updates
        if wandb.run.step == num_steps:
            return


def evaluate(model, val_loader, device, epoch):
    # Set model to evaluation mode
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    correct = 0
    loss = 0.0

    examples_table = Table(columns=["Input", "Predicted Output", "True Output"])

    # Loop over each batch from the validation set
    for batch in val_loader:
        # Copy data to device if needed
        batch = tuple(t.to(device) for t in batch)

        # Unpack the batch from the loader
        inputs, labels = batch

        # Forward pass
        with torch.no_grad():
            output = model(inputs)[-1, :, :]
            correct += (torch.argmax(output, dim=1) == labels).sum()
            loss += criterion(output, labels) * len(labels)

    acc = correct / len(val_loader.dataset)
    loss = loss / len(val_loader.dataset)

    examples_table = []
    for i in range(min(3, len(inputs))):
        input_example = str(list(inputs[i].cpu().numpy()))
        predicted_output = str(torch.argmax(output[i]).item())
        true_output = str(labels[i].item())
        examples_table.append([input_example, predicted_output, true_output])
        examples_table.append([input_example, predicted_output, true_output])

    columns = ["example", "prediction", "truth"]
    wandb_table = wandb.Table(data=examples_table, columns=columns)

    if wandb.run.step % 100 == 0:
        log_model_parameters_wandb(model)

    metrics = {
        f"Examples_epoch_{epoch}": wandb_table,
        "validation/accuracy": acc,
        "validation/loss": loss,
        "epoch": epoch,
    }
    wandb.log(metrics, commit=False)

    return acc, loss  # Return the metrics
