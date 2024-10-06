from math import ceil
import torch
from tqdm import tqdm
import wandb
from wandb import Table
import json

from data import get_data, binary_tokens
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

    train_loader, val_loader, op_token, eq_token, num_unique_tokens = get_data(
        config.operation,
        config.prime,
        config.training_fraction,
        config.batch_size,
        config.curriculum,
    )
    export_dataloader_data(train_loader, "train_data.json")
    export_dataloader_data(val_loader, "val_data.json")
    max_sequence_length = len(train_loader.dataset[0][0])
    config.max_sequence_length = max_sequence_length

    model = Transformer(
        num_layers=config.num_layers,
        dim_model=config.dim_model,
        num_heads=config.num_heads,
        num_tokens=num_unique_tokens,
        seq_len=max_sequence_length,
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
        train(model, train_loader, optimizer, scheduler, device, config)
        val_acc, val_loss = evaluate(model, val_loader, device, epoch, config)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_filename = f"models/best_model_{config.operation}.pt"
            model_config = {
                "num_layers": config.num_layers,
                "dim_model": config.dim_model,
                "num_heads": config.num_heads,
                "num_tokens": eq_token + 1,
                "seq_len": max_sequence_length,
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


def train(model, train_loader, optimizer, scheduler, device, config):
    # Set model to training mode
    model.train()
    if config.task_type == "sequence":
        criterion = torch.nn.CrossEntropyLoss(ignore_index=binary_tokens["<EOS>"])
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # Loop over each batch from the training set
    for batch in train_loader:

        # Copy data to device if needed
        batch = tuple(t.to(device) for t in batch)

        # Unpack the batch from the loader
        inputs, labels = batch

        # Zero gradient buffers
        optimizer.zero_grad()

        if config.task_type == "classification":
            # Zero gradient buffers
            optimizer.zero_grad()
            # Forward pass
            output = model(inputs)[-1, :, :]
            loss = criterion(output, labels)
            # Compute accuracy
            acc = (torch.argmax(output, dim=1) == labels).sum() / len(labels)
        elif config.task_type == "sequence":
            eq_positions = (inputs == binary_tokens["="]).nonzero(as_tuple=True)[1]

            # Prepare decoder input by shifting the entire sequence to the right
            decoder_input = inputs[:, :-1]  # [batch_size, seq_len - 1]
            target = inputs[:, 1:]  # [batch_size, seq_len - 1]

            # Forward pass through the model
            output = model(decoder_input)  # [seq_len -1, batch_size, num_tokens]

            # **Transpose the output to [batch_size, seq_len -1, num_tokens]**
            output = output.transpose(0, 1)  # Now [batch_size, seq_len -1, num_tokens]

            # Unpack the dimensions correctly
            batch_size, seq_len, _ = output.size()

            # Create a mask for positions after "=" token
            mask = torch.arange(seq_len, device=device).unsqueeze(0).expand(
                batch_size, seq_len
            ) > eq_positions.unsqueeze(1)

            # Flatten the output and target tensors
            flat_output = output.contiguous().view(
                -1, output.size(-1)
            )  # [batch_size * seq_len, num_tokens]
            flat_target = target.contiguous().view(-1)  # [batch_size * seq_len]

            # Apply the mask to select only the relevant positions
            mask_flat = mask.view(-1)  # [batch_size * seq_len]
            masked_output = flat_output[mask_flat]  # [num_selected, num_tokens]
            masked_target = flat_target[mask_flat]  # [num_selected]

            # Calculate loss
            loss = criterion(masked_output, masked_target)

            # Calculate accuracy
            preds = torch.argmax(masked_output, dim=1)
            correct = (preds == masked_target).float()
            total = (masked_target != binary_tokens["<EOS>"]).float()
            acc = correct.sum() / total.sum()

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
        if wandb.run.step == config.num_steps:
            return


def evaluate(model, val_loader, device, epoch, config):
    # Set model to evaluation mode
    model.eval()

    if config.task_type == "classification":
        criterion = torch.nn.CrossEntropyLoss()
    elif config.task_type == "sequence":
        criterion = torch.nn.CrossEntropyLoss(ignore_index=binary_tokens["<EOS>"])
    else:
        raise ValueError(f"Unsupported task type: {config.task_type}")

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    examples_table = []

    # Loop over each batch from the validation set
    for batch in val_loader:
        # Copy data to device if needed
        batch = tuple(t.to(device) for t in batch)

        # Unpack the batch from the loader
        inputs, labels = batch

        # Forward pass
        with torch.no_grad():
            if config.task_type == "classification":
                output = model(inputs)[-1, :, :]
                loss = criterion(output, labels)
                preds = torch.argmax(output, dim=1)
                correct = (preds == labels).sum().item()
                total_samples += len(labels)

            elif config.task_type == "sequence":
                eq_positions = (inputs == binary_tokens["="]).nonzero(as_tuple=True)[1]
                decoder_input = inputs[:, :-1]
                target = inputs[:, 1:]
                output = model(decoder_input)  # [seq_len -1, batch_size, num_tokens]

                # **Transpose the output to [batch_size, seq_len -1, num_tokens]**
                output = output.transpose(
                    0, 1
                )  # Now [batch_size, seq_len -1, num_tokens]

                batch_size, seq_len, _ = output.size()
                mask = torch.arange(seq_len, device=device).unsqueeze(0).expand(
                    batch_size, seq_len
                ) > eq_positions.unsqueeze(1)

                reshaped_output = output.contiguous().view(
                    -1, output.size(-1)
                )  # [batch_size * seq_len, num_tokens]
                reshaped_target = target.contiguous().view(-1)  # [batch_size * seq_len]

                mask_flat = mask.contiguous().view(-1)  # [batch_size * seq_len]
                masked_output = reshaped_output[mask_flat]  # [num_selected, num_tokens]
                masked_target = reshaped_target[mask_flat]  # [num_selected]

                loss = criterion(masked_output, masked_target)
                preds = torch.argmax(masked_output, dim=1)
                correct = (preds == masked_target).float().sum().item()
                total_samples += (masked_target != binary_tokens["<EOS>"]).sum().item()

        total_loss += loss.item() * inputs.size(0)
        total_correct += correct

        # Collect examples for visualization
        input_example = str(list(inputs[0].cpu().numpy()))
        if config.task_type == "classification":
            predicted_output = str(preds[0].item())
            true_output = str(labels[0].item())
        else:  # sequence
            start_idx = 0
            end_idx = seq_len
            predicted_output = str(preds[start_idx:end_idx].cpu().numpy())
            true_output = str(masked_target[start_idx:end_idx].cpu().numpy())
        examples_table.append([input_example, predicted_output, true_output])

    # Calculate average loss and accuracy
    avg_loss = total_loss / len(val_loader.dataset)
    accuracy = total_correct / total_samples

    # Create wandb table for examples
    columns = ["example", "prediction", "truth"]
    wandb_table = wandb.Table(data=examples_table, columns=columns)

    # Log model parameters if needed
    if wandb.run.step % 10 == 0:
        log_model_parameters_wandb(model)

    # Log metrics
    metrics = {
        f"Examples_epoch_{epoch}": wandb_table,
        "validation/accuracy": accuracy,
        "validation/loss": avg_loss,
        "epoch": epoch,
    }
    wandb.log(metrics, commit=True)

    return accuracy, avg_loss
