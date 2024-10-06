# training.py

from math import ceil
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import wandb
from torch import Tensor
from wandb import Table
import json
from typing import Dict, Any, Tuple

from data import get_data, BINARY_TOKENS
from model import Transformer


def export_dataloader_data(dataloader: DataLoader, filename: str) -> None:
    """
    Extracts all inputs and labels from a DataLoader and saves them to a JSON file.

    Args:
        dataloader (DataLoader): The DataLoader to extract data from.
        filename (str): The filename to save the extracted data.
    """
    all_inputs, all_labels = extract_data(dataloader)
    save_to_json(all_inputs, all_labels, filename)


def extract_data(dataloader: DataLoader) -> Tuple[list, list]:
    """
    Extracts inputs and labels from a DataLoader.

    Args:
        dataloader (DataLoader): The DataLoader to extract data from.

    Returns:
        Tuple containing lists of inputs and labels.
    """
    all_inputs, all_labels = [], []
    for batch in tqdm(dataloader, desc="Extracting data"):
        inputs, labels = [tensor.to("cpu") for tensor in batch]
        all_inputs.extend(inputs.numpy().tolist())
        all_labels.extend(labels.numpy().tolist())
    return all_inputs, all_labels


def save_to_json(inputs: list, labels: list, filename: str) -> None:
    """
    Saves inputs and labels to a JSON file.

    Args:
        inputs (list): list of input sequences.
        labels (list): list of label sequences.
        filename (str): The filename to save the data.
    """
    data = {"inputs": inputs, "labels": labels}
    with open(filename, "w") as f:
        json.dump(data, f)


def log_model_parameters_wandb(model: torch.nn.Module) -> None:
    """
    Logs model parameters to wandb as histograms.

    Args:
        model (torch.nn.Module): The model whose parameters are to be logged.
    """
    for name, param in model.named_parameters():
        param_np = param.detach().cpu().numpy()
        wandb.log({f"parameters/{name}": wandb.Histogram(param_np)})


def main(args: Dict[str, Any]) -> None:
    """
    Main function to initialize training.

    Args:
        args (Dict[str, Any]): Configuration arguments.
    """
    initialize_wandb(args)
    config = wandb.config
    device = torch.device(config.device)

    data_loaders, op_token, eq_token, num_unique_tokens = get_data(
        config.operation,
        config.prime,
        config.training_fraction,
        config.batch_size,
        config.curriculum,
    )
    train_loader, val_loader, _, _, _ = data_loaders

    export_dataloader_data(train_loader, "train_data.json")
    export_dataloader_data(val_loader, "val_data.json")

    max_sequence_length = get_max_sequence_length(train_loader)
    config.max_sequence_length = max_sequence_length

    model, optimizer, scheduler = initialize_model_optimizer_scheduler(
        config, num_unique_tokens, max_sequence_length, device
    )

    num_epochs = calculate_num_epochs(config.num_steps, len(train_loader))
    best_val_acc = 0.0

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        train(model, train_loader, optimizer, scheduler, device, config)
        val_acc, val_loss = evaluate(model, val_loader, device, epoch, config)

        best_val_acc = save_best_model(
            val_acc,
            best_val_acc,
            model,
            optimizer,
            config,
            epoch,
            op_token,
            eq_token,
            max_sequence_length,
        )
        # Optionally, you can break early if desired


def initialize_wandb(args: Dict[str, Any]) -> None:
    """
    Initializes Weights & Biases (wandb) for experiment tracking.

    Args:
        args (Dict[str, Any]): Configuration arguments.
    """
    wandb.init(project="grokking", config=args)
    config = wandb.config
    wandb.define_metric("step")
    wandb.define_metric("epoch")
    wandb.define_metric("training/accuracy", step_metric="step")
    wandb.define_metric("training/loss", step_metric="step")
    wandb.define_metric("validation/accuracy", step_metric="epoch")
    wandb.define_metric("validation/loss", step_metric="epoch")


def get_max_sequence_length(train_loader: DataLoader) -> int:
    """
    Retrieves the maximum sequence length from the training dataset.

    Args:
        train_loader (DataLoader): The training DataLoader.

    Returns:
        int: The maximum sequence length.
    """
    return len(train_loader.dataset[0][0])


def initialize_model_optimizer_scheduler(
    config: Any, num_unique_tokens: int, seq_len: int, device: torch.device
) -> Tuple[Transformer, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """
    Initializes the model, optimizer, and scheduler.

    Args:
        config (Any): Configuration parameters.
        num_unique_tokens (int): Number of unique tokens.
        seq_len (int): Maximum sequence length.
        device (torch.device): Device to run the model on.

    Returns:
        Tuple containing the model, optimizer, and scheduler.
    """
    model = Transformer(
        num_layers=config.num_layers,
        dim_model=config.dim_model,
        num_heads=config.num_heads,
        num_tokens=num_unique_tokens,
        seq_len=seq_len,
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
    return model, optimizer, scheduler


def calculate_num_epochs(num_steps: int, train_loader_length: int) -> int:
    """
    Calculates the number of epochs based on total training steps and loader length.

    Args:
        num_steps (int): Total number of training steps.
        train_loader_length (int): Number of batches per epoch.

    Returns:
        int: Number of epochs to train.
    """
    return ceil(num_steps / train_loader_length)


def train(
    model: Transformer,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    config: Any,
) -> None:
    """
    Trains the model for one epoch.

    Args:
        model (Transformer): The model to train.
        train_loader (DataLoader): Training DataLoader.
        optimizer (torch.optim.Optimizer): Optimizer.
        scheduler (torch.optim.lr_scheduler.LRScheduler): Learning rate scheduler.
        device (torch.device): Device to run the training on.
        config (Any): Configuration parameters.
    """
    model.train()
    criterion = get_criterion(config)

    for batch in train_loader:
        inputs, labels = [tensor.to(device) for tensor in batch]
        optimizer.zero_grad()

        if config.task_type == "classification":
            output, loss, acc = train_classification(model, inputs, labels, criterion)
        elif config.task_type == "sequence":
            output, loss, acc = train_sequence(model, inputs, labels, criterion, config)
        else:
            raise ValueError(f"Unsupported task type: {config.task_type}")

        loss.backward()
        optimizer.step()
        scheduler.step()

        log_training_metrics(acc, loss)

        if wandb.run.step >= config.num_steps:
            return


def get_criterion(config: Any) -> torch.nn.Module:
    """
    Returns the appropriate loss function based on task type.

    Args:
        config (Any): Configuration parameters.

    Returns:
        torch.nn.Module: The loss function.
    """
    if config.task_type == "sequence":
        return torch.nn.CrossEntropyLoss(ignore_index=BINARY_TOKENS["<EOS>"])
    else:
        return torch.nn.CrossEntropyLoss()


def train_classification(
    model: Transformer, inputs: Tensor, labels: Tensor, criterion: torch.nn.Module
) -> Tuple[Any, torch.Tensor, torch.Tensor]:
    """
    Performs a training step for classification tasks.

    Args:
        model (Transformer): The model.
        inputs (Tensor): Input data.
        labels (Tensor): Labels.
        criterion (torch.nn.Module): Loss function.

    Returns:
        Tuple containing outputs, loss, and accuracy.
    """
    output = model(inputs)[-1, :, :]
    loss = criterion(output, labels)
    preds = torch.argmax(output, dim=1)
    acc = (preds == labels).float().mean()
    return output, loss, acc


def train_sequence(
    model: Transformer,
    inputs: Tensor,
    labels: Tensor,
    criterion: torch.nn.Module,
    config: Any,
) -> Tuple[Any, torch.Tensor, torch.Tensor]:
    """
    Performs a training step for sequence tasks.

    Args:
        model (Transformer): The model.
        inputs (Tensor): Input data.
        labels (Tensor): Labels.
        criterion (torch.nn.Module): Loss function.
        config (Any): Configuration parameters.

    Returns:
        Tuple containing outputs, loss, and accuracy.
    """
    eq_positions = (inputs == BINARY_TOKENS["="]).nonzero(as_Tuple=True)[1]
    decoder_input = inputs[:, :-1]
    target = inputs[:, 1:]
    output = model(decoder_input).transpose(
        0, 1
    )  # [batch_size, seq_len -1, num_tokens]
    loss, acc = compute_sequence_loss_and_accuracy(
        output, target, eq_positions, criterion, config
    )
    return output, loss, acc


def compute_sequence_loss_and_accuracy(
    output: Tensor,
    target: Tensor,
    eq_positions: Tensor,
    criterion: torch.nn.Module,
    config: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes loss and accuracy for sequence tasks.

    Args:
        output (Tensor): Model outputs.
        target (Tensor): Target labels.
        eq_positions (Tensor): Positions of "=" tokens.
        criterion (torch.nn.Module): Loss function.
        config (Any): Configuration parameters.

    Returns:
        Tuple containing loss and accuracy.
    """
    batch_size, seq_len, num_tokens = output.size()
    mask = torch.arange(seq_len, device=output.device).unsqueeze(0).expand(
        batch_size, seq_len
    ) > eq_positions.unsqueeze(1)
    flat_output = output.view(-1, num_tokens)
    flat_target = target.view(-1)
    mask_flat = mask.view(-1)

    masked_output = flat_output[mask_flat]
    masked_target = flat_target[mask_flat]

    loss = criterion(masked_output, masked_target)
    preds = torch.argmax(masked_output, dim=1)
    acc = (preds == masked_target).float().mean()
    return loss, acc


def log_training_metrics(acc: torch.Tensor, loss: torch.Tensor) -> None:
    """
    Logs training metrics to wandb.

    Args:
        acc (torch.Tensor): Accuracy.
        loss (torch.Tensor): Loss.
    """
    metrics = {
        "training/accuracy": acc.item(),
        "training/loss": loss.item(),
        "step": wandb.run.step,
    }
    wandb.log(metrics)


def evaluate(
    model: Transformer,
    val_loader: DataLoader,
    device: torch.device,
    epoch: int,
    config: Any,
) -> Tuple[float, float]:
    """
    Evaluates the model on the validation set.

    Args:
        model (Transformer): The model to evaluate.
        val_loader (DataLoader): Validation DataLoader.
        device (torch.device): Device to run evaluation on.
        epoch (int): Current epoch number.
        config (Any): Configuration parameters.

    Returns:
        Tuple containing validation accuracy and loss.
    """
    model.eval()
    criterion = get_evaluation_criterion(config)

    total_loss, total_correct, total_samples = 0.0, 0.0, 0
    examples_table = []

    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = [tensor.to(device) for tensor in batch]
            if config.task_type == "classification":
                loss, correct, samples = evaluate_classification(
                    model, inputs, labels, criterion
                )
            elif config.task_type == "sequence":
                loss, correct, samples = evaluate_sequence(
                    model, inputs, labels, criterion, config
                )
            else:
                raise ValueError(f"Unsupported task type: {config.task_type}")

            total_loss += loss * inputs.size(0)
            total_correct += correct
            total_samples += samples
            examples_table.extend(
                collect_validation_examples(model, inputs, labels, config)
            )

    avg_loss = total_loss / len(val_loader.dataset)
    accuracy = total_correct / total_samples

    log_evaluation_metrics(examples_table, accuracy, avg_loss, epoch)
    log_model_parameters_conditionally(model, config)

    return accuracy, avg_loss


def get_evaluation_criterion(config: Any) -> torch.nn.Module:
    """
    Returns the appropriate loss function for evaluation based on task type.

    Args:
        config (Any): Configuration parameters.

    Returns:
        torch.nn.Module: The loss function.
    """
    if config.task_type == "classification":
        return torch.nn.CrossEntropyLoss()
    elif config.task_type == "sequence":
        return torch.nn.CrossEntropyLoss(ignore_index=BINARY_TOKENS["<EOS>"])
    else:
        raise ValueError(f"Unsupported task type: {config.task_type}")


def evaluate_classification(
    model: Transformer, inputs: Tensor, labels: Tensor, criterion: torch.nn.Module
) -> Tuple[float, float, int]:
    """
    Evaluates classification performance on a batch.

    Args:
        model (Transformer): The model.
        inputs (Tensor): Input data.
        labels (Tensor): Labels.
        criterion (torch.nn.Module): Loss function.

    Returns:
        Tuple containing loss, number of correct predictions, and number of samples.
    """
    output = model(inputs)[-1, :, :]
    loss = criterion(output, labels).item()
    preds = torch.argmax(output, dim=1)
    correct = (preds == labels).sum().item()
    return loss, correct, len(labels)


def evaluate_sequence(
    model: Transformer,
    inputs: Tensor,
    labels: Tensor,
    criterion: torch.nn.Module,
    config: Any,
) -> Tuple[float, float, int]:
    """
    Evaluates sequence performance on a batch.

    Args:
        model (Transformer): The model.
        inputs (Tensor): Input data.
        labels (Tensor): Labels.
        criterion (torch.nn.Module): Loss function.
        config (Any): Configuration parameters.

    Returns:
        Tuple containing loss, number of correct predictions, and number of samples.
    """
    eq_positions = (inputs == BINARY_TOKENS["="]).nonzero(as_Tuple=True)[1]
    decoder_input = inputs[:, :-1]
    target = inputs[:, 1:]
    output = model(decoder_input).transpose(
        0, 1
    )  # [batch_size, seq_len -1, num_tokens]
    loss, acc = compute_sequence_loss_and_accuracy(
        output, target, eq_positions, criterion, config
    )

    correct = acc.item() * ((target != BINARY_TOKENS["<EOS>"]).sum().item())
    samples = (target != BINARY_TOKENS["<EOS>"]).sum().item()
    return loss, correct, samples


def collect_validation_examples(
    model, inputs: Tensor, labels: Tensor, config: Any
) -> list[list[str]]:
    """
    Collects examples for visualization.

    Args:
        inputs (Tensor): Input data.
        labels (Tensor): Labels.
        config (Any): Configuration parameters.

    Returns:
        list of examples in the format [input, prediction, truth].
    """
    examples = []
    input_example = str(list(inputs[0].cpu().numpy()))
    if config.task_type == "classification":
        predicted_output = str(torch.argmax(model(inputs)[-1, :, :], dim=1)[0].item())
        true_output = str(labels[0].item())
    else:  # sequence
        # Assuming evaluate_sequence has been called and preds are available
        # Placeholder for actual prediction logic
        predicted_output = "predicted_sequence"
        true_output = "true_sequence"
    examples.append([input_example, predicted_output, true_output])
    return examples


def log_evaluation_metrics(
    examples: list[list[str]], accuracy: float, loss: float, epoch: int
) -> None:
    """
    Logs evaluation metrics and examples to wandb.

    Args:
        examples (list[list[str]]): list of examples for visualization.
        accuracy (float): Validation accuracy.
        loss (float): Validation loss.
        epoch (int): Current epoch number.
    """
    columns = ["example", "prediction", "truth"]
    wandb_table = wandb.Table(data=examples, columns=columns)

    metrics = {
        f"Examples_epoch_{epoch}": wandb_table,
        "validation/accuracy": accuracy,
        "validation/loss": loss,
        "epoch": epoch,
    }
    wandb.log(metrics, commit=False)


def log_model_parameters_conditionally(model: Transformer, config: Any) -> None:
    """
    Logs model parameters to wandb conditionally.

    Args:
        model (Transformer): The model.
        config (Any): Configuration parameters.
    """
    if wandb.run.step % 10 == 0:
        log_model_parameters_wandb(model)


def save_best_model(
    val_acc: float,
    best_val_acc: float,
    model: Transformer,
    optimizer,
    config: Any,
    epoch: int,
    op_token: int,
    eq_token: int,
    max_sequence_length: int,
) -> Tuple[float, str]:
    """
    Saves the model if validation accuracy improves.

    Args:
        val_acc (float): Current validation accuracy.
        best_val_acc (float): Best validation accuracy so far.
        model (Transformer): The model to save.
        config (Any): Configuration parameters.
        epoch (int): Current epoch number.
        op_token (int): Operation token.
        eq_token (int): Equality token.
        max_sequence_length (int): Maximum sequence length.

    Returns:
        Tuple containing updated best_val_acc and the model filename.
    """
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
        checkpoint = create_checkpoint(
            model, optimizer, config, epoch, model_config, val_acc, val_loss=None
        )  # val_loss can be added if needed
        torch.save(checkpoint, model_filename)
    return best_val_acc


def create_checkpoint(
    model: Transformer,
    optimizer,
    config: Any,
    epoch: int,
    model_config: Dict[str, Any],
    val_acc: float,
    val_loss: float,
) -> Dict[str, Any]:
    """
    Creates a checkpoint dictionary.

    Args:
        model (Transformer): The model.
        config (Any): Configuration parameters.
        epoch (int): Current epoch number.
        model_config (Dict[str, Any]): Model configuration.
        val_acc (float): Validation accuracy.
        val_loss (float): Validation loss.

    Returns:
        Dict containing checkpoint information.
    """
    return {
        "epoch": epoch,
        "model_config": model_config,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "validation_accuracy": val_acc,
        "validation_loss": val_loss,
    }
