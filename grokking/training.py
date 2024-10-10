# training.py

from math import ceil
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import wandb
from torch import Tensor
from typing import Dict, Any, Tuple, List
import json

from data import get_data, BINARY_TOKENS
from model import Transformer

# Add these at the top of training.py after imports

from typing import List, Dict

# Create a reverse mapping from token IDs to tokens
ID_TO_TOKEN = {v: k for k, v in BINARY_TOKENS.items()}


def decode_sequence(sequence: Tensor, id_to_token: Dict[int, str]) -> str:
    return "".join([id_to_token.get(token_id, "?") for token_id in sequence.tolist()])


def export_dataloader_data(dataloader: DataLoader, filename: str) -> None:
    all_inputs, all_labels = extract_data(dataloader)
    save_to_json(all_inputs, all_labels, filename)


def extract_data(dataloader: DataLoader) -> Tuple[List[List[int]], List[List[int]]]:
    all_inputs, all_labels = [], []
    for batch in tqdm(dataloader, desc="Extracting data"):
        inputs, labels = [tensor.to("cpu") for tensor in batch]
        all_inputs.extend(inputs.numpy().tolist())
        all_labels.extend(labels.numpy().tolist())
    return all_inputs, all_labels


def save_to_json(
    inputs: List[List[int]], labels: List[List[int]], filename: str
) -> None:
    data = {"inputs": inputs, "labels": labels}
    with open(filename, "w") as f:
        json.dump(data, f)


def log_model_parameters_wandb(model: torch.nn.Module) -> None:
    for name, param in model.named_parameters():
        param_np = param.detach().cpu().numpy()
        wandb.log({f"parameters/{name}": wandb.Histogram(param_np)}, commit=False)


def clear_logs() -> None:
    # replace logs/validation_examples.json with an empty file
    with open("logs/validation_examples.json", "w") as f:
        f.write("")


def main(args: Dict[str, Any]) -> None:
    clear_logs()
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
        train_metrics = train(model, train_loader, optimizer, scheduler, device, config)
        val_metrics = evaluate(model, val_loader, device, epoch, config)

        best_val_acc = save_best_model(
            val_metrics["accuracy"],
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


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_wandb(args: Dict[str, Any]) -> None:
    wandb.init(project="grokking", config=args)
    config = wandb.config
    wandb.define_metric("step")
    wandb.define_metric("epoch")
    wandb.define_metric("training/accuracy", step_metric="step")
    wandb.define_metric("training/loss", step_metric="step")
    wandb.define_metric("validation/accuracy", step_metric="epoch")
    wandb.define_metric("validation/loss", step_metric="epoch")


def get_max_sequence_length(train_loader: DataLoader) -> int:
    return len(train_loader.dataset[0][0])


def initialize_model_optimizer_scheduler(
    config: Any, num_unique_tokens: int, seq_len: int, device: torch.device
) -> Tuple[Transformer, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    model = Transformer(
        num_layers=config.num_layers,
        dim_model=config.dim_model,
        num_heads=config.num_heads,
        num_tokens=num_unique_tokens,
        seq_len=seq_len,
    ).to(device)

    total_params = count_parameters(model)
    wandb.log({"total_parameters": total_params}, commit=False)

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
) -> Dict[str, float]:
    """
    Trains the model for one epoch.

    Args:
        model (Transformer): The model to train.
        train_loader (DataLoader): Training DataLoader.
        optimizer (torch.optim.Optimizer): Optimizer.
        scheduler (torch.optim.lr_scheduler.LRScheduler): Learning rate scheduler.
        device (torch.device): Device to run the training on.
        config (Any): Configuration parameters.

    Returns:
        Dict containing training metrics.
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
            break

    return {"accuracy": acc.item(), "loss": loss.item()}


def get_criterion(config: Any) -> torch.nn.Module:
    """
    Returns the appropriate loss function based on task type.

    Args:
        config (Any): Configuration parameters.

    Returns:
        torch.nn.Module: The loss function.
    """
    if config.task_type == "sequence":
        return torch.nn.CrossEntropyLoss()
    else:
        return torch.nn.CrossEntropyLoss()


def train_classification(
    model: Transformer, inputs: Tensor, labels: Tensor, criterion: torch.nn.Module
) -> Tuple[Tensor, torch.Tensor, torch.Tensor]:
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
) -> Tuple[Tensor, torch.Tensor, torch.Tensor]:
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
    eq_positions = get_eq_positions(inputs)
    decoder_input = inputs[:, :-1]
    target = inputs[:, 1:]
    output = model(decoder_input).transpose(0, 1)
    # [batch_size, seq_len -1, num_tokens]
    loss, acc = compute_sequence_loss_and_accuracy(
        output, target, eq_positions, criterion, config
    )
    return output, loss, acc


def get_eq_positions(inputs: Tensor) -> Tensor:
    """
    Returns the positions of the "=" tokens in the input tensor.

    Args:
        inputs (Tensor): Input data.

    Returns:
        Tensor: Positions of the "=" tokens.
    """
    return (inputs == BINARY_TOKENS["="]).nonzero(as_tuple=True)[1] - 1


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
    mask = create_mask_from_positions(seq_len, batch_size, eq_positions, output.device)
    masked_output = mask_tensor(output, mask)
    masked_target = mask_tensor(target, mask)

    loss = criterion(masked_output, masked_target)
    # loss = criterion(output.transpose(1, 2), target)
    preds = torch.argmax(masked_output, dim=1)
    # preds = torch.argmax(output.transpose(1, 2), dim=1)
    acc = (preds == masked_target).float().mean()
    return loss, acc


def generate_flat_tensors(inputs: Tensor) -> Tuple[Tensor, Tensor]:
    # depending on the dimensions of the inputs, we need to flatten the inputs
    if len(inputs.shape) == 3:
        batch_size, seq_len, num_tokens = inputs.size()
        flat_inputs = inputs.contiguous().view(-1, num_tokens)
    else:
        flat_inputs = inputs.contiguous().view(-1)
    return flat_inputs


def create_mask_from_positions(seq_len, batch_size, positions, device) -> Tensor:
    mask = torch.arange(seq_len, device=device).unsqueeze(0).expand(
        batch_size, seq_len
    ) > positions.unsqueeze(1)
    return mask


def mask_tensor(tensor: Tensor, mask: Tensor) -> Tensor:
    """
    Masks a tensor.

    Args:
        tensor (Tensor): The tensor to mask.
        mask (Tensor): The mask to apply.

    Returns:
        Tensor: The masked tensor.
    """
    return tensor[mask]


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
    wandb.log(metrics, commit=False)


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
                total_loss, total_correct, total_samples = evaluate_classification(
                    model, inputs, labels, criterion
                )
            elif config.task_type == "sequence":
                loss, acc, num_samples, preds, trues = evaluate_sequence(
                    model, inputs, labels, criterion, config
                )
                total_loss += loss.item() * inputs.size(0)
                total_correct += acc * num_samples
                total_samples += num_samples
                # Collect validation examples with actual predictions
                examples_table.extend(
                    collect_validation_examples(
                        model, inputs, labels, config, preds, trues
                    )
                )

    # Calculate average loss and accuracy
    avg_loss = total_loss / len(val_loader.dataset)
    accuracy = total_correct / total_samples

    if wandb.run.step % 500 == 0:
        log_model_parameters_wandb(model)
        with open("logs/validation_examples.json", "a") as f:
            json.dump({f"Step {wandb.run.step}": examples_table}, f, indent=4)

    metrics = {
        "validation/accuracy": accuracy,
        "validation/loss": avg_loss,
        "epoch": epoch,
    }
    wandb.log(metrics)

    return {"accuracy": accuracy, "loss": avg_loss}


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
        return torch.nn.CrossEntropyLoss()
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
) -> Tuple[float, float, int, List[str], List[str]]:
    """
    Evaluates the model on a sequence task.

    Args:
        model (Transformer): The model to evaluate.
        inputs (Tensor): Input data.
        labels (Tensor): Labels.
        criterion (torch.nn.Module): Loss function.
        config (Any): Configuration parameters.

    Returns:
        Tuple containing loss, accuracy, number of samples, predicted sequences, and true sequences.
    """
    eq_positions = (inputs == BINARY_TOKENS["="]).nonzero(as_tuple=True)[1]
    decoder_input = inputs[:, :-1]
    target = inputs[:, 1:]
    output = model(decoder_input).transpose(
        0, 1
    )  # [batch_size, seq_len -1, num_tokens]
    loss, acc = compute_sequence_loss_and_accuracy(
        output, target, eq_positions, criterion, config
    )

    # Get predicted tokens by taking the argmax
    preds = torch.argmax(output, dim=2)  # [batch_size, seq_len -1]

    # Calculate the number of correct predictions
    correct = acc.item() * len(target)
    num_samples = len(target)

    return loss, acc, num_samples, preds, target


def collect_validation_examples(
    model,
    inputs: Tensor,
    labels: Tensor,
    config: Any,
    preds: List[str],
    trues: List[str],
) -> List[List[str]]:
    """
    Collects examples for visualization.

    Args:
        inputs (Tensor): Input data.
        labels (Tensor): Labels.
        config (Any): Configuration parameters.
        preds (List[str]): Predicted sequences.
        trues (List[str]): True sequences.

    Returns:
        List of examples in the format [input, prediction, truth].
    """
    examples = []
    input_example = decode_sequence(inputs[0], ID_TO_TOKEN)

    if config.task_type == "classification":
        predicted_output = str(torch.argmax(model(inputs)[-1, :, :], dim=1)[0].item())
        true_output = str(labels[0].item())
    else:  # sequence
        split_pos = (inputs[0] == BINARY_TOKENS["="]).nonzero()[0][0]
        predicted_output = decode_sequence(preds[0][split_pos:], ID_TO_TOKEN)
        true_output = decode_sequence(trues[0][split_pos:], ID_TO_TOKEN)

    examples.append(
        {"example": input_example, "prediction": predicted_output, "truth": true_output}
    )
    return examples


def save_best_model(
    val_acc: float,
    best_val_acc: float,
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    config: Any,
    epoch: int,
    op_token: int,
    eq_token: int,
    max_sequence_length: int,
) -> float:
    """
    Saves the model if validation accuracy improves.

    Args:
        val_acc (float): Current validation accuracy.
        best_val_acc (float): Best validation accuracy so far.
        model (Transformer): The model to save.
        optimizer (torch.optim.Optimizer): Optimizer.
        config (Any): Configuration parameters.
        epoch (int): Current epoch number.
        op_token (int): Operation token.
        eq_token (int): Equality token.
        max_sequence_length (int): Maximum sequence length.

    Returns:
        Updated best_val_acc.
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
    optimizer: torch.optim.Optimizer,
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
        optimizer (torch.optim.Optimizer): Optimizer.
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
