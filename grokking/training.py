from math import ceil
import torch
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from torch import Tensor
from typing import Dict, Any, Tuple, List
import json
from data_encoder import Encoder
from collections import defaultdict


from data import get_data, BINARY_TOKENS
from multitask_data import get_multitask_data
from model import Transformer


def decode_sequence(sequence: Tensor, id_to_token: Dict[int, str]) -> str:
    return " ".join([id_to_token.get(token_id, "?") for token_id in sequence.tolist()])


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
    weight_norms = []
    for name, param in model.named_parameters():
        param_np = param.detach().cpu()
        weight_norms.append(torch.norm(param_np).item())
        # wandb.log({f"parameters/{name}": wandb.Histogram(param_np)}, commit=False)
    total_weight_norm = sum(weight_norms) / len(weight_norms)
    wandb.log({"model_weights/weights_norm": total_weight_norm}, commit=False)


def clear_logs() -> None:
    # replace logs/validation_examples.json with an empty file
    with open("logs/validation_examples_in_domain.json", "w") as f:
        f.write("")
    with open("logs/validation_examples_out_of_domain.json", "w") as f:
        f.write("")


def main(args: Dict[str, Any]) -> None:
    clear_logs()
    initialize_wandb(args)
    config = wandb.config
    device = torch.device(config.device)
    operation = config.operation
    generate_data = get_data
    if config.multitask:
        operation = config.operation.split(",")
        generate_data = get_multitask_data

    # Retrieve DataLoaders
    (
        train_loader,
        val_in_loader,
        val_out_loader,
        op_token,
        eq_token,
        num_unique_tokens,
        encoder,
    ) = generate_data(
        operation,
        task_type=config.task_type,
        max_bit_length_train=config.max_bit_length_train,  # e.g., 6
        max_bit_length_val_out=config.max_bit_length_val_out,  # e.g., 7
        training_fraction=config.training_fraction,
        batch_size=config.batch_size,
        curriculum=config.curriculum,
        base=config.base,
        fixed_sequence_length=config.fixed_sequence_length,
    )

    # Check that training data has no duplicates
    assert len(train_loader.dataset) == len(
        set(train_loader.dataset)
    ), "Training data has duplicates"

    config.training_set_size = len(train_loader.dataset)
    config.op_token = op_token
    config.eq_token = eq_token
    # log training set size
    wandb.log({"training_set_size": config.training_set_size}, commit=False)

    # Optionally export data
    export_dataloader_data(train_loader, "train_data.json")
    export_dataloader_data(val_in_loader, "val_in_data.json")
    export_dataloader_data(val_out_loader, "val_out_data.json")

    max_sequence_length = get_max_sequence_length(train_loader)
    config.max_sequence_length = max_sequence_length

    model, optimizer, scheduler = initialize_model_optimizer_scheduler(
        config, num_unique_tokens, max_sequence_length, device
    )
    optimizer.training_steps = 0

    num_epochs = calculate_num_epochs(config.num_steps, config.training_set_size)
    best_val_acc = 0.0

    current_epoch = 0

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        # Training
        train_metrics = train(
            model, encoder, train_loader, optimizer, scheduler, device, config
        )

        if current_epoch % 10 == 0:
            # In-Domain Validation
            val_in_metrics = evaluate(
                model,
                encoder,
                val_in_loader,
                device,
                optimizer,
                config,
                validation_type="in_domain",
            )

            # Out-of-Domain Validation
            val_out_metrics = evaluate(
                model,
                encoder,
                val_out_loader,
                device,
                optimizer,
                config,
                validation_type="out_of_domain",
            )

        if val_in_metrics["accuracy"] >= 0.99 and not config.continue_training:
            wandb.finish()
            print(
                f"Training completed successfully after {optimizer.training_steps} steps."
            )
            break


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_wandb(args: Dict[str, Any]) -> None:
    if args.wandb_tracking == "disabled":
        os.environ["WANDB_DISABLED"] = "true"
    else:
        os.environ["WANDB_DISABLED"] = "false"
    wandb.init(project="grokking", config=args)
    wandb.define_metric("Optimization Steps")
    wandb.define_metric("epoch")
    wandb.define_metric("training_set_size")
    wandb.define_metric("training/accuracy", step_metric="Optimization Steps")
    wandb.define_metric("training/loss", step_metric="Optimization Steps")
    wandb.define_metric("learning_rate", step_metric="Optimization Steps")
    wandb.define_metric(
        "validation_in_domain/accuracy", step_metric="Optimization Steps"
    )
    wandb.define_metric("validation_in_domain/loss", step_metric="Optimization Steps")
    wandb.define_metric(
        "validation_out_of_domain/accuracy", step_metric="Optimization Steps"
    )
    wandb.define_metric(
        "validation_out_of_domain/loss", step_metric="Optimization Steps"
    )


def get_max_sequence_length(train_loader: DataLoader) -> int:
    return len(train_loader.dataset[0][0])


def initialize_model_optimizer_scheduler(
    config: Any, num_unique_tokens: int, seq_len: int, device: torch.device
) -> Tuple[Transformer, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    model = Transformer(
        config=config,
        num_layers=config.num_layers,
        dim_model=config.dim_model,
        num_heads=config.num_heads,
        num_tokens=num_unique_tokens,
        seq_len=seq_len,
        dropout=config.dropout,
    ).to(device)

    total_params = count_parameters(model)
    wandb.log({"total_parameters": total_params}, commit=False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=9
    )
    return model, optimizer, scheduler


def calculate_num_epochs(num_steps: int, train_loader_length: int) -> int:
    return ceil(num_steps / train_loader_length)


def track_grads(model: torch.nn.Module) -> Dict[str, float]:
    gradient_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            abs_grad_sum = torch.sum(torch.abs(param.grad)).item()
            gradient_norms[f"gradients/{name}_abs_sum"] = abs_grad_sum

    # Add total gradient norm across all parameters
    total_grad_norm = sum(gradient_norms.values())
    gradient_norms["gradients/total_abs_sum"] = total_grad_norm
    return {"gradients/total_abs_sum": total_grad_norm}


def train(
    model: Transformer,
    encoder: Encoder,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    config: Any,
) -> Dict[str, float]:
    model.train()
    criterion = get_criterion(config, encoder)

    for batch in train_loader:
        inputs, labels = [tensor.to(device) for tensor in batch]
        optimizer.zero_grad()

        if config.task_type == "classification":
            output, loss, acc = train_classification(model, inputs, labels, criterion)
        elif config.task_type == "sequence":
            output, loss, acc = train_sequence(
                model, encoder, inputs, labels, criterion, config
            )
        else:
            raise ValueError(f"Unsupported task type: {config.task_type}")

        loss.backward()

        optimizer.step()
        optimizer.training_steps += 1
        scheduler.step()

        if config.wandb_tracking != "minimal":
            gradient_norms = track_grads(model)
        else:
            gradient_norms = {}

        log_training_metrics(
            optimizer.training_steps,
            acc,
            loss,
            scheduler.get_last_lr()[0],
            gradient_norms,
        )

        if wandb.run.step >= config.num_steps:
            break

    return {"accuracy": acc.item(), "loss": loss.item()}


def get_criterion(config: Any, encoder: Encoder) -> torch.nn.Module:
    if config.task_type == "sequence":
        return torch.nn.CrossEntropyLoss(ignore_index=encoder.pad_token)
    else:
        return torch.nn.CrossEntropyLoss()


def train_classification(
    model: Transformer, inputs: Tensor, labels: Tensor, criterion: torch.nn.Module
) -> Tuple[Tensor, torch.Tensor, torch.Tensor]:
    output = model(inputs)[-1, :, :]
    loss = criterion(output, labels)
    preds = torch.argmax(output, dim=1)
    acc = (preds == labels).float().mean()
    return output, loss, acc


def train_sequence(
    model: Transformer,
    encoder: Encoder,
    inputs: Tensor,
    labels: Tensor,
    criterion: torch.nn.Module,
    config: Any,
) -> Tuple[Tensor, torch.Tensor, torch.Tensor]:
    output, target, eq_positions = get_logits(
        model, encoder, inputs
    )  # [batch_size, seq_len -1, num_tokens]
    loss, acc, _ = compute_sequence_loss_and_accuracy(
        output, target, eq_positions, criterion, config, encoder
    )
    return output, loss, acc


def get_eq_positions(inputs: Tensor, encoder: Encoder) -> Tensor:
    eq_tokens = torch.tensor(encoder.list_of_eq_tokens, device=inputs.device)
    return torch.isin(inputs, eq_tokens).nonzero(as_tuple=True)[1]


def compute_sequence_loss_and_accuracy(
    output: Tensor,
    target: Tensor,
    eq_positions: Tensor,
    criterion: torch.nn.Module,
    config: Any,
    encoder: Encoder,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size, seq_len, num_tokens = output.size()
    mask = create_mask_from_positions(
        encoder, seq_len, batch_size, eq_positions, output.device, target
    )
    masked_output = output[mask]
    masked_target = target[mask]

    loss = criterion(masked_output, masked_target)
    # loss = criterion(output.transpose(1, 2), target)
    preds = torch.argmax(masked_output, dim=1)
    # preds = torch.argmax(output.transpose(1, 2), dim=1)
    acc = (preds == masked_target).float().mean()
    return loss, acc, mask


def generate_flat_tensors(inputs: Tensor) -> Tuple[Tensor, Tensor]:
    # depending on the dimensions of the inputs, we need to flatten the inputs
    if len(inputs.shape) == 3:
        batch_size, seq_len, num_tokens = inputs.size()
        flat_inputs = inputs.contiguous().view(-1, num_tokens)
    else:
        flat_inputs = inputs.contiguous().view(-1)
    return flat_inputs


def create_mask_from_positions(
    encoder: Encoder,
    seq_len: int,
    batch_size: int,
    positions: Tensor,
    device: torch.device,
    target: Tensor,
) -> Tensor:
    """
    Creates a mask that combines position-based masking (after equals sign) with padding masking.

    Args:
        seq_len: Length of the sequence
        batch_size: Size of the batch
        positions: Tensor containing positions of equals signs
        device: Device to create mask on
        target: Target tensor to check for padding tokens

    Returns:
        Tensor: Boolean mask where True indicates positions to include in loss/accuracy
    """
    # Create position-based mask (True for positions after equals sign)
    position_mask = torch.arange(seq_len, device=device).unsqueeze(0).expand(
        batch_size, seq_len
    ) > positions.unsqueeze(1)

    # Create padding mask (True for non-padding tokens)
    padding_mask = target != encoder.pad_token

    # Combine both masks - we only care about non-padding tokens after equals sign
    return position_mask & padding_mask


def log_training_metrics(
    step: int, acc: torch.Tensor, loss: torch.Tensor, lr, gradient_norms={}
) -> None:

    metrics = {
        "training/accuracy": acc.item(),
        "training/loss": loss.item(),
        # "learning_rate": lr,
        "Optimization Steps": step,
        **gradient_norms,  # Add all gradient metrics
    }
    wandb.log(metrics, commit=False)


# training.py


def evaluate(
    model: Transformer,
    encoder: Encoder,
    val_loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    config: Any,
    validation_type: str,  # 'in_domain' or 'out_of_domain'
) -> Dict[str, float]:
    model.eval()
    criterion = get_evaluation_criterion(config, encoder)

    total_loss, total_correct, total_samples = 0.0, 0.0, 0
    examples_table = []

    if config.multitask:
        eq_token_to_task = {v: k for k, v in encoder.task_to_eq_token.items()}
        per_task_correct = defaultdict(int)
        per_task_total = defaultdict(int)

    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = [tensor.to(device) for tensor in batch]
            if config.task_type == "classification":
                loss, correct, num_samples, preds, trues = evaluate_classification(
                    model, inputs, labels, criterion
                )
                total_loss += loss * num_samples
                total_correct += correct
                total_samples += num_samples
            elif config.task_type == "sequence":
                loss, acc, num_samples, preds, trues, sequence_accuracy = (
                    evaluate_sequence(model, encoder, inputs, labels, criterion, config)
                )
                total_loss += loss.item() * num_samples
                total_correct += acc * num_samples
                total_samples += num_samples

            if config.multitask and wandb.run.step % 50 == 0:
                # Determine the task for each sample in the batch
                eq_tokens = get_eq_tokens_from_inputs(inputs, encoder)
                for eq_token, pred, true in zip(eq_tokens, preds, trues):
                    task = eq_token_to_task.get(eq_token.item(), "Unknown")
                    if config.task_type == "classification":
                        per_task_total[task] += 1
                        if pred.item() == true.item():
                            per_task_correct[task] += 1
                    elif config.task_type == "sequence":
                        per_task_total[task] += true.shape[0]
                        per_task_correct[task] += (pred == true).int().sum().item()

            examples_table.extend(
                collect_validation_examples(
                    model, encoder, inputs, labels, config, preds, trues
                )
            )

    # Calculate average loss and accuracy
    avg_loss = total_loss / len(val_loader.dataset)
    accuracy = total_correct / total_samples

    if config.multitask and wandb.run.step % 50 == 0:
        # Calculate and log per-task accuracies
        for task, correct in per_task_correct.items():
            total = per_task_total[task]
            task_accuracy = correct / total if total > 0 else 0.0
            wandb.log(
                {
                    f"validation_{validation_type}/accuracy_{task}": task_accuracy,
                    "Optimization Steps": optimizer.training_steps,
                },
                commit=False,
            )

    if wandb.run.step % 50 == 0:
        with open(f"logs/validation_examples_{validation_type}.json", "a") as f:
            json.dump({f"Step {wandb.run.step}": examples_table}, f, indent=4)
        if config.wandb_tracking != "minimal":
            log_model_parameters_wandb(model)

    # Log metrics with appropriate prefixes
    metrics = {
        f"validation_{validation_type}/accuracy": accuracy,
        f"validation_{validation_type}/loss": avg_loss,
        "Optimization Steps": optimizer.training_steps,
    }

    if config.task_type == "sequence":
        metrics[f"validation_{validation_type}/sequence_accuracy"] = sequence_accuracy

    next_step = True if validation_type == "in_domain" else False

    wandb.log(metrics, commit=next_step)

    return {"accuracy": accuracy, "loss": avg_loss}


def get_eq_tokens_from_inputs(inputs: Tensor, encoder: Encoder) -> Tensor:
    """
    Extracts the eq_token from each input sequence in a vectorized manner.

    Args:
        inputs (Tensor): Input tensor of shape [batch_size, seq_len].
        encoder (Encoder): Encoder instance containing task mappings.

    Returns:
        Tensor: Tensor of eq_token ids for each sample in the batch.
    """
    eq_token_ids = torch.tensor(
        list(encoder.task_to_eq_token.values()), device=inputs.device
    )
    # Create a mask of shape [batch_size, seq_len, num_tasks]
    mask = inputs.unsqueeze(2) == eq_token_ids.unsqueeze(0).unsqueeze(0)
    # Any position matching any eq_token
    mask_any = mask.any(dim=2)  # Shape: [batch_size, seq_len]

    # Find the first occurrence of any eq_token per sample
    # Set positions without eq_tokens to seq_len (out of bounds)
    first_eq_positions = torch.argmax(mask_any.long(), dim=1)
    # Check if any eq_token exists in the sample
    has_eq_token = mask_any.any(dim=1)

    # Gather the eq_token ids using the first_eq_positions
    eq_tokens = inputs[torch.arange(inputs.size(0)), first_eq_positions]
    # For samples without any eq_token, set to -1 or a default value
    eq_tokens = eq_tokens * has_eq_token + (-1) * (~has_eq_token)

    return eq_tokens


def get_evaluation_criterion(config: Any, encoder: Encoder) -> torch.nn.Module:
    if config.task_type == "classification":
        return torch.nn.CrossEntropyLoss()
    elif config.task_type == "sequence":
        return torch.nn.CrossEntropyLoss(ignore_index=encoder.pad_token)
    else:
        raise ValueError(f"Unsupported task type: {config.task_type}")


def evaluate_classification(
    model: Transformer, inputs: Tensor, labels: Tensor, criterion: torch.nn.Module
) -> Tuple[float, float, int]:
    output = model(inputs)[-1, :, :]
    loss = criterion(output, labels).item()
    preds = torch.argmax(output, dim=1)
    correct = (preds == labels).sum().item()
    # Get examples where the prediction is incorrect
    # incorrect_indices = (preds != labels).nonzero(as_tuple=True)[0]
    return loss, correct, len(labels), preds, labels


def get_logits(model: Transformer, encoder: Encoder, inputs: Tensor) -> Tensor:
    eq_positions = get_eq_positions(inputs, encoder)
    decoder_input = inputs[:, :-1]
    target = inputs[:, 1:]
    output = model(decoder_input).transpose(0, 1)
    return output, target, eq_positions


def evaluate_sequence(
    model: Transformer,
    encoder: Encoder,
    inputs: Tensor,
    labels: Tensor,
    criterion: torch.nn.Module,
    config: Any,
) -> Tuple[float, float, int, List[str], List[str]]:
    output, target, eq_positions = get_logits(model, encoder, inputs)
    loss, acc, mask = compute_sequence_loss_and_accuracy(
        output, target, eq_positions, criterion, config, encoder
    )

    # Get predicted tokens by taking the argmax
    preds = torch.argmax(output, dim=2)  # [batch_size, seq_len -1]

    # Calculate the number of correct predictions
    correct = acc.item() * len(target)
    num_samples = len(target)

    # Calculate the sequence accuracy
    sequence_correct = torch.all((preds == target) | ~mask, dim=1).float().mean()

    return loss, acc, num_samples, preds, target, sequence_correct


def collect_validation_examples(
    model,
    encoder,
    inputs: Tensor,
    labels: Tensor,
    config: Any,
    preds: List[str],
    trues: List[str],
) -> List[List[str]]:
    examples = []
    if config.task_type == "classification":
        # Find op and eq tokens and convert to strings
        input_example = decode_sequence(inputs[0], model.id_to_token)
        predicted_output = str(torch.argmax(model(inputs)[-1, :, :], dim=1)[0].item())
        true_output = str(labels[0].item())
    else:  # sequence
        input_example = decode_sequence(inputs[0], encoder.id_to_token)
        # Find equals token position
        eq_tokens = torch.tensor(encoder.list_of_eq_tokens, device=inputs.device)
        split_pos = torch.isin(inputs, eq_tokens).nonzero(as_tuple=True)[1][0].item()

        # Get predictions and truth after equals, excluding padding
        pred_seq = preds[0, split_pos:]
        true_seq = trues[0, split_pos:]

        # Create mask for non-padding tokens
        valid_mask = true_seq != encoder.pad_token

        # Convert only non-padding tokens to strings
        predicted_output = decode_sequence(pred_seq[valid_mask], encoder.id_to_token)
        true_output = decode_sequence(true_seq[valid_mask], encoder.id_to_token)

    match = True if predicted_output == true_output else False

    examples.append(
        {
            "example": input_example,
            "prediction": predicted_output,
            "truth": true_output,
            "match": match,
        }
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
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        operation = config.operation.replace("/", "div")
        model_filename = f"./models/best_model_{operation}.pt"
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
        )
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
    return {
        "epoch": epoch,
        "model_config": model_config,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "validation_accuracy": val_acc,
        "validation_loss": val_loss,
    }
