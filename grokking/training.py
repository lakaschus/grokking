from math import ceil
import torch
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from torch import Tensor
from typing import Dict, Any, Tuple, List
import json

from data import get_data, BINARY_TOKENS
from model import Transformer


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
    weight_norm = []
    for name, param in model.named_parameters():
        param_np = param.detach().cpu()
        weight_norm.append(torch.norm(param_np).item())
        wandb.log({f"parameters/{name}": wandb.Histogram(param_np)}, commit=False)
    wandb.log(
        {"parameters/weight_norm": sum(weight_norm) / len(weight_norm)}, commit=False
    )


def clear_logs() -> None:
    # replace logs/validation_examples.json with an empty file
    with open("logs/validation_examples_in_domain.json", "w") as f:
        f.write("")
    with open("logs/validation_examples_out_of_domain.json", "w") as f:
        f.write("")


# training.py


def main(args: Dict[str, Any]) -> None:
    clear_logs()
    initialize_wandb(args)
    config = wandb.config
    device = torch.device(config.device)

    # Retrieve DataLoaders
    (
        train_loader,
        val_in_loader,
        val_out_loader,
        op_token,
        eq_token,
        num_unique_tokens,
    ) = get_data(
        config.operation,
        max_bit_length_train=config.max_bit_length_train,  # e.g., 6
        max_bit_length_val_out=config.max_bit_length_val_out,  # e.g., 7
        training_fraction=config.training_fraction,
        batch_size=config.batch_size,
        curriculum=config.curriculum,
    )

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

    num_epochs = calculate_num_epochs(config.num_steps, len(train_loader))
    best_val_acc = 0.0

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        # Training
        train_metrics = train(model, train_loader, optimizer, scheduler, device, config)

        # In-Domain Validation
        val_in_metrics = evaluate(
            model, val_in_loader, device, optimizer, config, validation_type="in_domain"
        )

        # Out-of-Domain Validation
        if args.task_type != "classification":
            val_out_metrics = evaluate(
                model,
                val_out_loader,
                device,
                optimizer,
                config,
                validation_type="out_of_domain",
            )

        # Save the best model based on In-Domain Validation Accuracy
        best_val_acc = save_best_model(
            val_acc=val_in_metrics["accuracy"],
            best_val_acc=best_val_acc,
            model=model,
            optimizer=optimizer,
            config=config,
            epoch=epoch,
            op_token=op_token,
            eq_token=eq_token,
            max_sequence_length=max_sequence_length,
        )

        if val_in_metrics["accuracy"] >= 1.0:
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
    return gradient_norms


def train(
    model: Transformer,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    config: Any,
) -> Dict[str, float]:
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


def get_criterion(config: Any) -> torch.nn.Module:
    if config.task_type == "sequence":
        return torch.nn.CrossEntropyLoss()
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
    inputs: Tensor,
    labels: Tensor,
    criterion: torch.nn.Module,
    config: Any,
) -> Tuple[Tensor, torch.Tensor, torch.Tensor]:
    output, target, eq_positions = get_logits(
        model, inputs
    )  # [batch_size, seq_len -1, num_tokens]
    loss, acc = compute_sequence_loss_and_accuracy(
        output, target, eq_positions, criterion, config
    )
    return output, loss, acc


def get_eq_positions(inputs: Tensor) -> Tensor:
    return (inputs == BINARY_TOKENS["="]).nonzero(as_tuple=True)[1] - 1


def compute_sequence_loss_and_accuracy(
    output: Tensor,
    target: Tensor,
    eq_positions: Tensor,
    criterion: torch.nn.Module,
    config: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    return tensor[mask]


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
    val_loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    config: Any,
    validation_type: str,  # 'in_domain' or 'out_of_domain'
) -> Dict[str, float]:
    model.eval()
    criterion = get_evaluation_criterion(config)

    total_loss, total_correct, total_samples = 0.0, 0.0, 0
    examples_table = []

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
                loss, acc, num_samples, preds, trues = evaluate_sequence(
                    model, inputs, labels, criterion, config
                )
                total_loss += loss.item() * num_samples
                total_correct += acc * num_samples
                total_samples += num_samples
                # Collect validation examples
            examples_table.extend(
                collect_validation_examples(model, inputs, labels, config, preds, trues)
            )

    # Calculate average loss and accuracy
    avg_loss = total_loss / len(val_loader.dataset)
    accuracy = total_correct / total_samples

    if wandb.run.step % 5 == 0:
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

    next_step = True if validation_type == "in_domain" else False

    wandb.log(metrics, commit=next_step)

    return {"accuracy": accuracy, "loss": avg_loss}


def get_evaluation_criterion(config: Any) -> torch.nn.Module:
    if config.task_type == "classification":
        return torch.nn.CrossEntropyLoss()
    elif config.task_type == "sequence":
        # return torch.nn.CrossEntropyLoss(ignore_index=BINARY_TOKENS["<PAD>"])
        return torch.nn.CrossEntropyLoss()
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


def get_logits(model: Transformer, inputs: Tensor) -> Tensor:
    eq_positions = get_eq_positions(inputs)
    decoder_input = inputs[:, :-1]
    target = inputs[:, 1:]
    output = model(decoder_input).transpose(0, 1)
    return output, target, eq_positions


def evaluate_sequence(
    model: Transformer,
    inputs: Tensor,
    labels: Tensor,
    criterion: torch.nn.Module,
    config: Any,
) -> Tuple[float, float, int, List[str], List[str]]:
    output, target, eq_positions = get_logits(model, inputs)
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
    examples = []
    if config.task_type == "classification":
        # Find op and eq tokens and convert to strings
        input_example = decode_sequence(inputs[0], model.id_to_token)
        predicted_output = str(torch.argmax(model(inputs)[-1, :, :], dim=1)[0].item())
        true_output = str(labels[0].item())
    else:  # sequence
        input_example = decode_sequence(inputs[0], ID_TO_TOKEN)
        split_pos = (inputs[0] == BINARY_TOKENS["="]).nonzero()[0][0]
        predicted_output = decode_sequence(preds[0][split_pos:], ID_TO_TOKEN)
        true_output = decode_sequence(trues[0][split_pos:], ID_TO_TOKEN)
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


# training.py


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
