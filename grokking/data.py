# data.py

from math import ceil
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, List, Dict

DIVISION_MODULO_OPERATIONS = {
    "x/y": lambda x, y, p: (x * y % p, y, x),
}

ALL_MODULO_OPERATIONS = {
    "x+y": lambda x, y, _: (x, y, x + y),
    "x+y_binary": lambda x, y, _: (x, y, x + y),
    "x-y": lambda x, y, _: (x, y, x - y),
    **DIVISION_MODULO_OPERATIONS,
}

ALL_OPERATIONS = {
    **ALL_MODULO_OPERATIONS,
}

BINARY_TOKENS = {"0": 0, "1": 1, "+": 2, "=": 3, "<EOS>": 4}  # Example token index

# Update op_token and eq_token if necessary
BINARY_OP_TOKEN = BINARY_TOKENS["+"]
BINARY_EQ_TOKEN = BINARY_TOKENS["="]


def operation_mod_p_data(operation: str, p: int) -> Tuple[Tensor, Tensor, int, int]:
    """
    Generate data for operations involving modulo.

    Args:
        operation (str): The operation to perform.
        p (int): The modulo value.

    Returns:
        Tuple containing inputs, labels, op_token, and eq_token.
    """
    x, y = generate_cartesian_product(p)
    x, y, labels = ALL_OPERATIONS[operation](x, y, p)
    op_token, eq_token = define_tokens(labels)
    inputs = create_input_sequences(x, y, op_token, eq_token)
    return inputs, labels, op_token, eq_token


def generate_cartesian_product(p: int) -> Tuple[Tensor, Tensor]:
    x = torch.arange(0, p)
    y = torch.arange(0, p)
    return torch.cartesian_prod(x, y).T


def define_tokens(labels: Tensor) -> Tuple[int, int]:
    op_token = torch.max(labels).item() + 1
    eq_token = op_token + 1
    return op_token, eq_token


def create_input_sequences(
    x: Tensor, y: Tensor, op_token: int, eq_token: int
) -> Tensor:
    eq = torch.ones_like(x) * eq_token
    op = torch.ones_like(x) * op_token
    return torch.stack([x, op, y, eq], dim=1)


def binary_addition_data(
    variable_length: bool = True, num_samples: int = None, max_bit_length: int = 8
) -> Tuple[List[List[int]], List[List[int]], int, int]:
    """
    Generate binary addition data with variable lengths (no leading zeros).

    Args:
        variable_length (bool): If True, generate variable-length binary numbers.
        num_samples (int, optional): Number of samples to generate.
                                     If None, generate all possible combinations up to a certain bit length.
        max_bit_length (int): Maximum bit length for binary numbers.

    Returns:
        Tuple containing input sequences, label sequences, op_token, and eq_token.
    """
    x, y = generate_binary_operands(variable_length, num_samples, max_bit_length)
    sum_xy = x + y
    inputs, labels = encode_binary_sequences(x, y, sum_xy)
    return inputs, labels, BINARY_OP_TOKEN, BINARY_EQ_TOKEN


def generate_binary_operands(
    variable_length: bool, num_samples: int, max_bit_length: int
) -> Tuple[Tensor, Tensor]:
    if variable_length:
        if num_samples is None:
            x = torch.arange(0, 2**max_bit_length)
            y = torch.arange(0, 2**max_bit_length)
            x, y = torch.cartesian_prod(x, y).T
        else:
            x = torch.randint(0, 2**max_bit_length, (num_samples,))
            y = torch.randint(0, 2**max_bit_length, (num_samples,))
    else:
        x = torch.arange(0, 2**max_bit_length)
        y = torch.arange(0, 2**max_bit_length)
        x, y = torch.cartesian_prod(x, y).T
    return x, y


def encode_binary_sequences(
    x: Tensor, y: Tensor, sum_xy: Tensor
) -> Tuple[List[List[int]], List[List[int]]]:
    inputs = []
    labels = []
    for a, b, c in zip(x, y, sum_xy):
        a_bin = bin(a)[2:] if a != 0 else "0"
        b_bin = bin(b)[2:] if b != 0 else "0"
        c_bin = bin(c)[2:] if c != 0 else "0"

        input_seq = encode_sequence(a_bin, b_bin)
        label_seq = encode_label_sequence(c_bin)
        inputs.append(input_seq)
        labels.append(label_seq)
    return inputs, labels


def encode_sequence(a_bin: str, b_bin: str) -> List[int]:
    return (
        [BINARY_TOKENS[bit] for bit in a_bin]
        + [BINARY_TOKENS["+"]]
        + [BINARY_TOKENS[bit] for bit in b_bin]
        + [BINARY_TOKENS["="]]
    )


def encode_label_sequence(c_bin: str) -> List[int]:
    return [BINARY_TOKENS[bit] for bit in c_bin] + [BINARY_TOKENS["<EOS>"]]


def get_data(
    operation: str,
    max_number: int = 10,
    training_fraction: float = 0.8,
    batch_size: int = 32,
    curriculum: str = "random",
) -> Tuple[DataLoader, DataLoader, int, int, int]:
    """
    Get data loaders for the specified operation.

    Args:
        operation (str): The operation to perform ('x+y', 'x+y_binary', etc.).
        max_number (int, optional): Max number for modulo operations or bit length for binary addition.
        training_fraction (float): Fraction of data to use for training.
        batch_size (int): Batch size for data loaders.
        curriculum (str): Curriculum learning strategy.

    Returns:
        Tuple containing train_loader, val_loader, op_token, eq_token, num_unique_tokens
    """
    if operation in ALL_OPERATIONS and "binary" not in operation:
        inputs, labels, op_token, eq_token = operation_mod_p_data(operation, max_number)
        num_unique_tokens = eq_token + 1
        dataset = TensorDataset(inputs, labels)
    elif "binary" in operation:
        inputs, labels, op_token, eq_token = binary_addition_data(
            variable_length=True, num_samples=None, max_bit_length=max_number
        )
        inputs_padded, labels_padded = pad_binary_sequences(inputs, labels)
        dataset = TensorDataset(
            torch.tensor(inputs_padded), torch.tensor(labels_padded)
        )
        num_unique_tokens = BINARY_TOKENS["<EOS>"] + 1
        op_token = BINARY_OP_TOKEN
        eq_token = BINARY_EQ_TOKEN
    else:
        raise ValueError(f"Unsupported operation: {operation}")

    return (
        create_data_loaders(dataset, training_fraction, batch_size, curriculum),
        op_token,
        eq_token,
        num_unique_tokens,
    )


def pad_binary_sequences(
    inputs: List[List[int]], labels: List[List[int]]
) -> Tuple[Tensor, Tensor]:
    concatenated_inputs = [
        torch.cat((torch.tensor(input_t), torch.tensor(label_t)))
        for input_t, label_t in zip(inputs, labels)
    ]
    concatenated_inputs_padded = pad_sequence(
        concatenated_inputs, batch_first=True, padding_value=BINARY_TOKENS["<EOS>"]
    )
    labels_padded = pad_sequence(
        [torch.tensor(seq) for seq in labels],
        batch_first=True,
        padding_value=BINARY_TOKENS["<EOS>"],
    )
    return concatenated_inputs_padded, labels_padded


def create_data_loaders(
    dataset: TensorDataset, training_fraction: float, batch_size: int, curriculum: str
) -> Tuple[DataLoader, DataLoader, int, int, int]:
    train_loader, val_loader, op_token, eq_token = split_dataset(
        dataset, training_fraction, curriculum
    )
    num_unique_tokens = eq_token + 1
    batch_size = min(batch_size, ceil(len(dataset) / 2))
    return (
        DataLoader(
            train_loader, batch_size=batch_size, shuffle=(curriculum == "random")
        ),
        DataLoader(val_loader, batch_size=batch_size, shuffle=(curriculum == "random")),
        op_token,
        eq_token,
        num_unique_tokens,
    )


def split_dataset(
    dataset: TensorDataset, training_fraction: float, curriculum: str
) -> Tuple[TensorDataset, TensorDataset, int, int]:
    train_size = int(training_fraction * len(dataset))
    val_size = len(dataset) - train_size

    if curriculum == "random":
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
    elif curriculum == "ordered":
        train_dataset, val_dataset = dataset[:train_size], dataset[train_size:]
    elif curriculum == "domain_separated":
        # Implement domain separation logic if applicable
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
    else:
        raise ValueError(f"Unsupported curriculum: {curriculum}")

    # Placeholder tokens; adjust as necessary
    op_token = 0
    eq_token = 0

    return train_dataset, val_dataset, op_token, eq_token
