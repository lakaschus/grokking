# data.py

from math import ceil
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, List, Dict

DIVISION_MODULO_OPERATIONS = {
    "x/y": lambda x, y, p: ((x * y) % p, y, x),
}

ALL_MODULO_OPERATIONS = {
    "x+y": lambda x, y, _: (x, y, x + y),
    "x+y_binary": lambda x, y, _: (x, y, x + y),
    "x+y_binary_flipped": lambda x, y, _: (x, y, x + y),
    "x-y": lambda x, y, _: (x, y, x - y),
    **DIVISION_MODULO_OPERATIONS,
}

ALL_OPERATIONS = {
    **ALL_MODULO_OPERATIONS,
}

BINARY_TOKENS = {
    "0": 0,
    "1": 1,
    "+": 2,
    "=": 3,
    "<EOS>": 4,
    "<PAD>": 5,
}  # Example token index

# Update op_token and eq_token if necessary
BINARY_OP_TOKEN = BINARY_TOKENS["+"]
BINARY_EQ_TOKEN = BINARY_TOKENS["="]
BINARY_PAD_TOKEN = BINARY_TOKENS["<PAD>"]


def operation_mod_p_data(operation: str, p: int) -> Tuple[Tensor, Tensor, int, int]:
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
    variable_length: bool = True,
    num_samples: int = None,
    min_bit_length: int = 1,
    max_bit_length: int = 6,
    flipped: bool = False,
) -> Tuple[List[List[int]], List[List[int]], int, int]:
    x, y = generate_binary_operands(
        variable_length, num_samples, min_bit_length, max_bit_length
    )
    sum_xy = x + y
    inputs, labels = encode_binary_sequences(x, y, sum_xy, flipped=flipped)
    return inputs, labels, BINARY_OP_TOKEN, BINARY_EQ_TOKEN


def compute_bit_length(x: Tensor) -> Tensor:
    bit_length = torch.zeros_like(x)
    mask = x > 0
    # Handle x > 0
    bit_length[mask] = x[mask].floor().log2().ceil().long() + 1
    # Handle x = 0
    bit_length[~mask] = 1
    return bit_length


# data.py


def generate_binary_operands(
    variable_length: bool,
    num_samples: int,
    min_bit_length: int,
    max_bit_length: int,
) -> Tuple[Tensor, Tensor]:
    if variable_length:
        if num_samples is None:
            x = torch.arange(0, 2**max_bit_length)
            y = torch.arange(0, 2**max_bit_length)
            x_bit_length = compute_bit_length(x)
            y_bit_length = compute_bit_length(y)
            x = x[x_bit_length >= min_bit_length]
            y = y[y_bit_length >= min_bit_length]
            x, y = torch.cartesian_prod(x, y).T
        else:
            x = torch.randint(0, 2**max_bit_length, (num_samples,))
            y = torch.randint(0, 2**max_bit_length, (num_samples,))
            x_bit_length = compute_bit_length(x)
            y_bit_length = compute_bit_length(y)
            mask = (x_bit_length >= min_bit_length) & (y_bit_length >= min_bit_length)
            x = x[mask]
            y = y[mask]
    else:
        x = torch.arange(0, 2**max_bit_length)
        y = torch.arange(0, 2**max_bit_length)
        x_bit_length = compute_bit_length(x)
        y_bit_length = compute_bit_length(y)
        x = x[x_bit_length >= min_bit_length]
        y = y[y_bit_length >= min_bit_length]
        x, y = torch.cartesian_prod(x, y).T
    return x, y


def encode_binary_sequences(
    x: Tensor, y: Tensor, sum_xy: Tensor, flipped: bool = False
) -> Tuple[List[List[int]], List[List[int]]]:
    inputs = []
    labels = []
    for a, b, c in zip(x, y, sum_xy):
        a_bin = bin(a)[2:] if a != 0 else "0"
        b_bin = bin(b)[2:] if b != 0 else "0"
        c_bin = bin(c)[2:] if c != 0 else "0"

        if flipped:
            a_bin = a_bin[::-1]  # Flip the digits
            b_bin = b_bin[::-1]
            c_bin = c_bin[::-1]

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
    max_bit_length_train: int,
    max_bit_length_val_out: int,
    training_fraction: float = 0.8,
    batch_size: int = 32,
    curriculum: str = "random",
) -> Tuple[DataLoader, DataLoader, DataLoader, int, int, int]:
    if operation == "x+y_binary_flipped":
        # Generate training and in-domain validation data
        inputs_train_val, labels_train_val, _, _ = binary_addition_data(
            variable_length=True,
            num_samples=None,
            min_bit_length=1,
            max_bit_length=max_bit_length_train,
            flipped=True,
        )
        inputs_train_val_padded, labels_train_val_padded = pad_binary_sequences(
            inputs_train_val, labels_train_val
        )
        max_input_length = (
            2 * max_bit_length_val_out + (max_bit_length_val_out + 1) + 3
        )  # two summands, result which can be have one more digit, plus token, eq token, eos token
        max_label_length = (max_bit_length_val_out + 1) + 1
        inputs_train_val_padded = pad_sequence_to_length(
            inputs_train_val_padded, max_input_length
        )
        labels_train_val_padded = pad_sequence_to_length(
            labels_train_val_padded, max_label_length
        )
        dataset_train_val = TensorDataset(
            torch.tensor(inputs_train_val_padded), torch.tensor(labels_train_val_padded)
        )

        # Split into training and in-domain validation
        train_size = int(training_fraction * len(dataset_train_val))
        val_in_size = len(dataset_train_val) - train_size
        train_dataset, val_in_dataset = torch.utils.data.random_split(
            dataset_train_val, [train_size, val_in_size]
        )

        # Generate out-of-domain validation data
        inputs_val_out, labels_val_out, _, _ = binary_addition_data(
            variable_length=True,
            num_samples=None,
            min_bit_length=max_bit_length_train + 1,
            max_bit_length=max_bit_length_val_out,
            flipped=True,
        )
        inputs_val_out_padded, labels_val_out_padded = pad_binary_sequences(
            inputs_val_out, labels_val_out
        )
        dataset_val_out = TensorDataset(
            torch.tensor(inputs_val_out_padded), torch.tensor(labels_val_out_padded)
        )

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=(curriculum == "random")
        )
        val_in_loader = DataLoader(val_in_dataset, batch_size=batch_size, shuffle=False)
        val_out_loader = DataLoader(
            dataset_val_out, batch_size=batch_size, shuffle=False
        )

        num_unique_tokens = BINARY_TOKENS["<PAD>"] + 1
        op_token = BINARY_OP_TOKEN
        eq_token = BINARY_EQ_TOKEN

        return (
            train_loader,
            val_in_loader,
            val_out_loader,
            op_token,
            eq_token,
            num_unique_tokens,
        )
    else:
        raise ValueError(f"Unsupported operation: {operation}")


def pad_sequence_to_length(
    padded, max_length, batch_first=True, padding_value=BINARY_TOKENS["<PAD>"]
):

    # If padded sequences are shorter than max_length, pad further
    if padded.shape[1] < max_length:
        pad_size = max_length - padded.shape[1]
        if batch_first:
            padded = torch.nn.functional.pad(padded, (0, pad_size), value=padding_value)
        else:
            padded = torch.nn.functional.pad(
                padded, (0, 0, 0, pad_size), value=padding_value
            )

    # If padded sequences are longer than max_length, truncate
    elif padded.shape[1] > max_length:
        if batch_first:
            padded = padded[:, :max_length]
        else:
            padded = padded[:max_length, :]

    return padded


def pad_binary_sequences(
    inputs: List[List[int]], labels: List[List[int]]
) -> Tuple[Tensor, Tensor]:
    concatenated_inputs = [
        torch.cat((torch.tensor(input_t), torch.tensor(label_t)))
        for input_t, label_t in zip(inputs, labels)
    ]
    concatenated_inputs_padded = pad_sequence(
        concatenated_inputs, batch_first=True, padding_value=BINARY_TOKENS["<PAD>"]
    )
    labels_padded = pad_sequence(
        [torch.tensor(seq) for seq in labels],
        batch_first=True,
        padding_value=BINARY_TOKENS["<PAD>"],
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
