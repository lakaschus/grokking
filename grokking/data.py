from math import ceil
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, List
import numpy as np

DIVISION_MODULO_OPERATIONS = {
    "x/y": lambda x, y, p: ((x * y) % p, y, x),
    "x/y_binary": lambda x, y, p: ((x * y) % p, y, x),
    "x/y_binary_flipped": lambda x, y, p: ((x * y) % p, y, x),
}

ALL_MODULO_OPERATIONS = {
    "x+y_mod": lambda x, y, p: (x, y, (x + y) % p),
    "x-y_mod": lambda x, y, p: (x, y, (x - y) % p),
    "x^2+y^2_mod": lambda x, y, p: (x, y, (x**2 + y**2) % p),
    "x^3+xy+y^2_mod": lambda x, y, p: (x, y, (x**3 + x * y + y**2) % p),
    "x^3+xy+y^2+x_mod": lambda x, y, p: (x, y, (x**3 + x * y + y**2 + x) % p),
    **DIVISION_MODULO_OPERATIONS,
}

NEW_OPERATIONS = {
    "x+y": lambda x, y, _: (x, y, x + y),
    "x+y_binary": lambda x, y, _: (x, y, x + y),
    "x+y_binary_flipped": lambda x, y, _: (x, y, x + y),
    "x-y": lambda x, y, _: (x, y, x - y),
    "x,y_max": lambda x, y, _: (x, y, np.maximum(x, y)),
    "x,y_min": lambda x, y, _: (x, y, np.minimum(x, y)),
    "x,y_avg": lambda x, y, _: (x, y, (x + y) // 2),
    "x,y_sqrt": lambda x, y, _: (x, y, ((x * y) ** (1 / 2)).long()),
}

ALL_OPERATIONS = {
    **ALL_MODULO_OPERATIONS,
    **NEW_OPERATIONS,
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


def is_prime(n):
    """Check if a number is prime"""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


def get_next_prime(n):
    """Find the smallest prime number greater than or equal to n"""
    # Handle small numbers
    if n <= 2:
        return 2

    # Start with n and work upwards
    # If n is even, start with next odd number
    current = n if n % 2 == 1 else n + 1

    while True:
        if is_prime(current):
            return current
        current += 2  # Only check odd numbers


def operation_mod_p_data(
    operation: str, p: int, increment_eq_token: int = 0, increment_op_token: int = 0
) -> Tuple[Tensor, Tensor, int, int]:
    x, y = generate_cartesian_product(operation, p)
    p = get_next_prime(p)
    x, y, labels = ALL_OPERATIONS[operation](x, y, p)
    op_token, eq_token = define_tokens(labels)
    eq_token += increment_eq_token
    op_token += increment_op_token
    inputs = create_input_sequences(x, y, op_token, eq_token)
    return inputs, labels, op_token, eq_token


def generate_cartesian_product(operation: str, p: int) -> Tuple[Tensor, Tensor]:
    x = torch.arange(0, p)
    y = torch.arange(0 if operation not in DIVISION_MODULO_OPERATIONS else 1, p)
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
    out_domain: bool = False,
    min_bit_length: int = 1,
    max_bit_length: int = 6,
    flipped: bool = False,
) -> Tuple[List[List[int]], List[List[int]], int, int]:
    x, y = generate_binary_operands(out_domain, min_bit_length, max_bit_length)
    sum_xy = x + y
    inputs, labels = encode_binary_sequences(x, y, sum_xy, flipped=flipped)
    return inputs, labels, BINARY_OP_TOKEN, BINARY_EQ_TOKEN


def binary_divison_data(
    out_domain: bool = False,
    min_bit_length: int = 1,
    max_bit_length: int = 6,
    flipped: bool = False,
) -> Tuple[List[List[int]], List[List[int]], int, int]:
    x, y = generate_binary_operands(out_domain, min_bit_length, max_bit_length, y_min=1)
    p = get_next_prime(2**max_bit_length)
    xx = (x * y) % p
    z = x
    inputs, labels = encode_binary_sequences(xx, y, z, flipped=flipped)
    return inputs, labels, BINARY_OP_TOKEN, BINARY_EQ_TOKEN


def generate_binary_operands(
    out_domain: bool,
    min_bit_length: int = 1,
    max_bit_length: int = 6,
    x_min: int = 0,
    y_min: int = 0,
) -> Tuple[Tensor, Tensor]:
    if out_domain:
        x = torch.arange(x_min, 2**max_bit_length)
        y = torch.arange(y_min, 2**max_bit_length)
        y_ = y[y >= 2**min_bit_length]
        x_1, y_1 = torch.cartesian_prod(x, y_).T
        x_ = x[x >= 2**min_bit_length]
        x_2, y_2 = torch.cartesian_prod(x_, y).T
        # Concatenate x_1 and x_2
        x = torch.cat([x_1, x_2])
        # Concatenate y_1 and y_2
        y = torch.cat([y_1, y_2])
    else:
        x = torch.arange(x_min, 2**max_bit_length)
        y = torch.arange(y_min, 2**max_bit_length)
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
    increment_eq_token: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, int, int, int]:
    if "binary" in operation:
        flipped = True if "flipped" in operation else False
        if "x+y" in operation:
            task_method = binary_addition_data
        elif "x/y" in operation:
            task_method = binary_divison_data
        # Generate training and in-domain validation data
        inputs_train_val, labels_train_val, _, _ = task_method(
            max_bit_length=max_bit_length_train,
            flipped=flipped,
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
            torch.as_tensor(inputs_train_val_padded),
            torch.as_tensor(labels_train_val_padded),
        )

        # Split into training and in-domain validation
        train_dataset, val_in_dataset = split_dataset(
            dataset_train_val, training_fraction, curriculum
        )

        # Generate out-of-domain validation data
        inputs_val_out, labels_val_out, _, _ = task_method(
            out_domain=True,
            min_bit_length=max_bit_length_train,
            max_bit_length=max_bit_length_val_out,
            flipped=flipped,
        )
        inputs_val_out_padded, labels_val_out_padded = pad_binary_sequences(
            inputs_val_out, labels_val_out
        )
        dataset_val_out = TensorDataset(
            torch.as_tensor(inputs_val_out_padded),
            torch.as_tensor(labels_val_out_padded),
        )

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=(curriculum == "random")
        )
        val_in_loader = DataLoader(
            val_in_dataset, batch_size=batch_size, shuffle=(curriculum == "random")
        )
        val_out_loader = DataLoader(
            dataset_val_out, batch_size=batch_size, shuffle=(curriculum == "random")
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

    elif operation in ALL_OPERATIONS:
        # Generate out-of-domain validation data with larger modulo
        p_out = max_bit_length_val_out
        # TODO: Val Out not working for multitask because id of op and eq token change compared to train and val in set
        inputs_out, labels_out, _, _ = operation_mod_p_data(
            operation, p_out, increment_eq_token
        )

        # Generate training and validation data for modulo operations
        p_diff = max_bit_length_val_out - max_bit_length_train
        p = max_bit_length_train
        inputs, labels, op_token, eq_token = operation_mod_p_data(
            operation, p, increment_eq_token + p_diff, p_diff
        )

        # Create dataset
        dataset = TensorDataset(inputs, labels)

        train_dataset, val_in_dataset = split_dataset(
            dataset, training_fraction, curriculum
        )

        # Select only those examples that are not in the training_dataset set or val_in_dataset
        inputs_out, labels_out = select_out_of_domain_examples(
            inputs_out, labels_out, dataset
        )
        val_out_dataset = TensorDataset(inputs_out, labels_out)

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=(curriculum == "random")
        )
        val_in_loader = DataLoader(
            val_in_dataset, batch_size=batch_size, shuffle=(curriculum == "random")
        )
        val_out_loader = DataLoader(
            val_out_dataset, batch_size=batch_size, shuffle=(curriculum == "random")
        )

        num_unique_tokens = max_bit_length_val_out + 2

        return (
            train_loader,
            val_in_loader,
            val_out_loader,
            op_token,
            eq_token,
            num_unique_tokens,
        )


def select_out_of_domain_examples(
    inputs_out: Tensor, labels_out: Tensor, dataset: TensorDataset
) -> Tuple[Tensor, Tensor]:
    """
    Select examples from inputs_out and labels_out that are not present in the dataset.

    Args:
        inputs_out (Tensor): Input tensor containing out-of-domain examples
        labels_out (Tensor): Label tensor containing out-of-domain examples
        dataset (TensorDataset): Dataset containing training examples to exclude

    Returns:
        Tuple[Tensor, Tensor]: Filtered inputs and labels tensors containing only out-of-domain examples
    """
    # Convert training dataset inputs and labels to sets of tuples for efficient lookup
    training_examples = set(
        tuple(inp.tolist()) + tuple([lab.item()])
        for inp, lab in zip(dataset.tensors[0], dataset.tensors[1])
    )

    # Create masks for examples that are truly out of domain
    out_domain_masks = []
    for inp, lab in zip(inputs_out, labels_out):
        example = tuple(inp.tolist()) + tuple([lab.item()])
        out_domain_masks.append(example not in training_examples)

    # Convert mask to tensor
    mask = torch.tensor(out_domain_masks)

    # Filter inputs and labels using the mask
    filtered_inputs = inputs_out[mask]
    filtered_labels = labels_out[mask]

    return filtered_inputs, filtered_labels


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
        train_dataset = TensorDataset(
            dataset.tensors[0][:train_size], dataset.tensors[1][:train_size]
        )
        val_dataset = TensorDataset(
            dataset.tensors[0][train_size:], dataset.tensors[1][train_size:]
        )
    else:
        raise ValueError(f"Unsupported curriculum: {curriculum}")

    return train_dataset, val_dataset
