import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, List
import numpy as np
from data_encoder import Encoder

DIVISION_MODULO_OPERATIONS = {
    "x/y_mod": lambda x, y, p: ((x * y) % p, y, x),
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
    "xy_abs_max": lambda x, y, _: (x, y, np.abs(np.maximum(x, y) - 1)),
    "xy_abs_min": lambda x, y, _: (x, y, np.abs(np.minimum(x, y) - 1)),
    "xy_avg": lambda x, y, _: (x, y, (x + y) // 2),
    "xy_sqrt": lambda x, y, _: (x, y, ((x * y) ** (1 / 2)).long()),
    "xy_abs_diff": lambda x, y, _: (x, y, np.abs(x - y)),
    # "x,y_abs_todo": lambda x, y, _: (x, y, np.maximum(x, y) + np.abs(x - y)),
    # "x,y_complex_program_1": lambda x, y, _: (x, y, x // 10 if x % 10 == 0 ... TODO),
}

ALL_OPERATIONS = {
    **ALL_MODULO_OPERATIONS,
    **NEW_OPERATIONS,
}

BINARY_ENCODER = Encoder(base=2)
BINARY_OP_TOKEN = BINARY_ENCODER.op_token
BINARY_EQ_TOKEN = BINARY_ENCODER.eq_token
BINARY_EOS_TOKEN = BINARY_ENCODER.eos_token
BINARY_PAD_TOKEN = BINARY_ENCODER.pad_token

BINARY_TOKENS = {
    "0": 0,
    "1": 1,
    "+": BINARY_OP_TOKEN,
    "=": BINARY_EQ_TOKEN,
    "<EOS>": BINARY_EOS_TOKEN,
    "<PAD>": BINARY_PAD_TOKEN,
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


def sequence_task_data(
    operation: str,
    out_domain: bool = False,
    min_bit_length: int = 1,
    max_bit_length: int = 6,
    flipped: bool = False,
    encoder: Encoder = BINARY_ENCODER,
    fixed_sequence_length: bool = False,
    increment_eq_token: int = 0,
    increment_op_token: int = 0,
) -> Tuple[List[List[int]], List[List[int]], int, int]:
    x, y = generate_binary_operands(out_domain, min_bit_length, max_bit_length, y_min=1)
    p = get_next_prime(2**max_bit_length)
    x, y, z = ALL_OPERATIONS[operation](x, y, p)
    inputs, labels = encode_generic_sequences(
        x,
        y,
        z,
        encoder=encoder,
        flipped=flipped,
        fixed_sequence_length=fixed_sequence_length,
    )
    return inputs, labels, encoder.op_token, encoder.eq_token


def operation_mod_p_data(
    operation: str,
    max_number: int,
    p: int,
    increment_eq_token: int = 0,
    increment_op_token: int = 0,
) -> Tuple[Tensor, Tensor, int, int]:
    x, y = generate_cartesian_product(operation, max_number)
    p = get_next_prime(p)
    x, y, labels = ALL_OPERATIONS[operation](x, y, p)
    op_token, eq_token = define_tokens(p)
    eq_token += increment_eq_token
    op_token += increment_op_token
    inputs = create_input_sequences(x, y, op_token, eq_token)
    return inputs, labels, op_token, eq_token


def generate_cartesian_product(operation: str, p: int) -> Tuple[Tensor, Tensor]:
    x = torch.arange(0, p)
    y = torch.arange(0 if operation not in DIVISION_MODULO_OPERATIONS else 1, p)
    return torch.cartesian_prod(x, y).T


def define_tokens(p: int) -> Tuple[int, int]:
    op_token = p + 1
    eq_token = op_token + 1
    return op_token, eq_token


def create_input_sequences(
    x: Tensor, y: Tensor, op_token: int, eq_token: int
) -> Tensor:
    eq = torch.ones_like(x) * eq_token
    op = torch.ones_like(x) * op_token
    return torch.stack([x, op, y, eq], dim=1)


def encode_generic_sequences(
    x: Tensor,
    y: Tensor,
    result: Tensor,
    encoder: Encoder,
    flipped: bool = False,
    fixed_sequence_length: bool = False,
) -> Tuple[List[List[int]], List[int]]:
    inputs = []
    labels = []
    for a, b, c in zip(x, y, result):
        # Encode the operation sequence: a + b = c
        operation_seq = encoder.encode_sequence(
            a.item(),
            b.item(),
            operation="+",
            flipped=flipped,
            fixed_sequence_length=fixed_sequence_length,
        )
        # Append the encoded answer to the input sequence
        answer_seq = encoder.encode_label_sequence(
            c.item(), flipped=flipped, fixed_sequence_length=fixed_sequence_length
        )
        input_seq = operation_seq + answer_seq
        # Labels are the answer sequence only
        label_seq = answer_seq
        inputs.append(input_seq)
        labels.append(label_seq)
    return inputs, labels


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
    return encode_generic_sequences(
        x, y, sum_xy, encoder=BINARY_ENCODER, flipped=flipped
    )


def encode_sequence(a_bin: str, b_bin: str) -> List[int]:
    return (
        [BINARY_TOKENS[bit] for bit in a_bin]
        + [BINARY_TOKENS["+"]]
        + [BINARY_TOKENS[bit] for bit in b_bin]
        + [BINARY_TOKENS["="]]
    )


def encode_label_sequence(c_bin: str) -> List[int]:
    return [BINARY_TOKENS[bit] for bit in c_bin] + [BINARY_TOKENS["<EOS>"]]


def pad_generic_sequences(
    inputs: List[List[int]], labels: List[List[int]], encoder: Encoder
) -> Tuple[Tensor, Tensor]:
    """
    Pad input and label sequences based on the encoder's requirements.

    Args:
        inputs (List[List[int]]): List of input sequences.
        labels (List[List[int]]): List of label sequences.
        encoder (Encoder): Encoder instance.

    Returns:
        Tuple[Tensor, Tensor]: Padded input and label tensors.
    """
    # Determine maximum lengths based on the base and task
    max_input_length = max(len(seq) for seq in inputs)
    max_label_length = max(len(seq) for seq in labels)

    inputs_padded = pad_sequence(
        [torch.tensor(seq) for seq in inputs],
        batch_first=True,
        padding_value=encoder.pad_token,
    )
    labels_padded = pad_sequence(
        [torch.tensor(seq) for seq in labels],
        batch_first=True,
        padding_value=encoder.pad_token,
    )
    # Ensure all sequences are of the same length
    inputs_padded = pad_sequence_to_length(
        inputs_padded,
        max_input_length,
        batch_first=True,
        padding_value=encoder.pad_token,
    )
    labels_padded = pad_sequence_to_length(
        labels_padded,
        max_label_length,
        batch_first=True,
        padding_value=encoder.pad_token,
    )

    return inputs_padded, labels_padded


def get_data(
    operation: str,
    task_type: str,
    max_bit_length_train: int,
    max_bit_length_val_out: int,
    base: int = 2,
    training_fraction: float = 0.8,
    batch_size: int = 32,
    curriculum: str = "random",
    increment_eq_token: int = 0,
    fixed_sequence_length: bool = False,  # New parameter
) -> Tuple[DataLoader, DataLoader, DataLoader, int, int, int]:
    """
    Generate DataLoaders for training, in-domain validation, and out-of-domain validation.

    Args:
        operation (str): The operation to perform (e.g., "x+y_binary", "x+y").
        max_bit_length_train (int): Maximum bit length for training data.
        max_bit_length_val_out (int): Maximum bit length for out-of-domain validation data.
        base (int): Numerical base for encoding (default is 2 for binary).
        training_fraction (float): Fraction of data to use for training.
        batch_size (int): Batch size for DataLoaders.
        curriculum (str): Curriculum strategy ("random" or "ordered").
        increment_eq_token (int): Token increment for the equal sign.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader, int, int, int]: DataLoaders and token info.
    """

    if task_type == "sequence":
        encoder = Encoder(
            base=base, max_number=get_next_prime(2**max_bit_length_val_out)
        )
        if increment_eq_token != 0:
            encoder.increment_eq_token(increment_eq_token)
        flipped = "flipped" in operation

        # Generate training and in-domain validation data
        inputs_train_val, labels_train_val, _, _ = sequence_task_data(
            operation=operation,
            out_domain=False,
            min_bit_length=1,
            max_bit_length=max_bit_length_train,
            flipped=flipped,
            encoder=encoder,
            fixed_sequence_length=fixed_sequence_length,
        )
        inputs_train_val_padded, labels_train_val_padded = pad_generic_sequences(
            inputs_train_val, labels_train_val, encoder=encoder
        )

        max_digits = encoder.theoretical_max_number_of_digits()
        max_input_length = (
            3 * max_digits + 3
        )  # two summands, result which can be have one more digit, plus token, eq token, eos token
        max_label_length = max_digits  # + 1

        inputs_train_val_padded = pad_sequence_to_length(
            inputs_train_val_padded, max_input_length, padding_value=encoder.pad_token
        )
        labels_train_val_padded = pad_sequence_to_length(
            labels_train_val_padded, max_label_length, padding_value=encoder.pad_token
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
        inputs_val_out, labels_val_out, _, _ = sequence_task_data(
            operation=operation,
            out_domain=True,
            min_bit_length=max_bit_length_train,
            max_bit_length=max_bit_length_val_out,
            flipped=flipped,
            encoder=encoder,
            fixed_sequence_length=fixed_sequence_length,
        )
        inputs_val_out_padded, labels_val_out_padded = pad_generic_sequences(
            inputs_val_out, labels_val_out, encoder=encoder
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

        num_unique_tokens = encoder.get_num_unique_tokens()
        op_token = encoder.op_token
        eq_token = encoder.eq_token

        return (
            train_loader,
            val_in_loader,
            val_out_loader,
            op_token,
            eq_token,
            num_unique_tokens,
            encoder,
        )

    elif operation in ALL_OPERATIONS:
        encoder = Encoder(
            base=get_next_prime(max_bit_length_val_out),
            max_number=get_next_prime(max_bit_length_val_out),
        )
        if increment_eq_token != 0:
            encoder.increment_eq_token(increment_eq_token)
        # Generate out-of-domain validation data with larger modulo
        p_out = max_bit_length_val_out
        # TODO: Val Out not working for multitask because id of op and eq token changed compared to train and val in set
        inputs_out, labels_out, _, _ = operation_mod_p_data(
            operation, max_bit_length_val_out, p_out, increment_eq_token
        )

        # Generate training and validation data for modulo operations
        inputs, labels, op_token, eq_token = operation_mod_p_data(
            operation,
            max_bit_length_train,
            p_out,
            increment_eq_token,
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

        num_unique_tokens = eq_token + 1

        return (
            train_loader,
            val_in_loader,
            val_out_loader,
            op_token,
            eq_token,
            num_unique_tokens,
            encoder,
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
