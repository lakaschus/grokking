from math import ceil
import torch
from torch.nn.utils.rnn import pad_sequence

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

binary_tokens = {"0": 0, "1": 1, "+": 2, "=": 3, "<EOS>": 4}  # Example token index

# Update op_token and eq_token if necessary
binary_op_token = binary_tokens["+"]
binary_eq_token = binary_tokens["="]


def operation_mod_p_data(operation: str, p: int):
    """
    x◦y (mod p) for 0 <= x < p, 1 <= y < p if operation in DIVISION_MODULO_OPERATIONS
    x◦y (mod p) for 0 <= x, y < p otherwise
    """
    x = torch.arange(0, p)
    y = torch.arange(0, p)
    x, y = torch.cartesian_prod(x, y).T

    x, y, labels = ALL_OPERATIONS[operation](x, y, p)

    op_token = torch.max(labels) + 1
    eq_token = op_token + 1
    eq = torch.ones_like(x) * eq_token
    op = torch.ones_like(x) * op_token

    inputs = torch.stack([x, op, y, eq], dim=1)

    return inputs, labels, op_token, eq_token


def binary_addition_data(variable_length: bool = True, num_samples: int = None):
    """
    Generate binary addition data with variable lengths (no leading zeros).

    Args:
        variable_length (bool): If True, generate variable-length binary numbers.
        num_samples (int, optional): Number of samples to generate.
                                     If None, generate all possible combinations up to a certain bit length.

    Returns:
        inputs (List[List[int]]): Token sequences for inputs.
        labels (List[List[int]]): Token sequences for labels.
    """
    if variable_length:
        if num_samples is None:
            # Define a maximum bit length for practical purposes
            max_bit_length = 8  # Adjust as needed
            x = torch.arange(0, 2**max_bit_length)
            y = torch.arange(0, 2**max_bit_length)
            x, y = torch.cartesian_prod(x, y).T
        else:
            # Randomly sample numbers with varying bit lengths
            max_bit_length = 8  # Adjust as needed
            x = torch.randint(0, 2**max_bit_length, (num_samples,))
            y = torch.randint(0, 2**max_bit_length, (num_samples,))
    else:
        # Fixed bit length (for reference)
        max_bit_length = 8
        x = torch.arange(0, 2**max_bit_length)
        y = torch.arange(0, 2**max_bit_length)
        x, y = torch.cartesian_prod(x, y).T

    sum_xy = x + y

    inputs = []
    labels = []

    for a, b, c in zip(x, y, sum_xy):
        # Convert to binary strings without leading zeros
        a_bin = bin(a)[2:] if a != 0 else "0"
        b_bin = bin(b)[2:] if b != 0 else "0"
        c_bin = bin(c)[2:] if c != 0 else "0"

        # Encode input sequence: a_bin + '+' + b_bin + '=' + <EOS>
        input_seq = (
            [binary_tokens[bit] for bit in a_bin]
            + [binary_tokens["+"]]
            + [binary_tokens[bit] for bit in b_bin]
            + [binary_tokens["="]]
        )

        # Encode label sequence: c_bin + <EOS>
        label_seq = [binary_tokens[bit] for bit in c_bin] + [binary_tokens["<EOS>"]]

        inputs.append(input_seq)
        labels.append(label_seq)

    return inputs, labels, binary_tokens["+"], binary_tokens["="]


def get_data(
    operation: str,
    max_number: int = 10,
    training_fraction: float = 0.8,
    batch_size: int = 32,
    curriculum: str = "random",
):
    """
    Get data loaders for the specified operation.

    Args:
        operation (str): The operation to perform ('x+y', 'x+y_binary', etc.).
        max_number (int, optional): max_number number for modulo operations.
        max_number (int, optional): Bit length for binary addition.
        training_fraction (float): Fraction of data to use for training.
        batch_size (int): Batch size for data loaders.
        curriculum (str): Curriculum learning strategy.

    Returns:
        train_loader, val_loader, op_token, eq_token
    """
    if operation in ALL_OPERATIONS and "binary" not in operation:
        # Handle existing operations
        if max_number is None:
            raise ValueError("max_number must be specified for modulo operations.")
        inputs, labels, op_token, eq_token = operation_mod_p_data(operation, max_number)
        dataset = torch.utils.data.TensorDataset(inputs, labels)
        num_unique_tokens = eq_token + 1
    elif "binary" in operation:
        if max_number is None:
            raise ValueError("Bit length must be specified for binary addition.")
        inputs, labels, op_token, eq_token = binary_addition_data(max_number)

        input_tensors = [torch.tensor(seq) for seq in inputs]
        label_tensors = [torch.tensor(seq) for seq in labels]
        concatenated_inputs = [
            torch.cat((input_t, label_t))
            for input_t, label_t in zip(input_tensors, label_tensors)
        ]

        concatenated_inputs_padded = pad_sequence(
            concatenated_inputs, batch_first=True, padding_value=binary_tokens["<EOS>"]
        )
        labels_padded = pad_sequence(
            label_tensors, batch_first=True, padding_value=binary_tokens["<EOS>"]
        )

        dataset = torch.utils.data.TensorDataset(
            concatenated_inputs_padded, labels_padded
        )
        num_unique_tokens = binary_tokens["<EOS>"] + 1
    else:
        raise ValueError(f"Unsupported operation: {operation}")

    train_size = int(training_fraction * len(dataset))
    val_size = len(dataset) - train_size

    if curriculum == "random":
        shuffle = True
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
    elif curriculum == "ordered":
        shuffle = False
        train_dataset = dataset[:train_size]
        val_dataset = dataset[train_size:]
    elif curriculum == "domain_separated":
        if "binary" not in operation:
            # Existing domain separated logic
            shuffle = False
            threshold = ceil(max_number * training_fraction)
            x = inputs[:, 0]
            y = inputs[:, 2]
            mask_train = (x < threshold) & (y < threshold)
            mask_val = ~mask_train
            train_dataset = torch.utils.data.TensorDataset(
                inputs[mask_train], labels[mask_train]
            )
            val_dataset = torch.utils.data.TensorDataset(
                inputs[mask_val], labels[mask_val]
            )
        else:
            # For binary addition, use a different strategy or default to random
            shuffle = True
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size]
            )
    else:
        raise ValueError(f"Unsupported curriculum: {curriculum}")

    batch_size = min(batch_size, ceil(len(dataset) / 2))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=shuffle
    )

    return train_loader, val_loader, op_token, eq_token, num_unique_tokens
