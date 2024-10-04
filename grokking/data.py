from math import ceil
import torch

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


binary_tokens = {
    "0": 0,  # Example token index
    "1": 1,
    "+": 2,
    "=": 3,
}

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


def binary_addition_data(max_number: int, num_samples: int = None):
    """
    Generate binary addition data with fixed max_number.

    Args:
        max_number (int): Number of bits for each binary number.
        num_samples (int, optional): Number of samples to generate.
                                     If None, generate all possible combinations.

    Returns:
        inputs (List[List[int]]): Token sequences for inputs.
        labels (List[List[int]]): Token sequences for labels.
    """

    if num_samples is None:
        # Generate all possible combinations within max_number
        x = torch.arange(0, 2**max_number)
        y = torch.arange(0, 2**max_number)
    else:
        # Randomly sample
        x = torch.randint(0, 2**max_number, (num_samples,))
        y = torch.randint(0, 2**max_number, (num_samples,))

    sum_xy = x + y

    inputs = []
    labels = []

    for a, b, c in zip(x, y, sum_xy):
        # Convert to binary strings with leading zeros
        a_bin = format(a, f"0{max_number}b")
        b_bin = format(b, f"0{max_number}b")
        c_bin = format(c, f"0{max_number + 1}b")  # +1 for possible carry

        # Encode input sequence: a_bin + '+' + b_bin + '='
        input_seq = (
            [binary_tokens[bit] for bit in a_bin]
            + [binary_op_token]
            + [binary_tokens[bit] for bit in b_bin]
            + [binary_eq_token]
        )

        # Encode label sequence: c_bin
        label_seq = [binary_tokens[bit] for bit in c_bin]

        inputs.append(input_seq)
        labels.append(label_seq)

    return inputs, labels


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
    elif "binary" in operation:
        if max_number is None:
            raise ValueError("Bit length must be specified for binary addition.")
        inputs, labels = binary_addition_data(max_number)
        # For binary addition, define op_token and eq_token if not already defined
        # Assuming they are already defined globally
    else:
        raise ValueError(f"Unsupported operation: {operation}")

    if "binary" not in operation:
        dataset = torch.utils.data.TensorDataset(inputs, labels)
    else:
        # For binary addition, inputs and labels are lists of token sequences
        # Convert them to tensors with padding
        from torch.nn.utils.rnn import pad_sequence

        # Convert lists to tensors
        input_tensors = [torch.tensor(seq) for seq in inputs]
        label_tensors = [torch.tensor(seq) for seq in labels]

        # Pad sequences
        inputs_padded = pad_sequence(input_tensors, batch_first=True, padding_value=0)
        labels_padded = pad_sequence(label_tensors, batch_first=True, padding_value=0)

        dataset = torch.utils.data.TensorDataset(inputs_padded, labels_padded)

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

    return train_loader, val_loader, op_token, eq_token
