from math import ceil
import torch

DIVISION_MODULO_OPERATIONS = {
    "x/y": lambda x, y, p: (x * y % p, y, x),
}

ALL_MODULO_OPERATIONS = {
    "x+y": lambda x, y, _: (x, y, x + y),
    "x-y": lambda x, y, _: (x, y, x - y),
    **DIVISION_MODULO_OPERATIONS,
}

ALL_OPERATIONS = {
    **ALL_MODULO_OPERATIONS,
}


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


def get_data(operation: str, prime: int, training_fraction: float, batch_size: int):
    inputs, labels, op_token, eq_token = operation_mod_p_data(operation, prime)
    dataset = torch.utils.data.TensorDataset(inputs, labels)

    train_size = int(training_fraction * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    batch_size = min(batch_size, ceil(len(dataset) / 2))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )

    return train_loader, val_loader, op_token, eq_token
