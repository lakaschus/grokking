import torch
from torch import Tensor
from typing import Tuple, List

DIVISION_MODULO_OPERATIONS = {
    "x/y": lambda x, y, p: ((x * y) % p, y, x),
    "x/y_binary": lambda x, y, p: ((x * y) % p, y, x),
    "x/y_binary_flipped": lambda x, y, p: ((x * y) % p, y, x),
}

ALL_MODULO_OPERATIONS = {
    "x+y_mod": lambda x, y, p: (x, y, (x + y) % p),
    "x-y_mod": lambda x, y, p: (x, y, (x - y) % p),
    "x^2+y^2_mod": lambda x, y, p: (x, y, (x**2 + y**2) % p),
    "3x^2+y^2": lambda x, y, p: (x, y, (3 * (x**2) + y**2)),
    "3x^2+y^2_binary": lambda x, y, p: (x, y, (3 * (x**2) + y**2)),
    "3x^2+y^2_binary_flipped": lambda x, y, p: (x, y, (3 * (x**2) + y**2)),
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
}


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
