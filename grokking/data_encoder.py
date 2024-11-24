from typing import Dict, List


class Encoder:
    def __init__(self, base: int, include_operations: bool = True):
        """
        Initialize the encoder with a specific base.

        Args:
            base (int): The numerical base for encoding (e.g., 2 for binary, 10 for decimal).
            include_operations (bool): Whether to include operation tokens.
        """
        self.base = base
        self.include_operations = include_operations
        self.token_dict = self._create_token_dict()
        self.pad_token = self.token_dict["<PAD>"]
        self.eos_token = self.token_dict["<EOS>"]
        self.op_token = self.token_dict.get("+", None)  # Example operation token
        self.eq_token = self.token_dict.get("=", None)  # Example equal token
        self.id_to_token = {v: k for k, v in self.token_dict.items()}

    def _create_token_dict(self) -> Dict[str, int]:
        """
        Create a token dictionary based on the base.

        Returns:
            Dict[str, int]: Mapping from token to unique integer.
        """
        token_dict = {}
        # Add digit tokens
        for digit in range(self.base):
            token_dict[str(digit)] = digit
        current_index = self.base
        # Add operation tokens if needed
        if self.include_operations:
            operations = ["+", "=", "<EOS>", "<PAD>"]
            for op in operations:
                token_dict[op] = current_index
                current_index += 1
        else:
            token_dict["<EOS>"] = current_index
            token_dict["<PAD>"] = current_index + 1
        return token_dict

    def encode_number(self, number: int, flipped: bool = False) -> str:
        """
        Encode a number into its string representation in the specified base.

        Args:
            number (int): The number to encode.
            flipped (bool): Whether to flip the digit order.

        Returns:
            str: The encoded number as a string.
        """
        if number == 0:
            digits = "0"
        else:
            digits = ""
            n = number
            while n > 0:
                digits = str(n % self.base) + digits
                n = n // self.base
        if flipped:
            digits = digits[::-1]
        return digits

    def encode_sequence(
        self, a: int, b: int, operation: str = "+", flipped: bool = False
    ) -> List[int]:
        """
        Encode input sequences based on the specified base.

        Args:
            a (int): First operand.
            b (int): Second operand.
            operation (str): Operation symbol (e.g., "+").
            flipped (bool): Whether to flip the digit order.

        Returns:
            List[int]: Encoded input sequence.
        """
        a_str = self.encode_number(a, flipped)
        b_str = self.encode_number(b, flipped)
        sequence = (
            [self.token_dict[char] for char in a_str]
            + [self.token_dict[operation]]
            + [self.token_dict[char] for char in b_str]
            + [self.token_dict["="]]
        )
        return sequence

    def encode_label_sequence(self, c: int, flipped: bool = False) -> List[int]:
        """
        Encode label sequences based on the specified base.

        Args:
            c (int): The result number.
            flipped (bool): Whether to flip the digit order.

        Returns:
            List[int]: Encoded label sequence.
        """
        c_str = self.encode_number(c, flipped)
        sequence = [self.token_dict[char] for char in c_str] + [
            self.token_dict["<EOS>"]
        ]
        return sequence

    def get_num_unique_tokens(self) -> int:
        """
        Get the number of unique tokens in the dictionary.

        Returns:
            int: Number of unique tokens.
        """
        return len(self.token_dict)
