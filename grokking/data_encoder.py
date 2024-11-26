from typing import Dict, List


class Encoder:
    def __init__(
        self, base: int, max_number: int = 97, include_operations: bool = True
    ):
        """
        Initialize the encoder with a specific base.

        Args:
            base (int): The numerical base for encoding (e.g., 2 for binary, 10 for decimal).
            include_operations (bool): Whether to include operation tokens.
        """
        self.base = base
        self.max_number = max_number
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

    def encode_number(
        self, number: int, flipped: bool = False, fixed_sequence_length: bool = False
    ) -> List[int]:
        """
        Encode a number into its digit representation based on the base.
        Optionally pad with leading zeros to a fixed length.

        Args:
            number (int): The number to encode.
            flipped (bool): Whether to flip the digit order.
            fixed_length (Optional[int]): Fixed length to pad the digits.

        Returns:
            List[int]: List of encoded digits.
        """
        digits = []
        if number == 0:
            digits = [0]
        else:
            n = number
            while n > 0:
                digits.insert(0, n % self.base)
                n = n // self.base
        if flipped:
            digits = digits[::-1]
        if fixed_sequence_length:
            length = self.theoretical_max_number_of_digits()
            if len(digits) < length:
                # Pad with leading zeros
                digits = [0] * (length - len(digits)) + digits
            elif len(digits) > length:
                # Truncate the digits if they exceed length
                digits = digits[-length:]
        return digits

    def encode_sequence(
        self,
        a: int,
        b: int,
        operation: str = "+",
        flipped: bool = False,
        fixed_sequence_length: bool = False,
    ) -> List[int]:
        a_digits = self.encode_number(a, flipped, fixed_sequence_length)
        b_digits = self.encode_number(b, flipped, fixed_sequence_length)
        sequence = (
            [self.token_dict[str(digit)] for digit in a_digits]
            + [self.token_dict[operation]]
            + [self.token_dict[str(digit)] for digit in b_digits]
            + [self.token_dict["="]]
        )
        return sequence

    def encode_label_sequence(
        self, c: int, flipped: bool = False, fixed_sequence_length: bool = False
    ) -> List[int]:
        c_digits = self.encode_number(c, flipped, fixed_sequence_length)

        sequence = [self.token_dict[str(digit)] for digit in c_digits]
        if not fixed_sequence_length:
            sequence = sequence + [self.token_dict["<EOS>"]]
        return sequence

    def get_num_unique_tokens(self) -> int:
        """
        Get the number of unique tokens in the dictionary.

        Returns:
            int: Number of unique tokens.
        """
        return len(self.token_dict)

    def theoretical_max_number_of_digits(self) -> int:
        """
        Calculate the maximum number of digits needed to represent self.max_number in the current base.

        This implementation avoids floating-point precision issues by using integer division.

        Returns:
            int: Maximum number of digits needed
        """
        if self.max_number == 0 or self.max_number < self.base:
            return 1

        # Count digits by repeatedly dividing by base
        n = self.max_number
        digits = 0
        while n > 0:
            digits += 1
            n //= self.base

        return digits
