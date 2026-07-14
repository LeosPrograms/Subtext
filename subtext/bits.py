"""Bit-manipulation utilities for arithmetic coding."""

from typing import Iterable, List


def int_to_bits(n: int, num_bits: int) -> List[int]:
    """Integer → little-endian bit list.  int_to_bits(5, 4) == [1,0,1,0]."""
    return [(n >> i) & 1 for i in range(num_bits)]


def bits_to_int(bits: Iterable[int]) -> int:
    """Little-endian bit list → integer.  bits_to_int([1,0,1,0]) == 5."""
    return sum(b << i for i, b in enumerate(bits))


def matching_prefix_len(a: List[int], b: List[int]) -> int:
    """Number of equal elements from the start of two equal-length lists."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def bytes_to_bits(data: bytes) -> List[int]:
    bits: List[int] = []
    for byte in data:
        bits.extend(int_to_bits(byte, 8))
    return bits


def bits_to_bytes(bits: List[int]) -> bytes:
    pad = (-len(bits)) % 8
    padded = bits + [0] * pad
    return bytes(bits_to_int(padded[i:i + 8]) for i in range(0, len(padded), 8))
