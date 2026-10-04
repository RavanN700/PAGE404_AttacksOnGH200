#!/usr/bin/env python3
"""Compute the per-channel transmission error rate of the covert channel.

For each channel it compares the bit stream the sender transmitted against the
bits the receiver recovered, using Levenshtein (edit) distance normalized by
the received length. Edit distance (rather than a plain bit-by-bit XOR) is used
because the receiver can occasionally insert or drop a bit, which would
otherwise desynchronize every subsequent comparison.

Input files (written by sender.cu / receiver.cu):
    ./texts/parallel_channel/sender_message_<i>   # transmitted bits
    ./texts/parallel_channel/message_bits_<i>     # recovered bits

Usage:
    python error_rate.py <num_bits> <num_channels> [start]

    num_bits      length of the receiver window to compare (bits)
    num_channels  number of parallel channels to evaluate
    start         1-based offset into the receiver stream (default: 1)
"""

import sys


def read_binary_file(filepath):
    """Read a file of '0'/'1' characters, returning a list of ints (0/1)."""
    with open(filepath, 'r') as f:
        content = f.read().strip()
    return [int(c) for c in content if c in ('0', '1')]


def levenshtein_distance(a, b):
    """Return the Levenshtein (edit) distance between sequences `a` and `b`."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],
                                   dp[i][j - 1],
                                   dp[i - 1][j - 1])
    return dp[m][n]


def main():
    if len(sys.argv) < 3:
        print("Usage: python error_rate.py <num_bits> <num_channels> [start]")
        sys.exit(1)

    num_bits     = int(sys.argv[1])
    num_channels = int(sys.argv[2])
    start        = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    error_rates = []
    for i in range(num_channels):
        sender_file   = f"./texts/parallel_channel/sender_message_{i}"
        receiver_file = f"./texts/parallel_channel/message_bits_{i}"

        seq_sent     = read_binary_file(sender_file)
        seq_received = read_binary_file(receiver_file)

        # Compare only the requested window of the received stream.
        seq_received = seq_received[(start - 1):(start - 1 + num_bits)]
        if not seq_received:
            print(f"Channel {i}: no received bits in window; skipping")
            continue

        distance   = levenshtein_distance(seq_sent, seq_received)
        error_rate = distance / len(seq_received) * 100
        error_rates.append(error_rate)

        print(f"Channel {i}: sent={len(seq_sent)} bits, "
              f"received={len(seq_received)} bits | "
              f"edit distance={distance}, error rate={error_rate:.2f}%")

    if not error_rates:
        print("\nNo channels evaluated.")
        return

    avg = sum(error_rates) / len(error_rates)
    print(f"\nAverage error rate across {len(error_rates)} channel(s): {avg:.2f}%")


if __name__ == "__main__":
    main()
