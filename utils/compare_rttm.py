#!/usr/bin/env python3
"""Compare two RTTM files for consistent speaker labeling."""

import sys


def read_speakers(path):
    speakers = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            speakers.append(line.split()[7])
    return speakers


def compare_rttm(file_a, file_b):
    spk_a = read_speakers(file_a)
    spk_b = read_speakers(file_b)

    if len(spk_a) != len(spk_b):
        print(f"WARNING: different number of lines ({len(spk_a)} vs {len(spk_b)})")
        n_lines = min(len(spk_a), len(spk_b))
        spk_a = spk_a[:n_lines]
        spk_b = spk_b[:n_lines]

    # Build mapping from file_a labels to file_b labels
    mapping = {}
    mismatches = []

    for i, (a, b) in enumerate(zip(spk_a, spk_b), 1):
        if a in mapping:
            if mapping[a] != b:
                mismatches.append((i, a, b, mapping[a]))
        else:
            mapping[a] = b

    # Print label mapping
    print(f"Comparing: {file_a} -> {file_b}")
    print(f"Lines: {len(spk_a)}")
    print(f"\nLabel mapping ({file_a} -> {file_b}):")
    for a, b in sorted(mapping.items(), key=lambda x: int(x[0])):
        print(f"  spk {a} -> spk {b}")

    if mismatches:
        print(f"\nINCONSISTENT labeling! {len(mismatches)} conflict(s):")
        for line, a, b, expected in mismatches:
            print(f"  Line {line}: spk {a} mapped to spk {expected}, but got spk {b}")
        return False
    else:
        print(
            "\nResult: CONSISTENT labeling (same speaker grouping, different numeric labels)"
        )
        return True


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <file_a.rttm> <file_b.rttm>")
        sys.exit(1)
    ok = compare_rttm(sys.argv[1], sys.argv[2])
    sys.exit(0 if ok else 1)
