"""Shared helpers for the VBx Python-side tests."""

import numpy as np


def normalize_labels(labels):
    """Remap labels so the first occurrence gets 0, next new label gets 1, etc.

    Lets us compare partitions between implementations that agree on the
    clustering but disagree on which integer ID each cluster receives —
    notably scipy's `fcluster` vs the C++ `fcluster_distance` (see
    `test_clusterization.test_fcluster_distance_label_order_differs`).
    """
    mapping = {}
    out = []
    for lab in labels:
        key = int(lab)
        if key not in mapping:
            mapping[key] = len(mapping)
        out.append(mapping[key])
    return np.array(out, dtype=int)
