"""Miscellaneous utility methods."""
import os
import math
import random
import errno


def dict_update(d, u):
    """Update a nested dictionary recursively recursively."""
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def ensure_dir(path):
    """Ensure that the directory specified exists, and if not, create it."""
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    return path
