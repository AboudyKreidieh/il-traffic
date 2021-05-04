"""Miscellaneous utility methods."""


def dict_update(d, u):
    """Update a nested dictionary recursively recursively."""
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d
