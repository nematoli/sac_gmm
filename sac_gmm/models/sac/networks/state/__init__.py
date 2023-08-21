from collections import OrderedDict


def rmap(func, x):
    """Recursively applies `func` to list or dictionary `x`."""
    if isinstance(x, dict):
        return OrderedDict([(k, rmap(func, v)) for k, v in x.items()])
    if isinstance(x, list):
        return [rmap(func, v) for v in x]
    return func(x)
