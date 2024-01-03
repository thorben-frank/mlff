from .observable import Energy
from .observable_sparse import EnergySparse


def get_observable_module(name, h):
    if name == 'energy':
        return Energy(**h)
    elif name == 'energy_sparse':
        return EnergySparse(**h)
    else:
        msg = "No observable module implemented for `module_name={}`".format(name)
        raise ValueError(msg)
