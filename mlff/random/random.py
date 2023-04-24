import numpy as np


def set_seeds(seed=0):
    try:
        random.seed(seed)
    except NameError:
        import random
        random.seed(seed)

    np.random.seed(seed)
