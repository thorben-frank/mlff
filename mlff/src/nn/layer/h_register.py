from typing import Dict
from .so3krates_layer import So3kratesLayer


def get_layer(name: str, h: Dict):
    if name == 'so3krates_layer':
        return So3kratesLayer(**h)
    else:
        msg = "Layer with `module_name={}` is not implemented.".format(name)
        raise NotImplementedError(msg)
