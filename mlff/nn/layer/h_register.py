from typing import Dict
from .so3krates_layer import So3kratesLayer
from .so3kratace_layer import So3krataceLayer
from .schnet_layer import SchNetLayer


def get_layer(name: str, h: Dict):
    if name == 'so3krates_layer':
        return So3kratesLayer(**h)
    elif name == 'so3kratace_layer':
        return So3krataceLayer(**h)
    elif name == 'schnet_layer':
        return SchNetLayer(**h)
    elif name == 'spookynet_layer':
        raise NotImplementedError('SpookyNet not implemented!')
        return SpookyNetLayer(**h)
    elif name == 'painn_layer':
        raise NotImplementedError('PaiNN not implemented!')
        return PaiNNLayer(**h)
    else:
        msg = f"Layer with `module_name={name}` is not implemented."
        raise NotImplementedError(msg)
