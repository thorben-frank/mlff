from typing import Dict
from .embed import (AtomTypeEmbed,
                    GeometryEmbed,
                    OneHotEmbed
                    )


def get_embedding_module(name: str, h: Dict):
    if name == 'atom_type_embed':
        return AtomTypeEmbed(**h)
    elif name == 'geometry_embed':
        return GeometryEmbed(**h)
    elif name == 'one_hot_embed':
        return OneHotEmbed(**h)
    else:
        msg = "No embedding module implemented for `module_name={}`".format(name)
        raise ValueError(msg)
