from typing import Dict
from .embed import (AtomTypeEmbed,
                    GeometryEmbed,
                    )


def get_embedding_module(name: str, h: Dict):
    if name == 'atom_type_embed':
        return AtomTypeEmbed(**h)
    elif name == 'geometry_embed':
        return GeometryEmbed(**h)
    else:
        msg = "No embedding module implemented for `module_name={}`".format(name)
        raise ValueError(msg)
