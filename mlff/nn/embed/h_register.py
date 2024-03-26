from typing import Dict

from .embed import (
    AtomTypeEmbed,
    GeometryEmbed,
    OneHotEmbed,
)

from .embed_sparse import (
    GeometryEmbedSparse,
    GeometryEmbedE3x,
    AtomTypeEmbedSparse,
    SpinEmbedSparse,
    ChargeEmbedSparse
)


def get_embedding_module(name: str, h: Dict):
    if name == 'atom_type_embed':
        return AtomTypeEmbed(**h)
    elif name == 'spin_embed_sparse':
        return SpinEmbedSparse(**h)
    elif name == 'charge_embed_sparse':
        return ChargeEmbedSparse(**h)
    elif name == 'atom_type_embed_sparse':
        return AtomTypeEmbedSparse(**h)
    elif name == 'geometry_embed':
        return GeometryEmbed(**h)
    elif name == 'geometry_embed_e3x':
        return GeometryEmbedE3x(**h)
    elif name == 'geometry_embed_sparse':
        return GeometryEmbedSparse(**h)
    elif name == 'one_hot_embed':
        return OneHotEmbed(**h)
    else:
        msg = "No embedding module implemented for `module_name={}`".format(name)
        raise ValueError(msg)
