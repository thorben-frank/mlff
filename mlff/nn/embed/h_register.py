from typing import Dict

from .embed import (
    AtomTypeEmbed,
    GeometryEmbed,
    OneHotEmbed,
)

from .embed_sparse import (
    GeometryEmbedSparse,
    AtomTypeEmbedSparse
)


def get_embedding_module(name: str, h: Dict):
    if name == 'atom_type_embed':
        return AtomTypeEmbed(**h)
    elif name == 'atom_type_embed_sparse':
        return AtomTypeEmbedSparse(**h)
    elif name == 'geometry_embed':
        return GeometryEmbed(**h)
    elif name == 'geometry_embed_sparse':
        return GeometryEmbedSparse(**h)
    elif name == 'one_hot_embed':
        return OneHotEmbed(**h)
    else:
        msg = "No embedding module implemented for `module_name={}`".format(name)
        raise ValueError(msg)
