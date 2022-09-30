from typing import Dict
from .embed import (AtomTypeEmbed,
                    VectorFeatureEmbed,
                    ChargeEmbed,
                    SpinEmbed,
                    GeometryEmbed,
                    AtomCenteredBasisFunctionEmbed,
                    )


def get_embedding_module(name: str, h: Dict):
    if name == 'atom_type_embed':
        return AtomTypeEmbed(**h)
    elif name == 'geometry_embed':
        return GeometryEmbed(**h)
    elif name == 'charge_embed':
        return ChargeEmbed(**h)
    elif name == 'spin_embed':
        return SpinEmbed(**h)
    elif name == 'atom_centered_basis_function_embed':
        return AtomCenteredBasisFunctionEmbed(**h)
    elif name == 'vector_feature_embed':
        return VectorFeatureEmbed(**h)
    else:
        msg = "No embedding module implemented for `module_name={}`".format(name)
        raise ValueError(msg)