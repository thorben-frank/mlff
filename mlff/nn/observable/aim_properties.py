import jax.numpy as jnp
import flax.linen as nn
from jax.ops import segment_sum
from typing import Any, Callable, Dict
from mlff.nn.base.sub_module import BaseSubModule

class PartialCharge(BaseSubModule):
    prop_keys: Dict
    regression_dim: int = None
    activation_fn: Callable[[Any], Any] = lambda u: u
    output_is_zero_at_init: bool = True
    module_name: str = 'partial_charge'

    def setup(self):
        # self.partial_charge_key = self.prop_keys.get('partial_charge')
        # self.atomic_type_key = self.prop_keys.get('atomic_type')
        # self.total_charge_key = self.prop_keys.get('total_charge')

        if self.output_is_zero_at_init:
            self.kernel_init = nn.initializers.zeros_init()
        else:
            self.kernel_init = nn.initializers.lecun_normal()        

    @nn.compact
    def __call__(self,
                 inputs: Dict,
                 *args,
                 **kwargs) -> Dict[str, jnp.ndarray]:
        """
        Predict partial charges, from atom-wise features `x`, atomic types `atomic_numbers` and the total charge of the system `total_charge`.

        Args:
            inputs (Dict):
                x (Array): Node features, (num_nodes, num_features)
                atomic_numbers (Array): Atomic numbers, (num_nodes)
                total_charge (Array): Total charge, shape: (1)
                batch_segments (Array): Batch segments, (num_nodes)
                node_mask (Array): Node mask, (num_nodes)
                graph_mask (Array): Graph mask, (num_graphs)
            *args ():
            **kwargs ():

        #Returns: Dictionary of form {'q': Array}, where Array are the predicted partial charges, shape: (n,1)

        """
        x = inputs['x']  # (num_nodes, num_features)
        atomic_numbers = inputs['atomic_numbers']  # (num_nodes)
        batch_segments = inputs['batch_segments']  # (num_nodes)
        #        point_mask = inputs['point_mask']
        node_mask = inputs['node_mask']  # (num_nodes)
        graph_mask = inputs['graph_mask']  # (num_graphs)
        # total_charge = inputs[self.total_charge_key]
        total_charges = inputs['total_charges']

        num_graphs = len(graph_mask)
        num_nodes = len(node_mask)
        # n = (point_mask != 0).sum()  # shape: (1)

        #q_ - element-dependent bias
        q_ = nn.Embed(num_embeddings=100, features=1)(atomic_numbers).squeeze(axis=-1)  # shape: (n)
        # x_ = MLP(features=[x.shape[-1], 1],
        #          activation_fn=silu)(x).squeeze(axis=-1)  # shape: (n)
        
        if self.regression_dim is not None:
            y = nn.Dense(
                self.regression_dim,
                kernel_init=nn.initializers.lecun_normal(),
                name='charge_dense_regression'
            )(x)  # (num_nodes, regression_dim)
            y = self.activation_fn(y)  # (num_nodes, regression_dim)
            x_ = nn.Dense(
                1,
                kernel_init=self.kernel_init,
                name='charge_dense_final'
            )(y).squeeze(axis=-1)  # (num_nodes)
        else:
            x_ = nn.Dense(
                1,
                kernel_init=self.kernel_init,
                name='charge_dense_final'
            )(x).squeeze(axis=-1)  # (num_nodes)

        # x_q = safe_scale(x_ + q_, scale=point_mask)  # shape: (n)
        x_q = jnp.where(node_mask, x_ + q_, 
                                  jnp.asarray(0., dtype=x_q.dtype))  # (num_nodes)

        # partial_charges = x_q + (1 / n) * (total_charges - x_q.sum(axis=-1))  # shape: (n)
        partial_charges = x_q + (1 / num_nodes) * (total_charges - x_q.sum(axis=-1))  # shape: (num_nodes)
        return dict(partial_charges=partial_charges)
    
        # if self.output_convention == 'per_structure':
        #     energy = segment_sum(
        #         atomic_energy,
        #         segment_ids=batch_segments,
        #         num_segments=num_graphs
        #     )  # (num_graphs)
        #     energy = jnp.where(graph_mask, energy, jnp.asarray(0., dtype=energy.dtype))

        #     return dict(energy=energy)
        # elif self.output_convention == 'per_atom':
        #     energy = atomic_energy  # (num_nodes)

        #     return dict(energy=energy)
        # else:
        #     raise ValueError(
        #         f'{self.output_convention} is invalid argument for attribute `output_convention`.'
        #     )
        # return {self.partial_charge_key: safe_scale(partial_charges, scale=point_mask)[:, None]}  # shape: (n,1)

    # @staticmethod
    # def reset_output_convention(self, *args, **kwargs):
    #     logging.warning('Can not reset `output_convention` for `PartialCharge` module, since partial charges are '
    #                     'only defined per atom.')

    # def __dict_repr__(self) -> Dict[str, Dict[str, Any]]:
    #     return {self.module_name: {'prop_keys': self.prop_keys}}

# class HirshfeldVolume(BaseSubModule):
#     prop_keys: Dict
#     module_name = 'hirshfeld_volume'

#     def setup(self):
#         self.atomic_type_key = self.prop_keys.get('atomic_type')
#         self.hirshfeld_volume_key = self.prop_keys.get('hirshfeld_volume')

#     @nn.compact
#     def __call__(self,
#                  inputs: Dict,
#                  *args,
#                  **kwargs) -> Dict[str, jnp.ndarray]:
#         """
#         Predict Hirshfeld volumes from atom-wise features `x` and atomic types `atomic_numbers`.

#         Args:
#             inputs (Dict):
#                 x (Array): Atomic features, shape: (n,F)
#                 atomic_numbers (Array): Atomic types, shape: (n)
#                 point_mask (Array): Mask for atom-wise operations, shape: (n)
#             *args ():
#             **kwargs ():

#         Returns: Dictionary of form {'v_eff': Array}, where Array are the predicted Hirshfeld volumes, shape: (n,1)

#         """
#         point_mask = inputs['point_mask']
#         x = inputs['x']
#         atomic_numbers = inputs[self.atomic_type_key]

#         F = x.shape[-1]

#         v_shift = nn.Embed(num_embeddings=100, features=1)(atomic_numbers).squeeze(axis=-1)  # shape: (n)
#         q = nn.Embed(num_embeddings=100, features=int(F / 2))(atomic_numbers)  # shape: (n,F/2)
#         k = MLP(features=[x.shape[-1], int(F / 2)],
#                 activation_fn=silu)(x)  # shape: (n,F/2)

#         q_x_k = safe_scale((q * k / jnp.sqrt(k.shape[-1])).sum(axis=-1), scale=point_mask)  # shape: (n)
#         v_eff = v_shift + q_x_k  # shape: (n)

#         return {self.hirshfeld_volume_key: safe_scale(v_eff, scale=point_mask)[:, None]}  # shape: (n,1)

#     @staticmethod
#     def reset_output_convention(self, *args, **kwargs):
#         logging.warning('Can not reset `output_convention` for `HirshfeldVolume` module, since Hirshfeld volumes are '
#                         'only defined per atom.')

#     def __dict_repr__(self) -> Dict[str, Dict[str, Any]]:
#         return {self.module_name: {'prop_keys': self.prop_keys}
#                 }


# class HirshfeldVolumeRatio(BaseSubModule):
#     prop_keys: Dict
#     module_name = 'hirshfeld_volume_ratio'

#     def setup(self):
#         self.atomic_type_key = self.prop_keys.get(pn.atomic_type)
#         self.hirshfeld_volume_ratio_key = self.prop_keys.get(pn.hirshfeld_volume_ratio)

#     @nn.compact
#     def __call__(self,
#                  inputs: Dict,
#                  *args,
#                  **kwargs) -> Dict[str, jnp.ndarray]:
#         """
#         Predict Hirshfeld volumes from atom-wise features `x` and atomic types `atomic_numbers`.

#         Args:
#             inputs (Dict):
#                 x (Array): Atomic features, shape: (n,F)
#                 atomic_numbers (Array): Atomic types, shape: (n)
#                 point_mask (Array): Mask for atom-wise operations, shape: (n)
#             *args ():
#             **kwargs ():

#         Returns: Dictionary of form {'v_eff': Array}, where Array are the predicted Hirshfeld volumes, shape: (n,1)

#         """
#         point_mask = inputs['point_mask']
#         x = inputs['x']
#         atomic_numbers = inputs[self.atomic_type_key]

#         F = x.shape[-1]

#         v_shift = nn.Embed(num_embeddings=100, features=1)(atomic_numbers).squeeze(axis=-1)  # shape: (n)
#         q = nn.Embed(num_embeddings=100, features=int(F / 2))(atomic_numbers)  # shape: (n,F/2)
#         k = MLP(features=[x.shape[-1], int(F / 2)],
#                 activation_fn=silu)(x)  # shape: (n,F/2)

#         q_x_k = safe_scale((q * k / jnp.sqrt(k.shape[-1])).sum(axis=-1), scale=point_mask)  # shape: (n)
#         v_eff = v_shift + q_x_k  # shape: (n)

#         return {self.hirshfeld_volume_ratio_key: safe_scale(v_eff, scale=point_mask)[:, None]}  # shape: (n,1)

#     @staticmethod
#     def reset_output_convention(self, *args, **kwargs):
#         logging.warning('Can not reset `output_convention` for `HirshfeldVolumeRatio` module, since Hirshfeld '
#                         'volume ratios are only defined per atom.')

#     def __dict_repr__(self) -> Dict[str, Dict[str, Any]]:
#         return {self.module_name: {'prop_keys': self.prop_keys}
#                 }