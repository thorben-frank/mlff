class PartialCharge(BaseSubModule):
    prop_keys: Dict
    module_name: str = 'partial_charge'

    def setup(self):
        self.partial_charge_key = self.prop_keys.get('partial_charge')
        self.atomic_type_key = self.prop_keys.get('atomic_type')
        self.total_charge_key = self.prop_keys.get('total_charge')

    @nn.compact
    def __call__(self,
                 inputs: Dict,
                 *args,
                 **kwargs) -> Dict[str, jnp.ndarray]:
        """
        Predict partial charges, from atom-wise features `x`, atomic types `z` and the total charge of the system `Q`.

        Args:
            inputs (Dict):
                x (Array): Atomic features, shape: (n,F)
                z (Array): Atomic types, shape: (n)
                Q (Array): Total charge, shape: (1)
                point_mask (Array): Mask for atom-wise operations, shape: (n)
            *args ():
            **kwargs (): Q - Total charge, shape: (1)

        Returns: Dictionary of form {'q': Array}, where Array are the predicted partial charges, shape: (n,1)

        """
        point_mask = inputs['point_mask']
        x = inputs['x']
        z = inputs[self.atomic_type_key]
        Q = inputs[self.total_charge_key]

        n = (point_mask != 0).sum()  # shape: (1)
        q_ = nn.Embed(num_embeddings=100, features=1)(z).squeeze(axis=-1)  # shape: (n)
        x_ = MLP(features=[x.shape[-1], 1],
                 activation_fn=silu)(x).squeeze(axis=-1)  # shape: (n)

        x_q = safe_scale(x_ + q_, scale=point_mask)  # shape: (n)
        partial_charges = x_q + (1 / n) * (Q - x_q.sum(axis=-1))  # shape: (n)

        return {self.partial_charge_key: safe_scale(partial_charges, scale=point_mask)[:, None]}  # shape: (n,1)

    @staticmethod
    def reset_output_convention(self, *args, **kwargs):
        logging.warning('Can not reset `output_convention` for `PartialCharge` module, since partial charges are '
                        'only defined per atom.')

    def __dict_repr__(self) -> Dict[str, Dict[str, Any]]:
        return {self.module_name: {'prop_keys': self.prop_keys}}


class HirshfeldVolume(BaseSubModule):
    prop_keys: Dict
    module_name = 'hirshfeld_volume'

    def setup(self):
        self.atomic_type_key = self.prop_keys.get('atomic_type')
        self.hirshfeld_volume_key = self.prop_keys.get('hirshfeld_volume')

    @nn.compact
    def __call__(self,
                 inputs: Dict,
                 *args,
                 **kwargs) -> Dict[str, jnp.ndarray]:
        """
        Predict Hirshfeld volumes from atom-wise features `x` and atomic types `z`.

        Args:
            inputs (Dict):
                x (Array): Atomic features, shape: (n,F)
                z (Array): Atomic types, shape: (n)
                point_mask (Array): Mask for atom-wise operations, shape: (n)
            *args ():
            **kwargs ():

        Returns: Dictionary of form {'v_eff': Array}, where Array are the predicted Hirshfeld volumes, shape: (n,1)

        """
        point_mask = inputs['point_mask']
        x = inputs['x']
        z = inputs[self.atomic_type_key]

        F = x.shape[-1]

        v_shift = nn.Embed(num_embeddings=100, features=1)(z).squeeze(axis=-1)  # shape: (n)
        q = nn.Embed(num_embeddings=100, features=int(F / 2))(z)  # shape: (n,F/2)
        k = MLP(features=[x.shape[-1], int(F / 2)],
                activation_fn=silu)(x)  # shape: (n,F/2)

        q_x_k = safe_scale((q * k / jnp.sqrt(k.shape[-1])).sum(axis=-1), scale=point_mask)  # shape: (n)
        v_eff = v_shift + q_x_k  # shape: (n)

        return {self.hirshfeld_volume_key: safe_scale(v_eff, scale=point_mask)[:, None]}  # shape: (n,1)

    @staticmethod
    def reset_output_convention(self, *args, **kwargs):
        logging.warning('Can not reset `output_convention` for `HirshfeldVolume` module, since Hirshfeld volumes are '
                        'only defined per atom.')

    def __dict_repr__(self) -> Dict[str, Dict[str, Any]]:
        return {self.module_name: {'prop_keys': self.prop_keys}
                }


class HirshfeldVolumeRatio(BaseSubModule):
    prop_keys: Dict
    module_name = 'hirshfeld_volume_ratio'

    def setup(self):
        self.atomic_type_key = self.prop_keys.get(pn.atomic_type)
        self.hirshfeld_volume_ratio_key = self.prop_keys.get(pn.hirshfeld_volume_ratio)

    @nn.compact
    def __call__(self,
                 inputs: Dict,
                 *args,
                 **kwargs) -> Dict[str, jnp.ndarray]:
        """
        Predict Hirshfeld volumes from atom-wise features `x` and atomic types `z`.

        Args:
            inputs (Dict):
                x (Array): Atomic features, shape: (n,F)
                z (Array): Atomic types, shape: (n)
                point_mask (Array): Mask for atom-wise operations, shape: (n)
            *args ():
            **kwargs ():

        Returns: Dictionary of form {'v_eff': Array}, where Array are the predicted Hirshfeld volumes, shape: (n,1)

        """
        point_mask = inputs['point_mask']
        x = inputs['x']
        z = inputs[self.atomic_type_key]

        F = x.shape[-1]

        v_shift = nn.Embed(num_embeddings=100, features=1)(z).squeeze(axis=-1)  # shape: (n)
        q = nn.Embed(num_embeddings=100, features=int(F / 2))(z)  # shape: (n,F/2)
        k = MLP(features=[x.shape[-1], int(F / 2)],
                activation_fn=silu)(x)  # shape: (n,F/2)

        q_x_k = safe_scale((q * k / jnp.sqrt(k.shape[-1])).sum(axis=-1), scale=point_mask)  # shape: (n)
        v_eff = v_shift + q_x_k  # shape: (n)

        return {self.hirshfeld_volume_ratio_key: safe_scale(v_eff, scale=point_mask)[:, None]}  # shape: (n,1)

    @staticmethod
    def reset_output_convention(self, *args, **kwargs):
        logging.warning('Can not reset `output_convention` for `HirshfeldVolumeRatio` module, since Hirshfeld '
                        'volume ratios are only defined per atom.')

    def __dict_repr__(self) -> Dict[str, Dict[str, Any]]:
        return {self.module_name: {'prop_keys': self.prop_keys}
                }