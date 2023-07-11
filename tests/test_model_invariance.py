from .test_data import load_data


def test_SO3_invariance_so3krates():
    from mlff.nn import So3krates
    from mlff.properties import md17_property_keys as prop_keys

    net = So3krates(F=36,
                    n_layer=2,
                    prop_keys=prop_keys)

    _test_SO3_invariance_model(net)


def test_padding_invariance_so3krates():
    from mlff.nn import So3krates
    from mlff.properties import md17_property_keys as prop_keys

    net = So3krates(F=36,
                    n_layer=2,
                    prop_keys=prop_keys)

    _test_padding_invariance_model(net)


def test_SO3_invariance_so3kratace():
    from mlff.nn import So3kratACE
    from mlff.properties import md17_property_keys as prop_keys

    data = dict(load_data('ethanol.npz'))
    net = So3kratACE(prop_keys=prop_keys,
                     F=32,
                     n_layer=2,
                     atomic_types=data['z'].tolist(),
                     geometry_embed_kwargs={'degrees': [1, 2],
                                            'mic': False,
                                            'r_cut': 5.},
                     so3kratace_layer_kwargs={'degrees': [1, 2],
                                              'max_body_order': 4,
                                              'bo_features': 32})

    _test_SO3_invariance_model(net)


def test_padding_invariance_so3kratace():
    from mlff.nn import So3kratACE
    from mlff.properties import md17_property_keys as prop_keys

    data = dict(load_data('ethanol.npz'))
    net = So3kratACE(prop_keys=prop_keys,
                     F=32,
                     n_layer=2,
                     atomic_types=data['z'].tolist(),
                     geometry_embed_kwargs={'degrees': [1, 2],
                                            'mic': False,
                                            'r_cut': 5.},
                     so3kratace_layer_kwargs={'degrees': [1, 2],
                                              'max_body_order': 4,
                                              'bo_features': 32})

    _test_padding_invariance_model(net)


def _test_SO3_invariance_model(net):
    import jax
    import jax.numpy as jnp
    import numpy as np

    from mlff.nn import get_obs_and_force_fn
    from mlff.properties import md17_property_keys as prop_keys
    from mlff.properties import property_names as pn
    from mlff.data import DataSet, DataTuple
    from mlff.geometric import get_rotation_matrix, apply_rotation

    obs_fn = get_obs_and_force_fn(net)
    obs_fn = jax.jit(jax.vmap(obs_fn, in_axes=(None, 0)))

    r_cut = 5
    data = dict(load_data('ethanol.npz'))
    data_set = DataSet(data=data, prop_keys=prop_keys)
    data_set.random_split(n_train=2,
                          n_valid=1,
                          n_test=None,
                          mic=False,
                          r_cut=r_cut,
                          training=True,
                          seed=0
                          )
    d = data_set.get_data_split()

    data_tuple = DataTuple(inputs=[pn.atomic_position, pn.atomic_type, pn.node_mask, pn.idx_i, pn.idx_j],
                           targets=[pn.energy, pn.force],
                           prop_keys=prop_keys)

    ds = data_tuple(d['train'])

    params = net.init(jax.random.PRNGKey(0), jax.tree_map(lambda x: jnp.array(x[0, ...]), ds[0]))

    inputs = jax.tree_map(lambda x: jnp.array(x), ds[0])
    base = obs_fn(params, inputs)
    E_base = base['E']
    F_base = base['F']

    for _ in range(5):
        M_rot = get_rotation_matrix(euler_axes='xyz', angles=np.random.rand(3)*360, degrees=True)
        inputs_rot = {k: v for (k, v) in inputs.items()}
        inputs_rot['R'] = apply_rotation(inputs['R'], m_rot=M_rot)
        output_rot = obs_fn(params, inputs_rot)
        E_rot = output_rot['E']
        F_rot = output_rot['F']

        assert np.isclose(E_rot.reshape(-1), E_base.reshape(-1)).all()
        assert ~np.isclose(F_rot.reshape(-1), F_base.reshape(-1)).all()
        assert np.isclose(apply_rotation(F_base, M_rot).reshape(-1), F_rot.reshape(-1), atol=1e-5).all()


def _test_padding_invariance_model(net):
    import jax
    import jax.numpy as jnp
    import numpy as np

    from mlff.nn import get_obs_and_force_fn
    from mlff.properties import md17_property_keys as prop_keys
    from mlff.properties import property_names as pn
    from mlff.data import DataSet, DataTuple
    from mlff.padding import pad_indices, pad_coordinates, pad_atomic_types

    obs_fn = get_obs_and_force_fn(net)
    obs_fn = jax.jit(jax.vmap(obs_fn, in_axes=(None, 0)))

    r_cut = 5
    data = dict(load_data('ethanol.npz'))
    data_set = DataSet(data=data, prop_keys=prop_keys)
    data_set.random_split(n_train=2,
                          n_valid=1,
                          n_test=None,
                          mic=False,
                          r_cut=r_cut,
                          training=True,
                          seed=0
                          )
    d = data_set.get_data_split()

    data_tuple = DataTuple(inputs=[pn.atomic_position, pn.atomic_type, pn.node_mask, pn.idx_i, pn.idx_j],
                           targets=[pn.energy, pn.force],
                           prop_keys=prop_keys)

    ds = data_tuple(d['train'])

    params = net.init(jax.random.PRNGKey(0), jax.tree_map(lambda x: jnp.array(x[0, ...]), ds[0]))

    inputs = jax.tree_map(lambda x: jnp.array(x), ds[0])
    base = obs_fn(params, inputs)
    E_base = base['E']
    F_base = base['F']

    # pad the coordinates
    # note that currently node_mask is not used in the stack net but only for the loss calculation
    print('Test padding of nodes ...')
    for n in range(1, 4):
        inputs_pad = {k: v for (k, v) in inputs.items()}
        inputs_pad['R'] = pad_coordinates(inputs['R'], n_max=F_base.shape[1] + int(n*2))
        inputs_pad['z'] = pad_atomic_types(inputs['z'], n_max=F_base.shape[1] + int(n*2))

        output_pad = obs_fn(params, inputs_pad)
        E_pad = output_pad['E']
        F_pad = output_pad['F']

        assert np.isclose(E_pad.reshape(-1), E_base.reshape(-1)).all()
        assert np.isclose(F_pad[:, F_base.shape[1]:, :].reshape(-1), 0).all()
        assert np.isclose(F_pad[:, :F_base.shape[1], :].reshape(-1), F_base.reshape(-1)).all()
    print('... done!')

    print('Test padding of neighbors ...')
    for n in range(1, 4):
        inputs_pad = {k: v for (k, v) in inputs.items()}
        padded_indices = pad_indices(inputs['idx_i'], inputs['idx_j'], n_pair_max=inputs['idx_i'].shape[1] + int(n*3))
        inputs_pad['idx_i'] = padded_indices[0]
        inputs_pad['idx_j'] = padded_indices[1]

        output_pad = obs_fn(params, inputs_pad)
        E_pad = output_pad['E']
        F_pad = output_pad['F']

        assert np.isclose(E_pad.reshape(-1), E_base.reshape(-1)).all()
        assert np.isclose(F_pad[:, :F_base.shape[1], :].reshape(-1), F_base.reshape(-1)).all()
    print('... done!')

    print('Test padding of nodes and neighbors ...')
    for n in range(1, 4):
        inputs_pad = {k: v for (k, v) in inputs.items()}
        padded_indices = pad_indices(inputs['idx_i'], inputs['idx_j'], n_pair_max=inputs['idx_i'].shape[1] + int(n * 3))

        inputs_pad['idx_i'] = padded_indices[0]
        inputs_pad['idx_j'] = padded_indices[1]
        inputs_pad['R'] = pad_coordinates(inputs['R'], n_max=F_base.shape[1] + int(n*2))
        inputs_pad['z'] = pad_atomic_types(inputs['z'], n_max=F_base.shape[1] + int(n*2))

        output_pad = obs_fn(params, inputs_pad)
        E_pad = output_pad['E']
        F_pad = output_pad['F']

        assert np.isclose(E_pad.reshape(-1), E_base.reshape(-1)).all()
        assert np.isclose(F_pad[:, F_base.shape[1]:, :].reshape(-1), 0).all()
        assert np.isclose(F_pad[:, :F_base.shape[1], :].reshape(-1), F_base.reshape(-1)).all()
    print('... done!')
