from .test_data import load_data


SO3KRATES_DEFAULT_ARGS = {'n_layers': 6,
                          'F': 132,
                          'fb_rad_filter_features': [132, 132],
                          'fb_sph_filter_features': [int(132/4), 132],
                          'gb_rad_filter_features': [132, 132],
                          'gb_sph_filter_features': [int(132/4), 132]}


def test_so3krates_import():
    from mlff import nn


def test_so3krates_init():
    from mlff import nn
    from mlff.properties import md17_property_keys as prop_keys

    n_layers = 4
    net = nn.So3krates(n_layer=n_layers,
                       prop_keys=prop_keys)

    h = net.__dict_repr__()

    stack_net_layers = h['stack_net']['layers']

    # test number of layers
    assert len(stack_net_layers) == n_layers

    def check_layer(u):
        assert u['degrees'] == [1, 2, 3]
        assert u['fb_attention'] == 'conv_att'
        assert u['gb_attention'] == 'conv_att'
        assert u['fb_filter'] == 'radial_spherical'
        assert u['gb_filter'] == 'radial_spherical'
        assert u['n_heads'] == 4
        assert not u['final_layer']
        assert u['parity']
        assert u['fb_rad_filter_features'] == SO3KRATES_DEFAULT_ARGS['fb_rad_filter_features']
        assert u['gb_rad_filter_features'] == SO3KRATES_DEFAULT_ARGS['gb_rad_filter_features']
        assert u['fb_sph_filter_features'] == SO3KRATES_DEFAULT_ARGS['fb_sph_filter_features']
        assert u['gb_sph_filter_features'] == SO3KRATES_DEFAULT_ARGS['gb_sph_filter_features']

    [check_layer(x['so3krates_layer']) for x in stack_net_layers]


def test_so3krates_init_with_kwargs():
    from mlff import nn
    from mlff.properties import md17_property_keys as prop_keys

    F = 22
    test_args = {'fb_rad_filter_features': [6, F],
                 'fb_sph_filter_features': [19, F],
                 'gb_rad_filter_features': [F],
                 'gb_sph_filter_features': [17, F]
                 }
    # test_args.update({'F': F})

    net = nn.So3krates(F=F,
                       so3krates_layer_kwargs=test_args,
                       prop_keys=prop_keys)

    h = net.__dict_repr__()

    stack_net_layers = h['stack_net']['layers']

    # test default number of layers
    assert len(stack_net_layers) == SO3KRATES_DEFAULT_ARGS['n_layers']

    def check_mlp_dimensions(u, name: str):
        assert u[name] == test_args[name]

    for n in test_args.keys():
        [check_mlp_dimensions(x['so3krates_layer'], name=n) for x in stack_net_layers]


def test_so3krates_param_init():
    import jax
    import jax.numpy as jnp
    from mlff import nn
    from mlff.properties import md17_property_keys as prop_keys

    import mlff.properties.property_names as pn

    net = nn.So3krates(F=36,
                       n_layer=2,
                       prop_keys=prop_keys)

    inputs = {prop_keys[pn.atomic_position]: jax.random.normal(jax.random.PRNGKey(0), shape=(7, 3)),
              prop_keys[pn.atomic_type]: jnp.ones(7),
              prop_keys[pn.idx_i]: jnp.array([0, 0, 1, 2, 3, 3, 3, 4, 5, 6, 6, 6]),
              prop_keys[pn.idx_j]: jnp.array([1, 6, 0, 6, 4, 5, 6, 3, 3, 0, 2, 3])}

    net.init(jax.random.PRNGKey(0), inputs)


def test_so3krates_training():
    import numpy as np
    import jax
    import jax.numpy as jnp
    import os

    from mlff.io import create_directory, bundle_dicts, save_dict
    from mlff.training import Coach, Optimizer, get_loss_fn, create_train_state
    from mlff.data import DataTuple, DataSet

    from mlff.nn import get_obs_and_force_fn
    from mlff import nn
    from mlff.properties import md17_property_keys as prop_keys

    import mlff.properties.property_names as pn

    data_path = 'test_data/ethanol.npz'
    save_path = '_test_train_so3krates'
    ckpt_dir = os.path.join(save_path, 'module')
    ckpt_dir = create_directory(ckpt_dir, exists_ok=False)

    data = dict(load_data('ethanol.npz'))

    r_cut = 5
    data_set = DataSet(data=data, prop_keys=prop_keys)
    data_set.random_split(n_train=50,
                          n_valid=10,
                          n_test=None,
                          r_cut=r_cut,
                          training=True,
                          seed=0)

    data_set.shift_x_by_mean_x(x=pn.energy)
    data_set.divide_x_by_std_y(x=pn.energy, y=pn.force)
    data_set.divide_x_by_std_y(x=pn.force, y=pn.force)

    data_set.save_splits_to_file(ckpt_dir, 'splits.json')
    data_set.save_scales(ckpt_dir, 'scales.json')

    d = data_set.get_data_split()

    net = nn.So3krates(F=32,
                       n_layer=2,
                       prop_keys=prop_keys,
                       geometry_embed_kwargs={'degrees': [1, 2]},
                       so3krates_layer_kwargs={'n_heads': 1,
                                               'degrees': [1, 2]})

    obs_fn = get_obs_and_force_fn(net)
    obs_fn = jax.vmap(obs_fn, in_axes=(None, 0))

    opt = Optimizer()
    tx = opt.get(learning_rate=1e-3)

    coach = Coach(inputs=[pn.atomic_position, pn.atomic_type, pn.idx_i, pn.idx_j, pn.node_mask],
                  targets=[pn.energy, pn.force],
                  epochs=4,
                  training_batch_size=2,
                  validation_batch_size=2,
                  loss_weights={pn.energy: 0.01, pn.force: 0.99},
                  ckpt_dir=ckpt_dir,
                  data_path=data_path,
                  net_seed=0,
                  training_seed=0)

    loss_fn = get_loss_fn(obs_fn=obs_fn,
                          weights=coach.loss_weights,
                          prop_keys=prop_keys)

    data_tuple = DataTuple(inputs=coach.inputs,
                           targets=coach.targets,
                           prop_keys=prop_keys)

    train_ds = data_tuple(d['train'])
    valid_ds = data_tuple(d['valid'])

    inputs = jax.tree_map(lambda x: jnp.array(x[0, ...]), train_ds[0])
    params = net.init(jax.random.PRNGKey(coach.net_seed), inputs)
    train_state, h_train_state = create_train_state(net,
                                                    params,
                                                    tx,
                                                    polyak_step_size=None,
                                                    plateau_lr_decay={'patience': 50,
                                                                      'decay_factor': 0.5},
                                                    scheduled_lr_decay={'exponential': {'transition_steps': 10_000,
                                                                                        'decay_factor': 0.9}},
                                                    lr_warmup={'init_value': 0,
                                                               'peak_value': 1,
                                                               'warmup_steps': 10}
                                                    )

    h_net = net.__dict_repr__()
    h_opt = opt.__dict_repr__()
    h_coach = coach.__dict_repr__()
    h_dataset = data_set.__dict_repr__()
    h = bundle_dicts([h_net, h_opt, h_coach, h_dataset, h_train_state])
    save_dict(path=ckpt_dir, filename='hyperparameters.json', data=h, exists_ok=True)

    coach.run(train_state=train_state,
              train_ds=train_ds,
              valid_ds=valid_ds,
              loss_fn=loss_fn,
              ckpt_overwrite=True,
              eval_every_t=50,
              log_every_t=1,
              restart_by_nan=True,
              use_wandb=False)

    assert os.path.isfile(os.path.join(ckpt_dir, 'scales.json'))
    assert os.path.isfile(os.path.join(ckpt_dir, 'splits.json'))
    assert os.path.isfile(os.path.join(ckpt_dir, 'hyperparameters.json'))

    from mlff.io import read_json
    _h = read_json(os.path.join(ckpt_dir, 'hyperparameters.json'))

    def check_h(x, y):
        assert x == y

    [check_h(u, v) for u, v in zip(h.items(), _h.items())]


def test_remove_dirs():
    try:
        import shutil
        shutil.rmtree('_test_train_so3krates')
    except FileNotFoundError:
        pass
