SCHNET_DEFAULT_ARGS = {'n_layers': 6,
                       'F': 128}


def test_schnet_import():
    from mlff.nn import SchNet


def test_schnet_init():
    from mlff import nn
    from mlff.properties import md17_property_keys as prop_keys

    F = 16
    n_layers = 4
    net = nn.SchNet(F=F,
                    n_layer=n_layers,
                    prop_keys=prop_keys)

    h = net.__dict_repr__()

    stack_net_layers = h['stack_net']['layers']

    # test number of layers
    assert len(stack_net_layers) == n_layers

    # test feature dimension
    def check_F(u):
        assert u['F'] == F

    assert [check_F(x['schnet_layer']) for x in stack_net_layers]


def test_schnet_init_with_kwargs():
    from mlff import nn
    from mlff.properties import md17_property_keys as prop_keys

    F = 22
    test_args = {
                 'in2f_features': [32, F],
                 'filter_features': [16, F],
                 'f2out_features': [63, F]
                 }
    test_args.update({'F': F})

    net = nn.SchNet(schnet_layer_kwargs=test_args,
                    prop_keys=prop_keys)

    h = net.__dict_repr__()

    stack_net_layers = h['stack_net']['layers']

    # test default number of layers
    assert len(stack_net_layers) == SCHNET_DEFAULT_ARGS['n_layers']

    # test feature dimension
    def check_F(u):
        assert u['F'] == F

    [check_F(x['schnet_layer']) for x in stack_net_layers]

    def check_mlp_dimensions(u, name: str):
        assert u[name] == test_args[name]

    for n in test_args.keys():
        [check_mlp_dimensions(x['schnet_layer'], name=n) for x in stack_net_layers]


def test_schnet_param_init():
    import jax
    import jax.numpy as jnp
    from mlff.nn import SchNet
    from mlff.properties import md17_property_keys as prop_keys

    import mlff.properties.property_names as pn

    net = SchNet(F=32,
                 n_layer=2,
                 prop_keys=prop_keys)

    inputs = {prop_keys[pn.atomic_position]: jax.random.normal(jax.random.PRNGKey(0), shape=(7, 3)),
              prop_keys[pn.atomic_type]: jnp.ones(7),
              prop_keys[pn.idx_i]: jnp.array([0, 0, 1, 2, 3, 3, 3, 4, 5, 6, 6, 6]),
              prop_keys[pn.idx_j]: jnp.array([1, 6, 0, 6, 4, 5, 6, 3, 3, 0, 2, 3])}

    net.init(jax.random.PRNGKey(0), inputs)
