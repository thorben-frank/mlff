import numpy as np
import jax
import jax.numpy as jnp
import os
import wandb
import portpicker
import ase.units as si

from mlff.io.io import create_directory, bundle_dicts, save_dict
from mlff.training import Coach, Optimizer, get_loss_fn, create_train_state
from mlff.data import DataTuple, DataSet

from mlff.nn.stacknet import get_obs_and_force_fn, get_observable_fn, get_energy_force_stress_fn
from mlff.nn import So3krates, Energy, ZBLRepulsion
from mlff.properties import md17_property_keys as prop_keys

import mlff.properties.property_names as pn

port = portpicker.pick_unused_port()
jax.distributed.initialize(f'localhost:{port}', num_processes=1, process_id=0)

# data_path = '/Users/thorbenfrank/Documents/data/mol-data/dissipation-curves/NaCl.npz'
# data_path = '/Users/thorbenfrank/Documents/data/MD22/nanotube.npz'
data_path = '/Users/thorbenfrank/Documents/data/mol-data/dft/ethanol_dft.npz'
save_path = 'ckpt_dir'

ckpt_dir = os.path.join(save_path, 'module')
ckpt_dir = create_directory(ckpt_dir, exists_ok=False)

E_key = prop_keys['energy']
F_key = prop_keys['force']

data = dict(np.load(data_path))

# convert data to eV from kcal/mol used in MD17 data
data[E_key] = data[E_key] * si.kcal / si.mol
data[F_key] = data[F_key] * si.kcal / si.mol

r_cut = 5
data_set = DataSet(data=data, prop_keys=prop_keys)
data_set.random_split(n_train=20,
                      n_valid=20,
                      n_test=None,
                      mic=False,
                      r_cut=r_cut,
                      training=True,
                      seed=0
                      )

data_set.shift_x_by_mean_x(x=pn.energy)

data_set.save_splits_to_file(ckpt_dir, 'splits.json')
data_set.save_scales(ckpt_dir, 'scales.json')

d = data_set.get_data_split()

opt = Optimizer()
tx = opt.get(learning_rate=1e-3)

coach = Coach(inputs=[pn.atomic_position, pn.atomic_type, pn.idx_i, pn.idx_j, pn.node_mask],
              targets=[pn.energy],
              epochs=1000,
              training_batch_size=1,
              validation_batch_size=1,
              loss_weights={pn.energy: 1.},
              ckpt_dir=ckpt_dir,
              data_path=data_path,
              net_seed=0,
              training_seed=0)

data_tuple = DataTuple(inputs=coach.inputs,
                       targets=coach.targets,
                       prop_keys=prop_keys)

train_ds = data_tuple(d['train'])
valid_ds = data_tuple(d['valid'])

obs = [Energy(prop_keys=prop_keys, zbl_repulsion=False)]
net = So3krates(F=32,
                n_layer=2,
                prop_keys=prop_keys,
                obs=obs,
                geometry_embed_kwargs={'degrees': [1, 2],
                                       'r_cut': r_cut
                                       },
                so3krates_layer_kwargs={'n_heads': 2,
                                        'degrees': [1, 2]})

obs_fn = get_obs_and_force_fn(net)
obs_fn = jax.vmap(obs_fn, in_axes=(None, 0))

loss_fn = get_loss_fn(obs_fn=obs_fn,
                      weights=coach.loss_weights,
                      prop_keys=prop_keys)

inputs = jax.tree_map(lambda x: jnp.array(x[0, ...]), train_ds[0])
params = net.init(jax.random.PRNGKey(coach.net_seed), inputs)
train_state, h_train_state = create_train_state(net,
                                                params,
                                                tx,
                                                polyak_step_size=None,
                                                plateau_lr_decay={'patience': 50,
                                                                  'decay_factor': 1.
                                                                  },
                                                scheduled_lr_decay={'exponential': {'transition_steps': 10_000,
                                                                                    'decay_factor': 0.9}
                                                                    }
                                                )

h_net = net.__dict_repr__()
h_opt = opt.__dict_repr__()
h_coach = coach.__dict_repr__()
h_dataset = data_set.__dict_repr__()
h = bundle_dicts([h_net, h_opt, h_coach, h_dataset, h_train_state])
save_dict(path=ckpt_dir, filename='hyperparameters.json', data=h, exists_ok=True)

wandb.init(config=h)
coach.run(train_state=train_state,
          train_ds=train_ds,
          valid_ds=valid_ds,
          loss_fn=loss_fn,
          log_every_t=1,
          restart_by_nan=True,
          use_wandb=True)
