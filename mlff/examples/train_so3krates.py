import numpy as np
import jax
import jax.numpy as jnp
import logging
import os
import wandb
import ase.units as si

from mlff.src.io import create_directory, bundle_dicts, save_dict
from mlff.src.training import Coach, Optimizer, get_loss_fn, create_train_state
from mlff.src.data import DataTuple, DataSet

from mlff.src.nn.stacknet import StackNet, get_obs_and_force_fn, get_observable_fn
from mlff.src.nn.embed import AtomTypeEmbed, GeometryEmbed
from mlff.src.nn.layer import So3kratesLayer
from mlff.src.nn.observable import Energy
from jax.config import config
#config.update("jax_enable_x64", True)

# config.update("jax_log_compiles", 1)
# config.update('jax_disable_jit', True)

logging.basicConfig(level=logging.INFO)

# data_path = '/Users/thorbenfrank/Documents/data/mol-data/dft/ethanol_dft.npz'
# data_path = '/Users/thorbenfrank/Documents/data/mol-data/cumulene/cumulene8_train.npz'
# data_path = 'example_data/qm7x_226.npz'
data_path = 'example_data/qm7x_250_small.npz'
save_path = '/Users/thorbenfrank/Desktop/ethanol'
ckpt_dir = os.path.join(save_path, 'module')
ckpt_dir = create_directory(ckpt_dir, exists_ok=True)

prop_keys = {'energy': 'E',
             'force': 'F',
             'atomic_type': 'z',
             'atomic_position': 'R',
             'hirshfeld_volume': 'V_eff',
             'total_charge': 'Q',
             'total_spin': None,
             'partial_charge': 'q',
             'total_dipole_moment': 'vDIP',
             'total_quadrupole_moment': 'vQT'
             }

# prop_keys = {'energy': 'ePBE0+MBD',
#              'force': 'totFOR',
#              'atomic_type': 'atNUM',
#              'atomic_position': 'atXYZ',
#              'hirshfeld_volume': 'hRAT',
#              'total_charge': 'Q',
#              'total_spin': None,
#              'partial_charge': 'q',
#              'total_dipole_moment': 'vDIP'
#              }
E_key = prop_keys['energy']
F_key = prop_keys['force']
R_key = prop_keys['atomic_position']
z_key = prop_keys['atomic_type']
# Vh_key = prop_keys['hirshfeld_volume']
Q_key = prop_keys['total_charge']
# mu_key = prop_keys['dipole_moment']

data = dict(np.load(data_path))
# data[E_key] = data[E_key] * si.kcal / si.mol
# data[F_key] = data[F_key] * si.kcal / si.mol
data[Q_key] = np.repeat(np.array([0.]), repeats=len(data[E_key]), axis=0)[:, None]

r_cut = 5
data_set = DataSet(data=data, prop_keys=prop_keys)
d = data_set.strat_split(n_train=100,
                         n_valid=50,
                         n_test=None,
                         r_cut=r_cut,
                         training=True,
                         seed=0,
                         strat_key=E_key)
data_set.save_splits_to_file(ckpt_dir, 'splits.json')
n_atoms = d['train'][z_key].shape[-1]
E_mean = d['train']['{}_mean'.format(E_key)]
F_scale = d['train']['{}_scale'.format(F_key)]

embeddings = [AtomTypeEmbed(num_embeddings=100, features=32, prop_keys=prop_keys)]
geometry_embeddings = [GeometryEmbed(degrees=[1, 2],
                                     radial_basis_function='phys',
                                     n_rbf=32,
                                     radial_cutoff_fn='cosine_cutoff_fn',
                                     r_cut=r_cut,
                                     prop_keys=prop_keys,
                                     sphc=True
                                     )]

so3krates_layer = [So3kratesLayer(fb_filter='radial_spherical',
                                  fb_rad_filter_features=[32, 32],
                                  fb_sph_filter_features=[32, 32],
                                  fb_attention='conv_att',
                                  gb_filter='radial_spherical',
                                  gb_rad_filter_features=[32, 32],
                                  gb_sph_filter_features=[32, 32],
                                  gb_attention='conv_att',
                                  degrees=[1, 2],
                                  n_heads=2,
                                  chi_cut=None,
                                  chi_cut_dynamic=False,
                                  feature_layer_norm=True,
                                  sphc_layer_norm=False
                                  ) for _ in range(2)]

obs = [Energy(per_atom_scale=[F_scale.tolist()] * 20, per_atom_shift=[(E_mean / n_atoms).tolist()] * 20,
              prop_keys=prop_keys)]

net = StackNet(geometry_embeddings=geometry_embeddings,
               feature_embeddings=embeddings,
               layers=so3krates_layer,
               observables=obs,
               prop_keys=prop_keys)

obs_fn = get_obs_and_force_fn(net)
obs_fn = jax.vmap(obs_fn, in_axes=(None, 0))

opt = Optimizer(clip_by_global_norm=1)

tx = opt.get(learning_rate=1e-3)

coach = Coach(input_keys=[R_key, z_key, 'idx_i', 'idx_j', Q_key],
              target_keys=[E_key, F_key],
              epochs=1000,
              training_batch_size=5,
              validation_batch_size=5,
              loss_weights={E_key: .0, F_key: 0.99},
              ckpt_dir=ckpt_dir,
              data_path=data_path,
              net_seed=0,
              training_seed=0)

loss_fn = get_loss_fn(obs_fn=obs_fn, weights=coach.loss_weights)

data_tuple = DataTuple(input_keys=coach.input_keys,
                       target_keys=coach.target_keys)
# `target_keys` determines which quantities must appear in output of the network and in `weights` of the loss_fn
# the network might return additional observables, but they will not be used for training (in the loss function)

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
                                                                                    'decay_factor': 0.5}}
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
          ckpt_overwrite=True,
          eval_every_t=50,
          log_every_t=1,
          restart_by_nan=True)
