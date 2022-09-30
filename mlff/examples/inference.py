import numpy as np
import jax
import logging
import os

from mlff.src.training import Coach
from mlff.src.data.data import DataTuple
from mlff.src.data.dataset import DataSet
from mlff.src.nn.stacknet import get_obs_and_force_fn

logging.basicConfig(level=logging.INFO)
from mlff.src.io.io import read_json
from mlff.src.nn.stacknet import init_stack_net
from flax.training import checkpoints


# We start by giving the path to the data we want to train our model on, as well as the path where we want
# to save the model checkpoints as well as the hyperparamter file.

data_path = '/Users/thorbenfrank/Documents/data/mol-data/dft/ethanol_dft.npz'
data_path = '/Users/thorbenfrank/Documents/data/mol-data/cumulene/cumulene8_test.npz'
save_path = '/Users/thorbenfrank/Desktop/cumulene/'
ckpt_dir = os.path.join(save_path, 'module')

# next we define the property keys. The keys in the dictionary should not be changed, but the values can be
# changed arbitrarily such that they match the names in the data file.

prop_keys = {'energy': 'E',
             'force': 'F',
             'atomic_type': 'z',
             'atomic_position': 'R',
             'hirshfeld_volume': 'V_eff',
             'total_charge': None,
             'total_spin': None,
             'partial_charge': None
             }

R_key = prop_keys['atomic_position']
E_key = prop_keys['energy']
F_key = prop_keys['force']
z_key = prop_keys['atomic_type']
Vh_key = prop_keys['hirshfeld_volume']
q_key = prop_keys['partial_charge']
Q_key = prop_keys['total_charge']

# prop_keys_new = {'energy': 'ePBE0+MBD',
#                  'force': 'totFOR',
#                  'atomic_type': 'atNUM',
#                  'atomic_position': 'atXYZ',
#                  'hirshfeld_volume': 'hRAT',
#                  'total_charge': 'Q',
#                  'total_spin': None,
#                  'partial_charge': 'hCHG'
#                  }

h = read_json(os.path.join(ckpt_dir, 'hyperparameters.json'))
coach = Coach(**h['coach'])

data = dict(np.load(data_path))
data_set = DataSet(prop_keys=prop_keys, data=data)
d = data_set.strat_split(n_train=1,
                         n_valid=1,
                         n_test=None,
                         training=False,
                         r_cut=np.inf,
                         strat_key=E_key,
                         seed=0)

data_tuple = DataTuple(input_keys=coach.input_keys,
                       target_keys=coach.target_keys)
test_input, test_obs = data_tuple(d['test'])

test_net = init_stack_net(h)
obs_fn = get_obs_and_force_fn(test_net)
obs_fn = jax.jit(jax.vmap(obs_fn, in_axes=(None, 0)))
params = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=None, prefix='checkpoint_loss_')['params']
# params['record'] = {}
#
#
# def apply_fn(x):
#     return test_net.apply(params, x, mutable=['record'])
#
#
# apply_fn = jax.jit(jax.vmap(apply_fn, in_axes=0))
# obs_pred, records = apply_fn(jax.tree_map(lambda x: jnp.array(x), test_input))
# pprint(records)

obs_pred = obs_fn(params, test_input)
import matplotlib.pyplot as plt
plt.scatter(np.arange(len(obs_pred[E_key])), obs_pred[E_key].reshape(-1))
print('done')