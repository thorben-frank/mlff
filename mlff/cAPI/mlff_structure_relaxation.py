import argparse
import numpy as np
import os
import logging

from ase.units import *
from ase import Atoms
import ase.optimize as ase_opt

from mlff.md.calculator import mlffCalculator
from mlff.io import read_json, create_directory, load_params_from_ckpt_dir
from mlff.nn.stacknet import init_stack_net
from mlff.cAPI.process_argparse import StoreDictKeyPair, default_access
from mlff.data import DataSet
from mlff.random.random import set_seeds

logging.basicConfig(level=logging.INFO)

ns = 10 ** 6 * fs


def run_relaxation():
    # Create the parser
    parser = argparse.ArgumentParser(description='Structure relaxation using ASE Calculator and MLFF NN.')

    # Add the arguments
    parser.add_argument('--ckpt_dir', type=str, required=False, default=os.getcwd(),
                        help='Path to the checkpoint directory. Defaults to the current directory.')

    parser.add_argument('--start_geometry', type=str, required=False, default=None,
                        help='Path to data file that the model should be applied to. '
                             'Defaults to the training data file.')

    parser.add_argument('--from_split', type=str, required=False, default=None,
                        help='Use random configuration from the test set of a given data split.')

    parser.add_argument("--units", action=StoreDictKeyPair,
                        metavar="energy='kcal/mol',force='kcal/(mol*Ang)',KEY3=VAL3",
                        default=None,
                        help='Units in which the NN has been trained on. If you trained your NN in default '
                             'ASE units you do not need to specify anything here.')

    parser.add_argument("--n_interactions_max", type=int, default=None,
                        help='Maximal number of pairwise interactions. Improves the efficiency of the MD. Is set by default'
                             'to n^2 where n is the number of atoms in the system.')

    parser.add_argument("--prop_keys", action=StoreDictKeyPair, metavar="KEY1=VAL1,KEY2=VAL2...", default=None,
                        help='Property keys of the data set. Needs only to be specified, if e.g. the keys of the '
                             'properties in the data set from which the start geometry should be fetched differ from'
                             'the property keys the network has been trained on.')

    parser.add_argument('--save_dir', type=str, required=False, default=None,
                        help='Where to save the trajectory and the log file of the MD. Defaults to the current working'
                             'directory.')

    parser.add_argument('--x64', type=bool, required=False, default=False)

    parser.add_argument('--qn_tol', type=float, required=False, default=1e-4)
    parser.add_argument('--qn_max_steps', type=int, required=False, default=200)

    parser.add_argument('--optimizer', type=str, required=False, default='QuasiNewton')

    parser.add_argument('--mic', type=str, required=False, default=None,
                        help='Minimal image convention.')

    parser.add_argument('--random_seed', type=int, required=False, default=0,
                        help='Random seed for the MD.')

    parser.add_argument("--use_mdx", action="store_true", default=False, help="Use mdx for structure relaxation.")

    parser.add_argument('--mdx_dtype', type=str, required=False, default='x64',
                        help='Dtype used in the mdx MD. Defaults to x64, since x32 can lead to instabilities in the '
                             'optimizer.')

    args = parser.parse_args()

    # Read arguments
    ckpt_dir = args.ckpt_dir
    start_geometry = args.start_geometry
    from_split = args.from_split
    units = args.units
    prop_keys = args.prop_keys
    n_interactions_max = args.n_interactions_max
    # md_name = args.name
    restart = False
    save_dir = os.path.join(os.getcwd(), 'relax') if args.save_dir is None else os.path.join(args.save_dir, 'relax')
    save_dir = create_directory(save_dir, exists_ok=restart)

    mic = args.mic

    x64 = args.x64

    if args.mdx_dtype == 'x64':
        from jax import config
        config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    _mdx_dtype = jnp.float64 if args.mdx_dtype == 'x64' else jnp.float32

    # random seed
    seed = args.random_seed

    # quasi newton parameters
    qn_tol = args.qn_tol
    qn_max_steps = args.qn_max_steps

    # set the seed
    set_seeds(seed)

    if x64:
        from jax import config
        config.update("jax_enable_x64", True)

    h = read_json(os.path.join(ckpt_dir, 'hyperparameters.json'))

    if start_geometry is None:
        start_geometry = h['coach']['data_path']
        logging.info('No start geometry specified. Default to randomly picking a frame from the training data set {}'
                     .format(start_geometry))

    net = init_stack_net(h)
    _prop_keys = net.prop_keys
    if prop_keys is not None:
        _prop_keys.update(prop_keys)
        net.reset_prop_keys(prop_keys=_prop_keys)
    prop_keys = net.prop_keys

    R_key = prop_keys['atomic_position']
    z_key = prop_keys['atomic_type']

    # TODO: Default to some force key if it is missing from prop keys and just assume that NN has only be trained on
    #  energies. Give a warning though.
    try:
        F_key = prop_keys['force']
    except KeyError:
        msg = "The NN needs to output forces to be used in force field calculations. Your prop_keys are missing the " \
              "'force' key which can be either due to wrongly set prop_keys or since the network does not output forces." \
              "Without forces we can not run the MD. If you trained your model only on energies and this is why the " \
              "StackNet does not have the 'force' key, you can just add `--prop_keys force=F` as argument to the" \
              "`run_md` command."
        raise ValueError(msg)

    try:
        E_key = prop_keys['energy']
    except KeyError:
        msg = "The NN needs to predict energies to be used in an MD. Your prop_keys are missing the `energy` key which " \
              "can be either due to wrongly set prop_keys or since the network does not output energies."
        raise RuntimeError(msg)

    # read the units
    conversion_table = {}
    if units is not None:
        for (q, v) in units.items():
            k = prop_keys[q]
            conversion_table[k] = eval(v)

    # load the parameters
    params = load_params_from_ckpt_dir(ckpt_dir)

    # load the start geometry
    def load_start_geometry(f: str) -> Atoms:
        import os.path
        import numpy as np
        extension = os.path.splitext(f)[1]

        if extension == '.npz':
            loaded_data = dict(np.load(file=f))
        else:
            from mlff.data import AseDataLoader
            data_loader = AseDataLoader(f, load_stress=False, load_energy_and_forces=False)
            loaded_data = data_loader.load_all()

        if from_split is not None:
            test_data_set = DataSet(prop_keys=prop_keys, data=loaded_data)
            test_data_set.load_split(file=os.path.join(ckpt_dir, 'splits.json'),
                                     n_train=0,
                                     n_valid=0,
                                     n_test=None,
                                     r_cut=None,
                                     split_name=from_split)
            data = test_data_set.get_data_split()['test']
        else:
            data = loaded_data

        R: np.ndarray = data[R_key]
        z: np.ndarray = data[z_key]
        _idx = np.random.randint(low=0, high=len(R))
        if R.ndim > 2:
            R = R[_idx]
        if z.ndim > 1:
            z = z[_idx]
        if mic is not None:
            unit_cell_key = prop_keys['unit_cell']
            unit_cell: np.ndarray = data[unit_cell_key][_idx]
            pbc_key = prop_keys['pbc']
            pbc_array: np.ndarray = data[pbc_key][_idx]
            _molecule = Atoms(positions=R, numbers=z, cell=unit_cell, pbc=pbc_array)
        else:
            _molecule = Atoms(positions=R, numbers=z)
        return _molecule

    # load the start geometry
    molecule = load_start_geometry(f=start_geometry)
    if args.use_mdx:
        from mlff import mdx

        potential = mdx.MLFFPotential.create_from_ckpt_dir(ckpt_dir=ckpt_dir, add_shift=True, dtype=_mdx_dtype)

        atomsx = mdx.AtomsX.create(molecule, dtype=_mdx_dtype)
        atomsx = atomsx.init_spatial_partitioning(cutoff=potential.cutoff,
                                                  skin=0.,
                                                  capacity_multiplier=1.25
                                                  )

        optimizer = mdx.LBFGS.create(atoms=atomsx,
                                     potential=potential,
                                     save_dir=save_dir)
        optimizer.minimize(atomsx, max_steps=qn_max_steps, tol=qn_tol)
    else:
        from mlff import mdx

        # if n_interactions_max is None:
        #     n_atoms = len(molecule.get_atomic_numbers())
        #     n_interactions_max = n_atoms ** 2
        #
        # scales = read_json(os.path.join(ckpt_dir, 'scales.json'))

        calc = mlffCalculator.create_from_ckpt_dir(
            ckpt_dir=ckpt_dir,
            capacity_multiplier=1.25,
            add_energy_shift=False,
            F_to_eV_Ang=default_access(conversion_table, key=F_key, default=eV),
            E_to_eV=default_access(conversion_table, key=E_key, default=eV),
            dtype=np.float64,
        )

        molecule.set_calculator(calc)

        # save the structure before the relaxation
        from ase.io import write
        write(os.path.join(save_dir, 'init_structure.xyz'), molecule)
        # do a geometry relaxation
        qn = getattr(ase_opt, args.optimizer)(molecule)
        converged = qn.run(qn_tol, qn_max_steps)
        if converged:
            write(os.path.join(save_dir, 'relaxed_structure.xyz'), molecule)
        else:
            raise RuntimeError('Geometry optimization did not converge!')


if __name__ == '__main__':
    run_relaxation()
