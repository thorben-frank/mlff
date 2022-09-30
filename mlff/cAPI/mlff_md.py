import argparse
import os
import logging
import numpy as np

from flax.training import checkpoints
from ase.units import fs, eV
from ase import Atoms
from ase.optimize import QuasiNewton
from ase.md.langevin import Langevin
from ase.md import MDLogger
from ase.io import read
from ase.io.trajectory import Trajectory

from mlff.src.md.calculator import mlffCalculator
from mlff.src.io import read_json
from mlff.src.nn.stacknet import init_stack_net
from mlff.cAPI.process_argparse import StoreDictKeyPair, default_access

logging.basicConfig(level=logging.INFO)


ns = 10**6 * fs


def run_md():

    # Create the parser
    parser = argparse.ArgumentParser(description='Run an MD with a NN.')

    # Add the arguments
    parser.add_argument('--ckpt_dir', type=str, required=False, default=os.getcwd(),
                        help='Path to the checkpoint directory. Defaults to the current directory.')

    parser.add_argument('--start_geometry', type=str, required=False, default=None,
                        help='Path to data file that the model should be applied to. '
                             'Defaults to the training data file.')

    parser.add_argument("--units", action=StoreDictKeyPair, metavar="energy='kcal/mol',force='kcal/(mol*Ang)',KEY3=VAL3",
                        default=None, help='Units in which the NN has been trained on. If you trained your NN in default '
                                           'ASE units you do not need to specify anything here.')

    parser.add_argument("--n_interactions_max", type=int, default=None,
                        help='Maximal number of pairwise interactions. Improves the efficiency of the MD. Is set by default'
                             'to n^2 where n is the number of atoms in the system.')

    parser.add_argument("--prop_keys", action=StoreDictKeyPair, metavar="KEY1=VAL1,KEY2=VAL2...", default=None,
                        help='Property keys of the data set. Needs only to be specified, if e.g. the keys of the '
                             'properties in the data set from which the start geometry should be fetched differ from'
                             'the property keys the network has been trained on.')

    parser.add_argument('--name', type=str, required=False, default=None,
                        help='Name of the MD.')

    parser.add_argument('--save_dir', type=str, required=False, default=os.getcwd(),
                        help='Where to save the trajectory and the log file of the MD. Defaults to the current working'
                             'directory.')

    parser.add_argument('--time_step', type=float, required=False, default=0.2,
                        help='Time step in femto seconds (fs). Defaults to 0.2 fs.')

    parser.add_argument('--temperature', type=float, required=False, default=300,
                        help='MD temperature in Kelvin (K). Defaults to 300 K.')

    parser.add_argument('--friction', type=float, required=False, default=0.002,
                        help='Friction. Defaults to 0.002.')

    parser.add_argument('--total_steps', type=int, required=False, default=None)

    parser.add_argument('--total_time', type=float, required=False, default=None,
                        help='Total time of the MD in ns.')

    parser.add_argument('--log_every_step', type=int, required=False, default=None,
                        help='MD quantities are logged every t steps. Defaults to the number of MD steps that follow '
                             'from the `--resolution` argument. Assuming default values this corresponds to logging '
                             'every 10 MD steps.')

    parser.add_argument('--resolution', type=float, required=False, default=2.,
                        help='Resolution of the MD, meaning at what time interval in fs the atomic positions are '
                             'stored to the .traj file. Defaults to 2 fs, which corresponds to saving the positions '
                             'every 10 MD steps, assuming the default `--time_step` of 0.2 fs.')

    parser.add_argument('--x64', type=bool, required=False, default=True)

    parser.add_argument('--qn_tol', type=float, required=False, default=1e-4)
    parser.add_argument('--qn_max_steps', type=float, required=False, default=200)

    args = parser.parse_args()

    # Read arguments
    ckpt_dir = args.ckpt_dir
    start_geometry = args.start_geometry
    units = args.units
    prop_keys = args.prop_keys
    n_interactions_max = args.n_interactions_max
    md_name = args.name
    save_dir = args.save_dir
    resolution = args.resolution * fs

    time_step = args.time_step * fs
    friction = args.friction
    temperature = args.temperature
    total_steps = args.total_steps
    total_time = args.total_time
    log_every_step = args.log_every_step

    x64 = args.x64

    # quasi newton parameters
    qn_tol = args.qn_tol
    qn_max_steps = args.qn_max_steps

    if x64:
        from jax.config import config
        config.update("jax_enable_x64", True)

    if resolution < time_step:
        msg = 'The resolution time step can not be smaller than the MD time step.'
        raise ValueError(msg)

    traj_interval = resolution / time_step
    if np.isclose(resolution % time_step, 0):
        traj_interval_new = np.floor(traj_interval)
        logging.info('Resolution % time_step is unequal zero. We changed the interval at which positions are saved. '
                     'Old resolution was {} fs and new resolution is {} fs.'
                     .format(traj_interval*time_step/fs, traj_interval_new*time_step/fs))
        traj_interval = int(np.rint(traj_interval_new))
    else:
        traj_interval = int(np.rint(traj_interval))

    if log_every_step is None:
        log_every_step = traj_interval
    print(log_every_step)

    if total_time is None and total_steps is None:
        msg = "Either `--steps` or `--total_time` needs to be specified."
        raise ValueError(msg)
    elif total_time is not None and total_steps is not None:
        msg = "Only `--steps` or `--total_time` can be specified."
        raise ValueError(msg)
    elif total_time is not None and total_steps is None:
        steps = total_time * ns / time_step
    elif total_time is None and total_steps is not None:
        steps = total_steps
    else:
        msg = 'One should not end up here. This is likely due to a bug in the mlff package. Please report to ' \
              'https://github.com/thorben-frank/mlff'
        raise RuntimeError(msg)

    h = read_json(os.path.join(ckpt_dir, 'hyperparameters.json'))

    if start_geometry is None:
        start_geometry = h['coach']['data_path']
        logging.info('No start geometry specified. Default to randomly picking a frame from the training data set {}'
                     .format(start_geometry))

    if md_name is None:
        import datetime
        e = datetime.datetime.now()
        md_name = "MD_{}{}{}_{}{}{}".format(e.year, e.month, e.day, e.hour, e.minute, e.second)

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
    params = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=None, prefix='checkpoint_loss_')['params']

    # load the start geometry
    def load_start_geometry(f: str) -> Atoms:
        import os.path
        import numpy as np
        extension = os.path.splitext(f)[1]
        if extension == '.npz':
            data = dict(np.load(file=f))
            R: np.ndarray = data[R_key]
            z: np.ndarray = data[z_key]
            _idx = np.random.randint(low=0, high=len(R))
            if R.ndim > 2:
                R = R[_idx]
            if z.ndim > 1:
                z = z[_idx]
            _molecule = Atoms(positions=R, numbers=z)
            return _molecule

        elif extension == '.xyz':
            # TODO: deal with the case that xyz has multiple configurations. Not sure if read supports that by default.
            _molecule = read(filename=f)
            return _molecule
        else:
            msg = 'Unsupported file format {} to load start geometry from.'.format(extension)
            raise ValueError(msg)

    # load the start geometry
    molecule = load_start_geometry(f=start_geometry)

    if n_interactions_max is None:
        n_atoms = len(molecule.get_atomic_numbers())
        n_interactions_max = n_atoms ** 2

    calc = mlffCalculator(params=params,
                          stack_net=net,
                          n_interactions_max=n_interactions_max,
                          F_to_eV_Ang=default_access(conversion_table, key=F_key, default=eV),
                          E_to_eV=default_access(conversion_table, key=E_key, default=eV))

    molecule.set_calculator(calc)

    # do a geometry relaxation
    qn = QuasiNewton(molecule)
    qn.run(qn_tol, qn_max_steps)

    # # set the momenta corresponding to T=300K
    # MaxwellBoltzmannDistribution(mol, temperature_K=300)
    # Stationary(mol)  # zero linear momentum
    # ZeroRotation(mol)  # zero angular momentum

    traj = Trajectory(os.path.join(save_dir, "{}.traj".format(md_name)), 'w', molecule)
    dyn = Langevin(atoms=molecule, timestep=time_step, temperature_K=temperature, friction=friction)

    dyn.attach(traj.write, interval=traj_interval)
    dyn.attach(MDLogger(dyn,
                        atoms=molecule,
                        logfile=os.path.join(save_dir, "{}.log".format(md_name)),
                        header=True,
                        stress=False,
                        peratom=True,
                        mode="a"),
               interval=log_every_step)

    # run the MD
    dyn.run(steps=np.rint(steps))
    traj.close()


if __name__ == '__main__':
    run_md()
