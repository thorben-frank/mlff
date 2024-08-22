import argparse
import os
import logging
import numpy as np
import jax

from ase.units import *
from ase import Atoms
from ase.optimize import QuasiNewton

from mlff.md.calculator import mlffCalculator
from mlff.io import read_json, create_directory, load_params_from_ckpt_dir
from mlff.nn.stacknet import init_stack_net
from mlff.cAPI.process_argparse import StoreDictKeyPair, default_access, str2bool
from mlff.data import DataSet
from mlff.md.integrator import NoseHoover, Langevin, VelocityVerlet
from mlff.md.simulate import Simulator
from mlff.random.random import set_seeds

logging.basicConfig(level=logging.INFO)

ns = 10 ** 6 * fs


def run_md():
    # Create the parser
    parser = argparse.ArgumentParser(description='Run an MD with a NN.')

    parser.add_argument('--thermostat', type=str, required=True,
                        help='Thermostat used for MD.')

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

    # parser.add_argument('--name', type=str, required=False, default=None,
    #                     help='Name of the MD.')

    parser.add_argument('--save_dir', type=str, required=False, default=None,
                        help='Where to save the trajectory and the log file of the MD. Defaults to the current working'
                             'directory.')

    parser.add_argument('--time_step', type=float, required=False, default=0.2,
                        help='Time step in femto seconds (fs). Defaults to 0.2 fs.')

    parser.add_argument('--temperature', type=float, required=False,
                        help='MD temperature in Kelvin (K).')

    parser.add_argument('--temperature_init', type=float, required=False, default=None,
                        help='Init temperature in Kelvin (K). Defaults to the MD temperature.')

    parser.add_argument("--zero_linear_momentum", type=str2bool, nargs='?', const=True, default=True,
                        help='Set the linear momentum (Defaults to True).')

    parser.add_argument("--zero_angular_momentum", type=str2bool, nargs='?', const=True, default=True,
                        help='Set the linear momentum (Defaults to True).')

    # parser.add_argument('--zero_linear_momentum', type=bool, required=False, default=True,
    #                     help='Set the linear momentum (Defaults to True).')
    #
    # parser.add_argument('--zero_angular_momentum', type=bool, required=False, default=True,
    #                     help='Set the angular momentum (Defaults to True).')

    parser.add_argument('--friction', type=float, required=False, default=0.002,
                        help='Friction. Defaults to 0.002.')

    parser.add_argument('--total_steps', type=int, required=False, default=None)

    parser.add_argument('--total_time', type=float, required=False, default=None,
                        help='Total time of the MD in ns.')

    parser.add_argument('--save_frequency', type=float, required=False, default=100,
                        help='At what frequency MD information is saved. Defaults to 100 time steps.')

    # parser.add_argument('--x64', type=bool, required=False, default=False)
    # parser.add_argument("--x64", action="store_true", default=False, help="Dtype for ASE ")
    # parser.add_argument("--x64", type=str2bool, nargs='?', const=True, default=False)

    parser.add_argument('--qn_tol', type=float, required=False, default=1e-4)
    parser.add_argument('--qn_max_steps', type=float, required=False, default=200)

    parser.add_argument('--mic', type=str, required=False, default=None,
                        help='Minimal image convention.')

    parser.add_argument('--random_seed', type=int, required=False, default=0,
                        help='Random seed for the MD.')

    parser.add_argument("--use_mdx", type=str2bool, nargs='?', const=True, default=False,
                        help='Use mdx as package for molecular dynamics. WARNING: mdx is experimental and under '
                             'heavy development!!'
                        )

    # parser.add_argument('--use_mdx', type=bool, required=False, default=False,
    #                     help='Use mdx as package for molecular dynamics. WARNING: mdx is experimental and under '
    #                          'heavy development!!')

    parser.add_argument('--mdx_skin', type=float, required=False, default=0.,
                        help='Skin for neighborhood calculation. Defaults to 0.')

    parser.add_argument('--mdx_unfolding_skin', type=float, required=False, default=0.,
                        help='Skin for spatial unfolding. Defaults to 0.')

    parser.add_argument('--capacity_multiplier', type=float, required=False, default=1.25,
                        help='Capacity multiplier for neighborhood calculation. Defaults to 1.25')

    parser.add_argument('--mdx_capacity_multiplier', type=float, required=False, default=None,
                        help='Capacity multiplier for neighborhood calculation. Defaults to 1.25')

    parser.add_argument('--mdx_scan_interval', type=int, required=False, default=1000.,
                        help='Scan interval for the number of MD steps. Defaults to 1000.')

    parser.add_argument('--mdx_dtype', type=str, required=False, default='x32',
                        help='Dtype used in the mdx MD.')

    # parser.add_argument('--mdx_heat_flux', type=bool, required=False, default=False,
    #                     help='Calculate the heat flux.')

    parser.add_argument('--mdx_save_frequency', type=float, required=False, default=1,
                        help='At what frequency MD information is saved. Defaults to 1 scan.')

    parser.add_argument('--mdx_opt', type=str, required=False, default=None,
                        help='Optimizer used for structure relaxation.')

    parser.add_argument('--mdx_opt_max_steps', type=int, required=False, default=5000,
                        help='Maximal number of steps used for structure relaxation.')

    parser.add_argument('--mdx_opt_tol', type=float, required=False, default=1e-3,
                        help='Maximal L2 norm of atomic forces for which convergence is assumed. Defaults to '
                             '0.001 eV/Ang.')

    parser.add_argument('--mdx_opt_lr', type=float, required=False, default=1e-2,
                        help='Step size for steps during structure relaxation. Defaults to 1e-2.')

    parser.add_argument('--mdx_langevin_fixrot', action="store_true", default=False,
                        help='')

    parser.add_argument('--mdx_langevin_fixcm', action="store_true", default=False,
                        help='')

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
    save_dir = os.path.join(os.getcwd(), 'md') if args.save_dir is None else os.path.join(args.save_dir, 'md')
    save_dir = create_directory(save_dir, exists_ok=restart)
    save_frequency = args.save_frequency

    _thermostat = args.thermostat

    time_step = args.time_step * fs
    friction = args.friction

    temperature = None
    if args.temperature is not None:
        temperature = args.temperature * kB

    # defaults to args.temperature if not set
    temperature_init = temperature if args.temperature_init is None else args.temperature_init * kB
    total_steps = args.total_steps
    total_time = args.total_time * ns
    # log_every_step = args.log_every_step
    use_mdx = args.use_mdx

    mdx_skin = args.mdx_skin
    if mdx_skin != 0:
        raise NotImplementedError('--mdx_skin != 0, not supported yet!')

    if args.mdx_capacity_multiplier is not None:
        raise DeprecationWarning('`--mdx_capacity_multiplier` is outdated. Use `--capacity_multiplier` instead.')

    capacity_multiplier = args.capacity_multiplier
    mdx_scan_interval = args.mdx_scan_interval

    if args.mdx_dtype == 'x64':
        from jax import config
        config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    _mdx_dtype = jnp.float64 if args.mdx_dtype == 'x64' else jnp.float32

    mic = args.mic

    # x64 = args.x64

    # random seed
    seed = args.random_seed

    # quasi newton parameters
    qn_tol = args.qn_tol
    qn_max_steps = args.qn_max_steps

    # set the seed
    set_seeds(seed)

    if _thermostat == 'velocity_verlet' and temperature is not None:
        raise RuntimeError(f'For thermostat {_thermostat} no temperature can be specified since it is not actively'
                           f'controlled during simulation. Rather, the `--temperature_init` argument should be '
                           f'specified, where `--temperature_init = 2*target_temperature`.')

    if _thermostat in ('langevin', 'nose_hoover'):
        try:
            assert temperature is not None
        except AssertionError:
            raise RuntimeError(f'For thermostat {_thermostat} a target temperature must be specified via '
                               f'the `--temperature` argument.')

    # if x64:
    #     from jax.config import config
    #     config.update("jax_enable_x64", True)

    if total_time is None and total_steps is None:
        msg = "Either `--steps` or `--total_time` needs to be specified."
        raise ValueError(msg)
    elif total_time is not None and total_steps is not None:
        msg = "Only `--steps` or `--total_time` can be specified."
        raise ValueError(msg)
    elif total_time is not None and total_steps is None:
        steps = total_time / time_step
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

    if n_interactions_max is None:
        n_atoms = len(molecule.get_atomic_numbers())
        n_interactions_max = n_atoms ** 2

    scales = read_json(os.path.join(ckpt_dir, 'scales.json'))

    if not use_mdx:
        from mlff import mdx

        potential = mdx.MLFFPotential.create_from_ckpt_dir(ckpt_dir=ckpt_dir, dtype=_mdx_dtype)
        calc = mlffCalculator(potential=potential,
                              capacity_multiplier=capacity_multiplier,
                              F_to_eV_Ang=default_access(conversion_table, key=F_key, default=eV),
                              E_to_eV=default_access(conversion_table, key=E_key, default=eV),
                              mic=mic)

        molecule.set_calculator(calc)

        # do a geometry relaxation
        qn = QuasiNewton(molecule)
        qn.run(qn_tol, qn_max_steps)

        if _thermostat == 'nose_hoover':
            integrator = NoseHoover(atoms=molecule,
                                    timestep=time_step,
                                    temperature=temperature,
                                    ttime=20,
                                    trajectory=None,
                                    logfile=None,
                                    loginterval=1, )
        elif _thermostat == 'langevin':
            integrator = Langevin(atoms=molecule,
                                  timestep=time_step,
                                  temperature=temperature,
                                  friction=friction,
                                  trajectory=None,
                                  logfile=None,
                                  loginterval=1, )
        elif _thermostat == 'velocity_verlet':
            integrator = VelocityVerlet(atoms=molecule,
                                        timestep=time_step,
                                        trajectory=None,
                                        logfile=None,
                                        loginterval=1, )
        else:
            msg = f"Unknown thermostat `{_thermostat}`."
            raise ValueError(msg)

        simulator = Simulator(molecule,
                              integrator,
                              temperature_init,
                              zero_linear_momentum=args.zero_linear_momentum,
                              zero_angular_momentum=args.zero_angular_momentum,
                              save_frequency=save_frequency,
                              save_dir=save_dir,
                              restart=False)

        simulator.run(int(np.rint(steps)))

    else:
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
        from mlff import mdx

        MaxwellBoltzmannDistribution(molecule, temp=temperature_init)

        potential = mdx.MLFFPotential.create_from_ckpt_dir(ckpt_dir=ckpt_dir, dtype=_mdx_dtype)

        # initialize the atomsx object
        atomsx = mdx.AtomsX.create(atoms=molecule, dtype=_mdx_dtype)

        # initialize spatial partitioning for structure relaxation
        atomsx = atomsx.init_spatial_partitioning(cutoff=potential.cutoff,
                                                  skin=mdx_skin,
                                                  capacity_multiplier=capacity_multiplier
                                                  )

        if args.mdx_opt is not None:
            raise ValueError('Structure relaxation has to be done using run_relaxation CLI.')

        # scale momenta to the initial temperature
        atomsx = mdx.scale_momenta(atomsx, T0=temperature_init)

        # zero linear and angular momentum
        if args.zero_linear_momentum:
            atomsx = mdx.zero_translation(atomsx, preserve_temperature=True)
        if args.zero_angular_momentum:
            atomsx = mdx.zero_rotation(atomsx, preserve_temperature=True)

        # set the calculator
        # if args.mdx_heat_flux:
        #     atomsx = atomsx.init_spatial_unfolding(cutoff=potential.effective_cutoff,
        #                                            skin=args.mdx_unfolding_skin)
        #     calc = mdx.HeatFluxCalculatorX.create(potential)
        # else:
        calc = mdx.CalculatorX.create(potential)

        atomsx = atomsx.init_spatial_partitioning(cutoff=potential.cutoff,
                                                  skin=mdx_skin,
                                                  capacity_multiplier=capacity_multiplier
                                                  )

        # choose an integrator
        if _thermostat == 'nose_hoover':
            integratorx = mdx.NoseHooverX.create(timestep=time_step,
                                                 ttime=20,
                                                 temperature=temperature,
                                                 calculator=calc)
        elif _thermostat == 'langevin':
            integratorx = mdx.LangevinX.create(timestep=time_step,
                                               temperature=temperature,
                                               calculator=calc,
                                               friction=friction)
        elif _thermostat == 'langevin_baoab':
            integratorx = mdx.BAOABLangevinX.create(timestep=time_step,
                                                    temperature=temperature,
                                                    calculator=calc,
                                                    gamma=friction,
                                                    fixrot=args.mdx_langevin_fixrot,
                                                    fixcm=args.mdx_langevin_fixcm)
        elif _thermostat == 'velocity_verlet':
            integratorx = mdx.VelocityVerletX.create(timestep=time_step,
                                                     calculator=calc)
        else:
            msg = f"Unknown thermostat mdx `{_thermostat}`."
            raise ValueError(msg)

        simulator = mdx.SimulatorX(n_atoms=len(molecule.get_atomic_numbers()),
                                   save_dir=save_dir,
                                   save_frequency=args.mdx_save_frequency,
                                   run_interval=mdx_scan_interval)

        simulator.run(integratorx, atomsx, steps=steps)


if __name__ == '__main__':
    run_md()
