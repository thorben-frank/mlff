import argparse
import numpy as np
import os
from pwtools.constants import fs, rcm_to_Hz
from pwtools.pydos import pdos
from ase.io.trajectory import Trajectory

from mlff.cAPI.process_argparse import StoreDictKeyPair


def analyse_md():
    # Create the parser
    parser = argparse.ArgumentParser(description='Run an MD with a NN.')

    parser.add_argument('--observable', type=str, required=True,
                        help='Observable to plot.')
    parser.add_argument("--kwargs", action=StoreDictKeyPair,
                        metavar="KEY1=VAL2,KEY2=VAL2,KEY3=VAL3,...",
                        default=None)

    args = parser.parse_args()
    # random seed
    obs = args.observable
    kwargs = args.kwargs

    if obs == "auto_correlation":
        calculate_auto_correlation(**kwargs)
    else:
        msg = f"Analyse function not implemented for {obs}."
        raise NotImplementedError(msg)


def calculate_auto_correlation(time_step):
    cwd = os.getcwd()
    tr = Trajectory(os.path.join(cwd, 'atom.traj'))

    # read velocities
    V = np.array([atoms.get_velocities() for atoms in tr])

    freq, dos = pdos(V, dt=time_step * fs, method='direct', npad=1)
    np.savez(os.path.join(cwd, 'auto_correlation.npz'), **{'freq': freq/rcm_to_Hz, 'dos': dos})


if __name__ == '__main__':
    analyse_md()
