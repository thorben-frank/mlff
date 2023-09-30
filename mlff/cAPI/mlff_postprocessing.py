import argparse
import numpy as np
import h5py

from ase import Atoms
from ase.io import write
from tqdm import tqdm


def trajectory_to_xyz():
    parser = argparse.ArgumentParser(description='Convert an .h5 trajectory file to an XYZ file.')

    parser.add_argument('--trajectory', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--save_every_i', type=int, required=False, default=1)
    parser.add_argument('--save_from_i', type=int, required=False, default=0)
    parser.add_argument('--append', type=str, required=False, default=False)
    parser.add_argument('--lazy_io', action="store_true", required=False, help='')

    args = parser.parse_args()

    trajectory_file = args.trajectory
    output_file = args.output
    save_from_i = args.save_from_i
    lazy_io = args.lazy_io

    if (save_from_i > 0) and lazy_io:
        raise NotImplementedError

    trajectory = h5py.File(trajectory_file)

    if not lazy_io:
        pos = np.array(trajectory['positions'][:])
        pos = pos.reshape(-1, *pos.shape[-2:])[save_from_i:][::args.save_every_i]
        # shape: (steps%save_every_i, n_atoms, 3)
        try:
            vel = np.array(trajectory['velocities'][:])
            vel = vel.reshape(-1, *vel.shape[-2:])[save_from_i:][::args.save_every_i]
            # shape: (steps%save_every_i, n_atoms, 3)
        except KeyError:
            import logging
            logging.warning('No velocities detected in trajectory. Converting to trajectory to XYZ without them.')
            vel = np.zeros_like(pos)

        z = np.array(trajectory['atomic_numbers'][:])
        z = z.reshape(-1, z.shape[-1])[0]  # shape: (n_atoms)

        trajectory.close()

        frames = []
        print(f'Creating frames from {trajectory_file} ... ')
        for p, v in tqdm(zip(pos, vel)):
            frames += [Atoms(positions=p, numbers=z, velocities=v)]
        print(f'... done!')

        print(f'Save frames to {output_file} ... ')
        write(filename=output_file, images=frames, format='xyz', append=args.append)
        print(f'... done!')
    else:
        z = np.array(trajectory['atomic_numbers'][0]).reshape(-1)

        def save_batch(pos_B, vel_B, r=None):
            frames = []
            start = 0 if r is None else args.save_every_i - r - 1

            if vel_B is None:
                for p, v in zip(pos_B[start::args.save_every_i], np.zeros_like(pos_B)[start::args.save_every_i]):
                    frames += [Atoms(positions=p, numbers=z, velocities=v)]
            else:
                for p, v in zip(pos_B[start::args.save_every_i], vel_B[start::args.save_every_i]):
                    frames += [Atoms(positions=p, numbers=z, velocities=v)]

            write(filename=output_file, images=frames, format='xyz', append=True)

            r = (len(pos_B) - start - 1) % args.save_every_i
            return r

        pos_lazy = trajectory['positions']
        try:
            vel_lazy = trajectory['velocities']
            print(f'Reading frames from {trajectory_file} and saving them to {output_file} ...')
            rest = None
            for pos_batch, vel_batch in tqdm(zip(pos_lazy, vel_lazy)):
                rest = save_batch(pos_batch, vel_batch, rest)

        except KeyError:
            import logging
            logging.warning('No velocities detected in trajectory. Converting to trajectory to XYZ without them.')
            print(f'Reading frames from {trajectory_file} and saving them to {output_file} ...')
            rest = None
            for pos_batch in tqdm(pos_lazy):
                rest = save_batch(pos_batch, None, rest)
        print(f'... done!')

        trajectory.close()
