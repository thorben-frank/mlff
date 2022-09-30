from setuptools import setup

setup(
    name='mlff',
    version='0.1',
    description='Build Neural Networks for Force Fields with JAX',
    packages=['mlff'],  # same as name
    install_requires=['numpy',
                      'jax',
                      'flax',
                      'optax',
                      'tensorflow',
                      'sklearn',
                      'ase',
                      'tqdm',
                      'wandb',
                      'pyyaml'],
    include_package_data=True,
    package_data={'': ['src/sph_ops/cgmatrix.npz']},
    entry_points={'console_scripts': ['evaluate=mlff.cAPI.mlff_eval:evaluate',
                                      'train=mlff.cAPI.mlff_train:train',
                                      'run_md=mlff.cAPI.mlff_md:run_md'],
                  }
)
