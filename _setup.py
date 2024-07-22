from setuptools import setup, find_packages

setup(
    name="mlff",
    version="1.0",
    description="Build Neural Networks for Force Fields with JAX",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "clu == 0.0.9",
        # "jax == 0.4.8",
        "e3x",
        "flax",
        "jaxopt",
        "jraph",
        "optax",
        "orbax-checkpoint",
        "portpicker",
        "pandas",
        # 'tensorflow',
        "scikit-learn",
        "ase",
        "tqdm",
        "wandb",
        "pyyaml",
        "pytest",
        "h5py",
        "ml_collections"
    ],
    include_package_data=True,
    package_data={"": ["sph_ops/cgmatrix.npz",
                       "sph_ops/u_matrix.pickle"]},
    entry_points={
        "console_scripts": [
            "evaluate=mlff.cAPI.mlff_eval:evaluate",
            "train=mlff.cAPI.mlff_train:train",
            "run_md=mlff.cAPI.mlff_md:run_md",
            "run_relaxation=mlff.cAPI.mlff_structure_relaxation:run_relaxation",
            "analyse_md=mlff.cAPI.mlff_analyse:analyse_md",
            "train_so3krates=mlff.cAPI.mlff_train_so3krates:train_so3krates",
            "train_so3kratACE=mlff.cAPI.mlff_train_so3kratace:train_so3kratace",
            "trajectory_to_xyz=mlff.cAPI.mlff_postprocessing:trajectory_to_xyz",
            "to_mlff_input=mlff.cAPI.mlff_input_processing:to_mlff_input",
            "train_so3krates_sparse=mlff.CLI.run_training:train_so3krates_sparse",
            "train_itp_net=mlff.CLI.run_training_itp_net:train_itp_net",
            "evaluate_itp_net=mlff.CLI.run_evaluation_itp_net:evaluate_itp_net",
            "evaluate_itp_net_on=mlff.CLI.run_evaluation_itp_net_on:evaluate_itp_net_on",
            "fine_tune_so3krates_sparse=mlff.CLI.run_fine_tuning:fine_tune_so3krates_sparse",
            "evaluate_so3krates_sparse=mlff.CLI.run_evaluation:evaluate_so3krates_sparse",
            "evaluate_so3krates_sparse_on=mlff.CLI.run_evaluation_on:evaluate_so3krates_sparse_on"
        ],
    },
)

# [build-system]
# requires = ["setuptools>=42", "wheel"]
# build-backend = "setuptools.build_meta"
#
# [project]
# name = "mlff"
# version = "1.0"
# description = "Build Neural Networks for Force Fields with JAX"
# requires-python = ">=3.9"
# dependencies = [
#     "numpy",
#     "clu == 0.0.9",
#     "e3x",
#     "flax",
#     "jaxopt",
#     "jraph",
#     "optax",
#     "orbax-checkpoint",
#     "portpicker",
#     "pandas",
#     "scikit-learn",
#     "ase",
#     "tqdm",
#     "wandb",
#     "pyyaml",
#     "pytest",
#     "h5py",
#     "ml_collections"
# ]
#
# [project.scripts]
# evaluate = "mlff.cAPI.mlff_eval:evaluate"
# train = "mlff.cAPI.mlff_train:train"
# run_md = "mlff.cAPI.mlff_md:run_md"
# run_relaxation = "mlff.cAPI.mlff_structure_relaxation:run_relaxation"
# analyse_md = "mlff.cAPI.mlff_analyse:analyse_md"
# train_so3krates = "mlff.cAPI.mlff_train_so3krates:train_so3krates"
# train_so3kratACE = "mlff.cAPI.mlff_train_so3kratace:train_so3kratace"
# trajectory_to_xyz = "mlff.cAPI.mlff_postprocessing:trajectory_to_xyz"
# to_mlff_input = "mlff.cAPI.mlff_input_processing:to_mlff_input"
# train_so3krates_sparse = "mlff.CLI.run_training:train_so3krates_sparse"
# train_itp_net = "mlff.CLI.run_training_itp_net:train_itp_net"
# evaluate_itp_net = "mlff.CLI.run_evaluation_itp_net:evaluate_itp_net"
# evaluate_itp_net_on = "mlff.CLI.run_evaluation_itp_net_on:evaluate_itp_net_on"
# fine_tune_so3krates_sparse = "mlff.CLI.run_fine_tuning:fine_tune_so3krates_sparse"
# evaluate_so3krates_sparse = "mlff.CLI.run_evaluation:evaluate_so3krates_sparse"
# evaluate_so3krates_sparse_on = "mlff.CLI.run_evaluation_on:evaluate_so3krates_sparse_on"
