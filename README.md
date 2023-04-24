# MLFF
Repository for training, testing and developing machine learned force fields using the `So3krates` transformer.
## Installation
Assuming you have already set up an virtual environment with python version `>= 3.8.` In order to ensure compatibility
with CUDA `jax/jaxlib` have to be installed manually. Therefore **before** you install `MLFF` run 
```
pip install --upgrade pip
# Installs the wheel compatible with CUDA 11 and cuDNN 8.6 or newer.
# Note: wheels only available on linux.
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
for details check the official [`jax`](https://github.com/google/jax#pip-installation-gpu-cuda) repository.   

Next clone the `mlff` repository by running
```
git clone https://github.com/thorben-frank/mlff.git
```
**In case you have an Apply Mx chip, check the next sub-section before continuing.** 

Now do
```
cd mlff
pip install -e .
```
which completes the installation and installs remaining dependencies.
### Apple Mx
In case you have an Apple Mx chip, there is a known issue when installing tensorflow using the 
`setup.py` command. Thus, you have to install the appropriate version by hand, by executing
```
do Apple m chip installation for tensorflow
```
## Weights and Bias
If you do not have a weights and bias account already you can create on [here](https://wandb.ai/site). After installing
``mlff`` run
```
wandb login
```
and log in with your account.
# Quickstart
Following we will give a quick start how to train, evaluate and run an MD simulation with the 
`So3krates` model.
## Training
Train your fist `So3krates` model by running
```
train_so3krates --data_file data.xyz  --n_train 1000 --n_valid 100
```
The `--data_file` can be any format digestible by the `ase.io.read` method. In case minimal image convention
should be applied, add `--mic` to the command. The model parameters will be saved per default to `/module`. Another 
directory can be specified using `--ckpt_dir $CKPT_DIR`, which will safe the model parameters to `$CKPT_DIR/module`. 
More details on training can be found in the detailed training section below.
## Evaluation
After training, change into the model directory, e.g. and run the `evaluate` command
```
cd module
evaluate
``` 
As before, when your data is not in eV and Angstrom add the `--units` keyword. The reported metrics are then in eV and
Angstrom (e.g. `--units energy='kcal/mol',force='kcal/(mol*Ang)'` if the energy in your data is in `kcal/mol`).
## Molecular Dynamics
You can use the `mdx` package which is the `mlff` internal MD package, fully relying on `jax` and thus fully 
optimized for XLA compilation on GPU. Before you can use it make sure you install the `glp` package by cloning
the `glp` repository and install it
```
git clone https://github.com/sirmarcel/glp-dev.git
git cd glp-dev
pip install -e .
```
First, lets create a relaxed structure, using the LBFGS optimizer
```
run_relaxation  --qn_max_steps 1000 --qn_tol 0.0001
```
which will save the relaxed geometry to `relaxed_structure.h5`. Next, convert the `.h5` file to an 
`xyz` file, by running
```
trajectory_to_xyz --trajectory relaxed_structure.h5 --output relaxed_structure.xyz
```
We now run an MD with the relaxed structure as start geometry
```
run_md --start_geometry relaxed_structure.xyz --thermostat velocity_verlet --temperature_init 600 --time_step 0.5 --total_time 1 --use_mdx
```
Temperature is in Kelvin, time step in femto seconds and total time in nano seconds. It will save a `trajectory.h5` 
file to the current working directory.
### Analysis
After the MD is finished you can either work with the `trajectory.h5` using e.g. a `jupyter notebook` and `h5py`. 
Alternatively, you can run
```
trajectory_to_xyz --trajectory trajectory.h5 --output trajectory.xyz
```  
which will create an `xyz` file. The resulting `xyz` file can be used as input to the 
[`MDAnalysis`](https://docs.mdanalysis.org/stable/index.html) python package, which provides a broad range of functions 
to analyse the MD simulations. The central `Universe` object can be creates easily as
```python
import MDAnalysis as mda

# Load MD simulation results from xyz
u = mda.Universe('trajectory.xyz')
```
## Run the tests
The test suite can be run with pytest as:
```
pytest tests/
```
## Cite
If you use parts of the code please cite the corresponding paper
```
@article{frank2022so3krates,
  title={So3krates: Equivariant attention for interactions on arbitrary length-scales in molecular systems},
  author={Frank, Thorben and Unke, Oliver and M{\"u}ller, Klaus-Robert},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={29400--29413},
  year={2022}
}
```
