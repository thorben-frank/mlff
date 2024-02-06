import jax.numpy as jnp
import jax
import numpy as np
from ase.io import read
from ase.units import Bohr, Hartree
from ase.units import alpha as fine_structure
import scipy.optimize as opt

def _get_ref_data():
    fname = '/Users/adil.kabylda/Documents/mlff_v1.0/train_runs/ref.dat'
    A = np.loadtxt(fname, skiprows=1, usecols=[1,2,3], dtype=float)
    species = np.loadtxt(fname, skiprows=1, usecols=[0], dtype=str)
    alphas = A[:,0]
    C6s = A[:,1]
    return species, alphas, C6s

def QDO_params(alpha, C6):
    # This function returns gamma = mu*omega based on the vdW-OQDO parametrization
    # It is enough to have just 'gamma' to compute the dispersion energy
    
    # Flattening matrices of atomic pairs and taking only unique values for convenience
    N = alpha.shape[0]
    iu = jnp.triu_indices(N)
    a0 = alpha[iu].flatten()
    C6 = C6[iu].flatten()

    # Starting points for the larger root that we need
    x0 = 0.5 * jnp.ones(a0.shape)
    # Setting tolerance
    tol = 1e-5
    
    def fun(x,a,b):
        p = 1 - jnp.exp(-b*x)*(1 + (2*b*x)/2 + (2*b*x)**2/8 + (2*b*x)**3/48 + (2*b*x)**4/6/48)
        f = a*jnp.exp(b*x) - (2*x**2 + x/b)/p
        return f

    b = 2*fine_structure**(-8/21)*a0**(2/7)
    a = 9/64*fine_structure**(4/3)
    x = opt.fsolve(fun, x0, args=(a,b), xtol=tol, factor=1)
    
    # Reshaping the solutions obtained back to NxN symmetric matrix
    gamma = jnp.zeros((N,N))
    # gamma[iu] = x
    gamma = gamma.at[iu].set(x)
    gamma = gamma + gamma.T
    # omega = 4*C6/3/alpha**2
    # q = jnp.sqrt(x*omega*alpha)
    # mu = x/omega
    
    return gamma

def vdw_QDO_disp_damp(R, gamma, C6):
    #  Computing the vdW-QDO dispersion energy and returning it in eV
    C8 = 5/gamma*C6
    C10 = 245/8/gamma**2*C6
    f6 = Damp(R, gamma, 3)
    f8 = Damp(R, gamma, 4)
    f10 = Damp(R, gamma, 5)
    V1 = -f6*C6/R**6
    V2 = -f6*C6/R**6 - f8*C8/R**8
    V3 = -f6*C6/R**6 - f8*C8/R**8 - f10*C10/R**10
    
    return V3*Hartree

def Damp(R, gamma, n):
    # Computes the QDO damping function of the order 2n
    f = 1
    for k in range(n+1):
        f += -jnp.exp(-gamma*R**2/2)* gamma**k * R**(2*k)/2**k/jax.scipy.special.factorial(k)
    return f

species, alphas, C6s = _get_ref_data()

# Reading the molecule
mol = read('/Users/adil.kabylda/Documents/mlff_v1.0/train_runs/qm7x_disp_train.extxyz', index="20", format='extxyz')

# Getting positions and converting them to a.u.
positions = mol.get_positions() / Bohr
hirshfeld_ratios = mol.arrays['hirsh_ratios']

# Getting atomic numbers (needed to link to the free-atom reference values)
at_nums = mol.get_atomic_numbers()
N = len(at_nums)
alpha = alphas[at_nums-1] * hirshfeld_ratios
C6 = C6s[at_nums-1] * hirshfeld_ratios**2  

# Treating all atomic pairs in one shot by 2d arrays
# Computing mixed parameters
a0_mat = (alpha[:, None] + alpha[None, :])/2
C6_mat = 2 * C6[:, None] * C6[None, :] * alpha[None, :] * alpha[:, None]
C6_mat *= 1/(alpha[None, :]**2 * C6[:, None] + alpha[:, None]**2 * C6[None, :])

# Computing all pairwise distances
Rs = positions[:, None, :] - positions[None, :, :]
dists = jnp.sqrt(jnp.sum(Rs**2, -1))

# Getting the QDO parameters for all pairs
gamma = QDO_params(a0_mat, C6_mat)

# Masking the diagonal elements to avoid division by zero
gamma = gamma[~jnp.eye(len(gamma), dtype=bool)].reshape(len(gamma), -1)
dists = dists[~jnp.eye(len(dists), dtype=bool)].reshape(len(dists), -1)
C6_mat = C6_mat[~jnp.eye(len(C6_mat), dtype=bool)].reshape(len(C6_mat), -1)

# Computing the dispersion energy
ene_mat = vdw_QDO_disp_damp(dists, gamma, C6_mat)
disp_energy = 0.5 * jnp.sum(ene_mat)
print(disp_energy)