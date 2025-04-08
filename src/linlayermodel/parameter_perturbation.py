import numpy as np
from linlayermodel.LLMParams import Constants, LinearSolutions
import pathlib
from mpi4py import MPI
from copy import deepcopy
import pickle
from pathlib import Path
from pint import UnitRegistry
import typing


ureg = UnitRegistry()
m2s2 = ureg.meter**2/ureg.second**2
per_sec = 1/ureg.second
per_day = 1/ureg.day
unitless = ureg.dimensionless
K = ureg.kelvin
deg = ureg.degree

# typing
Units = ureg.Quantity

# set cache directory
cache_dir = Path('../../cache')
cache_dir.mkdir(exist_ok=True)

def norm_vec(vec):
    return vec/np.linalg.norm(vec)

def save_obj(obj, save_path):
    file_name = f'obj_n={obj.nlayers}_x0={obj.phys_params["x0"] * 1e-3:.0f}km_theta={obj.phys_params["theta0"]}deg.pkl'
    with open( str(save_path / file_name), 'wb') as f:
        pickle.dump(obj, f)
    print(f'Object saved as {str(save_path / file_name)}')

def all_format(x, fmt):
    """
    Formats all elements of an array or a single value to a given format.
    AUTHOR: FA
    """
    return [f'{i:{fmt}}' for i in x] if isinstance(x, np.ndarray) else f'{x:{fmt}}'

def _new_val(val: Units, 
             default_vals: dict[str, float], 
             pert_var: str, i: int, 
             repr_indx:int = 0) -> tuple[Units, str]:
    """
    Takes in pert_var name and value, and returns a new value based on the perturbation.
    i: int, perturbation index
    """

    if val.units == (ureg('1./day')):
        new_val = default_vals[pert_var] * (1/ureg.second) + val.to(1/ureg.second) * i
        dv = all_format( (default_vals[pert_var] * (1/ureg.second)).to(1/ureg.day),'.2e' )
        
        if isinstance(new_val, np.ndarray):
            pert_var_string = '_'.join([f'{perts.pert_vars}={dv[repr_indx].magnitude}'])
        else:
            pert_var_string = '_'.join([f'{perts.pert_vars}={dv.magnitude}'])

    else:            
        new_val = default_vals[pert_var] * val.units + val * i
        
        if isinstance(new_val.magnitude, np.ndarray):
            pert_var_string = '_'.join([f'{perts.pert_vars}={new_val[repr_indx].magnitude}'])
        else:
            pert_var_string = '_'.join([f'{perts.pert_vars}={new_val.magnitude}'])
        
    return new_val, pert_var_string


def _run_save_file(pert_vars: str, 
                   pert_var_string : str, 
                   rad_file_path: str, 
                   conv_file_path: str,
                   phys_params: dict[str, float]) -> None:
    """
    Runs the model and saves the object to a file.
    """
    file_path = cache_dir/'parameter_perturbation'/f'{pert_vars}_n={nlayers}'/f'{pert_var_string}'
    pathlib.Path(file_path).mkdir(parents=True, exist_ok=True)

    np.save(str( file_path/f'phys_params_n={nlayers}.npy'), phys_params)
    obj = LinearSolutions(constants, phys_params, nlayers, pinterfaces, 
                          rad_file_path, conv_file_path)
    obj.partial_solve()
    save_obj(obj, file_path)
    print(f'File saved to {str(file_path)}')


class Pert:
    def __init__(self):
        self.pert_vars = None
        self.pert_vals = None
        self.default_vals = None
        self.VAR_TYPE = None
        self.x0_string = None

    @property
    def pert_vars(self):
        return self._pert_vars
    
    @pert_vars.setter
    def pert_vars(self, pert_vars):
        self._pert_vars = pert_vars
    
    @property
    def pert_vals(self):
        return self._pert_vals      
    
    @pert_vals.setter
    def pert_vals(self, pert_vals):
        self._pert_vals = pert_vals

nlayers = 4

sbar_pert = np.eye(nlayers) * 0.5 * K # diagonal perturbation
qbar_pert = np.eye(nlayers) * 0.5 * K
epsilon_cr_pert = np.zeros((nlayers, nlayers)) * 1e-1 * unitless  # % a fractional perturbation

pert_dict = { 'DT': 2.5e6 * m2s2, 'Dq': 2.5e6 * m2s2, 'epsilon_f': 2.5e-2 * per_day, 
             'kappas_pert': 0.5e-1 * per_day, 'fp': 2.5e-2 * unitless, 
             'sbar_pert' : sbar_pert, 'qbar_pert': qbar_pert,
             'epsilon_cr_pert': epsilon_cr_pert,
             'theta0': 1 * K, 'thermo_pert': 0.25e-1 * K, 
             'x0': 5e5 * ureg.meter, 'Ts0': 1.0 * K, 'theta0': 1 * deg }

pert_range = { k : range(-2, 8) for k in pert_dict.keys() }
pert_range['theta0'] = np.arange(-2.5, 10.0, 2.5)

if __name__ == '__main__':

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    rad_file_path = str ( next ( cache_dir.resolve().glob(f'epsilon_rad_matrix_n={nlayers}.npy') ) )
    conv_file_path = str ( next ( cache_dir.resolve().glob(f'epsilon_conv_matrix_n={nlayers}.npy') ) )

    phys_params_file = str( cache_dir / f'phys_params_n={nlayers}.npy' )
    phys_params = np.load(phys_params_file, allow_pickle=True).item()
    constants = Constants().constants
    pinterfaces = phys_params['pinterfaces']

    perts = Pert()

    perts.pert_vars = 'epsilon_cr_pert' #'qbar_pert'  #'sbar_pert'
    perts.pert_vals = pert_dict[perts.pert_vars] 
    pert_var = perts.pert_vars
    repr_indx = 0

    # if perts.pert_vars not in ['sbar_pert', 'qbar_pert']: 
    default_vals = { perts.pert_vars: phys_params[f'{perts.pert_vars}'] }
    # else:
    #     default_vals = { perts.pert_vars: phys_params[perts.pert_vars] }
            
    if rank == 0:
        # print('Running on rank 0')
        l = np.array_split(pert_range[perts.pert_vars], nprocs)
    else:
        l = None
    
    data_l = comm.scatter(l, root = 0)
    x0_string = f'x0={str((phys_params["x0"] * 1e-3))}'

    TEST = True

    for i in data_l:
        
        phys_params_copy = deepcopy(phys_params)
        val = perts.pert_vals

        print(np.ndim(val))
        
        if np.ndim(val) < 2:
            new_val, pert_var_string = _new_val(val, default_vals, pert_var, i, repr_indx)
            phys_params_copy[pert_var] = new_val.magnitude

            if not TEST:
                _run_save_file(perts.pert_vars, pert_var_string, rad_file_path, 
                            conv_file_path, phys_params_copy)

        elif np.ndim(val) == 2:

            for j in range(val.shape[0]):
            
                new_val, pert_var_string = _new_val(val[j], default_vals, pert_var, i, j)
                phys_params_copy[pert_var] = new_val.magnitude
                pert_var_string = "_".join([pert_var_string, f'indx={j}'])
                print(f'Rank {rank} new_val: {new_val}, pert_string: {pert_var_string}')

                if not TEST:
                    _run_save_file(perts.pert_vars, pert_var_string, rad_file_path, 
                                conv_file_path, phys_params_copy)
                    
                    
        else:
            raise ValueError('Invalid perturbation value dimension, must be <=2.')

        




