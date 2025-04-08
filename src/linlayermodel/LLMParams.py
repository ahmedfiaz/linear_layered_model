import numpy as np
import pathlib
from linlayermodel.thermodynamic_functions import qs_calc, qsi_calc
# from thermodynamic_functions import qs_calc, qsi_calc
import xarray as xr
from scipy.stats import linregress
from scipy import integrate
import copy
import sympy as sp
import pickle
import cmath
from copy import deepcopy
from scipy.io import loadmat
import scipy.linalg as spl
import re
import warnings
from itertools import zip_longest
import re
import linlayermodel
import importlib.resources as res
import logging
import sys

logger = logging.getLogger()
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))



import climlab
from climlab.domain import field


arm_file = next( res.files(linlayermodel).joinpath('data/').glob('*.nc') )
params_file = next( res.files(linlayermodel).joinpath('data/').glob('*.out') )
# arm_path = pathlib.Path('/home/fiaz/LLM/files') 
# arm_file = [str(f) for f in arm_path.glob('twparmbeatmC2.c1*')][0]  # get ARM Nauru file for the year 2000 
# rad_param_path = pathlib.Path('/home/fiaz/LLM/files') / 'three_layer_rad_coeffs.dat'
# params_file = '/home/fiaz/LLM/analysis/par.out'

prof_res = 10 # vertical resolution in hPa

flip_if_decreasing = lambda x, y: y[::-1] if np.all(np.diff(x)<0) else y 

def _interpolate_profile(pres_new, pres_old, prof_old):
    
    # if pressure is decreasing with height, flip it
    xnew = flip_if_decreasing(pres_new, pres_new)
    xold = flip_if_decreasing(pres_old, pres_old)
    yold = flip_if_decreasing(pres_old, prof_old)
    
    ynew = np.interp(xnew, xold, yold)  # interpolated profile, pressure increasing 
    ynew = ynew[::-1] if np.all(np.diff(pres_old)<0) else ynew # if original pressure levels are decreasing, flip interpolated profile
    return ynew

def gaussian(x, mu, sigma):
    """
    return gaussian distribution for input
    x, mean (mu) and std. dev. (sigma)
    """
    norm = 1/(sigma*np.sqrt(2*np.pi))
    return norm * np.exp(-0.5 * pow((x - mu)/sigma, 2))


class Constants:
    def __init__(self) -> None:
        self.constants = dict(CPD = 1004., DELP = 800_00, 
                    HLATENT = 2.4e6, GRAVITY = 9.8, 
                    RD = 287.0, RV = 461.0)


# define functions to specify divergence structures in each layer


class LLMParams:
    
    """
    Class to construct an n-layered linear model (LLM)
    
    Inputs: constants (dict): dictionary of constants
            phys_params (dict): dictionary of physical parameters
            methodVBVLVU (function): function to compute the structure of divergence in each layer
            nlayers (int): number of layers in the model
            pinterface (array): pressure interface levels (should be one more than nlayers)
            fil (str): parameters file for LLM,
            arm_file (str): file location of ARM data, used to compute static stabilities
            rad_param_path (str): file location of radiative parameterizations
    -------
    """
                 
    def __init__(self, constants, phys_params,  
                 nlayers, pinterface, fil = params_file, 
                 arm_file = arm_file):
        
        self.fil = fil
        self.arm_file = arm_file
        self.constants = constants
        self.K_to_mm = constants['CPD']*constants['DELP']/(constants['GRAVITY' ] * constants['HLATENT'])
        self.mm_to_K = 1./self.K_to_mm
        self.phys_params = phys_params

        self.nlayers = nlayers
        self.pinterface = pinterface
        assert len(pinterface) == nlayers + 1, "number of pressure interfaces should be one more than number of layers"
        # self.deltap = abs(np.diff(pinterface))
        self.deltap = []
        self.ind_interface = {}

        self.pref = None
        self.pref_flip = None

        self.vert_struct_params = {}
        self.mom_params = {}
        self.dse_params = {}
        self.q_params = {}

        self.Tsref = 301.0  # reference surface temperature

        # initilize perturbations dict (Ti & Ts in K; qi in kg/kg; prc in mm/hr)
        self.perts = {f'{v}{k}' : 0 for v in ['T', 'q'] for k in range(self.nlayers) } | dict(Ts = 0, prc = 0)       
        
        self.params = {}
        self.geopotential_params = {}
        self.downdraft_params = {}

        self.qsat_params = {}    
        self.diag_params = {} 
        # compute Coriolis parameter
        f = 2 * self.phys_params['Omega'] * np.sin(np.deg2rad(self.phys_params['theta0']))
        self.phys_params.update(f = f)

    # def natural_sort(self, l): 
    #     convert = lambda text: int(text) if text.isdigit() else text.lower()
    #     alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    #     l_string = [str(i) for i in l]
    #     sorted_list = sorted(l_string, key = alphanum_key)
    #     # take list and return expressions in sorted order
    #     return [self.symbols[k.split('[')[0]][int( re.search(r"\[([0-9]+)\]", k).group(1) )]  for k in sorted_list]

    def _natural_sort(self, l): 
        """
        Sort a list of strings in a natural order.
        """
        convert = lambda text: int(text) if re.fullmatch(r"-?\d+", text) else text.lower()
        alphanum_key = lambda key: [abs(convert(c)) if isinstance(convert(c), int) else convert(c) 
                                    for c in re.split(r"(-?\d+)", key)]
        
        l_string = [str(i) for i in l]
        sorted_list = sorted(l_string, key=alphanum_key)

        return [
            self.symbols[k.split('[')[0]][int(re.search(r"\[(-?\d+)\]", k).group(1))]  
            for k in sorted_list
            ]


    def read_profiles(self):

        fpar = np.loadtxt(self.fil)
       
        # interpolate profiles to coarser resolution to facilitate RRTMG computations
        self.res = prof_res # hPa
        pres_old = fpar[:,0]
        pres_new = np.arange(pres_old[0], pres_old[-1] - self.res, -self.res)  # generate new pressure levels
        
        a1, V1, Omega, b, Tref = [_interpolate_profile(pres_new, pres_old, fpar[:,i]) for i in (1, 2, 3, 4, 9)]

        # make a1 and b1 unity
        # a1 = np.ones_like(a1)
        # b = np.ones_like(a1)
        
        self.profiles = dict(pres = pres_new, a1 = a1,
                             a1full = np.copy(a1),
                             b = b, V1 = V1, 
                             Omega = Omega, Tref = Tref)
        
        pres = self.profiles['pres']
        self.dp = abs(np.diff(pres)[0]) 
        self.prange = pres[0] - pres[-1]

        # extend pressure levels to stratosphere assuming the pressure is decreasing
        self.pref = np.append(self.profiles['pres'], 
                              np.arange(10, self.profiles['pres'].min(), prof_res)[::-1])  # extend stratosphere from 150 hPa to 10 hPa

        self.pref_flip = self.pref[::-1]  # flipped array to have pressure increasing
    
        # compute indices for three layers 
        pres = self.profiles['pres']
        self.deltap = []
        for i in range(self.nlayers):
        
            if i < self.nlayers - 1:        
                inds = np.where(np.logical_and(pres <= self.pinterface[i], 
                                            pres > self.pinterface[i+1]))[0]
            else:
                inds = np.where(np.logical_and(pres <= self.pinterface[i], 
                                            pres >= self.pinterface[i+1]))[0]

            self.ind_interface[i] = inds 
            self.deltap.append(pres[inds[0]] - pres[inds[-1]])
        
        self.deltap = np.asarray(self.deltap)

    def get_reference_thermo_profiles(self):

        """
        Read ARM data for reference thermodynamic profiles, zonal wind,
        and interpolate to QTCM levels. Assumes that pressure levels 
        are monotonically decreasing.
        """

        ds = xr.open_dataset(self.arm_file)
        
        hus_mean = ds.hus  # get average specific humidity
        temp_mean = ds.T_p # get average temperature 
        u_mean = ds.u_p  # get average zonal wind

        p = self.profiles['pres']

        # interpolate to QTCM pressure levels
        qinterp = _interpolate_profile(p, hus_mean.p, hus_mean)
        Tinterp = _interpolate_profile(p, temp_mean.p, temp_mean)    
        uinterp = _interpolate_profile(p, u_mean.p, u_mean)    

        ds.close()

        phi_ref = self.compute_geopotential(Tinterp, qinterp)
        qsat_ref = qs_calc(p, Tinterp)


        self.profiles.update(qref = qinterp, Tref = Tinterp, 
                             phiref = phi_ref, qsat_ref = qsat_ref,
                             uref = uinterp)
                            
    def compute_ab_profiles(self):
            
            """
            Compute vertical structure functions a, a+ and b
            from the reference temperature and moisture profiles
            for each vertical layer.
            """

            a1 = self.profiles['a1']
            b = self.profiles['b']
            dp = self.res
            pres = self.profiles['pres']
            
            RD = self.constants['RD']
            qref = self.profiles['qref']
            epsilon = RD / self.constants['RV']

            a_profiles = {}
            b_profiles = {}
            aplus_profiles = {}

            a_layer_ave = {}
            aplus_layer_ave = {}
            b_layer_ave = {}

            for i in range(self.nlayers):
                inds = self.ind_interface[i]

                a_profiles[i] = a1[inds]/a1[inds[0]]
                b_profiles[i] = b[inds]/b[inds[0]]

                # express geopotential perturbations to K
                integrand_Aprof = a_profiles[i] * (1 + epsilon * qref[inds])/pres[inds]  
                aplus_profiles[i] = integrate.cumulative_trapezoid(integrand_Aprof, dx = dp, initial = 0.)

                # layer averages
                a_layer_ave[i] = self.layer_ave(a_profiles[i], inds)
                aplus_layer_ave[i] = self.layer_ave(aplus_profiles[i], inds)
                b_layer_ave[i] = self.layer_ave(b_profiles[i], inds)

            self.vert_struct_params.update(a_profiles = a_profiles, 
                                           b_profiles = b_profiles, 
                                           aplus_profiles = aplus_profiles, 
                                           a_layer_ave = a_layer_ave, 
                                           aplus_layer_ave = aplus_layer_ave, 
                                           b_layer_ave = b_layer_ave)
            

    def compute_geopotential(self, T, q, phi_s = 0):

        """
        For given temperature (K) and water vapor (kg/kg) profiles,
        compute geopotential height in K. 
        Default surface geopotential height = 0.
        """
        pres = self.profiles['pres']
        RD = self.constants['RD']
        CPD = self.constants['CPD']

        phi = np.zeros_like(T) + phi_s
        phi[0] = 0
        dp = self.res  # pressure thickness in hPa
        dlnp = dp/pres  # take logarithm of pressure

        Tv = T * (1 + 0.608 * q)  # virtual temperature
        phi[1:] =  (RD/CPD) * integrate.cumulative_trapezoid(Tv * dlnp)
        return phi  

    def distort_ab(self):
        
        """
        Change the vertical structure of a1 in the boundary layer such that
        a_p = Exner function (p, ps, qrb) (DSE conserving)
        and a1(pb) = 1 in the free troposphere.
        
        Change the vertical structure of b such that it = 1 in the boundary layer
        and the top of the bl sp. humidity = 1
        """

        a1 = self.profiles['a1']
        b = self.profiles['b']
        p = self.profiles['pres']
        pb = self.pb
        ps = self.ps
        ind_bl = self.ind_bl

        qrb = self.layer_ave(self.profiles['qref'], layer = 'bl')

        kappa = self.constants['RD']/self.constants['CPD']
        epsilon =  self.constants['RD']/self.constants['RV']
        exner = pow(p / ps, kappa * (1 + epsilon * qrb) )  # compute Exner function
        a1[self.inds_bl] = exner[self.inds_bl]  # dry adiabat, dse conserving
        a1[self.inds_ft] = a1[self.inds_ft]/a1[self.inds_ft[0]]  # moist adiabat, mse conserving
        
        b[self.inds_bl] = 1  # constant 
        b[self.inds_ft] = b[self.inds_ft]/b[self.inds_ft[0]]


    # def read_rad_params(self):

    #     """
    #     Read linear coefficients associated with radiative parameterizations.
    #     Convert per day to per second.
    #     """

    #     gcpinv = self.constants['GRAVITY']/self.constants['CPD']

    #     with open(self.rad_param_path, 'rb') as f:
    #         rad_param_dict = pickle.load(f)

    #     var_rad = ['Tas', 'qb', 'Ts', 'Tul', 'qul']
    #     for k,v in rad_param_dict.items():
            
    #         var = k.split('_')[0]
    #         layer = k.split('_')[1]
    #         key = f'eps_rad_{var}_{layer}'
            
    #         if layer == 'bl' or layer == 'srf':
    #             mult = self.delta_pb *1e2
    #         elif layer == 'lft':
    #             mult = self.delta_pLFT *1e2
    #         elif layer == 'uft':
    #             mult = self.delta_pUFT *1e2

    #         if var in var_rad:
    #             self.phys_params[key] = v * gcpinv  / (mult )


    def compute_thermo_params(self):
        
        p = self.profiles['pres']
        a1 = self.profiles['a1']
        b1 = self.profiles['b']
        Tref = self.profiles['Tref']
        qref = self.profiles['qref']
        qsref = self.profiles['qsat_ref']
        RD = self.constants['RD']
        CPD = self.constants['CPD']
        epsilon = RD/self.constants['RV']
        kappa = RD/CPD
        ind_bl = self.ind_bl
        ind_lft = self.ind_lft
        kg_per_kg_to_K = self.constants['HLATENT']/self.constants['CPD']

        pres = self.profiles['pres']

        # parameter to convert T1 to qsat_lft perturbations
        alpha_T1_to_qsat_lft = self.layer_ave(a1 * qsref/pow(Tref,2), layer = 'lft')
        alpha_T1_to_qsat_lft = kg_per_kg_to_K * (self.constants['HLATENT']/self.constants['RV']) * alpha_T1_to_qsat_lft  # express in units of K/K
        self.params.update(alpha_T1_to_qsat_lft = alpha_T1_to_qsat_lft)
        delta_kappa_LFT_qsat = self.params['delta_kappa_LFT_qsat']
        self.params.update(delta_kappa_LFT_qsat = delta_kappa_LFT_qsat * alpha_T1_to_qsat_lft)

        # layer coefficients for geopotential computation 
        Aprof = self.profiles['a1']
        Bprof = self.profiles['b']

        # Layer indices
        # renormalize A- and B-profiles such that they equal 1 at the bottom of each layer
        Aprof[self.inds_bl] = Aprof[self.inds_bl]/Aprof[self.inds_bl[0]]     
        
        Aprof[self.inds_lft] = Aprof[self.inds_lft]/Aprof[self.inds_lft[0]]     
        Aprof[self.inds_uft] = Aprof[self.inds_uft]/Aprof[self.inds_uft[0]]  

        Bprof[self.inds_lft] = Bprof[self.inds_lft]/Bprof[self.inds_lft[0]]  
        Bprof[self.inds_uft] = Bprof[self.inds_uft]/Bprof[self.inds_uft[0]]  

        ## layer averaged a and b profiles
        a1BL = self.layer_ave(Aprof, layer = 'bl')
        a1LFT = self.layer_ave(Aprof, layer = 'lft')
        a1UFT = self.layer_ave(Aprof, layer = 'uft')
        
        b1BL = self.layer_ave(Bprof, layer = 'bl')
        b1LFT = self.layer_ave(Bprof, layer = 'lft')
        b1UFT = self.layer_ave(Bprof, layer = 'uft')

        self.params.update(a1BL = a1BL, a1LFT = a1LFT, a1UFT = a1UFT, 
                           b1BL = b1BL, b1LFT = b1LFT, b1UFT = b1UFT)

        integrand_Aprof = Aprof * (1 + epsilon * qref)  # express geopotential perturbations to K
        integrand_Bprof = epsilon * Bprof * Tref / kg_per_kg_to_K  # express geopotential perturbations to K

        # boundary layer
        Ab_plus, Ab_plus_bl_ave = self.log_layer_integral(integrand_Aprof, 'bl')
        Bb_plus, Bb_plus_bl_ave = self.log_layer_integral(integrand_Bprof , 'bl')
        # _, ab_plus_bl_ave = self.log_layer_integral( epsilon * a1 / kg_per_kg_to_K , 'bl')

        Ab_plus_pb = Ab_plus[-1]
        ab_plus_bl_ave = Ab_plus_pb - Ab_plus_bl_ave

        Bb_plus_pb = Bb_plus[-1]
        bb_plus_bl_ave = Bb_plus_pb - Bb_plus_bl_ave

        #lft
        a1_plus_lft, a1_plus_lft_ave = self.log_layer_integral(integrand_Aprof, layer = 'lft')  # used in lft geopotential computation
        b1_plus_lft, b1_plus_lft_ave = self.log_layer_integral(integrand_Bprof, layer = 'lft')

        #uft
        A1_plus_uft, A1_plus_uft_ave = self.log_layer_integral(integrand_Aprof, layer = 'uft')  # used in uft geopotential computation
        _, B1_plus_uft_ave = self.log_layer_integral(integrand_Bprof, layer = 'uft')

        ind_pL = self.inds_lft[-1]
        AL_lft_top = Aprof[ind_pL]
        bL_lft_top = Bprof[ind_pL]

        a1_plus_lft_top = a1_plus_lft[-1]
        b1_plus_lft_top = b1_plus_lft[-1]

        A1_plus_uft_top = A1_plus_uft[-1]

        a1_plus_uft_ave = a1_plus_lft_top + AL_lft_top * A1_plus_uft_ave
        b1_plus_uft_ave = b1_plus_lft_top + bL_lft_top * B1_plus_uft_ave
        a1_plus_uft_top = a1_plus_lft_top + AL_lft_top * A1_plus_uft_top

        self.geopotential_params.update(ab_plus_bl_ave = ab_plus_bl_ave,
                                        bb_plus_bl_ave = bb_plus_bl_ave,
                                        al_plus_lft_ave = a1_plus_lft_ave, 
                                        bl_plus_lft_ave = b1_plus_lft_ave,
                                        au_plus_uft_ave = a1_plus_uft_ave,
                                        bu_plus_uft_ave = b1_plus_uft_ave,
                                        AL_lft_top = AL_lft_top, 
                                        bL_lft_top = bL_lft_top,
                                        Ab_plus_pb = Ab_plus_pb,
                                        a1_plus_lft_top = a1_plus_lft_top,
                                        a1_plus_uft_top = a1_plus_uft_top)
        
        self.profiles.update(Aprof = Aprof, Bprof = Bprof)

    def compute_MsMq(self):

        """
        Get averaged Ms and Mq for each layer.
        Also compute s_dagger and q_dagger (interfacial terms)
        
        Note: 
        ----
            For unit divergence, Ms = avg. dse and Mq = avg. q
        
        """

        Tref = self.profiles['Tref']
        qref = (self.constants['HLATENT']/self.constants['CPD']) * self.profiles['qref'] # convert kg/kg to K
        phiref = self.profiles['phiref']
        dse_ref = Tref + phiref
        uref = self.profiles['uref']
        pres = self.profiles['pres']

        Msref  = {}
        Mqref = {}
        Vave = {}

        # horizontal advection parameters
        AdvT_ref = {}
        Advq_ref = {}

        sdagger_ref = {}
        qdagger_ref = {}
        Vprofiles = {}
        sbar_pert = self.phys_params['sbar_pert']
        qbar_pert = self.phys_params['qbar_pert']

        # V profiles in each layer
        for i in range(self.nlayers):

            a1 = self.vert_struct_params['a_profiles'][i]
            b1 = self.vert_struct_params['b_profiles'][i]

            inds = self.ind_interface[i]
            Vprofiles[i] = np.ones_like(inds)   # unit divergence
            
            dse_ref_mod = np.copy(dse_ref)
            dse_ref_mod[inds] += sbar_pert[i]

            qref_mod = np.copy(qref)
            qref_mod[inds] += qbar_pert[i]

            # make topmost layer have linear V profiles
            # if i <= self.nlayers - 1:
            #     pres_sub = pres[inds]
            #     Vprofiles[i] = 1 + (pres_sub[0] - pres_sub)/(pres_sub[0] - pres_sub[-1])

            Msref[i] = self.layer_ave(dse_ref_mod[inds]  * Vprofiles[i], inds) 
            Mqref[i] = self.layer_ave(qref_mod[inds] * Vprofiles[i], inds) 

            AdvT_ref[i] = self.layer_ave(uref[inds] * a1, inds) 
            Advq_ref[i] = self.layer_ave(uref[inds] * b1, inds) 

            Vave[i]  = 1

            # compute dagger (interfacial) terms as a centered difference
            # for n layers there are n+1 interfacial terms. The top and 
            # bottom interfacial terms are set to zero.

            key = f'{i-1, i}'  
            if i > 0 :
                inds_m1 = self.ind_interface[i-1]      
                sdagger_ref[key] = (dse_ref_mod[inds[0]] + dse_ref_mod[inds_m1[-1]]) * 0.5    
                qdagger_ref[key] = (qref_mod[inds[0]] + qref_mod[inds_m1[-1]]) * 0.5    

            elif i == 0:
                sdagger_ref[key] = 0
                qdagger_ref[key] = 0

        key = f'{self.nlayers - 1, self.nlayers}'  
        sdagger_ref[key] = 0
        qdagger_ref[key] = 0

        self.vert_struct_params.update(Msref = Msref, Mqref = Mqref, 
                                       sdagger_ref = sdagger_ref, 
                                       qdagger_ref = qdagger_ref, 
                                       AdvT_ref = AdvT_ref, 
                                       Advq_ref = Advq_ref,
                                       Vave = Vave)
        
        self.profiles.update(Vprofiles = Vprofiles)

    def get_Msref_unmod(self):
        # use unmodified dse profile
        Tref = self.profiles['Tref']
        phiref = self.profiles['phiref']
        dse_ref = Tref + phiref
        Msref = {}
        sdagger_ref = {}

        
        for i in range(self.nlayers):
            inds = self.ind_interface[i]
            Msref[i] = self.layer_ave(dse_ref[inds], inds)

            key = f'{i-1, i}'  
            if i > 0 :
                inds_m1 = self.ind_interface[i-1]      
                sdagger_ref[key] = (dse_ref[inds[0]] + dse_ref[inds_m1[-1]]) * 0.5    

            elif i == 0:
                sdagger_ref[key] = 0

        key = f'{self.nlayers - 1, self.nlayers}'  
        sdagger_ref[key] = 0

        return Msref, sdagger_ref

    def compute_coeffs_momentum(self):

        """
        Compute coefficients that appear in the momentum equation.
        The temperature coefficients are in terms of timescales and phase speeds.
        This in turn requires information about the stabilities. 
        """

        RD = self.constants['RD']

        # epsilon_b = self.phys_params['epsilon_b']
        epsilon_f = self.phys_params['epsilon_f']
        f = self.phys_params['f']

        # test
        # f0 = 2 * self.phys_params['Omega'] * np.sin(np.deg2rad(2.5))

        tau = np.zeros((self.nlayers))
        ci = np.zeros((self.nlayers))
        mbi = np.zeros((self.nlayers))

        generate_tau = lambda x, f: x/(pow(x, 2) + pow(f, 2))
        tau = generate_tau( epsilon_f, f )

        # tau0 = generate_tau( epsilon_f, f0 )
        # tau[1:] = tau0[1:]  # boundary layer tau is unchanged with f

        tau_if = tau * self.deltap/(tau @ self.deltap)
        # tau_if0 = tau0 * self.deltap/(tau0 @ self.deltap)

        Msref = self.vert_struct_params['Msref']
        # Msref, _  = self.get_Msref_unmod()
        Msb = Msref[0]

        for i in range(self.nlayers):
            aplus_layer_ave = self.vert_struct_params['aplus_layer_ave'][i]
            Msi = Msref[i]
            ci[i] = np.sqrt(RD * aplus_layer_ave * Msi) # phase speed
            mbi[i] = Msb/Msi
        
        # replacing tau_i with tau0
        self.mom_params.update(tau_i = tau, tau_if = tau_if, 
                               ci = ci, mbi = mbi)

    def compute_coeffs_dse(self):

        """
        Compute the coefficients appearing in the 
        DSE equation. 
        """

        Msi_dagger = np.zeros((self.nlayers))
        Msi_ddagger = np.zeros((self.nlayers))

        Msref = self.vert_struct_params['Msref']
        sdagger_ref = self.vert_struct_params['sdagger_ref']
        # Msref, sdagger_ref = self.get_Msref_unmod()

        Msb = Msref[0]
        # Msb0 = Msref0[0]

        for i in range(self.nlayers):

            key_p1 = f'{i, i+1}'
            key_m1 = f'{i-1, i}'

            Msi = Msref[i]
            # Msi0 = Msref0[i]
            Vi = self.vert_struct_params['Vave'][i]
            sdagger_i_ip1 = sdagger_ref[key_p1]
            sdagger_im1_i = sdagger_ref[key_m1]
        
            # sdagger_i_ip1_0 = sdagger_ref[key_p1]
            # sdagger_im1_i_0 = sdagger_ref[key_m1]

            Msi_dagger[i] = (sdagger_i_ip1 * Vi - Msi) / Msb

            if i == 0:
                Msi_ddagger[i] = - (Vi  * sdagger_i_ip1) / Msb
                # Msi_ddagger[i] = 0

            Msi_ddagger[i] = Vi * (sdagger_im1_i - sdagger_i_ip1) / Msb


        self.dse_params.update(Msi_dagger_ref = Msi_dagger, 
                               Msi_ddagger_ref = Msi_ddagger)

    def compute_coeffs_moisture(self):

        """
        Compute the coefficients appearing in the 
        DSE equation. 
        """

        Mqi_dagger = np.zeros((self.nlayers))
        Mqi_ddagger = np.zeros((self.nlayers))
        Msref = self.vert_struct_params['Msref']
        # Msref, _ = self.get_Msref_unmod()
        Msb = Msref[0]
        

        for i in range(self.nlayers):
            key_p1 = f'{i, i+1}'
            key_m1 = f'{i-1, i}'

            Mqi = self.vert_struct_params['Mqref'][i]
            Vi = self.vert_struct_params['Vave'][i]
            qdagger_i_ip1 = self.vert_struct_params['qdagger_ref'][key_p1]
            qdagger_im1_i = self.vert_struct_params['qdagger_ref'][key_m1]
            
            Mqi_dagger[i] = (qdagger_i_ip1 * Vi - Mqi) / Msb
            Mqi_ddagger[i] = Vi * (qdagger_im1_i - qdagger_i_ip1) / Msb

        self.q_params.update(Mqi_dagger_ref = Mqi_dagger, 
                             Mqi_ddagger_ref = Mqi_ddagger)

    remove_tilde = lambda self, x: x.replace('_tilde','')
    rearrange_tilde = lambda self, x: x.replace('tilde','') + '_tilde'

    def _ft_norm(self, V):
        """
        Get the L2 norm of a vector V
        """
        return np.sqrt(integrate.trapezoid(V * V, dx = self.res))
    

    def layer_ave(self, var, inds:np.ndarray):
        
        """
        Input:
        -----
            var: array of variable to be averaged
        """
        layer_depth = self.profiles['pres'][inds[0]] - self.profiles['pres'][inds[-1]]
        # print(layer, layer_depth)
        return integrate.trapezoid(var, dx = self.res)/layer_depth


    def vert_integ(self, x, dp):
        return np.sum(x[1:]+x[:-1]) * 0.5 * dp * 1e2/self.constants['GRAVITY']
    
        
    def extend_to_strat(self, temp_in, sphum_in):

        """
        Extend the vertical profile out to the 
        stratosphere (10 hPa)
        """

        pres = self.profiles['pres']
        
        temp_ext = np.zeros_like(self.pref)
        sphum_ext = np.zeros_like(self.pref)

        temp_ext[:pres.size] = temp_in
        temp_ext[pres.size:] = temp_in[-1]
        sphum_ext[:pres.size] = sphum_in

        return temp_ext, sphum_ext
    
    def compute_perturbed_profiles(self, **perts_update):
        
        K_to_kg_per_kg = self.constants['CPD']/self.constants['HLATENT']
        Tspert = perts_update['Ts']
        
        # set perturbation temperature profile #
        Tpert = np.zeros_like(self.profiles['Tref'])
        qpert = np.zeros_like(self.profiles['qref'])

        for i in range(self.nlayers):

            Tkey, qkey = f'T{i}', f'q{i}'
            inds = self.ind_interface[i]

            a = self.vert_struct_params['a_profiles'][i]
            b = self.vert_struct_params['b_profiles'][i]

            Tpert[inds] = perts_update[Tkey] * a
            qpert[inds] = perts_update[qkey] * b * K_to_kg_per_kg
        
        temp = self.profiles['Tref'] + Tpert
        sphum = self.profiles['qref'] + qpert
        Tsurf = self.Tsref + Tspert

        temp_ext, sphum_ext = self.extend_to_strat(temp, sphum)

        return temp, sphum, temp_ext, sphum_ext, Tsurf


    def compute_profiles_from_perturbations(self, **perts):

        """
        perts is a dict with perturbations to input temp, sphum and precipitation.
        """
        perts_input = copy.deepcopy(self.perts)  # make deep copy of default dict (zero perturbations)
        
        if len(perts) > 0:  # if input perturbations exist
            perts_input.update({k:v for k,v in perts.items() if k in perts_input.keys()})
        
        _, _, self.temp_ext, self.sphum_ext, self.Tsurf = self.compute_perturbed_profiles(**perts_input)

        assert all(self.sphum_ext >= 0), "negative specific humidity values"

    def main(self):
        self.read_profiles()  # read idealized profiles
        self.get_reference_thermo_profiles()  # read reference profiles from ARM Nauru
        self.compute_ab_profiles()
        self.compute_MsMq()
        self.compute_coeffs_momentum()
        self.compute_coeffs_dse()
        self.compute_coeffs_moisture()
        

class ConvectiveParams(LLMParams):
    
    def __init__(self, constants, phys_params, nlayers, pinterfaces, 
                    conv_file, save_dir, fil = params_file, arm_file = arm_file):
        super().__init__(constants, phys_params, nlayers, pinterfaces, fil, arm_file)
        super().main()
        self.conv_file = conv_file
        self.save_dir = save_dir

        self.epsilonc = {}
        self.pres = None
        self.nQ = None
        self.nT = None
        self.phi = None
        self.projA = None

    def read_lrf(self):

        kd = loadmat(self.conv_file)
        M = kd['M']
        self.pres = kd['pres'].squeeze()
        tmean, qmean = kd['tmean'], kd['qmean']
        # zrce = kd['z']
        nQ, nT = kd['nQ'].item(), kd['nT'].item()
        self.nQ, self.nT = nQ, nT
        self.pres_lrf = None

        # flip eigenvalue to get steady state matrix
        lambdas, P = spl.eig(M)
        lambdas[5] *=  -1
        Pinv = spl.inv(P)
        M2 = np.real(P @ np.diag(lambdas) @ Pinv)

        # extract convective heating and moistening timescales
        self.epsilonc['TT'] = M2[:nT, :nT]  # T perturbations on convective heating
        self.epsilonc['Tq'] = np.zeros_like(self.epsilonc['TT'])  # q perturbations on convective heating
        self.epsilonc['qT'] = np.zeros_like(self.epsilonc['TT'])  # T perturbations on convective moistening
        self.epsilonc['qq'] = np.zeros_like(self.epsilonc['TT'])  # q perturbations on convective moistening

        self.epsilonc['Tq'][:nT, :nQ] = M2[:nT, nT:]  # q perturbations on convective heating
        self.epsilonc['qT'][:nQ, :nT] = M2[nT:, :nT]  # T perturbations on convective moistening
        self.epsilonc['qq'][:nQ, :nQ] = M2[nT:, nT:]   # q perturbations on convective moistening
    
    # vertical integral: redefining parent method    
    vert_integ = staticmethod(lambda x, p: (x * abs(np.diff(p))).sum())


    def interpolate_lrf(self):

        presT = self.pres[:self.nT]
        pres = self.profiles['pres']

        # create pressure levels for interpolation
        pres_bl = np.arange(1000, 875, -25)
        pres_ft = np.arange(850, 100, -50)

        pres_lrf = np.concatenate([pres_bl, pres_ft])

        # interpolate to LRF pressure levels
        epsilonc_lrf = {k: np.zeros((len(pres), self.nT)) for k in self.epsilonc.keys()}

        ind_sort = np.argsort(presT)
        for i in range(self.nT):
            for k in self.epsilonc.keys():
                epsilonc = self.epsilonc[k]
                epsilonc_lrf[k][:, i] = np.interp(pres[::-1], presT[ind_sort], epsilonc[:, i][ind_sort])[::-1]

        self.pres_lrf = pres_lrf
        self.epsilonc_lrf = epsilonc_lrf


    def rescale_conv_moistening(self, epsilonc_T, epsilonc_q, pres):

        """
        The convective moistening terms are rescaled such that the vertical integral 
        of convective moistening equals that of convective heating.
        """

        vert_axis = 0
        vert_integ_Q1 = np.apply_along_axis(self.vert_integ, vert_axis, epsilonc_T, pres)
        vert_integ_Q2 = np.apply_along_axis(self.vert_integ, vert_axis, epsilonc_q, pres)
        rescale_factor = np.divide(vert_integ_Q2, vert_integ_Q1, out=np.zeros_like(vert_integ_Q2), where=vert_integ_Q1!=0)
        return epsilonc_T * np.expand_dims(abs(rescale_factor), vert_axis)
    
    @staticmethod
    def generate_basis_functions(p, i):

        """
        For a given pressure level and index i
        generate the vertical structure of the basis function. 
        """

        basis = np.zeros_like(p)
        basis[i] = 1
        return basis
        # return np.zeros_like(p)
    
    # @staticmethod
    # def generate_gaussian_basis_functions(p, i):
        #     ps = p[0]
        #     Delta_ps = 30
        #     Delta_pi = 75
        #     if i == 0:
        #         arg = (p - ps) / Delta_ps
        #     else:
        #         arg = (p - ps)/Delta_pi + (i - 0.5) 
        #     return np.exp(-arg ** 2)


    def compute_orthogonal_projection(self):

        self.phi = np.zeros((self.nT, self.nT))
        # generate basis functions for each layer index
        for i in range(self.nT):
                self.phi[:, i] = self.generate_basis_functions(self.pres[:self.nT], i)   

        # Generate the projection matrix (contains all the inner products of the basis functions)
        self.projA = self.phi.T @ self.phi 


    def compute_perturbed_profiles(self, **perts_update):

        """
        For a given perturbation variable in temperature or moisture,
        compute the perturbation profile
        This overrides the parent class method. 
        """
                
        # set perturbation temperature profile #
        Tq_pert = np.zeros_like(self.profiles['Tref'])
        pert_key = [k for k,v in perts_update.items() if v != 0][0] # find non-zero perturbation key
        var_name, var_layer = re.findall(r'[A-Za-z]+|\d+', pert_key) # check if this is a T or a q perturbation

        # get vertical profiles of a or b
        if var_name == 'T':
            ab_prof = self.vert_struct_params['a_profiles']
            factor = 1
        elif var_name == 'q':
            ab_prof = self.vert_struct_params['b_profiles'] 
            factor = 1
        else:
            raise ValueError('Invalid perturbation key: should be Tx or qx')

        for i in range(self.nlayers):   

            if int(var_layer) == i:
                inds = self.ind_interface[i]
                Tq_pert[inds] = perts_update[pert_key] * ab_prof[i] * factor


        # interpolate to LRF pressure levels
        presT = self.pres[:self.nT]
        pres_in = self.profiles['pres']
        ind_sort = np.argsort(pres_in)
        Tq_pert_interp = np.interp(presT[::-1], pres_in[ind_sort], Tq_pert[ind_sort])[::-1]

        return Tq_pert_interp


    def compute_profiles_from_perturbations(self, **perts):

        """
        Wrapper around compute_perturbed_profiles. 
        perts is a dict with perturbations to input temp, sphum and precipitation.
        Since the dict is updated for every perturbation, this wrapper compies the perts into a new dict.
        The default dict (zero perts) is unchanged.
        Overriding the parent class method to only return a single thermodynamic profile
        """
        perts_input = copy.deepcopy(self.perts)  # make deep copy of default dict (zero perturbations)
        
        if len(perts) > 0:  # if input perturbations exist
            perts_input.update({k:v for k,v in perts.items() if k in perts_input.keys()})
        
        self.thermo_prof = self.compute_perturbed_profiles(**perts_input)

    def generate_conv_coeffs(self):


        # epsilonc_conv_coeffs = {k: np.zeros((len(pres_lrf), self.nlayers)) for k in self.epsilonc.keys()}

        epsilonc_conv_coeffs = {k: np.zeros((self.nlayers, self.nlayers)) for k in self.epsilonc.keys()}
        # self.rescale_conv_moistening()

        # pres_lrf = self.profiles['pres']
        print(f'Generating convection coefficients')

        for pert_var in list(self.perts.keys())[:-2]: # all pert keys except precip.
            pert_dict = {pert_var: 1}
            self.compute_profiles_from_perturbations( **pert_dict)

            # get coefficients of the basis that project the perturbation profile
            b = self.phi.T @ self.thermo_prof
            alpha_coeffs = np.linalg.solve(self.projA, b)

            # get layer index
            _, layer_index = re.findall(r'[A-Za-z]+|\d+', pert_var) 
            layer_index = int(layer_index)

            for k in epsilonc_conv_coeffs.keys():
                epsc = (alpha_coeffs * self.epsilonc_lrf[k]).sum(axis = 1)

                # layer averages
                for il in range(self.nlayers):
                    inds = self.ind_interface[il]
                    epsilonc_conv_coeffs[k][il, layer_index] = self.layer_ave(epsc[inds], inds)

        for k in ['T', 'q']:
            keyT, keyq = f'T{k}', f'q{k}'
            epsilonc_conv_coeffs[keyT] = self.rescale_conv_moistening(epsilonc_conv_coeffs[keyT], 
                                                                      epsilonc_conv_coeffs[keyq], self.pinterface) 

            # test to check if rescaling works
            # a1 = np.apply_along_axis(self.vert_integ, 0, epsilonc_conv_coeffs[keyT], self.pinterface)
            # a2 = np.apply_along_axis(self.vert_integ, 0, epsilonc_conv_coeffs[keyq], self.pinterface)
            # print(a1 + a2)


        self.epsilonc_conv_coeffs = epsilonc_conv_coeffs

    def save_conv_file(self):

        """
        Save adjustment coefficients to file
        """

        print('Saving convective coeffs.')
        fil_out = self.save_dir + f'epsilon_conv_matrix_n={self.nlayers}.npy'
        days_to_seconds = 86400

        epsilonc_coeffs_matrix = np.zeros((4, self.nlayers, self.nlayers))
        for i,k in enumerate(self.epsilonc_conv_coeffs.keys()):
            epsilonc_coeffs_matrix[i, ...] = self.epsilonc_conv_coeffs[k] / days_to_seconds # convert to per seconds

        np.save(fil_out, epsilonc_coeffs_matrix)  
        print(f'Convective coefficients saved to {fil_out}')  

    def main_conv(self):
        self.read_lrf()
        self.interpolate_lrf()
        self.compute_orthogonal_projection()
        self.generate_conv_coeffs()
        self.save_conv_file()

class CloudProperties(LLMParams):

    def __init__(self, constants, phys_params, nlayers, pinterfaces, 
                 fil = params_file, arm_file=arm_file):
        super().__init__(constants, phys_params, nlayers, pinterfaces, fil, arm_file)
        super().main()
        
        # parameters 
        self.T_mid = 255.  # mid-tropospheric temp for peak liq. water concn in K 
        self.T_top = 220.  # upper trop. temperature in K
        self.T_freeze = 273.16 


    def compute_cloud_props(self, **perts):

        """
        Compute cloud properties of the grid using grid-scale
        temperature and humidity information.
        prc is precipitation in mm/hr
        """

        perts_input = copy.deepcopy(self.perts)  # make deep copy of default dict (zero perturbations)
        if len(perts) > 0:  # if input perturbations exist
            perts_input.update({k:v for k,v in perts.items() if k in perts_input.keys()})

        prc = perts_input['prc']

        self.liq_frac = self.calc_liq_cld_frac(self.temp_ext)

        self.rh, self.cldfrac = self.calc_cld_frac(self.pref, self.liq_frac, 
                                                   self.temp_ext, self.sphum_ext,
                                                   prc)
        
        self.wl = self.calc_incloud_liquid(self.temp_ext,self.sphum_ext, 
                                           self.pref, prc)

        self.re = 14 * self.liq_frac + 25 * (1 - self.liq_frac)  # effective droplet radius


    def generate_qiqc_bases(self):

        """
        For a set of nodes, this function will 
        generate top-hat basis profiles 
        """

        # flip array if pressure levels are not increasing
        pres = self.pref[::-1] if np.all(np.diff(self.pref)<=0) else self.pref  
        nodes = self.cre_dict['nodes']
        nbasis = nodes.size
        phi = np.zeros((pres.size, nbasis))
        
        for i in range(nbasis):
            start = pres[0] if i == 0 else nodes[i-1]            
            middle = nodes[i]
            end =  pres[-1] if i == nbasis-1 else nodes[i+1]
            cond1 = np.logical_and(pres>start, pres<=middle)
            cond2 = np.logical_and(pres>middle, pres<=end)
            
            phi[cond1,i] = (pres[cond1]-start)/(middle - start)
            phi[cond2,i] = (end-pres[cond2])/(end - middle)            
        
        self.phi = phi[::-1] if np.all(np.diff(self.pref)<=0)  else phi


    @staticmethod
    def calc_liq_cld_frac(temp, T0 = 273.16, Tice = 250.16):
        
        """
        Uses IFS formula for liquid cloud
        fraction computation.
        """
        f_l = np.maximum(0, temp - Tice ) / (T0 - Tice)  # numerator is zero if temp < Tice
        f_l = pow(f_l, 2)  # take square
        f_l = np.minimum(1., f_l)  # ensures that f_l <=  1
        return f_l

    @staticmethod
    def calc_rhc_clouds(pres, pres_surf, 
                                c_t = 0.7, c_s = 0.9, n = 4):

        """
        For pressure levels compute the critical rh
        above which clouds form. Following Quass (2012).
        """
        # parameters from Quass 2012, JGR.
        rh_c = c_t + (c_s - c_t) * np.exp(1-(pres_surf/pres)**n)
        return rh_c


    def calc_cld_frac(self, pres, f_l, temp, sphum, prc):

        """
        Use Sundqvist et al. 1989 scheme to predict cloud fraction
        as a function of grid rh and critical rh.
        """
        qsw = qs_calc(pres, temp)  # over water
        qsi = qsi_calc(pres, temp)  # over ice
        qs_mix = qsw * f_l + qsi * (1-f_l)  
        ind_base = self.ind_interface[0][-1]
        T_bot = temp[ind_base] # assume cloud base is at p = pb

        rh =  sphum/qs_mix  # convert temp. to sat. specific humidity

        ps = self.pinterface[0]  # surface pressure
        rh_c = self.calc_rhc_clouds(pres, ps)  # compute critical rh for cloud cover 
        rh = np.minimum(1, rh)  # ensure that rh does not exceed 1
        cldfrac = 1 - np.sqrt((1 - rh)/(1 - rh_c))
        cldfrac = np.maximum(0., cldfrac)  # ensure that cld fraction is > 0

        # w_l = gaussian(temp, T_mid, sigma) 
        sigma = 0.75 * (self.T_mid - self.T_top)  # Gaussian width 
        cldfrac = gaussian(temp, self.T_top, sigma) #+ 0.075 * gaussian(temp, T_bot, sigma * 0.25)
        cldfrac = (prc) * 0.5 * cldfrac/cldfrac.max()  # cloud fraction increases by 0.25 per mm/h or rain
        cldfrac[sphum <= 1e-5] = 0.  # zero out liq. water if the sp. humidity is too small (for stratosphere)
        cldfrac = np.minimum(1., np.maximum(0., cldfrac))  # ensure that cld fraction is > 0 and <1

        return rh, cldfrac


    def calc_incloud_liquid(self, temp, sphum, pres, prc):

        """
        Compute the in-cloud liquid water as a 
        function of precipitation and temperature.

        As the environment moves from non-precipitating to precipitation, the condensed
        water path has a sharp jump. This is captured using a linear function with a large 
        slope for prc <= 1 mm/hr. Further increases in precipitation result in smaller
        cwp increases. This is captured with a smaller slope.

        Non-precipitating environments are assigned a baseline cwp concentraion and profile.

        All parameters rely on ERA5 data.
        """
        
        # T_mid = 255.  # mid-tropospheric temp for peak liq. water concn in K 
        # T_top = 150.  # upper trop. temperature in K
        # T_freeze = 273.16 
        # sigma = 0.5 * (T_bot - T_mid)  # Gaussian width 
        
        ind_base = self.ind_interface[0][-1]
        pbase = self.profiles['pres'][ind_base]
        T_bot = temp[ind_base] # assume cloud base is at p = pb

        cwp_slope_large = 2600. #g/m^2 per mm/h of rain    
        cwp_slope_small = 10. #g/m^2 per mm/h of rain  
        cwp_non_prc = 200.  #g/m^2 baseline non-precipitating cwp

        # piecewise linear formulation for condensed water path
        # if prc <= 1 mm/h, use large slope, else use small slope
        cwp = cwp_slope_large * prc if prc<=1 else cwp_slope_large + cwp_slope_small * prc
        sigma = 0.5 * (T_bot - self.T_mid)  # Gaussian width 

        w_l = gaussian(temp, self.T_mid, sigma) 
        w_l[sphum <= 1e-5] = 0.  # zero out liq. water if the sp. humidity is too small (for stratosphere)
        w_l[pres > pbase] = 0. # zero out liq. water below the nominal cloud base
        w_l = (cwp/np.sum(w_l)) * w_l  # scale the liq. water profile such that the vertical sum gives the cwp

        w_l_non_prc = self.calc_incloud_liquid_nonprc(temp, sphum, self.T_freeze, self.T_top)
        w_l_non_prc[pres > pbase] = 0.0
        w_l_non_prc = (cwp_non_prc/np.sum(w_l_non_prc)) * w_l_non_prc

        w_l = np.heaviside(prc, 0) * w_l #+ (1 - np.heaviside(prc, 0)) * w_l_non_prc
        
        return w_l

    @staticmethod
    def calc_incloud_liquid_nonprc(temp, sphum, T_mid, T_top, wl_max = 0.18):

        """
        Compute the in-cloud liquid water as a 
        function of temperature following 
        Liu et al. 2021, GMD.
        """

        wl_min = 3e-4  #g/kg
        w_l = wl_max * np.minimum(1, (temp - T_top)**4/(T_mid - T_top)**4)
        w_l = np.maximum(wl_min, w_l)
        w_l = xr.where(sphum <= 1e-5, 0., w_l)

        return w_l

class RadiativeComputation(CloudProperties):

    def __init__(self, constants, phys_params, nlayers, pinterfaces, 
                 rad_dir, fil = params_file, arm_file = arm_file):
        super().__init__(constants, phys_params, nlayers, pinterfaces, fil, arm_file)

        # initialize 
        self.rad = None
        self.clim_dict = {}
        self.rad_fluxes = {k: {} for k in ['LW', 'SW']}

        # read profiles from ARM
        self.main()  

        self.rad_dir = rad_dir

        # define dict to hold coeffs
        self.epsilon = dict(LW = None, SW = None)

        # declare empty dicts for each layer
        self.epsilon['LW'] = {k : {} for k in range(self.nlayers)}
        self.epsilon['SW'] = {k : {} for k in range(self.nlayers)}

        # self.cre_dict = cre_dict # dict holding values for cloud-radiative effect computation.
        self.phi = None

        # self.pert_range = np.arange(-2, 2.25, 0.25)  # perturbation range for epsilon (in K)
        self.pert_range = np.arange(-1, 1.1, 0.1) * 1e-1  # perturbation range for epsilon (in K)

        # matrix to hold radiative coefficients
        # 2nlayers + 1 variables (T, q for each layer and surface)
        self.epsilon_rad_matrix = np.zeros((self.nlayers * 2 + 1, self.nlayers))

    def init_rad(self):

        """
        Initialize atmospheric column state for 
        RRTMG calculations
        """
        state = climlab.column_state(lev = self.pref_flip)
        sfc, atm = climlab.domain.single_column(lev = self.pref_flip, water_depth = 50.)
        self.clim_dict.update(state = state, sfc = sfc, atm = atm)


    def read_layer_flux_components(self, wavelen: str):

        """
        Read and store the layer radiative flux components.
        wavelen: longwave or shortwave
        """

        rad_dict = {}
        half_res = self.res / 2

        flux_up = f'{wavelen}_flux_up'
        flux_dwn = f'{wavelen}_flux_down'
        ds_rad = self.ds_rad
        for i, n in enumerate(range(-1, self.nlayers)):

            if n > - 1:
                pres_interface = self.pinterface[i]
                rad_dict[f'{n}_up'] = ds_rad[flux_up].sel(lev_bounds = pres_interface + half_res)
                rad_dict[f'{n}_dwn'] = ds_rad[flux_dwn].sel(lev_bounds = pres_interface + half_res)

            else:
                rad_dict[f'{n}_up'] = ds_rad[flux_up][-1]
                rad_dict[f'{n}_dwn'] = ds_rad[flux_dwn][-1]
        
        rad_dict = {key : rad_dict[key].item() for key in rad_dict.keys()}  # only store item

        return rad_dict
    
    def compute_layer_rad_flux(self):
        
        """
        Compute net radiative fluxes in and out of a layer
        Input:
        -----
            wavelen: longwave or shortwave
            layer: index of layer (-1 is surface)
            """

        for wavelen in ['LW', 'SW']:
            rad_dict = self.read_layer_flux_components(wavelen) # get 
        
            for n in range(-1, self.nlayers):

                if n == -1: # surface
                    rad_in = rad_dict[f'{n}_dwn']
                    rad_out = rad_dict[f'{n}_up']
                
                else:
                    rad_in = rad_dict[f'{n-1}_up'] + rad_dict[f'{n}_dwn']
                    rad_out = rad_dict[f'{n-1}_dwn'] + rad_dict[f'{n}_up']

                self.rad_fluxes[wavelen][n] = rad_in - rad_out

    @staticmethod
    def flip_array(x, pres):
        if all(pres[1:] < pres[:-1]):  # if pressure levels are strictly decreasing:
            return x[::-1]
        else:
            return x
        
    @staticmethod
    def compute_rad(temp, sphum, Tsurf, lev, 
                    clim_dict, cloud_props):

        """
        For any input temp, sphum and Tsurf,
        this methods computes the radiative
        diagnostics (including fluxes)
        """

        Tatm = field.Field(temp, domain = clim_dict['atm']) 
        Ts = field.Field(Tsurf, domain = clim_dict['sfc'])
        clim_dict['state'].update(Tatm = Tatm, Ts = Ts)

        rad = climlab.radiation.RRTMG(name='Column RRTMG',  # Model name
                                    state = clim_dict['state'],   # Initial condition
                                    specific_humidity = sphum,  # sp.humidity, flipped to ensure increasing pres. levels
                                    lev = lev,
                                    albedo = 0.08,  # surface shortwave albedo for ocean surface
                                    coszen = 1.0/np.pi, # daily-avg. cos zenith angle at equinox
                                    **cloud_props)  
        
        rad.compute_diagnostics()
        ds_rad = climlab.to_xarray(rad.diagnostics)
        return rad, ds_rad
    
    def setup_compute_rad(self, **perts):  # prc is precipitation in mm/hr

        self.compute_profiles_from_perturbations(**perts)
        self.compute_cloud_props(**perts)

        # flip arrays to lie on increasing pressure levels: consistent with climlab format
        arr = [self.temp_ext, self.sphum_ext, self.cldfrac, self.wl, self.re]
        temp,  sphum, cld_frac, clwp, re = [self.flip_array(i, self.pref) for i in arr]
        
        cloud_props = dict(icld = 2,  # maximum random overlap
                           iceflglw = 0, cldfrac = cld_frac,               
                           clwp = clwp, r_liq = re)
        
        self.rad, self.ds_rad = self.compute_rad(temp, sphum, self.Tsurf, 
                                                 self.pref_flip, self.clim_dict, 
                                                 cloud_props)

        self.compute_layer_rad_flux()

    def perturb_rrtm(self, pert_var, pert_range):
        
        """
        Perturb RRTM selfect for a given 
        variable and perturbation range
        """

        QLW = {}
        QSW = {}
        x = []

        for i in pert_range:

            
            try:
                pert_dict = {pert_var: i}
                self.setup_compute_rad( **pert_dict )
                x.append(i)
            
            except AssertionError:
                
                # if perturbation produces negative humidity
                # reduce size by a factor of 10

                pert_dict = {pert_var: i * 1e-1}
                self.setup_compute_rad( **pert_dict )
                x.append(i * 1e-1)

            # empty dicts for each layer
            QLW[i] = {k: {} for k in range(-1, self.nlayers)} 
            QSW[i] = {k: {} for k in range(-1, self.nlayers)} 

            for k in range(-1, self.nlayers):
                QLW[i][k] = self.rad_fluxes['LW'][k]
                QSW[i][k] = self.rad_fluxes['SW'][k]
            

        x = np.array(x)
        yLW = {k: {} for k in range(-1, self.nlayers)}  # slopes 
        ySW = {k: {} for k in range(-1, self.nlayers)}

        for l in range(-1, self.nlayers):

            yLW[l] = self.get_slope_intercept_from_dict(x, QLW, l) 
            ySW[l] = self.get_slope_intercept_from_dict(x, QSW, l) 

        return yLW, ySW

    @staticmethod
    def get_slope_intercept_from_dict(x, d1, ilayer):
        y = np.array([d1[k][ilayer] for k in d1.keys()])
        reg = linregress(x, y)
        ydict = dict(y = y, slope = reg[0], intercept = reg[1])
        return ydict


    def generate_rad_coeffs(self):

        epsilon_rad_coeffs = {}
        print(f'Generating clear-sky radiation coefficients')

        for pert_var in list(self.perts.keys())[:-1]: # all pert keys except precip.
                print(pert_var)
                epsilon_rad_coeffs[pert_var] = []
                assert pert_var in self.perts.keys(), 'Perturbation variable not in the list of perturbations'
                yLW, ySW = self.perturb_rrtm(pert_var, self.pert_range)

                for ilayer in range(-1, self.nlayers):
                        epsilon_rad_coeffs[pert_var].append(yLW[ilayer]['slope'] + ySW[ilayer]['slope'])
                
                print(epsilon_rad_coeffs[pert_var])
                print('==========================')

        l1 = list(self.perts.keys())[:-1]
        l1.insert(0, l1.pop(-1))

        # convert to per s
        for i, pert_var in enumerate(l1):
            norm = self.constants['CPD'] * self.deltap * 1e2 /self.constants['GRAVITY'] 
            self.epsilon_rad_matrix[i, :] = np.array(epsilon_rad_coeffs[pert_var])[1:]/norm


    def save_rad_coeffs(self):

        print('Saving radiative coeffs')
        fil_out = self.rad_dir + f'epsilon_rad_matrix_n={self.nlayers}.npy'
        np.save(fil_out, self.epsilon_rad_matrix)  
        print(f'Radiative coefficients saved to {fil_out}')  


class nLayerModel(LLMParams):
    
    """
    Build the n-layer model, starting with governing equations
    """

    def __init__(self, constants, phys_params, nlayers, pinterfaces, rad_fil, conv_file, 
                 fil=params_file, arm_file=arm_file):
        
        super().__init__(constants, phys_params, nlayers, pinterfaces, fil, arm_file)
        super().main()

        # parameters for matrix evaluation
        self.params_eval = {'mom': {}, 'dse': {}, 'moisture': {}}  # parameters for matrix evaluation

        self.matrices = {}

        self.epsilon_rad_matrix = np.load(rad_fil)
        self.epsilon_conv_matrix = np.load(conv_file) * self.phys_params['fp']  # scale adjustment timescales
        self.epsilon_conv_matrix[:2, ...] = self.epsilon_conv_matrix[:2, ...] * (1 + self.phys_params['epsilon_cre'])  # cloud-radiative effect
        
        # perturbation for adjustment timescales
        thermo_pert = self.phys_params['thermo_pert']

        # self.epsilon_conv_matrix[0, 1:, 1] += thermo_pert  # trop heating response to lft perturbations
        # self.epsilon_conv_matrix[0, 1:, 0] += thermo_pert  # trop heating response to bl perturbations
        # self.epsilon_conv_matrix[0, :1, 0] += thermo_pert  # bl heating response to bl perturbations
        # self.epsilon_conv_matrix[0, :1, 1] += thermo_pert  # bl heating response to ft perturbations

        # self.epsilon_conv_matrix[0, :, :2] += thermo_pert  # ft heating response to bl and ft perturbations
        
        # self.epsilon_conv_matrix[:2, 1:, 0]  *=  thermo_pert  # free-trop. heating response to boundary layer perturbations
        # self.epsilon_conv_matrix[:, 1:, :2]  *=  thermo_pert    # free-trop. heating response to boundary layer perturbations

        # self.epsilon_conv_matrix[0, 1:, 1] += thermo_pert  # trop heating response to lft perturbations
        # self.epsilon_conv_matrix[1, 1:, 1] += thermo_pert  # trop heating response to lft q pertrubations
        # self.epsilon_conv_matrix[1, 1:, 0] += thermo_pert  # trop heating response to bl q pertrubations

        # self.epsilon_conv_matrix[2, 1, 1] += thermo_pert  # trop drying response to lft T pertrubations
        # self.epsilon_conv_matrix[2, 1, 0] += thermo_pert  # trop drying response to bl T pertrubations
        # self.epsilon_conv_matrix[3, 1, 1] += thermo_pert  # trop drying response to lft T pertrubations
        # self.epsilon_conv_matrix[3, 1, 0] += thermo_pert  # trop drying response to bl T pertrubations

        # self.epsilon_conv_matrix[:2, 1:, :2] += thermo_pert  # ft heating response to lft_bl qT pertrubations

        # self.epsilon_conv_matrix[0, 1:, 0] += thermo_pert  # trop heating response to bl perturbations


        self.Mcoeff_all = {'conv': None, 'nconv': None} # final coefficient matrix with the governing equations
        self.Meval = {'conv': None, 'nconv': None}  # coefficient matrix numerically evaluated
        self.state_vec = None
        self.delta_vec = self.Tvec = self.qvec = None
        self.delta_sols = {}
        self.Mforced = None

        # assumptions for sympy variables
        self.args_rp = dict(real = True, positive = True)
        self.args_r = dict(real = True)
        self.ksymb = sp.symbols('k')

        self.symbols = {}
        self.params_Dm = {}
        self.diabatic_terms = {}

    def define_symbols(self):

        args_rp = self.args_rp
        args_r = self.args_r

        self.symbols['D_m'] = sp.symbols('D_m', **args_rp)
        self.symbols['D_T'] = sp.symbols('D_T', **args_rp)
        self.symbols['D_q'] = sp.symbols('D_q', **args_rp)
        
        self.symbols['epsilon_turb'] = sp.symbols('epsilon_turb', **args_rp)
        self.symbols['epsilon_mix'] = sp.IndexedBase('epsilon_mix', **args_rp)
        self.symbols['epsilon_m'] = sp.IndexedBase('epsilon_m', **args_rp)  # momentum mixing

        self.symbols['Adv_T'] = sp.IndexedBase('Adv_T', **args_r)
        self.symbols['Adv_q'] = sp.IndexedBase('Adv_q', **args_r)

        self.symbols['epsilon_sT'] = sp.IndexedBase('epsilon_sT', **args_r)
        self.symbols['epsilon_qq'] = sp.IndexedBase('epsilon_qq', **args_r)
        self.symbols['gamma_s'] = sp.symbols('gamma_s', **args_r)
        self.symbols['epsilon_Tsurf'] = sp.IndexedBase('epsilon_Ts', **args_r)

        self.symbols['Delta_p'] = sp.IndexedBase('\\Delta p', **args_rp)
        self.symbols['kappa_s'] = sp.symbols('kappa_s', **args_r)
        
        self.symbols['m_b'] = sp.IndexedBase('m_b', **args_r)
        self.symbols['tau'] = sp.IndexedBase('tau', **args_rp)
        self.symbols['tau_f'] = sp.IndexedBase('tau_f', **args_rp)
        self.symbols['c'] = sp.IndexedBase('c', **args_rp)

        # coefficients for matching
        self.symbols['A'] = sp.IndexedBase('A')
        self.symbols['B'] = sp.IndexedBase('B')

        self.symbols['a'] = sp.IndexedBase('a', **args_r)
        self.symbols['b'] = sp.IndexedBase('b', **args_r)

        # Ms dagger symbols
        self.symbols['Msd'] = sp.IndexedBase('M^\\dagger_s', **args_r)
        self.symbols['Msdd'] = sp.IndexedBase('M^\\ddagger_s', **args_r)
        
        self.symbols['Mqd'] = sp.IndexedBase('M^\\dagger_q', **args_r)
        self.symbols['Mqdd'] = sp.IndexedBase('M^\\ddagger_q', **args_r)

        # diabatic terms
        self.symbols['Qc'] = sp.symbols('Q_c', **args_r)
        self.symbols['Qr'] = sp.symbols('Q_r', **args_r)
        self.symbols['Qsh'] = sp.symbols('Q_sh', **args_r)

        # convective interface
        self.symbols['xc'] = sp.symbols('x_c', **args_r)


        
    def construct_square_matrix(self, ncols, sym):

        args_r = dict(real = True)
        self.symbols[sym] = sp.IndexedBase(sym, **args_r)
        a = sp.IndexedBase(sym, **args_r)
        """
        Construct a square matrix with ncols
        """
        elements = []
        for i in range(ncols):
            elements.append([a[i, j] for j in range(ncols)])
        return sp.Matrix(elements), a 

    def momentum_eqns_deltacoeffs(self):

        a = self.symbols['a']
        n = self.nlayers
        epsilon_turb_on = 1
        # epsilon_turb = self.symbols['epsilon_turb']
        epsilon_m = self.symbols['epsilon_m'] * epsilon_turb_on
        k = self.ksymb
        Dm = self.symbols['D_m']


        # create banded matrix for delta
        M = sp.banded(n, {0:[a[i,i] for i in range(n)], 
                          1:[a[i, i+1] for i in range( n - 1 )], 
                         -1:[a[i +1, i] for i in range( n - 1 )]})

        # diagonal terms
        for i in range(self.nlayers):

            # M = M.subs({a[i, i] : 1})

            if i == 0:
                M = M.subs({a[i, i] : 1 - Dm * pow(k,2) + epsilon_m[i]  })
                M = M.subs({a[i, i + 1] : -epsilon_m[i]})

            elif i == self.nlayers - 1:
                M = M.subs({a[i, i] : 1 - Dm * pow(k,2) + epsilon_m[n-2]})
                M = M.subs({a[i, i -1] : -epsilon_m[n-2] })

            else:
                M = M.subs({a[i, i] : 1 - Dm * pow(k,2) + (epsilon_m[i] + epsilon_m[i-1])})
                M = M.subs({a[i, i + 1] : -epsilon_m[i]})
                M = M.subs({a[i, i - 1] : -epsilon_m[i-1]})

        return M

    def momentum_eqns_Tcoeffs(self, M, a):

        args_rp = self.args_rp
        args_r = self.args_r

        k = self.ksymb

        m_b = self.symbols['m_b'] 
        tau = self.symbols['tau']
        tau_f = self.symbols['tau_f']
        c = self.symbols['c']

        M = M.subs([(a[0, i],  pow(c[i], 2)/pow(c[0], 2) * m_b[i] * tau_f[i]) for i in range(self.nlayers) if i > 0]) 
        M = M.subs({a[0, 0]: 1 -  tau_f[0]}) 
        M = M.subs([(a[i, 0], tau[i] * tau_f[0] / tau[0]) for i in range(self.nlayers)]) 

        # diagonal terms
        M = M.subs([(a[i, i], (tau[j] / tau[0]) * (pow(c[j], 2)/pow(c[0], 2)) * m_b[j] * (1 - tau_f[j])) for i in range(self.nlayers) for j in range(self.nlayers) if i == j and i > 0])
        
        # cross diagonal terms
        M = M.subs([(a[i, j], -(tau[j] / tau[0]) * (pow(c[j], 2)/pow(c[0], 2)) * m_b[j] * tau_f[j]) for i in range(self.nlayers) for j in range(self.nlayers) if i != j and i > 0 ])

        # include k^2 terms and switching sides from RHS to LHS
        M = M.elementary_row_op('n->kn', row1 = 0, k = -pow(k,2))
        for i in range(1, self.nlayers):
            M = M.elementary_row_op('n->kn', row1 = i, k = pow(k,2))

        # get parameters to numerically evaluate matrix
        mom_params_eval = {}
        mom_params = self.mom_params

        for key in mom_params.keys():
            for n in range(self.nlayers):
                if key in ['tau_i'] :
                    mom_params_eval.update({tau[n] : mom_params[key][n]})  
                elif key in ['tau_if']:
                    mom_params_eval.update({tau_f[n] : mom_params[key][n]})
                elif key in ['mbi']:
                    mom_params_eval.update({m_b[n] : mom_params[key][n]})
                elif key == 'ci':
                    mom_params_eval.update({c[n] : mom_params[key][n]})

        self.params_eval['mom'] = mom_params_eval

        return M

    def thermo_eqn_deltacoeffs(self, M, a, var = 'dse'):

        """"
        Set the coeffficients of divergence in the dry static energy equations
        """

        args_rp = self.args_rp
        args_r = self.args_r

        if var not in ['dse', 'q']:
            raise ValueError('Invalid variable name')

        if var == 'dse':
            Mxd = self.symbols['Msd']
            Mxdd = self.symbols['Msdd']
            thermo_params = self.dse_params

        elif var == 'q':
            Mxd = self.symbols['Mqd']
            Mxdd = self.symbols['Mqdd']
            thermo_params = self.q_params

        Deltap = self.symbols['Delta_p']

        M = M.subs([(a[i, i], -Mxd[i]) for i in range(self.nlayers)])
        M = M.subs([(a[i, j], (Deltap[j]/Deltap[i]) * Mxdd[i]) for i in range(self.nlayers) for j in range(self.nlayers) if i > j])
        M = M.subs([(a[i, j], 0) for i in range(self.nlayers) for j in range(self.nlayers) if i < j])
        
        # parameter file
        thermo_delta_params_eval = {}

        for key in thermo_params.keys(): 
            for n in range(self.nlayers):
                if key in ['Msi_dagger_ref', 'Mqi_dagger_ref']:
                    thermo_delta_params_eval.update({Mxd[n] : thermo_params[key][n]})   
                elif key in ['Msi_ddagger_ref', 'Mqi_ddagger_ref']:
                    thermo_delta_params_eval.update({Mxdd[n] : thermo_params[key][n]})
                
                thermo_delta_params_eval.update({Deltap[n] : self.deltap[n]})

        if var == 'dse':
            self.params_eval['dse']['delta'] = thermo_delta_params_eval

        elif var == 'q':
            self.params_eval['moisture']['delta'] = thermo_delta_params_eval
            
        return M

    def thermo_eqn_Tqcoeffs_symbolic(self, M, var = 'dse'):
            
        """
        Compute the coefficients for temperature in the
        dse/water vapor equation
        """

        args_rp = self.args_rp
        args_r = self.args_r

        if var not in ['dse', 'q']:
            raise ValueError('Invalid variable name')

        # declare coefficients
        k = self.ksymb

        epsilon_mix =  self.symbols['epsilon_mix']
        tau_f = self.symbols['tau_f']
        c = self.symbols['c']    
        kappa_s = self.symbols['kappa_s']
        Deltap = self.symbols['Delta_p']

        # flag for Advection and WISHE 
        WISHE_flag = int(self.phys_params['WISHE_on'])

        # WISHE terms
        Tas_bar = self.profiles['Tref'][0]
        DeltaTs_bar = self.phys_params['DeltaTs_bar']
        kg_per_kg_to_K = self.constants['HLATENT']/self.constants['CPD']
        
        # combined radiative convective coefficients
        if var == 'dse':

            Adv_flag = int(self.phys_params['TAdv_on'])  
            epsilon_1 =  self.symbols['epsilon_sT']
            abi = self.symbols['a']  # aprofile at the top of the layer
            DTq, AdvTq = self.symbols['D_T'], self.symbols['Adv_T']
            ABi = sp.IndexedBase('A', **args_r)
            kappa_w = sp.symbols('kappa_sh', **args_r)
            DiffOn = 1  # Flag for diffusion

        elif var == 'q':
            
            Adv_flag = int(self.phys_params['qAdv_on'])  
            epsilon_1 =  self.symbols['epsilon_qq']
            abi = self.symbols['b']  # aprofile at the top of the layer
            DTq, AdvTq = self.symbols['D_q'], self.symbols['Adv_q']
            ABi = sp.IndexedBase('B', **args_r)
            kappa_w = sp.symbols('kappa_lh', **args_r)
            DiffOn = 1 # Flag for diffusion

            qas_bar = self.profiles['qref'][0] 
            Deltaqs_bar = qs_calc(self.pref[0], Tas_bar + DeltaTs_bar) - qas_bar 
            Deltaqs_bar = Deltaqs_bar * kg_per_kg_to_K
 
        # fill in the matrix, symbolically

        n = self.nlayers 

        M = M.subs([(epsilon_1[0, 0], 
                     epsilon_1[0, 0] - kappa_s - epsilon_mix[0] * abi[0] + DTq * ABi[0] * pow(k,2) - AdvTq[0] * k * Adv_flag + k * kappa_w * (1 - tau_f[0]) * WISHE_flag)])
       
        M = M.subs([(epsilon_1[0, 1], 
                     epsilon_1[0, 1] + epsilon_mix[0] + k * kappa_w * tau_f[1] * (c[1] **2)/(c[0] **2) * WISHE_flag)])
       
        M = M.subs([(epsilon_1[0, i], 
                     epsilon_1[0, i] + k * kappa_w * tau_f[i] * (c[i] **2)/(c[0] **2) * WISHE_flag) for i in range(self.nlayers) if i > 1])

        M = M.subs([(epsilon_1[i, i], 
                     epsilon_1[i, i]  - (epsilon_mix[i - 1]  + epsilon_mix[i] * abi[i]) * (Deltap[0]/Deltap[i]) + DTq * ABi[i] * pow(k,2) - AdvTq[i] * k * Adv_flag) for i in range(self.nlayers) if i > 0 and i < n - 1])

        M = M.subs([(epsilon_1[i, i + 1], 
                     epsilon_1[i, i + 1] + epsilon_mix[i] * (Deltap[0]/Deltap[i])) for i in range(self.nlayers) if i > 0])
        
        M = M.subs([(epsilon_1[i, i - 1], 
                     epsilon_1[i, i - 1] + epsilon_mix[i - 1] * (Deltap[0]/Deltap[i] * abi[i - 1])) for i in range(self.nlayers) if i > 0])
        
        M = M.subs([(epsilon_1[n-1, n-1], 
                     epsilon_1[n-1, n-1]  - epsilon_mix[n - 2] * (Deltap[0]/Deltap[n-1]) + DiffOn * DTq * ABi[n - 1] * pow(k,2) - AdvTq[n-1] * k * Adv_flag)])



        return M

    def uwind_phi_symbolic(self):

        uwind_dict = {}
        phi_dict = {}
        
        k = self.ksymb
        Vi = self.vert_struct_params['Vave']
        tau_f = self.symbols['tau_f']
        tau = self.symbols['tau']
        c = self.symbols['c']
        mb = self.symbols['m_b']

        Tvec = self.Tvec
        uvec = self.uvec
        phivec = self.phivec

        aplus_layer_ave = self.vert_struct_params['aplus_layer_ave']
        aplus_profiles = self.vert_struct_params['aplus_profiles']

        for i, (usymb, Tsymb) in enumerate(zip(uvec, Tvec)):
            
            if i == 0:
                temp = sum([mb[j] * tau_f[j] * pow(c[j]/c[0], 2) * Tvec[j] for j in range(1, self.nlayers)])
                uwind_dict[usymb] = (k / Vi[i]) * ((1 - tau_f[0]) * Tsymb + temp)
            
            else:
                temp1 = -sum([mb[j] * tau_f[j] * pow(c[j]/c[0], 2) * Tvec[j] for j in range(1, self.nlayers) if j != i])
                temp2 = (1 - tau_f[i]) * mb[i] * pow(c[i]/c[0], 2) * Tsymb
                temp0 = tau_f[0] * Tvec[0]
                uwind_dict[usymb] = -(k / Vi[i]) * (tau[i]/tau[0]) * (temp0 + temp1 + temp2)

        # non-dimensionalized phi

        for i, (phisymb, Tsymb) in enumerate(zip(phivec[1:], Tvec)):

            if i == 0:
                temp = sum([mb[j] * tau_f[j] * pow(c[j] / c[0], 2) * Tvec[j] for j in range(1, self.nlayers)])
                phi_dict[phisymb] = Tsymb * pow(c[0]/c[0], 2) * tau_f[0] - temp

            else:
                temp0 = Tvec[0] * pow(c[0] / c[0], 2) * tau_f[0]
                temp1  = sum([mb[j] * pow(c[j] / c[0], 2) * ( aplus_profiles[j][-1]/aplus_layer_ave[j] - tau_f[j] ) * Tvec[j] for j in range(1, i + 1)])
                temp2 = -sum([mb[j] * tau_f[j] * pow(c[j] / c[0], 2) * Tvec[j] for j in range(i + 1, self.nlayers)])

                phi_dict[phisymb] = temp0 + temp1 + temp2

        # phi_dict[phivec[0]] = phi_dict[phivec[1]] - aplus_profiles[0][-1]/aplus_layer_ave[0] * Tvec[0]  # surface geopotential


        return uwind_dict, phi_dict

    def construct_forced_vector(self, M):

        kappa_s = self.symbols['kappa_s']
        epsilon_Ts = self.symbols['epsilon_Tsurf']
        gamma_s = self.symbols['gamma_s']
        n = self.nlayers
        Msref = self.vert_struct_params['Msref']
        # Msref, _ = self.get_Msref_unmod()
        Msbref = Msref[0]

        # surface flux coefficients
        M[n, 0] = (kappa_s + epsilon_Ts[0]) 
    
        if self.phys_params['q_on']:
            M[2 * n, 0] = kappa_s * gamma_s 
    
        # radiation coefficients
        for i in range(n + 1, 2 * n):
            M[i, 0] = epsilon_Ts[i - n]

        M = M * self.phys_params['Ts0'] /Msbref

        return M

    def return_surf_flux_terms(self):

        # Surface flux terms
        Vs = self.profiles['Vprofiles'][0][0]
        Vb = self.vert_struct_params['Vave'][0]
        Tas_bar = self.profiles['Tref'][0]
        DeltaTs_bar = self.phys_params['DeltaTs_bar']
        rhogCd_deltapBinv = self.phys_params['rhoCd'] * self.constants['GRAVITY']/(self.deltap[0] * 1e2)  # units: 1/m
        kg_per_kg_to_K = self.constants['HLATENT']/self.constants['CPD']
        WISHE_FLAG = int(self.phys_params['WISHE_on'])

        # parameters for WISHE
        qas_bar = self.profiles['qref'][0] 
        Deltaqs_bar = qs_calc(self.pref[0], Tas_bar + DeltaTs_bar) - qas_bar 
        Deltaqs_bar = Deltaqs_bar * kg_per_kg_to_K
        usurf = self.profiles['uref'][0]   

        return Vs, Vb, DeltaTs_bar, rhogCd_deltapBinv, WISHE_FLAG, Deltaqs_bar, usurf

    def thermo_eqn_Tqcoeffs_params_general(self, params_dict):
        
        """
        Compute the coefficients for temperature in the
        dse/water vapor equation
        """

        args_rp = self.args_rp
        args_r = self.args_r

        # Coefficients for non-dimensionalization
        ci = self.mom_params['ci']
        taui = self.mom_params['tau_i']
        tau_if = self.mom_params['tau_if']
        mbi = self.mom_params['mbi']

        cb = ci[0]
        taub = taui[0]
        Msref = self.vert_struct_params['Msref']
        # Msref, _ = self.get_Msref_unmod()
        MsBref = Msref[0]

        # surface flux terms
        Vs, Vb, DeltaTs_bar, rhogCd_deltapBinv, WISHE_FLAG, Deltaqs_bar, usurf = self.return_surf_flux_terms()
        kappa_wishe = (Vs/Vb) * cb * taub * rhogCd_deltapBinv * (1/MsBref) * WISHE_FLAG

        kappas_pert = self.phys_params['kappas_pert']

        # parameters for vapor disequilibrium
        print(usurf, rhogCd_deltapBinv)
        kappa_s_star = taub * ( rhogCd_deltapBinv  * abs(usurf)  + kappas_pert)  
        self.kappa_s_star = kappa_s_star

        Lv = self.constants['HLATENT']
        CPD = self.constants['CPD']
        RV = self.constants['RV']
        Ts = self.Tsref
        gamma_s = pow(Lv, 2) * qs_calc(1000, Ts)/ (CPD * RV * pow(Ts, 2))

        # general parameters
        Deltap = self.symbols['Delta_p']
        c = sp.IndexedBase('c', **args_rp)

        tau = sp.IndexedBase('tau', **args_rp)
        tau_f = sp.IndexedBase('tau_f', **args_rp)
        mb = sp.IndexedBase('m_b', **args_r)

        epsilon_mix = self.symbols['epsilon_mix']
        epsilon_m = self.symbols['epsilon_m']
        
        epsilon_mix_val = self.phys_params['epsilon_mix'] * taub
        epsilon_m_val = self.phys_params['epsilon_turb'] * taub
        
        # params_dict.update({self.symbols['epsilon_turb'] : self.phys_params['epsilon_turb'] * taub})
        params_dict.update({self.symbols['kappa_s'] : kappa_s_star})
        # params_dict.update({self.symbols['D_m'] : self.phys_params['Dm'] / (pow(cb, 2 ) * taub)})    
        params_dict.update({self.symbols['gamma_s'] : gamma_s})

        # surface radiative effects
        rad_vector = self.epsilon_rad_matrix[0, :] * taub
        params_dict.update({self.symbols['epsilon_Tsurf'][i] : rad_vector[i] for i in range(self.nlayers)})

        
        for i in range(self.nlayers):
            params_dict.update({Deltap[i] : self.deltap[i], c[i] : ci[i], 
                                tau_f[i] : tau_if[i], tau[i] : taui[i], mb[i] : mbi[i]})
            if i < self.nlayers - 1:
                params_dict.update({epsilon_mix[i] : epsilon_mix_val[i],
                                    epsilon_m[i] : epsilon_m_val[i]})

        for k1 in ['s', 'q']:

            # Diffusion 
            DTq = self.symbols['D_T'] if k1 == 's' else self.symbols['D_q']
            Diff_Tq = self.phys_params['DT'] if k1 == 's' else self.phys_params['Dq']
            params_dict.update({DTq : Diff_Tq / (pow(cb, 2 ) * taub)})    

            # WISHE
            kappa_w = sp.symbols('kappa_sh', **args_r) if k1 == 's' else sp.symbols('kappa_lh', **args_r)
            kappa_wishe_star = kappa_wishe * DeltaTs_bar if k1 == 's' else kappa_wishe * Deltaqs_bar
            params_dict.update({kappa_w : kappa_wishe_star})

            # Layer params

            # Advection param
            Adv_Tq = sp.IndexedBase('Adv_T', **args_r) if k1 == 's' else sp.IndexedBase('Adv_q', **args_r)
            AdvTq_ref = self.vert_struct_params['AdvT_ref'] if k1 == 's' else self.vert_struct_params['Advq_ref']

            # layer averages
            ab_layer_ave_symb = sp.IndexedBase('A', **args_r) if k1 == 's' else sp.IndexedBase('B', **args_r)
            ab_layer_ave = self.vert_struct_params['a_layer_ave'] if k1 == 's' else self.vert_struct_params['b_layer_ave']

            # profiles
            ab_profile_symb = self.symbols['a'] if k1 == 's' else self.symbols['b']
            ab_profile = self.vert_struct_params['a_profiles'] if k1 == 's' else self.vert_struct_params['b_profiles']

            # Dagger (interfacial) terms 
            Mxd = sp.IndexedBase(f'M^\\dagger_{k1}', **args_r)
            Mxdd = sp.IndexedBase(f'M^\\ddagger_{k1}', **args_r)

            Mxd_ref = self.dse_params['Msi_dagger_ref'] if k1 == 's' else self.q_params['Mqi_dagger_ref']
            Mxdd_ref = self.dse_params['Msi_ddagger_ref'] if k1 == 's' else self.q_params['Mqi_ddagger_ref']

            for i in range(self.nlayers):
                params_dict.update({Adv_Tq[i] : AdvTq_ref[i] / cb, 
                                    ab_layer_ave_symb[i] : ab_layer_ave[i], 
                                    Mxd[i] : Mxd_ref[i], Mxdd[i] : Mxdd_ref[i]})
                params_dict.update({ab_profile_symb[i] : ab_profile[i][-1]})        

    def thermo_eqn_Tqcoeffs_params_radconv(self, params_dict, opt = 'nconv'):
            
        # """
        # Compute the convective and radiative heating
        # coefficients in the
        # dse/water vapor equation
        # """

        args_rp = self.args_rp
        args_r = self.args_r
        pert_mat = self.phys_params['epsilon_cr_pert']  # matrix of perturbations

        conv_opt = 1 if opt == 'conv' else 0
        taub = self.mom_params['tau_i'][0]

        for c1, k1 in enumerate(['s', 'q']):

            # update convection-radiation param
            for ix, k2 in enumerate(['T', 'q'], start = c1 * 2): 
                k = f'{k1}{k2}'

                # radiation parameters
                if k == 'sT':
                    rad_matrix = self.epsilon_rad_matrix[1:self.nlayers + 1, :].T 
                elif k == 'sq':
                    rad_matrix = self.epsilon_rad_matrix[self.nlayers + 1:, :].T * (self.phys_params['qrad_on'])
                else:
                    rad_matrix = np.zeros((self.nlayers, self.nlayers))  # no radiation contribution to q equation

                # convection parameters
                # assert pert_mat.shape == Qmat.shape, 'Perturbation matrix shape mismatch'
                perts = np.ones_like(pert_mat) + pert_mat
                Qmat = (self.epsilon_conv_matrix[ix, :, :] * conv_opt + rad_matrix) * taub #* perts  # note that perts is a multiplicative factor
                epsilon = sp.IndexedBase(f'epsilon_{k}', **args_r)
                params_dict.update({epsilon[i, j] : Qmat[i,j] for i in range(self.nlayers) for j in range(self.nlayers)})

    def assign_Dm_param(self):
        cb = self.mom_params['ci'][0]
        taub = self.mom_params['tau_i'][0]
        Dm = self.phys_params['Dm'] / (pow(cb, 2 ) * taub)
        self.params_Dm = {self.symbols['D_m'] : Dm}

    def construct_governing_eqn_matrix(self):
        
        # initialize dicts of coefficient matrices
        Mcoeff_dict = {'mom': {} ,'dse': {}, 'moisture': {}}

        Mcoeff_dict['dse']['T'] = {'conv': None, 'nconv': None}
        Mcoeff_dict['dse']['q'] = {'conv': None, 'nconv': None}

        Mcoeff_dict['moisture']['T'] = {'conv': None, 'nconv': None}
        Mcoeff_dict['moisture']['q'] = {'conv': None, 'nconv': None}

        # dicts to collect the coefficients for governing equations
        Mcoeff_collected = {'mom': None, 'dse': {'conv': None , 'nconv': None}, 
                            'moisture': {'conv': None , 'nconv': None}}    
        
        # define column vectors for governing equations
        self.delta_vec = sp.Matrix(sp.symbols('\\delta_0:{}'.format(self.nlayers))) # horizontal divergence
        self.Tvec = sp.Matrix(sp.symbols('T_0:{}'.format(self.nlayers)))  # temperature
        self.qvec = sp.Matrix(sp.symbols('q_0:{}'.format(self.nlayers)))  # moisture
        self.uvec = sp.Matrix(sp.symbols('u_0:{}'.format(self.nlayers)))  # zonal wind
        self.phivec = sp.Matrix(sp.symbols('phi_0:{}'.format(self.nlayers + 1)))  # geopotential

        # ncols = self.nlayers * 3  # 3 variables (delta, T, q) per layer

        # momentum equations: delta contribution
        Mcoeff_dict['mom']['delta'] = self.momentum_eqns_deltacoeffs()

        # momentum equations: temperature contribution
        M, a = self.construct_square_matrix(self.nlayers, 'a')
        Mcoeff_dict['mom']['T'] = self.momentum_eqns_Tcoeffs(M, a) 

        # momentum equations: moisture contribution (zero: no virtual effects)
        Mcoeff_dict['mom']['q'] = sp.zeros(self.nlayers, self.nlayers) 

        # create matrix for momentum equations: divergence contribution is a diagonal matrix,
        # DRY STATIC ENERGY and MOISTURE EQUATIONS

        # delta: DSE
        M, a = self.construct_square_matrix(self.nlayers, 'a')
        Mcoeff_dict['dse']['delta'] = self.thermo_eqn_deltacoeffs(M, a, 'dse')

        # temperature: convective and non-convective components
        MT, _ = self.construct_square_matrix(self.nlayers, 'epsilon_sT')
        Mq, _ = self.construct_square_matrix(self.nlayers, 'epsilon_sq')

        Mcoeff_dict['dse']['T'] = self.thermo_eqn_Tqcoeffs_symbolic(MT, var = 'dse')
        Mcoeff_dict['dse']['q'] = Mq

        # WATER VAPOR EQUATIONS
        # delta: moisture
        M, a = self.construct_square_matrix(self.nlayers, 'a')
        Mcoeff_dict['moisture']['delta'] = self.thermo_eqn_deltacoeffs(M, a, 'q')

        # moisture: convective and non-convective components
        MT, _ = self.construct_square_matrix(self.nlayers, 'epsilon_qT')
        Mq, _ = self.construct_square_matrix(self.nlayers, 'epsilon_qq')
        
        # forced vector
        Mcoeff_dict['moisture']['T'] = MT
        Mcoeff_dict['moisture']['q'] = self.thermo_eqn_Tqcoeffs_symbolic(Mq, var = 'q')


        if self.phys_params['q_on']:

            Mforced = sp.zeros(3 * self.nlayers, 1) 

            self.thermo_state_vec = sp.Matrix.vstack(self.Tvec, self.qvec)

            self.state_vec = sp.Matrix.vstack(self.delta_vec, self.Tvec, self.qvec)
            Mcoeff_collected['mom'] = sp.Matrix.hstack(Mcoeff_dict['mom']['delta'], 
                                                       Mcoeff_dict['mom']['T'], 
                                                       Mcoeff_dict['mom']['q'])

            # solve for delta 
            self.Mmom = Mcoeff_collected['mom'].subs(self.params_dict['general']).subs(self.symbols['D_m'], self.phys_params['Dm'])
            self.delta_sols = sp.solve(self.Mmom * self.state_vec, self.delta_vec)
            
            # reduce factors in the delta sols
            # leading_term = sp.Poly(self.delta_sols[self.delta_vec[0]].as_numer_denom()[1], self.ksymb).all_coeffs()[-1]
            # leading_term_inv = abs( 1/leading_term )
            # reduce_order = lambda x: (x.simplify().as_numer_denom()[0] * leading_term_inv ).expand()/ (x.simplify().as_numer_denom()[1] * leading_term_inv )
            self.delta_sols = {k : v.simplify() for k,v in self.delta_sols.items()}

            self.delta_denom = self.delta_sols[self.delta_vec[0]].as_numer_denom()[1]  # extract the denominator for simplification

            Mcoeff_collected['dse'] = sp.Matrix.hstack(Mcoeff_dict['dse']['delta'], 
                                                        -Mcoeff_dict['dse']['T'], 
                                                        -Mcoeff_dict['dse']['q'])
            
            Mcoeff_collected['moisture'] = sp.Matrix.hstack(Mcoeff_dict['moisture']['delta'], 
                                                            -Mcoeff_dict['moisture']['T'], 
                                                            -Mcoeff_dict['moisture']['q'])
            
            # thermo matrix
            Mthermo = sp.Matrix.vstack(Mcoeff_collected['dse'],
                                       Mcoeff_collected['moisture'])

            self.Mthermo = Mthermo
            
            self.Mcoeff_all = self.simplify_matrix_with_delta_sub(Mthermo, 
                                                                  self.delta_sols, 
                                                                  self.thermo_state_vec)

            self.Mfull = sp.Matrix.vstack(Mcoeff_collected['mom'], Mcoeff_collected['dse'], Mcoeff_collected['moisture'])

        else:
            
            Mforced = sp.zeros(2 * self.nlayers, 1)
            self.state_vec = sp.Matrix.vstack(self.delta_vec, self.Tvec)
            self.thermo_state_vec = self.Tvec

            Mcoeff_collected['mom'] = sp.Matrix.hstack(Mcoeff_dict['mom']['delta'], 
                                                       Mcoeff_dict['mom']['T'])

            self.Mmom = Mcoeff_collected['mom'].subs(self.params_dict['general']).subs(self.symbols['D_m'], self.phys_params['Dm'])

            # solve for delta 
            self.delta_sols = sp.solve(self.Mmom * self.state_vec, self.delta_vec)
            self.delta_denom = self.delta_sols[self.delta_vec[0]].as_numer_denom()[1]  # extract the denominator for simplification

            Mcoeff_collected['dse'] = sp.Matrix.hstack(Mcoeff_dict['dse']['delta'], 
                                                      -Mcoeff_dict['dse']['T'])
            
            Mthermo = Mcoeff_collected['dse']
            self.Mthermo = Mthermo

            self.Mcoeff_all = self.simplify_matrix_with_delta_sub(Mthermo, 
                                                                  self.delta_sols, 
                                                                  self.thermo_state_vec)             

            self.Mfull = sp.Matrix.vstack(Mcoeff_collected['mom'], Mcoeff_collected['dse'])

        self.Mcoeff_collected = Mcoeff_collected

        # construct forced vector
        self.Mforced = self.construct_forced_vector(Mforced)

        # zonal wind
        self.uwind_dict, self.phi_dict = self.uwind_phi_symbolic()

        # diabatic heating terms
        self.diab_vec = sp.Matrix( (self.symbols['Qc'], self.symbols['Qr'], self.symbols['Qsh']) )

    def simplify_matrix_with_delta_sub(self, Mthermo, delta_sols, thermo_state_vec):

        """
        Simplify the matrix by substituting the solutions for delta
        """

        Mred = (Mthermo * self.state_vec).subs(delta_sols)
        Mred = Mred.applyfunc(lambda x: (x.together() * self.delta_denom).cancel())
        # Mred = Mred.applyfunc(lambda x: (x * self.delta_denom).as_numer_denom()[0].expand())  # simplify thermo matrix with delta solutions
        # Mred = Mred.applyfunc(sp.simplify)
        # Mred = Mred.applyfunc(sp.cancel)

        eqns_list = []
        for i in range(Mred.shape[0]):
            eqns_list.append(Mred.row(i)[0])
        
        Meq, _ = sp.linear_eq_to_matrix(eqns_list, list(thermo_state_vec))
        Meq = Meq.applyfunc(lambda x: x.simplify())

        return Meq

    def create_coeff_matrices(self):

        """
        Create coefficient matrices for the governing equations.
        Two (slightly) different coefficients for convective and non-convective
        """

        self.define_symbols()

        print(f'Estimating parameters for matrix evaluation')
        self.params_dict = {'general': {}, 'conv' : {}, 'nconv': {}}

        # general parameters        
        self.thermo_eqn_Tqcoeffs_params_general(self.params_dict['general'])
        
        # radiation and convective 
        for k in ['conv', 'nconv']:
            self.thermo_eqn_Tqcoeffs_params_radconv(self.params_dict[k], k)


        self.construct_governing_eqn_matrix()
        self.get_col_ave_heating_expr()  # expressions for column-averaged heating
        # self.get_col_ave_heating(conv_opt = 0)  # expressions for column-averaged heating in non-convective case

    def get_col_ave_heating_expr(self, conv_opt: int = 1):

        """
        Get expressions for column-averaged convective heating,
        radiative heating, and surface fluxes
        """

        taub = self.mom_params['tau_i'][0]

        conv_factor = abs( np.diff(self.pinterface) ) * self.constants['CPD'] * 1e2 / self.constants['GRAVITY']  # K/sec to W/m2; K is still non-dimensional

        rad_matrix_sT = self.epsilon_rad_matrix[1:self.nlayers + 1, :].T 
        rad_matrix_sq = self.epsilon_rad_matrix[self.nlayers + 1:, :].T * (self.phys_params['qrad_on'])
        conv_matrix = self.epsilon_conv_matrix

        QradT = sp.Matrix(rad_matrix_sT) @ self.Tvec
        Qradq = sp.Matrix(rad_matrix_sq) @ self.qvec

        # sympy.matrices.dense.matrix_multiply_elementwise(A, B), 
        Qradq_col = sp.ones(1, self.nlayers) @ Qradq.multiply_elementwise(sp.Matrix(conv_factor))
        QradT_col = sp.ones(1, self.nlayers) @ QradT.multiply_elementwise(sp.Matrix(conv_factor))

        Qc = conv_matrix[0, :, :]  @ self.Tvec + conv_matrix[1, :, :] @ self.qvec 
        Qc = Qc * conv_opt # convective heating is 0 or 1 
        Qconv_heating = sp.ones(1, self.nlayers) @ Qc.multiply_elementwise(sp.Matrix(conv_factor))

        Qsh = - (self.kappa_s_star / taub) * self.Tvec[0] * conv_factor[0]  # surface flux contribution to boundary layer budget
        # return (Qconv_heating + QradT_col + Qradq_col)[0] + Qsh

        self.diabatic_terms[ self.symbols['Qc'] ] = Qconv_heating[0]
        self.diabatic_terms[ self.symbols['Qr'] ] = QradT_col[0] + Qradq_col[0]
        self.diabatic_terms[ self.symbols['Qsh'] ] = Qsh



class LinearSolutions(nLayerModel):

    """
    Class to generate linear solutions from the 
    nLayerModel class, using the model hierarchy information.
    """

    def __init__(self, constants, phys_params, nlayers, pinterfaces, rad_fil, conv_fil, 
                 params_fil = params_file, arm_file = arm_file):
        super().__init__(constants, phys_params, nlayers, pinterfaces, rad_fil, conv_fil, params_fil, arm_file)
        super().main()


        self.Lx =  self.mom_params['ci'][0] * self.mom_params['tau_i'][0]

        self.free_sols = {'conv': {}, 'nconv': {}}
        self.free_vecs = {'conv': {}, 'nconv': {}}
        self.eigen_vecs = {'conv': {}, 'nconv': {}}
        
        self.forced_vec = {}

        self.omega_profile = {'conv': None, 'nconv': None}

        self.base_var = None
        self.base_var_pert = None # perturbation for base variable
        self.lambda_symb = sp.symbols('lambda')

        self.decay_scales = {'conv': None, 'nconv': None}
        self.decay_scales_trunc = {'conv': None, 'nconv': None}
        self.forced_basis = {'conv' : None}
        self.forced_weights = {'conv' : None}

        # compute budgets
        self.dse_budget = {'conv' : {}, 'nconv' : {}}
        self.q_budget = {'conv' : {}, 'nconv' : {}}

        # vertical profiles
        self.omega_profile = {'conv' : {}, 'nconv' : {}}
        self.delta_profile = {'conv' : {}, 'nconv' : {}}
        self.temp_profile = {'conv' : {}, 'nconv' : {}}
        self.q_profile = {'conv' : {}, 'nconv' : {}}
        self.u_profile = {'conv' : {}, 'nconv' : {}}
        self.phi_profile = {'conv' : {}, 'nconv' : {}}

        # eigenvalues and eigenvectors
        self.evecs = {'conv' : {}, 'nconv' : {}}
        self.evals = {'conv' : {}, 'nconv' : {}}

        # matching problem
        self.u_x0 = None
        self.phi_x0 = None
        self.q_x0 = None
        self.T_x0 = None
        self.delta_domain = None

        self.nderivatives = 1 #if self.phys_params['Dm'] == 0 else 3

        self.phi_eqns = {i:[] for i in range(self.nlayers)}
        
        self.q_eqns = { i: {j:[] for j in range(self.nlayers)} for i in range(self.nderivatives + 1) }
        self.T_eqns = { i: {j:[] for j in range(self.nlayers)} for i in range(self.nderivatives + 1) }
        self.delta_eqns = { i: {j:[] for j in range(self.nlayers)} for i in range(self.nderivatives + 1) }

        self.q_eqns_origin = { i: {j:[] for j in range(self.nlayers)} for i in range(self.nderivatives + 1) }
        self.T_eqns_origin = { i: {j:[] for j in range(self.nlayers)} for i in range(self.nderivatives + 1) }
        self.delta_eqns_origin = { i: {j:[] for j in range(self.nlayers)} for i in range(self.nderivatives + 1) }

        # self.dqdx_eqns = {i:[] for i in range(self.nlayers)}
        # self.dqdx_origin_eqns = {i:[] for i in range(self.nlayers)}

        # self.dqdxx_eqns = {i:[] for i in range(self.nlayers)}

        self.u_eqns = {i:[] for i in range(self.nlayers - 1)}
        self.delta_eqns_x0 = {i:[] for i in range(self.nlayers - 1)}
        self.Fs_eqns = {0 : []}
        self.Qc_eqns = []

        # self.dTdx_eqns = {i:[] for i in range(self.nlayers)}
        # self.dTdx_origin_eqns = {i:[] for i in range(self.nlayers)}

        # self.dTdxx_eqns = {i:[] for i in range(self.nlayers)}
        # self.dTdxxx_eqns = {i:[] for i in range(self.nlayers)}
        
        self.matching_coeffs = {'conv': {}, 'nconv': {}}

        # full solution
        self.x0 = self.phys_params['x0']
        res = self.phys_params['dx']
        self.xend = self.phys_params['xend']
        
        xrange_conv = np.arange(0, self.x0 + res , res)  # high resolution for x < xc
        xrange_nconv = np.arange(self.x0 +  res, self.xend + res, res)  # low resolution in non-convective zone for x > xc
        self.xrange  = np.append(xrange_conv, xrange_nconv) 

        self.sol_full = {} 
        self.sol_forced = {} 
        self.sol_modal = {'conv': {}, 'nconv': {}, 'forced': {}}

        self.LM = {}  # LM decomposition matrix

    def compute_eigenvectors(self, ksol, key, base_var):

        """
        For a given lambda solution, compute the eigenvectors
        as a function of Tas perturbation 
        ksol: dimensionless wavenumber solution
        key: convective or non-convective
        base_var: base variable for the eigenvector, default is T_0
        """

        #
        # sols = {k:v for k,v in zip(obj.thermo_state_vec,evecs[:,-2])}


        Meval = self.Meval[key].evalf(subs = {self.ksymb: ksol})
        var = [i for i in self.state_vec if i != base_var]  # express solution in terms of T0 (boundary layer temp.)
        eqns = Meval * self.thermo_state_vec
        evec = sp.solve([i for i in eqns[1:]], var)  # solve n-1 equations to 
                                                     # get the eigenvectors
        return evec 
    
    def get_norm_vec(self, var):

        """
        Get the norm for a given symbolic 
        variable.
        """
        Msref = self.vert_struct_params['Msref']
        # Msref, _ = self.get_Msref_unmod()
        Msbref = Msref[0]
        taub = self.mom_params['tau_i'][0]

        if var in list(self.Tvec) + list(self.qvec):
            norm = Msbref
        elif var in self.delta_vec:
            norm = 1./taub
        
        return norm

    def redim_vect(self, evec, base_var, pert = 1.0):
            
            """
            Redimensionalize the eigenvector
            """    
            evec_redim = {}
            base_norm = self.get_norm_vec(base_var)

            for key in evec.keys():
                norm = self.get_norm_vec(key)
                evec_redim[key] = evec[key].subs({base_var : pert/base_norm}) * norm
    
            return evec_redim

    # helper functions to decompose the matrix into powers of k^2 and others
    def get_ksqr_coeffs(self, expr):
        k = self.ksymb
        poly = sp.Poly(expr, k)
        coeffs = poly.all_coeffs()
        if len(coeffs) == 1:  # if the polynomial is constant
            return 0
        else:
            assert coeffs[1] == 0.0, f'Polynomial has a k term'
            return coeffs[0]

    def get_non_ksqr_coeffs(self, expr):
        k = self.ksymb
        poly = sp.Poly(expr, k)
        coeffs = poly.all_coeffs()
        return coeffs[-1]

    def product_ksqr(self, expr):
        k = self.ksymb
        return sp.expand(k**2 * expr)

    def convert_thermo_vec_to_full(self, 
                                   vec:np.ndarray, 
                                   ksol_dict:dict)->dict:
        """
        Method to take thermo vector array and compute delta and u,phi if required
        """
        thermo_evec = {k:v for k,v in zip(self.thermo_state_vec, vec)}

        delta_evec = {}
        for symb in self.delta_sols.keys():
            delta_evec[symb] = self.delta_sols[symb].evalf(subs = ksol_dict | thermo_evec | self.params_Dm)

        return thermo_evec | delta_evec



    def get_free_solutions(self):

        """
        For both the convective and non convective
        cases, solve for the free solutions of the 
        coefficient matrix (scale and eigenvector)
        """

        err_thresh = -1e-10   # threshold to filter solutions with real part ~ 0
        Lx = self.Lx  # distance scale
        base_pert_norm = self.base_var_pert / self.get_norm_vec(self.base_var)
        base_indx = [n for n,i in enumerate(self.thermo_state_vec) if i == self.base_var][0]  # index of base variable

        print('Free Solutions')

        for key in ['conv', 'nconv']:

            conv_opt = 1 if key == 'conv' else 0
            eval_dict = self.params_dict['general'] | self.params_dict[key] | self.params_Dm
            Meval = self.Meval[key]

            if Meval:

                # decompose matrix into M (k^2) coefficients and a linear matrix (only coefficients)
                M = Meval.applyfunc(self.get_ksqr_coeffs)
                L = Meval.applyfunc(self.get_non_ksqr_coeffs)
                Mcheck = ( Meval - (M.applyfunc(self.product_ksqr) + L) ).applyfunc(sp.simplify)
                
                # check if decomposition is successful
                assert Mcheck.is_zero_matrix, f'ksqr matrix decomposition failed'
                self.LM[key] = dict(L = L, M = M)

                L, M  = np.array(L).astype(np.float64), np.array(M).astype(np.float64)
                Minv = np.linalg.inv(M)
                A = -Minv @ L
                evals, evecs = np.linalg.eig(A)  # obtain k^2 and eigenvectors

                # store eigenvectors and eigenvalues
                self.evecs[key] = evecs
                self.evals[key] = evals

                # scale eigenvectors
                evecs = evecs * base_pert_norm/evecs[base_indx, :]
                
                for n, sol in enumerate(evals):

                    ksol_dict = {self.ksymb**2 : sol}

                    print(n, sol, key)
                    self.free_sols[key][n] = sol  #
                    # thermo_evec = {k:v for k,v in zip(self.thermo_state_vec, evecs[:,n])}

                    # delta_evec = {}
                    # for symb in self.delta_sols.keys():
                    #     delta_evec[symb] = self.delta_sols[symb].evalf(subs = ksol_dict | thermo_evec | self.params_Dm)

                    # evaluate u and phi
                    # uwind_evec = {}
                    # phi_evec = {}
                    # eval_dict = {self.ksymb : sol} | thermo_evec | eval_dict
                    # for usymb, phisymb in zip( self.uwind_dict.keys(), self.phi_dict.keys() ):
                    #     uwind_evec[usymb] = self.uwind_dict[usymb].evalf(subs = eval_dict)
                    #     phi_evec[phisymb] = self.phi_dict[phisymb].evalf(subs = eval_dict)

                    self.free_vecs[key][n] = self.convert_thermo_vec_to_full(evecs[:,n], ksol_dict)
                    # self.free_vecs[key][n] = thermo_evec | delta_evec #| uwind_evec | phi_evec #{k : v.subs({self.base_var : base_pert_norm}) for k, v in evec.items()} | {self.base_var : base_pert_norm}                    
                    ret = self.vert_profile(sol, self.free_vecs[key][n], conv_opt, compute_budget = False)
                    self.omega_profile[key][n], self.delta_profile[key][n] = ret[0], ret[1]
                    self.temp_profile[key][n], self.q_profile[key][n] = ret[2], ret[3]

            waveno = { k : np.sqrt(v) for k,v in self.free_sols[key].items() }
            ds =  {k: self.Lx * 1e-3 /np.real(waveno[k]) for k in waveno.keys()}  # decay scale in km (real part of k)
            self.decay_scales[key] = dict(sorted(ds.items(), key = lambda item: abs(item[1]), reverse=True))  # sort dict by length scale
            print('--'*20)


    @staticmethod
    def append_factor(vec, sol_dict, factor, li):
            """
            Append the factor to the solution vector
            """
            for symb in vec:
                li.append(sol_dict[symb] * factor)


    def vert_profile(self, ksol, sol_dict:dict, conv_opt, compute_budget = True):

        """
        sol_dict: is a dictionary that contains the solution vector (non-dimensionalized)
        For a solution, obtain the vertical profile of T, q, div and omega
        """

        div = []
        temp = []
        q = []
        phi = []
        u = []
        cb = self.mom_params['ci'][0]
        Msref = self.vert_struct_params['Msref']
        Msb = Msref[0]

        # redimensionalize the solution and store in div, temp and q lists
        self.append_factor(self.delta_vec, sol_dict, 
                           1.0 / self.mom_params['tau_i'][0], 
                           div)
        self.append_factor(self.Tvec, sol_dict, Msb, temp)
        
        # for symb in self.delta_vec:
        #     sol = sol_dict[symb]
        #     factor = 1.0 / self.mom_params['tau_i'][0]
        #     div.append(sol * factor)

        # for symb in self.Tvec:
        #     sol = sol_dict[symb]
        #     factor = self.vert_struct_params['Msref'][0]
        #     temp.append(sol * factor)

        # for symb in self.Tvec:
        #     sol = sol_dict[symb]
        #     factor = self.vert_struct_params['Msref'][0]
        #     u.append(sol * factor)


        # Vertical profiles
        omega, delta = self.compute_omega_profile(div)
        temp_profile = self.compute_Tq_profiles(temp, var = 'T')
        # u_profile = self.compute_uwind_profiles(u)
        # phi_profile = self.compute_phi_profiles(phi, temp)

        if self.phys_params['q_on']:

            self.append_factor(self.qvec, sol_dict, Msb, q)

            # for symb in self.qvec:
            #     sol = sol_dict[symb]
            #     factor = self.vert_struct_params['Msref'][0]
            #     q.append(sol * factor)

            q_profile = self.compute_Tq_profiles(q, var = 'q')
        else:
            q_profile = np.zeros_like(temp_profile)

        return omega, delta, temp_profile, q_profile#, u_profile, phi_profile, dse_budget, q_budget

    # def vert_profile(self, ksol, sol_dict:dict, conv_opt, compute_budget = True):

    #     """
    #     sol_dict: is a dictionary that contains the solution vector (non-dimensionalized)
    #     For a solution, obtain the vertical profile of T, q, div and omega
    #     """

    #     div = []
    #     temp = []
    #     q = []
    #     phi = []
    #     u = []
    #     cb = self.mom_params['ci'][0]
    #     Msref = self.vert_struct_params['Msref']
    #     Msb = Msref[0]

    #     uvec = {k for k in self.uwind_dict.keys()}
    #     phivec = {k for k in self.phi_dict.keys()}

    #     # redimensionalize the solution and store in div, temp and q lists
    #     self.append_factor(self.delta_vec, sol_dict, 1.0 / self.mom_params['tau_i'][0], div)
    #     self.append_factor(self.Tvec, sol_dict, Msb, temp)
    #     self.append_factor(uvec, sol_dict, cb, u)
    #     self.append_factor(phivec, sol_dict, pow(cb, 2) / self.constants['CPD'], phi)  # convert phi to K
        
    #     # for symb in self.delta_vec:
    #     #     sol = sol_dict[symb]
    #     #     factor = 1.0 / self.mom_params['tau_i'][0]
    #     #     div.append(sol * factor)

    #     # for symb in self.Tvec:
    #     #     sol = sol_dict[symb]
    #     #     factor = self.vert_struct_params['Msref'][0]
    #     #     temp.append(sol * factor)

    #     # for symb in self.Tvec:
    #     #     sol = sol_dict[symb]
    #     #     factor = self.vert_struct_params['Msref'][0]
    #     #     u.append(sol * factor)


    #     # Vertical profiles
    #     omega, delta = self.compute_omega_profile(div)
    #     temp_profile = self.compute_Tq_profiles(temp, var = 'T')
    #     u_profile = self.compute_uwind_profiles(u)
    #     phi_profile = self.compute_phi_profiles(phi, temp)

    #     if self.phys_params['q_on']:

    #         self.append_factor(self.qvec, sol_dict, Msb, q)

    #         # for symb in self.qvec:
    #         #     sol = sol_dict[symb]
    #         #     factor = self.vert_struct_params['Msref'][0]
    #         #     q.append(sol * factor)

    #         q_profile = self.compute_Tq_profiles(q, var = 'q')
    #     else:
    #         q_profile = np.zeros_like(temp_profile)

    #     # Budgets
    #     if compute_budget:
    #         dse_budget, q_budget = self.compute_budgets(ksol, div, temp, q, conv_opt)
    #     else:
    #         dse_budget, q_budget = {}, {}

    #     return omega, delta, temp_profile, q_profile, u_profile, phi_profile, dse_budget, q_budget

    # Reconstruct vertical profiles
    def compute_omega_profile(self, div):

        """
        Get profiles of the vertical velocity
        from a list of horizontal divergences in each layer
        """
        omega = np.zeros_like(self.profiles['pres'], dtype = complex)
        delta = np.zeros_like(self.profiles['pres'], dtype = complex)

        assert len(div) == self.nlayers, 'Number of divergences must match the number of layers'
        
        for i in range(self.nlayers):  # each layer
            
            ind = self.ind_interface[i]
            V = self.profiles['Vprofiles'][i]

            delta[ind] = div[i] * V

            if i ==0 :
                omega[ind] = div[i] * integrate.cumulative_trapezoid(V, dx = self.res, initial = 0)
            
            elif i > 0 and i < self.nlayers - 1:
                ind_m1 = self.ind_interface[i - 1][-1]
                omega_base = omega[ind_m1] # omega at top of layer below
                
                omega_within_layer = div[i] * integrate.cumulative_trapezoid(V, dx = self.res, initial = 0)
                omega[ind] = omega_within_layer + omega_base 

            
            elif i == self.nlayers - 1:
                omega[ind] = -div[i] * integrate.cumulative_trapezoid(V[::-1], dx = self.res, initial = 0)[::-1]
            
        return omega * 1e2, delta  # convert omega to Pa/s

    def compute_Tq_profiles(self, Tq_hor, var = 'T'):

        """
        Given temperature perturbations, compute the
        vertical profile of temperature
        """

        Tq = np.zeros_like(self.profiles['pres'], dtype = complex)

        basis_prof = self.vert_struct_params['a_profiles']  if var == 'T' else self.vert_struct_params['b_profiles']        

        assert len(Tq_hor) == self.nlayers, 'Number of variables must match the number of layers'
        
        for i in range(self.nlayers):  # each layer
            ind = self.ind_interface[i]
            Tq[ind] = Tq_hor[i] * basis_prof[i]

        return Tq

    def compute_uwind_profiles(self, u_hor):

        """
        Given temperature perturbations, compute the
        vertical profile of zonal wind
        """

        u = np.zeros_like(self.profiles['pres'], dtype = complex)

        basis_prof = self.profiles['Vprofiles']

        assert len(u_hor) == self.nlayers, 'Number of variables must match the number of layers'
        
        for i in range(self.nlayers):  # each layer
            ind = self.ind_interface[i]
            u[ind] = u_hor[i] * basis_prof[i]

        return u

    def compute_phi_profiles(self, phi_hor, temp_hor):
        
        """
        Given compute the interfactial values of geopotential
        and temperature, compute the vertical profile of 
        geopotential.
        """
            
        phi = np.zeros_like(self.profiles['pres'], dtype = complex)
        aplus_profiles = self.vert_struct_params['aplus_profiles'] 
        assert len(phi_hor) == self.nlayers, 'Number of variables must match the number of layers'
        
        for i in range(self.nlayers):
            ind = self.ind_interface[i]
            if i == 0:
                phis = phi_hor[i] - temp_hor[i] * aplus_profiles[i][-1]
                phi[ind] = phis + temp_hor[i] * aplus_profiles[i]
            else:
                phi[ind] = phi[ind - 1][-1] + temp_hor[i] * aplus_profiles[i]
        
        return phi

    def compute_budgets(self, ksol, div, temp, q, conv_opt):
        
        """
        ksol : non-dimensionalized k
        div, temp, q: lists of the divergence, temperature and moisture 
        """

        dse_budget = {}
        q_budget = {}

        for i in range(self.nlayers):  # each layer

            # compute surface fluxes
            if i == 0:
                dse_budget['sh_flux'] = -self.kappa_s_star * temp[i] * 86400 / self.mom_params['tau_i'][0]    
                if self.phys_params['q_on']:
                    q_budget['lh_flux'] = -self.kappa_s_star * q[i] * 86400 / self.mom_params['tau_i'][0]
                else:
                    q_budget['lh_flux'] = 0

            dse_budget['Qc'], q_budget['Qd'], dse_budget['QradT'], dse_budget['Qradq'] = self.compute_Q1_Q2(np.asarray(temp), np.asarray(q), conv_opt)
            dse_budget['adiab_cooling'], q_budget['adiab_moistening'] = self.compute_adiab_cooling_moistening(div)
            dse_budget['Tdiffusion'], q_budget['qdiffusion'] = self.compute_diffusion(np.asarray(temp), np.asarray(q), ksol)
            dse_budget['Tmixing'], q_budget['qmixing'] = self.compute_mixing(np.asarray(temp), np.asarray(q))
            dse_budget['Tadv'], q_budget['qadv'] = self.compute_advection(np.asarray(temp), np.asarray(q), ksol)

        return dse_budget, q_budget
        
    def get_forced_solutions(self):

        """
        Use the input wavenumber to obtain the forced solution
        """

        kn = self.phys_params['kn'] # input dimensionless wavenumber
        self.kf = 1J * ( kn / self.phys_params['x0'] ) * (np.pi/2) * self.Lx  # wavenumber with units 1/m
                                                                              # note imaginary for cosine forcing
        self.assign_Dm_param()
        ksol_dict = {self.ksymb**2 : self.kf**2}

        # Mfull = self.Mfull.subs(self.params_dict['conv'] | self.params_dict['general'] | self.params_Dm).subs(self.ksymb, self.kf)
        f = self.Mforced.subs(self.params_dict['general']) # multiply the rhs by delta denominator, which is factored out of the lhs
        f = sp.Matrix( self.Mforced[self.nlayers:]).subs(self.params_dict['general']) 

        for conv_opt in ['conv', 'nconv']:


            Mforced = self.Meval[conv_opt].subs(ksol_dict)
            Mforced = np.asarray(Mforced, complex)
            f = np.asarray(f, complex).squeeze()

            self.forced_vec[conv_opt] = np.linalg.solve(Mforced, f)



    def get_surf_contribution_to_col_dse(self, conv_opt):

        """
        Surface contribution to column dse budget (radiative heating and sensible heat flux)
        """

        conv_factor = abs( np.diff(self.pinterface) ) * self.constants['CPD'] * 1e2 / self.constants['GRAVITY']  # K/sec to W/m2
        Qsrf = self.dse_budget[conv_opt]['forced']['surf'] * (1./86400) * conv_factor # forced component surface heating from K/day to W/m^2
        Qsrf = sp.ones(1, self.nlayers) @ sp.Matrix(Qsrf)
        return Qsrf[0]


    def append_solution(self, symb, sol_dict, l1):

        """
        Check if symbol exists in the dictionary
        """

        if symb == self.base_var:
            l1.append(self.base_var_pert)
        else:
            # l1.append(sp.re(sol_dict[symb]))
            l1.append( sol_dict[symb] )

        return l1

    def check_if_even(self, x):

        """
        Check if polynomial is even
        """

        coeffs = sp.Poly(x, self.ksymb).all_coeffs()
        ind_zero = [i for i, x in enumerate(coeffs) if x == 0]

        if len(coeffs) == 1:  #if constant, return true
            return 1
        elif len(coeffs) > 1:  # if more than one coefficient, check if odd powers are zero
            if [i%2 !=0 for i in ind_zero]:
                return 1
            else:
                return 0
    
    # BUDGETS
    def compute_Q1_Q2(self, T, q, conv_opt):
        """
        For a given temperature, moisture
        compute Q1 = Qc + Qr
        """
        taub = self.mom_params['tau_i'][0]

        rad_matrix_sT = self.epsilon_rad_matrix[1:self.nlayers + 1, :].T 
        rad_matrix_sq = self.epsilon_rad_matrix[self.nlayers + 1:, :].T * (self.phys_params['qrad_on'])
        conv_matrix = self.epsilon_conv_matrix

        QradT = rad_matrix_sT @ T
        Qconv_heating = (conv_matrix[0, :, :] * conv_opt) @ T

        if self.phys_params['q_on']:

            Qconv_heating += (conv_matrix[1, :, :] * conv_opt) @ q
            Qconv_drying  = (conv_matrix[2, :, :] * conv_opt) @ T + \
                            (conv_matrix[3, :, :] * conv_opt) @ q
            
            Qradq = rad_matrix_sq @ q

        else:
            Qconv_drying = np.zeros_like(Qconv_heating)
            Qradq = np.zeros_like(QradT)

        Qconv_heating = Qconv_heating * 86400  # convert to K/day
        Qconv_drying  = Qconv_drying * 86400  # convert to K/day
        QradT = QradT * 86400
        Qradq = Qradq * 86400

        return Qconv_heating, Qconv_drying, QradT, Qradq

    # adiabatic cooling/moistening
    def compute_adiab_cooling_moistening(self, div):
        
        """
        For a given divergence, compute the 
        effects of adiabatic cooling and moistening
        """

        Msbref = self.vert_struct_params['Msref'][0]

        Msi_dagger_ref = self.dse_params['Msi_dagger_ref']
        Msi_ddagger_ref = self.dse_params['Msi_ddagger_ref']

        Mqi_dagger_ref = self.q_params['Mqi_dagger_ref']
        Mqi_ddagger_ref = self.q_params['Mqi_ddagger_ref']

        ac = np.zeros_like(div)
        am = np.zeros_like(div)

        for i in range(self.nlayers):

            ac[i] = Msi_dagger_ref[i] * div[i] * Msbref * 86400 # K/day
            am[i] = Mqi_dagger_ref[i] * div[i] * Msbref * 86400 if self.phys_params['q_on'] else 0.0

            ac2 = 0
            am2 = 0.0
            for j in range(i):
                ac2 +=  div[j] * 86400 * self.deltap[j]/self.deltap[i] #
                if self.phys_params['q_on']:
                    am2 += div[j] * 86400 * self.deltap[j]/self.deltap[i]
                
            ac[i] =  (ac[i] - Msbref * Msi_ddagger_ref[i] * ac2) 
            am[i] =  (am[i] - Msbref * Mqi_ddagger_ref[i] * am2) if self.phys_params['q_on'] else 0.0

        return ac, am

    # diffusion
    def compute_diffusion(self, T, q, k):
            
            """
            Compute the diffusion term
            """
    
            factor = 1 / pow(self.mom_params['ci'][0] * self.mom_params['tau_i'][0] , 2)

            DT = self.phys_params['DT'] * factor
            Dq = self.phys_params['Dq'] * factor if self.phys_params['q_on'] else 0.0

            Ai = np.asarray([v for v in self.vert_struct_params['a_layer_ave'].values()])
            Bi = np.asarray([v for v in self.vert_struct_params['b_layer_ave'].values()])

            T_diffusion = pow(k, 2) * DT * Ai * np.asarray(T) * 86400 
            q_diffusion = pow(k, 2) * Dq * Bi * np.asarray(q) * 86400 if self.phys_params['q_on'] else np.zeros_like(T_diffusion)

            return T_diffusion, q_diffusion

    # advection
    def compute_advection(self, T, q, k):
                
        """
        Compute the advection term
        """

        factor = 1 / (self.mom_params['ci'][0] * self.mom_params['tau_i'][0])
        Tadv_flag = int(self.phys_params['TAdv_on'])
        qadv_flag = int(self.phys_params['qAdv_on'])

        AdvTi = np.asarray([v for v in self.vert_struct_params['AdvT_ref'].values()])
        Advqi = np.asarray([v for v in self.vert_struct_params['Advq_ref'].values()])

        T_advection = -k * AdvTi * np.asarray(T) * Tadv_flag * factor * 86400
        q_advection = -k * Advqi * np.asarray(q) * qadv_flag * factor * 86400 if self.phys_params['q_on'] else np.zeros_like(T_advection)

        return T_advection, q_advection

    # surface contribution
    def compute_Tsurf_contrib(self):
            
            """
            Compute the surface contribution
            """
            n = self.nlayers
            taub = self.mom_params['tau_i'][0]
            Msb = self.vert_struct_params['Msref'][0]
            Mforced = np.array(self.Mforced.subs(self.params_dict['general']), dtype = float)

            dse_contrib = Mforced[n : 2*n, 0]
            q_contrib = Mforced[2*n:, 0] if self.phys_params['q_on'] else np.zeros_like(dse_contrib)

            dse_contrib = dse_contrib * (Msb/taub) * 86400 # convert to K/day
            q_contrib = q_contrib * (Msb/taub) * 86400 # convert to K/day

            return dse_contrib, q_contrib

    # mixing
    def compute_mixing(self, T, q):
            
        """
        Compute the mixing term
        """

        eps_mix = self.phys_params['epsilon_mix']
        Tmixing = np.zeros_like(T)
        qmixing = np.zeros_like(T)
         
        for i in range(self.nlayers):

            deltap_fac = self.deltap[0]/self.deltap[i]
            
            am1_top = self.vert_struct_params['a_profiles'][i-1][-1] if i > 0 else 0.0 
            a_bot = 1 if i > 0 else 0.0 
            a_top = self.vert_struct_params['a_profiles'][i][-1] if i < self.nlayers - 1 else 0.0
            
            T_p1 = T[i+1] if i < self.nlayers - 1 else 0.0
            T_m1 = T[i-1] if i > 0 else 0.0

            eps_bot = eps_mix[i - 1] if i > 0 else 0.0
            eps_top = eps_mix[i] if i < self.nlayers - 1 else 0.0

            Tmixing[i] = eps_top * (T_p1 - a_top * T[i]) + eps_bot * (am1_top * T_m1 - a_bot * T[i])
            Tmixing[i] = Tmixing[i] * deltap_fac * 86400

            if self.phys_params['q_on']:

                bm1_top = self.vert_struct_params['b_profiles'][i-1][-1] if i > 0 else 0.0 
                b_bot = 1 if i > 0 else 0.0 
                b_top = self.vert_struct_params['b_profiles'][i][-1] if i < self.nlayers - 1 else 0.0

                q_p1 = q[i+1] if i < self.nlayers - 1 else 0.0
                q_m1 = q[i-1] if i > 0 else 0.0
                qmixing[i] = eps_top * (q_p1 - b_top * q[i]) + eps_bot * (bm1_top * q_m1 - b_bot * q[i])
                qmixing[i] = qmixing[i] * deltap_fac * 86400

        return Tmixing, qmixing

    # project forced solution
    def project_forced(self, profile_dict : dict):

        """
        Project a given forced solution 
        into the vector space spanned by the free vectors
        """

        free_vecs = []
        conv_opt = 'conv'        
        free_vecs = [profile_dict[conv_opt][k] for k in self.forced_basis[conv_opt].keys()]

        # normalize free vec to have L2 norm = 1
        for i in range(len(free_vecs)):
            vec = free_vecs[i]
            norm = np.sqrt(vec @ vec.T)
            free_vecs[i] = free_vecs[i] / norm

        free_vecs = np.asarray(free_vecs)
        proj_mat = free_vecs @ free_vecs.T # projection matrix

        forced_normed_vec = profile_dict['forced']/np.sqrt(profile_dict['forced'] @ profile_dict['forced'].T)
        frhs = forced_normed_vec @ free_vecs.T  # rhs of projection equation
        alpha = np.linalg.solve(proj_mat, frhs) # get coefficients of projection
        # alpha = alpha / alpha.sum()  # normalize such that sum = 1

        self.forced_weights[conv_opt] = {k : v for k,v in zip(self.forced_basis[conv_opt].keys(), alpha)}

    # MATCHING CONDITIONS
    # check for complex conjugate solutions:
    @staticmethod
    def find_complex_conjugates(d : dict):
        """
        d is a dict with solutions
        """
        sol_dict = {}  # save solutions in this dict
        list_cc = {}  # list of complex conjugates

        for k in d.keys():

            l = { k: k1 for k1, v1 in sol_dict.items() 
                if np.isclose(np.real(v1), np.real(d[k])) and 
                np.isclose(np.imag(v1), -np.imag(d[k])) }
            if l:
                list_cc = list_cc | l

            sol_dict[k] = d[k]

        return list_cc

    # check for roots with same absolute real part (the horizontal scale):
    @staticmethod
    def find_same_scales(d : dict):
        """
        d is a dict with solutions
        """
        sol_dict = {}  # save solutions in this dict
        list_sol = {}  # list of solutions with same real part

        for k in d.keys():

            l = {k : k1 for k1, v1 in sol_dict.items() 
                if np.isclose( abs(np.real(v1)), abs(np.real(d[k])) ) and
                np.isclose(np.imag(v1), np.imag(d[k]))}
            
            if l:
                list_sol = list_sol | l

            sol_dict[k] = d[k]

        return list_sol

    @staticmethod
    def find_degenerate_roots(d : dict, evecs: dict):
        """
        d is a dict with solutions
        """
        sol_dict = {}  # save solutions in this dict
        list_sol = {}  # list of solutions with same real part

        for k in d.keys():

            l1 = [complex(v) for v in evecs[k].values()]

            for k1 in sol_dict.keys():
                l2 = [complex(v) for v in evecs[k1].values()]
                if all( np.isclose(l1, l2) ):
                    list_sol[k] = k1

            sol_dict[k] = d[k]

        return list_sol

    def get_matching_conditions(self):

        """
        FA: three sets of matching conditions:
        i) the solution vector is continuous at x = xc
        ii the first derivative of the solution vector is continuous at x=xc
        iii) the solution vector is symmetric about x = 0, so f(x) = f(-x) for x = x_c
        """

        nderivatives = self.nderivatives + 1 # number of derivatives to compute including 0th derivative

        x0_nd = self.symbols['xc'] 
        A, B = self.symbols['A'], self.symbols['B']
        Qc_expr = self.diabatic_terms[self.diab_vec[0]]

        thermo_matching = sp.zeros(2 * self.nlayers, 
                                   nderivatives + 1)  # one symmetry condition about the origin

        Qc_x0 = 0.0

        forced_vec_conv = self.forced_vec['conv']
        forced_vec_nconv = self.forced_vec['nconv']

        thermo_matching[:, -1] += sp.Matrix( forced_vec_conv * sp.exp(-self.kf * x0_nd) ).expand()

        for key in ['conv', 'nconv', 'forced']:

            for d in range(nderivatives):

                if key in ['forced']:    
                    logger.info(f"{d}")
                    x0_val = pow(self.kf, d) * ( forced_vec_conv - forced_vec_nconv ) * sp.exp(self.kf * x0_nd) 

                    thermo_matching[:, d] += sp.Matrix(x0_val).expand()

                    if d == 0:
                        ksol_dict = { self.ksymb**2 : self.kf**2 }
                        fv = self.convert_thermo_vec_to_full(forced_vec_conv, ksol_dict)
                        Qc_x0 += Qc_expr.subs(fv) * sp.exp(self.kf * x0_nd)

                elif key in ['conv', 'nconv']:
                    for i, ksqr in enumerate(self.evals[key]):

                        ksol = np.sqrt(ksqr)
                        thermo_vec = self.evecs[key][:, i]
                        thermo_vec = thermo_vec / thermo_vec[0]

                        if key == 'conv':                    
                            prefactor = sp.exp( ksol * x0_nd )  # convective solutions ~ exp(kx)
                            prefactor_neg = sp.exp( -ksol * x0_nd )  # convective solutions ~ exp(-kx)
                            coeff_pos = A[i + 1]
                            coeff_neg = A[-(i + 1)]
                            x0_val = sp.Matrix( thermo_vec * (  pow(ksol, d)  * coeff_pos * prefactor 
                                                              + pow(-ksol, d) * coeff_neg * prefactor_neg) ).expand()

                            if d == 0: 
                                thermo_matching[:, -1] += sp.Matrix( thermo_vec * (coeff_pos * prefactor_neg + coeff_neg * prefactor) ).expand() 
                                ksol_dict = { self.ksymb**2 : self.evals[key][i] }
                                fv = self.convert_thermo_vec_to_full(thermo_vec, ksol_dict)
                                Qc_x0 += ( Qc_expr.subs(fv) * ( coeff_pos * prefactor_neg + coeff_neg * prefactor ) ).expand()

                        elif key == 'nconv':  # check for complex conjugates
                            # non-convective solutions ~ exp(k(x - x0))
                            x0_val = -sp.Matrix( pow(ksol, d) * thermo_vec * B[i+1] ).expand() # negative sign to indicate terms on the rhs

                        thermo_matching[:, d] += ( x0_val ).expand()

        self.thermo_x0 = thermo_matching
        self.Qc_x0 = Qc_x0

    def solve_matching_problem(self):

        """
        Solve matching problem
        """
        
        nlayers = self.nlayers
        Msb = self.vert_struct_params['Msref'][0]

        # solve linear equation with x = x0
        # x0_nd = self.phys_params['x0'] / self.Lx # dimensionless x0
        # Mlinear = self.thermo_x0.applyfunc(lambda x: x.subs({self.symbols['xc']: x0_nd}).evalf())
        Meqns = self.thermo_x0.reshape(6 * nlayers, 1)
        
        # extract symbols
        li = list(Meqns.T)
        symb = list( li[0].free_symbols )
        symb.remove(sp.symbols('A'))
        symb.remove(sp.symbols('B'))
        eqns = [ li[i] for i in range( len(li) ) ]  # convert matrix to list

        assert len(symb) == 6 * nlayers + 1, f'Number of symbols {len(symb)} does not match number of layers {6 * nlayers + 1}'

        # # get the linear equations
        # A, b = sp.linear_eq_to_matrix(li, symb)
        # A = np.array(A, dtype=complex)
        # b = np.array(b, dtype=complex)

        # np.linalg.solve(A, b).squeeze()
        self.eqns = eqns
        self.symb = symb
        # symb = self._natural_sort(symb)
        
        sols, v  = self.get_Ab_x0(eqns, symb)  # For xc = x0, get sols and b
        # if self.phys_params['solve_xc']:
        #     aux_eqn = self.Qc_eqns
        #     # aux_eqn = (self.delta_eqns[0][0] / self.delta_eqns[0][0].args[0].args[0]).simplify()
        #     initial_guess = sols.copy()
        #     initial_guess[-1] = initial_guess[-1] * 0.75  # change initial guess 
        #     sols_temp, v_temp = self.non_linear_solve_xc(initial_guess, eqns, v, aux_eqn)  # solution for xc != x0 using sols as initial guess
        #     x0_new = np.real( sols_temp[-1] * self.Lx )

        #     if x0_new < self.phys_params['x0']:
        #         print(f'New x0: {x0_new * 1e-3 : .2f} km')
        #         sols = sols_temp
        #         v = v_temp
        #     else:
        #         print(f'New x0: {x0_new * 1e-3:.2f} km is larger than the initial x0 {self.phys_params["x0"] * 1e-3 : .2f} km')
        #         print('Using x0 = x0')

        sol_dict = {k : v for k,v in zip(v, sols)}
        # sort dict by key
        # sol_dict = {k : sol_dict[k] for k in self.natural_sort(sol_dict.keys())}
        self.sol_dict = sol_dict 
        for k, sol in sol_dict.items():

            if k == self.symbols['xc']:
                print(f'{k} : {sol * self.Lx * 1e-3 : .2f} km')
                self.matching_coeffs[k] = sol

            else:
                if k.base == self.symbols['A']: 
                    conv_opt = 'conv' 
                elif k.base == self.symbols['B']:
                    conv_opt = 'nconv'

                kidx = k.indices[0]
                print(f'{k} : {sol * Msb : .2e} K')
                self.matching_coeffs[conv_opt][kidx] = sol



    def truncate_modes(self, eqns, vars):
        
        """
        Add one equation alternatively from convective and non-convective side
        until the number of equations equals the number of variables. The modes
        are added in the decreasing order of decay scales.
        """

        # truncate modes

        print('Truncating modes')

        vars_trunc = set()
        eqns_trunc = []

        ds_conv = self.decay_scales['conv']
        ds_nconv = self.decay_scales['nconv']

        self.decay_scales_trunc['conv'] = {}
        self.decay_scales_trunc['nconv'] = {}

        for k1, k2 in zip_longest(ds_conv.keys(), ds_nconv.keys()):

            if k1 is not None:
                
                if k1 in self.dict_conv.keys():
                    vars_trunc.add(self.symbols['A'][self.dict_conv[k1]])
                else:
                    vars_trunc.add(self.symbols['A'][k1])    

                self.decay_scales_trunc['conv'][k1] = ds_conv[k1]                    
                if len(vars_trunc) == len(eqns):
                    break

            if k2 is not None:

                if k2 in self.dict_nconv.keys():
                    vars_trunc.add(self.symbols['B'][self.dict_nconv[k2]])
                else:
                    vars_trunc.add(self.symbols['B'][k2])    

                self.decay_scales_trunc['nconv'][k2] = ds_nconv[k2]
                if len(vars_trunc) == len(eqns):
                    break

        vars_dict = {v : 0 for v in vars if v not in vars_trunc}
        for eq in eqns:
            eqns_trunc.append( eq.subs(vars_dict) )

        vars = vars_trunc
        eqns = eqns_trunc

        return eqns, vars


    def get_Ab_x0(self, eqns, vars):
        
        xc = self.symbols['xc']
        x0_nd = self.phys_params['x0']/self.Lx
        eqns = [eqn.subs(xc, x0_nd).evalf() for eqn in eqns]
    
        v = list(vars).copy()
        v.remove(xc)
        v = self._natural_sort(v)

        A, b = sp.linear_eq_to_matrix(eqns, v)
        
        A = np.array(A).astype(complex) 
        b = np.array(b).astype(complex)
        b = np.real(b)

        self.Asol = A
        self.bforce = b
        self.list_vars = vars

        if np.isclose(np.linalg.det(A), 0):
            raise ValueError('Matrix is singular')

        COND = np.linalg.cond(A)
        print(f'Matrix Condition number: {COND}')
        print(f'Determinant: {np.linalg.det(A)}')

        sols = list( np.linalg.solve(A, b).squeeze() )
        sols.append(x0_nd)
        v.append( self.symbols['xc'] )

        return sols, v

    def non_linear_solve_xc(self, sols_x0, eqns, vars, aux_eqn):

        v = vars.copy()
        eqs = eqns.copy()

        eqs.append( aux_eqn )

        nonlinear_sol = sp.nsolve(eqs, v, sols_x0, tol = 1e-12)
        return list( np.array(nonlinear_sol).astype(complex).squeeze() ), v

    def _convert_forced_arr_to_vec(self, arr):
        ksol_dict = {self.ksymb**2 :self.kf**2}
        fv = self.convert_thermo_vec_to_full(arr, ksol_dict)
        return fv

    def get_full_solution(self):

        """
        Get full solution in both the convective and non-convective zones
        """

        print(f'Getting full solution')

        x0_nd = self.matching_coeffs[ self.symbols['xc'] ] # dimensionless x0
        x0 = x0_nd * self.Lx

        xconv = np.where(abs(self.xrange) <= x0)[0]
        xnconv_forced = np.where(np.logical_and( abs(self.xrange) > x0, 
                                                abs(self.xrange) <= self.phys_params['x0']) )[0]  # a forced non-convective region
        xnconv = np.where(abs(self.xrange) > x0)[0]

        self.sol_full = np.zeros( (2 * self.nlayers, self.xrange.size), dtype = complex)

        # uphi_list = list(self.uwind_dict.keys()) + list(self.phi_dict.keys())
        # uphi_dict = self.uwind_dict | self.phi_dict
        state_vec_list = list(self.state_vec)
        diab_heating_list = [ self.symbols['Qc'] ]

        for key in ['forced', 'conv', 'nconv']:

            if key in ['forced']:

                # forced_sol_conv = self._convert_forced_arr_to_vec( self.forced_vec['conv'] )
                # forced_sol_nconv = self._convert_forced_arr_to_vec( self.forced_vec['nconv'] )

                self.sol_modal[key] = {0: {}}  # forced solution
                argx = self.kf * self.xrange[xconv] / self.Lx
                argx_nconv = self.kf * self.xrange[xnconv_forced] / self.Lx

                self.sol_full[:, xconv] = self.forced_vec['conv'][:, None] * np.exp(argx) 
                var = (self.forced_vec['nconv'][:, None] * np.exp(argx_nconv) )
                self.sol_full[:, xnconv_forced] = self.forced_vec['nconv'][:, None] * np.exp(argx_nconv)
                # # for symb in uphi_list + state_vec_list + diab_heating_list:
                # for symb in state_vec_list + diab_heating_list:

                #     self.sol_full[symb] = np.zeros_like(self.xrange, dtype=complex)
                #     self.sol_forced[symb] = np.zeros_like(self.xrange, dtype=complex)
                #     self.sol_modal[key][0][symb] = np.zeros_like(self.xrange, dtype=complex)
                #     self.Qc_sol_x0 = 0.0 + 0.0j

                #     # if symb in uphi_list:
                #     #     eval_dict = self.params_dict['general'] | {self.ksymb:self.kf} | self.forced_vec['conv']
                #     #     var_conv = uphi_dict[symb].subs(eval_dict)
                #     #     var_nconv = uphi_dict[symb].subs(self.params_dict['general'] | {self.ksymb:self.kf} | self.forced_vec['nconv'])
                    
                #     if symb in state_vec_list:
                #         var_conv = forced_sol_conv[symb]
                #         var_nconv = forced_sol_nconv[symb]
                        
                #     elif symb in diab_heating_list:
                #         var_conv = self.diabatic_terms[symb].subs(forced_sol_conv)  # col-integrated convective heating
                #         self.Qc_sol_x0 += var_conv * np.exp(self.kf * x0_nd)  # col-integrated convective heating
                #         # var_nconv = 0.0 + 0.0j
                    
                #     self.sol_full[symb][xconv] += np.array(var_conv * np.exp(argx), dtype = complex) 
                #     self.sol_forced[symb][xconv] = np.array(var_conv * np.exp(argx), dtype = complex) 
                #     self.sol_modal[key][0][symb][xconv] = np.array(var_conv * np.exp(argx), dtype = complex)

                #     if symb not in diab_heating_list:
                #         self.sol_full[symb][xnconv_forced] += np.array(var_nconv * np.exp(argx_nconv), dtype = complex) 
                #         self.sol_forced[symb][xnconv_forced] = np.array(var_nconv * np.exp(argx_nconv), dtype = complex) 
                #         self.sol_modal[key][0][symb][xnconv_forced] = np.array(var_nconv * np.exp(argx_nconv), dtype = complex)
            
            if key in ['conv', 'nconv']:

                # self.sol_modal[key] = {}   # conv or non-convective solutions

                xsubset = xconv if key == 'conv' else xnconv
                argx = (x0_nd - self.xrange[xsubset]/self.Lx) if key == 'nconv' else self.xrange[xsubset]/self.Lx

                for k in self.matching_coeffs[key].keys():
                    
                    indx = abs(k) - 1 
                    # self.sol_modal[key][k] = {}  # solution for each mode
                    ksol = np.sign(k) * np.sqrt(self.evals[key][indx])
                    T0_val = self.matching_coeffs[key][k]
                    evec = self.evecs[key][:, indx]
                    self.sol_full[:, xsubset] += evec[:, None] * T0_val * np.exp(ksol * argx) / evec[0]  # normalize to have first element = T0_val
            
            #         evec = self.convert_thermo_vec_to_full( evec, {self.ksymb**2 : self.evals[key][indx]} )
            #         # evec = self.eigen_vecs[key][k]
            #         # evec = {j : v.subs({self.base_var : T0_val}) for j, v in evec.items()} | {self.base_var : T0_val}

            #         for symb in state_vec_list + diab_heating_list:
            # #         for symb in uphi_list + state_vec_list + diab_heating_list:
                        
            #             self.sol_modal[key][k][symb] = np.zeros_like(self.xrange, dtype=complex)
                            
            #             # if symb in uphi_list:
            #             #     eval_dict = self.params_dict['general'] | {self.ksymb : ksol} | evec
            #             #     var = uphi_dict[symb].subs(eval_dict)

            #             if symb in state_vec_list:
            #                 var = evec[symb]

            #             elif (symb in diab_heating_list):
            #                 if key in ['conv']:
            #                     var = self.diabatic_terms[symb].subs(evec)
            #                     self.Qc_sol_x0 += var * np.exp(ksol * x0_nd)

            #                 # elif key in ['nconv'] and symb not in [self.symbols['Qc']]:  # no convective heating in the non-convective zone
            #                 #     var = self.diabatic_terms[symb].subs(evec)

                        # sol_exp = np.array( var * np.exp(ksol * argx), dtype = complex) 
                        # self.sol_full[symb][xsubset] += sol_exp
                        # self.sol_modal[key][k][symb][xsubset] = sol_exp


    # def assemble_solve_matching_problem(self):

    #     """
    #     Assemble equations for matching problem.
    #     """
    #     nderivative = self.nderivatives + 1
    #     Asymb = sp.symbols('A')
    #     Bsymb = sp.symbols('B')
    #     Msb = self.vert_struct_params['Msref'][0]

    #     for n in range(self.nlayers):
            
    #         for key in ['forced', 'conv', 'nconv']:
    #             sign = -1 if key in ['nconv'] else 1

    #             for d in range(nderivative):
                    
    #                 self.q_eqns[d][n].append(sign * self.q_x0[key][d][n])
    #                 self.T_eqns[d][n].append(sign * self.T_x0[key][d][n])
    #                 self.delta_eqns[d][n].append(sign * self.delta_x0[key][d][n])

    #                 # if np.mod(d, 2) == 1:  # odd derivatives at the origin
    #                 if key in ['conv', 'forced']: # symmetry at x=\pm x0
    #                     self.q_eqns_origin[d][n].append(self.q_x0[key][d][n] - self.q_origin[key][d][n])
    #                     self.T_eqns_origin[d][n].append(self.T_x0[key][d][n] - self.T_origin[key][d][n])
    #                     self.delta_eqns_origin[d][n].append(self.delta_x0[key][d][n] - self.delta_origin[key][d][n])

    #             self.phi_eqns[n].append(sign * self.phi_x0[key][n])
                
                    
    #             if n < self.nlayers - 1:
    #                 self.u_eqns[n].append(sign * self.u_x0[key][n])
                
    #             if n == 0:
    #                 self.Fs_eqns[n].append(self.Fs_domain[key])
    #                 self.Qc_eqns.append(self.Qc_x0[key])

 
    #         # expand matching equations
    #         eqn_list = [self.q_eqns, self.T_eqns, self.T_eqns_origin, self.q_eqns_origin, 
    #                     self.delta_eqns, self.delta_eqns_origin]

    #         for l in eqn_list:
    #             for d in l.keys():
    #                 try:
    #                     l[d][n] = sp.expand(sum(l[d][n]))
    #                 except KeyError:
    #                     pass

    #     # get vector-matrix  to represent the linear equations
    #     eqns = []
    #     vars = set()

    #     delta_eqn_list = [self.delta_eqns, self.delta_eqns_origin]

    #     if self.phys_params['Dq'] > 0:
            
    #         eqns_x0 = [self.T_eqns, self.q_eqns]
    #         eqns_origin = [self.T_eqns_origin, self.q_eqns_origin] 

    #     elif self.phys_params['Dq'] == 0:
    #         eqns_x0 = [self.T_eqns]
    #         eqns_origin = [self.T_eqns_origin] 

    #     for d in range(nderivative):

    #         for l in eqns_x0:
    #             for n in l[d].keys():

    #                 # skip last layer for delta equations, since the divergence in this layer is not independent
    #                 if (l in delta_eqn_list) and n == self.nlayers - 1:
    #                     continue

    #                 eqns.append(l[d][n])  
    #                 vars = vars | l[d][n].free_symbols
    #                 vars.discard(Asymb)
    #                 vars.discard(Bsymb)

    #         if d<=1:            
    #             for l in eqns_origin:
    #                 for n in l[d].keys():

    #                     if (l in delta_eqn_list) and n == self.nlayers - 1:
    #                         continue

    #                     eqns.append(l[d][n])  
    #                     vars = vars | l[d][n].free_symbols
    #                     vars.discard(Asymb)
    #                     vars.discard(Bsymb)

    #     self.Qc_eqns = sp.expand(sum(self.Qc_eqns))
    #     vars.discard(self.symbols['xc'])

    #     if len(eqns) < len(vars):
    #         print(f'Number of equations: {len(eqns)} less than the number of variables: {len(vars)}')
    #         eqns, vars = self.truncate_modes(eqns, vars)
    #         assert len(eqns) == len(vars), 'Number of equations must equal the number of variables'
    #     else:
    #         self.decay_scales_trunc = self.decay_scales

    #     print(f'Number of equations: {len(eqns)}; number of variables: {len(vars)}')
    #     assert len(eqns) >= len(vars), f'Number of equations ({len(eqns)}) less than the number of variables ({len(vars)})'

    #     # if overdetermined, reduce the number of equations
    #     if len(eqns) > len(vars):
    #         eqns = eqns[:len(vars)]

    #     vars = self.natural_sort( list(vars) )
    #     self.vars = vars
    #     self.eqns = eqns

    #     # A, b = sp.linear_eq_to_matrix(eqns, vars)
    #     sols, v  = self.get_Ab_x0(eqns, vars)  # For xc = x0, get sols and b
    #     if self.phys_params['solve_xc']:
    #         aux_eqn = self.Qc_eqns
    #         # aux_eqn = (self.delta_eqns[0][0] / self.delta_eqns[0][0].args[0].args[0]).simplify()
    #         initial_guess = sols.copy()
    #         initial_guess[-1] = initial_guess[-1] * 0.75  # change initial guess 
    #         sols_temp, v_temp = self.non_linear_solve_xc(initial_guess, eqns, v, aux_eqn)  # solution for xc != x0 using sols as initial guess
    #         x0_new = np.real( sols_temp[-1] * self.Lx )

    #         if x0_new < self.phys_params['x0']:
    #             print(f'New x0: {x0_new * 1e-3 : .2f} km')
    #             sols = sols_temp
    #             v = v_temp
    #         else:
    #             print(f'New x0: {x0_new * 1e-3:.2f} km is larger than the initial x0 {self.phys_params["x0"] * 1e-3 : .2f} km')
    #             print('Using x0 = x0')

    #     sol_dict = {k : v for k,v in zip(v, sols)}
    #     # sort dict by key
    #     # sol_dict = {k : sol_dict[k] for k in self.natural_sort(sol_dict.keys())}
    #     self.sol_dict = sol_dict 
    #     for k, sol in sol_dict.items():

    #         if k == self.symbols['xc']:
    #             print(f'{k} : {sol * self.Lx * 1e-3 : .2f} km')
    #             self.matching_coeffs[k] = sol

    #         else:
    #             if k.base == self.symbols['A']: 
    #                 conv_opt = 'conv' 
    #             elif k.base == self.symbols['B']:
    #                 conv_opt = 'nconv'

    #             kidx = k.indices[0]
    #             print(f'{k} : {sol * Msb : .2e} K')
    #             self.matching_coeffs[conv_opt][kidx] = sol

    #     # self.update_matching_coeffs_cc()
    #     print('--'*20)

    def update_matching_coeffs_cc(self):

        for conv_opt in ['conv', 'nconv']:

            # check and append complex conjugates or solutions with same scales
            if conv_opt == 'conv':
                dict_cc = self.dict_conv

            else:
                dict_cc = self.dict_nconv
    
            temp = {}
            for k in self.matching_coeffs[conv_opt].keys():
                if k in list(dict_cc.values()):
                    temp = temp | {k1 : self.matching_coeffs[conv_opt][k] for k1, v1 in dict_cc.items() if v1 == k}

            self.matching_coeffs[conv_opt] = self.matching_coeffs[conv_opt] | temp

    # def get_full_solution(self):

    #     """
    #     Get full solution in both the convective and non-convective zones
    #     """

    #     print(f'Getting full solution')

    #     x0_nd = self.matching_coeffs[ self.symbols['xc'] ] # dimensionless x0
    #     x0 = x0_nd * self.Lx
    #     x1_nd = self.phys_params['x0'] / self.Lx # dimensionless x0

    #     xconv = np.where(abs(self.xrange) <= x0)
    #     xnconv_forced = np.where(np.logical_and( abs(self.xrange) > x0, abs(self.xrange) <= self.phys_params['x0']) )  # a forced non-convective region
    #     xnconv = np.where(abs(self.xrange) > x0)

    #     uphi_list = list(self.uwind_dict.keys()) + list(self.phi_dict.keys())
    #     uphi_dict = self.uwind_dict | self.phi_dict
    #     state_vec_list = list(self.state_vec)
    #     diab_heating_list = [ self.symbols['Qc'] ]

    #     for key in ['forced', 'conv', 'nconv']:

    #         if key in ['forced']:

    #             self.sol_modal[key] = {0: {}}  # forced solution
    #             argx = self.kf * self.xrange[xconv]/self.Lx
    #             argx_nconv = self.kf * self.xrange[xnconv_forced]/self.Lx

    #             for symb in uphi_list + state_vec_list + diab_heating_list:

    #                 self.sol_full[symb] = np.zeros_like(self.xrange, dtype=complex)
    #                 self.sol_forced[symb] = np.zeros_like(self.xrange, dtype=complex)
    #                 self.sol_modal[key][0][symb] = np.zeros_like(self.xrange, dtype=complex)
    #                 self.Qc_sol_x0 = 0.0 + 0.0j

    #                 if symb in uphi_list:
    #                     eval_dict = self.params_dict['general'] | {self.ksymb:self.kf} | self.forced_vec['conv']
    #                     var_conv = uphi_dict[symb].subs(eval_dict)
    #                     var_nconv = uphi_dict[symb].subs(self.params_dict['general'] | {self.ksymb:self.kf} | self.forced_vec['nconv'])
                        
    #                 elif symb in state_vec_list:
    #                     var_conv = self.forced_vec['conv'][symb]
    #                     var_nconv = self.forced_vec['nconv'][symb]
                        
    #                 elif symb in diab_heating_list:
    #                     var_conv = self.diabatic_terms[symb].subs(self.forced_vec['conv'])  # col-integrated convective heating
    #                     self.Qc_sol_x0 += var_conv * np.exp(self.kf * x0_nd)  # col-integrated convective heating
    #                     # var_nconv = 0.0 + 0.0j
                    
    #                 self.sol_full[symb][xconv] += np.array(var_conv * np.exp(argx), dtype = complex) 
    #                 self.sol_forced[symb][xconv] = np.array(var_conv * np.exp(argx), dtype = complex) 
    #                 self.sol_modal[key][0][symb][xconv] = np.array(var_conv * np.exp(argx), dtype = complex)

    #                 if symb not in diab_heating_list:
    #                     self.sol_full[symb][xnconv_forced] += np.array(var_nconv * np.exp(argx_nconv), dtype = complex) 
    #                     self.sol_forced[symb][xnconv_forced] = np.array(var_nconv * np.exp(argx_nconv), dtype = complex) 
    #                     self.sol_modal[key][0][symb][xnconv_forced] = np.array(var_nconv * np.exp(argx_nconv), dtype = complex)
            
    #         if key in ['conv', 'nconv']:

    #             self.sol_modal[key] = {}   # conv or non-convective solutions

    #             xsubset = xconv if key == 'conv' else xnconv
    #             argx = (self.xrange[xsubset]/self.Lx - x0_nd) if key == 'nconv' else self.xrange[xsubset]/self.Lx

    #             for k in self.matching_coeffs[key].keys():
                    
    #                 self.sol_modal[key][k] = {}  # solution for each mode
    #                 ksol = self.free_sols[key][k]
    #                 T0_val = self.matching_coeffs[key][k]

    #                 evec = self.eigen_vecs[key][k]
    #                 evec = {j : v.subs({self.base_var : T0_val}) for j, v in evec.items()} | {self.base_var : T0_val}

    #                 for symb in uphi_list + state_vec_list + diab_heating_list:
                        
    #                     self.sol_modal[key][k][symb] = np.zeros_like(self.xrange, dtype=complex)
                            
    #                     if symb in uphi_list:
    #                         eval_dict = self.params_dict['general'] | {self.ksymb : ksol} | evec
    #                         var = uphi_dict[symb].subs(eval_dict)

    #                     elif symb in state_vec_list:
    #                         var = evec[symb]

    #                     elif (symb in diab_heating_list):
    #                         if key in ['conv']:
    #                             var = self.diabatic_terms[symb].subs(evec)
    #                             self.Qc_sol_x0 += var * np.exp(ksol * x0_nd)

    #                         # elif key in ['nconv'] and symb not in [self.symbols['Qc']]:  # no convective heating in the non-convective zone
    #                         #     var = self.diabatic_terms[symb].subs(evec)

    #                     sol_exp = np.array( var * np.exp(ksol * argx), dtype = complex) 
    #                     self.sol_full[symb][xsubset] += sol_exp
    #                     self.sol_modal[key][k][symb][xsubset] = sol_exp

    def linear_solve(self):
        print(f'creating coefficient matrices')
        super().create_coeff_matrices()
        super().assign_Dm_param()

        # get free solution matrices
        print(f'Getting coeff. matrices')
        self.Mdet = {}

        for k in ['conv', 'nconv']:
            print(k)
            self.Meval[k] = self.Mcoeff_all.subs(self.params_dict[k] | self.params_dict['general'] | self.params_Dm)

        # perturbation for free solutions
        self.base_var = self.Tvec[0]
        self.base_var_pert = 1.0 # perturbation for base variable
        self.get_free_solutions()
        # self.get_free_solution_vertical_profiles()
        self.get_forced_solutions()

        # matching problem
        self.get_matching_conditions()
        self.assemble_solve_matching_problem()
        self.get_full_solution()
        # self.project_forced(self.delta_profile)  # project the forced solution into a truncated free vector space

        

            # self.Mdet[k] = self.solve_homogeneous_matrix(self.Meval[k])
            # ksols, nsols = self.get_solutions(self.Mdet[k])
            # print(f'{nsols} solutions in {k} regime')

    def partial_solve(self):
        """
        Only obtain the free solutions
        """
        print(f'creating coefficient matrices')
        super().create_coeff_matrices()
        super().assign_Dm_param()

        # get free solution matrices
        print(f'Getting coeff. matrices')
        self.Mdet = {}

        for k in ['conv', 'nconv']:
            print(k)
            self.Meval[k] = self.Mcoeff_all.subs(self.params_dict[k] | self.params_dict['general'] | self.params_Dm)

        # perturbation for free solutions
        self.base_var = self.Tvec[0]
        self.base_var_pert = 1.0 # perturbation for base variable
        self.get_free_solutions()
        self.get_forced_solutions()


class AnalyticSolutions(LinearSolutions):
    def __init__(self, constants, phys_params, nlayers, pinterfaces, rad_fil, conv_fil, 
                 params_fil=params_file, arm_file=arm_file):
        super().__init__(constants, phys_params, nlayers, pinterfaces, rad_fil, conv_fil, params_fil, arm_file)

    def create_symbols(self):
        self.symbols['gamma'] = sp.IndexedBase('gamma', real = True)
        self.symbols['kappa'] = sp.IndexedBase('kappa', real = True)
        self.symbols['lamda'] = sp.symbols('lambda', real = True)

    @staticmethod
    def elim_row_col(M, row1, row2, col):
        Mtemp = M.copy()
        return Mtemp.elementary_row_op(op = 'n->n+km', row1 = row1, row2 = row2,  k = -Mtemp[row1, col]/Mtemp[row2, col]) 

    

    def main(self):
        self.create_symbols()
        self.create_coeff_matrices()

    






