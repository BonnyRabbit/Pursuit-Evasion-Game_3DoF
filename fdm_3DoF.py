import numpy as np
import math
from load_mdl import ModelLoader
from saturation import saturation
from utils import interp, clamp, DCM
from aero import AeroForceCalculator
from prop import PropulsionCalculator
from gym import spaces

class fdm_3Dof():
    
    def __init__(self):

        # Iniialize model
        self.model_loader = ModelLoader()
        model_data = self.model_loader.load_mdl()
        # Pre-process data
        self.data_aero = model_data['data_aero']
        self.data_prop = model_data['data_prop']
        self.var_aero = model_data['var_aero']
        self.grid_axes_aero = model_data['grid_axes_aero']
        self.var_prop = model_data['var_prop']
        self.grid_axes_prop = model_data['grid_axes_prop']

        """
        Params
        """
        self.dt_AP = 0.005
        self.mass = 200
        self.g = 9.8
        self.RefArea = self.data_aero['RefArea']
        self.RefLen = self.data_aero['RefLen']
        self.RefSpan = self.data_aero['RefSpan']
        """
        State
        """
        # [x, y, z, x_t, y_t, z_t, tas, alpha, beta, gamma, chi, chi_t, mu]
        # speed
        self.tas = None
        self.mach = None
        # position
        self.x = None
        self.y = None
        self.z = None
        self.x_t = None
        self.y_t = None
        self.z_t = None
        self.alt = None
        # attitude angle
        self.phi = None
        self.tht = None
        self.psi = None
        # aero angle
        self.alpha = None
        self.beta = None
        # track angle
        self.gamma = None
        self.chi = None
        self.chi_t = None
        self.mu = None
        """
        Action
        """
        # [ny, nz, thr]
        self.n_y = None
        self.n_z = None
        self.thr = None
        """
        Saturation
        """

        # Inititalize aero_cal
        self.aero_force = AeroForceCalculator(
            data_aero=self.data_aero,
            var_aero=self.var_aero,
            grid_axes_aero=self.grid_axes_aero,
        )
        # Initialize prop_cal
        self.prop_force = PropulsionCalculator(
            data_prop=self.data_prop,
            var_prop=self.var_prop,
            grid_axes_prop=self.grid_axes_prop
        )     
        
 
    def cal_aero_force(self):
        D, L, Y = self.aero_force.cal_aero_force(
            alpha=self.alpha, beta=self.beta, mach=self.mach, tas=self.tas, alt=self.alt, RefArea=self.RefArea
        )
        return D, L, Y

    def cal_prop_force(self):
        Thrust = self.prop_force.cal_prop(self.mach, self.alt, self.thr)
        return Thrust

    def cal_mass(self):
        pass
    
    
    def cal_mach(self):
        temp= 273.15
        gamma = 1.4
        R = 287.05
        T = temp - 0.0065 * self.alt
        c = math.sqrt(gamma * R * T)
        self.mach = self.tas / c

        return self.mach

    def DCM_gk(self):
        Ly_gamma = DCM('y',self.gamma)
        Lz_chi = DCM('z', self.chi)
        L_kg = Ly_gamma @ Lz_chi
        L_gk = np.linalg.inv(L_kg)

        return L_gk
    
    def DCM_ka(self):
        Lx_mu = DCM('x', self.mu)
        L_ka = np.linalg.inv(Lx_mu)

        return L_ka
    
    def take_action(self, action):
        # Actions saturation
        action_dim = action.shape()
        min_val, max_val = saturation(action[i])
        for i in range(action_dim):
            action[i] = clamp(action[i], min_val, max_val) # TODO


#     # Example usage
# fdm = fdm_3Dof()
# fdm.mach = 0.8
# fdm.alt = 1000  # meters
# fdm.thr = 0.75  # throttle setting

# Thrust = fdm.cal_prop_force()
# print(f"Thrust: {Thrust}")

# fdm = fdm_3Dof()
# fdm.alpha = 5.0
# fdm.beta = 0
# fdm.mach = 0.8
# fdm.tas = 250  # m/s
# fdm.alt = 1000  # meters

# D, L, Y = fdm.cal_aero_force()
# print(f"Drag: {D}, Lift: {L}, Side Force: {Y}")